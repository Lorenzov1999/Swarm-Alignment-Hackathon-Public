"""Conservative provenance from direct Git and file commands.

Only cached observations are parsed; nothing in a recorded command is executed.
Missing exit codes and ambiguous compound-command output stay unknown. Local
paths, branches, and abbreviated hashes are never treated as global identities.
"""
import collections
import posixpath
import re
import shlex
from urllib.parse import quote

from .provenance_common import command_segments, file_key, git_key, in_scope, make_event
from .traces import repo_from_ref, resolve

HEX = re.compile(r"^[0-9a-f]{7,40}$", re.I)
FULL = re.compile(r"^[0-9a-f]{40}$", re.I)
COMMIT = re.compile(r"(?m)^\[[^\]\n]+?\s+([0-9a-f]{7,40})\]\s", re.I)
HEAD = re.compile(r"\bHEAD is now at ([0-9a-f]{7,40})\b", re.I)
RANGE = re.compile(r"(?m)^Updating ([0-9a-f]{7,40})\.\.([0-9a-f]{7,40})\s*$", re.I)
ERROR = re.compile(r"(?mi)^(?:fatal:|error:|CONFLICT\b|Automatic merge failed|Aborting\b|error\[)")
PASS = re.compile(r"(?:\b[1-9]\d* passed\b|(?m:^OK$)|test result: ok|Tests:\s+\d+ passed)")
FAIL = re.compile(r"(?:\b[1-9]\d* failed\b|(?m:^FAILED\b)|test result: FAILED|(?m:^FAILURES\b))")
BUILD = re.compile(r"(?:\bbuilt in [\d.]+\s*(?:ms|s)\b|Successfully built\b|BUILD SUCCESSFUL|BUILD SUCCESS|Finished .* target\(s\))", re.I)
ANSI = r'\x1b\[[0-?]*[ -/]*[@-~]'
REBASE_DONE = re.compile(r'(?:^|[\r\n])(?:' + ANSI + r')*Successfully rebased and updated [^\r\n]+\.[ \t]*(?:[\r\n]|$)')
REBASE_NOOP = re.compile(r'(?:^|[\r\n])(?:' + ANSI + r')*Current branch [^\r\n]+ is up to date\.[ \t]*(?:[\r\n]|$)')
MUTATIONS = {'commit', 'merge', 'cherry-pick', 'rebase', 'revert', 'reset', 'checkout', 'switch', 'pull'}
UNKNOWN_CWD = '/__unobserved_cwd__/'


def _evidence(record, pattern=None):
    spans = []
    for field in ('o', 'e'):
        text = record.get(field) or ''
        if pattern is None:
            if text:
                spans.append({'field': field, 'start': 0, 'end': len(text)})
        else:
            for match in pattern.finditer(text):
                spans.append({'field': field, 'start': match.start(), 'end': match.end()})
    return spans or [{'field': 'x', 'start': 0, 'end': len(record.get('x') or '')}]


def _head(argv):
    return posixpath.basename(argv[0]) if argv else ''


def _git(argv, cwd):
    """Return direct git (directory, verb, arguments), ignoring harmless globals."""
    if _head(argv) != 'git':
        return None
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg in ('-C', '-c') and index + 1 < len(argv):
            if arg == '-C':
                cwd = resolve(cwd, argv[index + 1])
                if cwd is None:
                    return None
            index += 2
        elif arg in ('--no-pager', '--no-optional-locks', '--literal-pathspecs'):
            index += 1
        else:
            break
    if index == len(argv) or argv[index].startswith('-'):
        return None
    return cwd, argv[index], argv[index + 1:]


def _redirect(argv):
    """Recognize one literal stdout destination; do not interpret expansion."""
    for index, arg in enumerate(argv):
        if arg in ('>', '>>') and index + 1 < len(argv):
            return index, argv[index + 1]
        if arg.startswith('>') and len(arg.lstrip('>')) > 0:
            return index, arg.lstrip('>')
    return None, None


def _literal(value):
    return bool(value) and not any(c in value for c in '$`*?[]{}\n\r')


def _positionals(args, takes_value=()):
    result, index = [], 0
    while index < len(args):
        arg = args[index]
        if arg == '--':
            break
        if arg in takes_value:
            index += 2
            continue
        if not arg.startswith('-'):
            result.append(arg)
        index += 1
    return result


def _direct_rebase_ref(args):
    harmless = {'--quiet', '-q', '--verbose', '-v', '--stat', '--no-stat'}
    if any(arg.startswith('-') and arg not in harmless for arg in args):
        return None
    refs = [arg for arg in args if arg not in harmless]
    return refs[0] if len(refs) == 1 else None


def _raw_segments(command):
    """Keep quoting for redirection checks; never execute shell text."""
    parts, current, quoted, escaped = [], [], None, False
    for char in command:
        if escaped:
            current.append(char)
            escaped = False
        elif char == '\\' and quoted != "'":
            current.append(char)
            escaped = True
        elif quoted:
            current.append(char)
            if char == quoted:
                quoted = None
        elif char in "\"'":
            current.append(char)
            quoted = char
        elif char in ';&|\n':
            if ''.join(current).strip():
                parts.append(''.join(current))
            current = []
        else:
            current.append(char)
    if ''.join(current).strip():
        parts.append(''.join(current))
    return parts


def _shell_surface(command):
    """Remove quoted/escaped text before checking control operators."""
    result, quoted, escaped = [], None, False
    for char in command:
        if escaped:
            escaped = False
            result.append(' ')
        elif char == '\\' and quoted != "'":
            escaped = True
            result.append(' ')
        elif quoted:
            if char == quoted:
                quoted = None
            result.append(' ')
        elif char in "\"'":
            quoted = char
            result.append(' ')
        else:
            result.append(char)
    return ''.join(result)


def _actual_redirect(argv, raw):
    """A quoted '>file' argument is not a shell redirection."""
    quoted, escaped = None, False
    for index, char in enumerate(raw):
        if escaped:
            escaped = False
        elif char == '\\' and quoted != "'":
            escaped = True
        elif quoted:
            if char == quoted:
                quoted = None
        elif char in "\"'":
            quoted = char
        elif char == '#':
            break
        elif char == '>':
            if index and raw[index - 1] in '0123456789&>':
                continue
            tail = raw[index + 1:].lstrip('>')
            try:
                targets = shlex.split(tail, comments=True)
            except ValueError:
                return None, None
            if targets:
                candidate = _redirect(argv)
                if candidate[1] == targets[0]:
                    return candidate
            return None, None
    return None, None


def _test_runner(argv):
    head = _head(argv)
    if head in ('pytest', 'py.test'):
        return 'test', 'pytest'
    if head.startswith('python') and argv[1:3] in (['-m', 'pytest'], ['-m', 'unittest']):
        return 'test', argv[2]
    if head in ('cargo', 'go') and len(argv) > 1 and argv[1] in ('test', 'build'):
        return ('test' if argv[1] == 'test' else 'build'), head + ':' + argv[1]
    if head in ('npm', 'yarn', 'pnpm'):
        args = argv[1:]
        if args[:1] == ['run']:
            args = args[1:]
        if args and args[0] in ('test', 'build', 'check'):
            return ('build' if args[0] == 'build' else 'test'), head + ':' + args[0]
    if head in ('make', 'mvn', 'gradle', 'gradlew') and len(argv) > 1:
        if argv[1] in ('test', 'check', 'build', 'package'):
            return ('test' if argv[1] in ('test', 'check') else 'build'), head + ':' + argv[1]
    return None


class GitExtractor:
    """State is per actor/session; remote-qualified entities can cross actors."""

    def __init__(self, actor):
        self.actor = actor
        self.cwd = {}
        self.roots = {}
        self.heads = {}
        self.clean = {}
        self.refs = {}
        self.record_repo = None
        self.record_session = None

    def _root(self, path, session):
        while path and path != '/':
            if (session, path) in self.roots:
                return path
            path = posixpath.dirname(path)
        return None

    def repo_for_path(self, path, session):
        root = self._root(path, session)
        return self.roots.get((session, root))

    def current_repo(self, session):
        if self.record_repo and str(session) == self.record_session:
            return self.record_repo
        return self.repo_for_path(self.cwd.get(session, '/home/computeruse'), session)

    def _location(self, path, session):
        root = self._root(path, session) or path
        repo = self.roots.get((session, root))
        if repo is None:
            repo = 'local:' + quote(self.actor, safe='') + ':' + quote(str(session), safe='') + ':' + root
        return repo, root, (session, root)

    def _learn(self, path, session, repo):
        if path and repo and not path.startswith(UNKNOWN_CWD):
            self.roots[(session, path)] = repo

    def _revision(self, value, state):
        if HEX.fullmatch(value or ''):
            return value.lower()
        if value == 'HEAD':
            return self.heads.get(state)
        return self.refs.get((state, value))

    def _event(self, record, operation, status, repo, sha=None, **kwargs):
        version = git_key(repo, sha) if sha else None
        return make_event(self.actor, record, 'git', operation, status, repo,
                          version=version, abbreviated=bool(sha and len(sha) < 40),
                          relation_authority=status == 'success',
                          evidence=kwargs.pop('evidence', _evidence(record)), **kwargs)

    def extract(self, record):
        self.record_repo, self.record_session = None, None
        if not in_scope(record['t']) or record.get('k') != 'shell':
            return []
        command = record.get('x') or ''
        segments = command_segments(command)
        if not segments:
            return []
        session = str(record.get('s', ''))
        cwd = self.cwd.get(session, '/home/computeruse')
        output = (record.get('o') or '') + '\n' + (record.get('e') or '')
        failed = bool(ERROR.search(output)) or record.get('exit_code', record.get('returncode')) not in (None, 0)
        rc_zero = record.get('exit_code', record.get('returncode')) == 0
        surface = _shell_surface(command).strip()
        linear = '||' not in surface and ';' not in surface and '\n' not in surface
        raw_parts = _raw_segments(command)
        raw_by_argv = {id(argv): raw for argv, raw in zip(segments, raw_parts)} if len(raw_parts) == len(segments) else {}
        entries = []
        cd_contexts = []
        for argv in segments:
            if _head(argv) == 'cd' and len(argv) == 2:
                target = resolve(cwd, argv[1])
                cd_contexts.append(target)
                # Failed cd or alternative branches must not overwrite context.
                confirmed_context = linear and ('&&' in surface or rc_zero)
                cd_failed = bool(re.search(r'(?m)(?:^|: )cd: ', output))
                if not cd_failed:
                    if target and confirmed_context:
                        cwd = target
                    else:
                        # A possible directory change invalidates the old known
                        # repo/HEAD. It does not establish a guessed real path.
                        cwd = UNKNOWN_CWD + str(record['i']) + '/' + str(len(cd_contexts))
                continue
            entries.append((argv, cwd, _git(argv, cwd)))
        self.cwd[session] = cwd
        git_entries = [entry for entry in entries if entry[2] is not None]
        verb_counts = collections.Counter(entry[2][1] for entry in git_entries)
        mutation_count = sum(verb_counts[v] for v in MUTATIONS)
        # Other output-producing programs could print a fake commit/merge marker.
        git_output_safe = all(git is not None for _, _, git in entries)
        events = []
        transport_repo = None
        transport_dir = None
        transport_prior = None
        git_dirs = {entry[2][0] for entry in git_entries}
        # A receipt identifies the actual repository even when a prior cd is
        # uncertain, provided every direct Git operation has one common context.
        # Custom-formatted log/show output could impersonate such a receipt.
        safe_verbs = {'add', 'commit', 'push', 'fetch', 'pull', 'status', 'rev-parse', 'config', 'diff', 'log', 'rebase'}
        safe_receipt_context = (git_output_safe and not failed and len(git_dirs) == 1
                                and len(cd_contexts) <= 1
                                and all(g[1] in safe_verbs for _, _, g in git_entries)
                                and not any(g[1] == 'log' and any(a.startswith(('--format', '--pretty')) for a in g[2])
                                            or g[1] == 'diff' and any(a.startswith('--word-diff') for a in g[2])
                                            or g[1] in ('push', 'fetch', 'pull') and any(a in ('-n', '--dry-run') for a in g[2])
                                            or g[1] == 'config' and any('remote.' in a for a in g[2])
                                            or g[1] == 'config' and not (
                                                len([a for a in g[2] if a not in ('--global', '--local')]) == 2
                                                and [a for a in g[2] if a not in ('--global', '--local')][0] in ('user.name', 'user.email'))
                                            or g[1] == 'rebase' and _direct_rebase_ref(g[2]) is None
                                            for _, _, g in git_entries))
        if safe_receipt_context:
            receipts = []
            if verb_counts['push'] == 1 and re.search(r'(?m)^\s+\S.* -> \S', output):
                receipts.extend(re.findall(r'(?m)^To (\S+)', output))
            if verb_counts['fetch'] + verb_counts['pull'] == 1:
                receipts.extend(re.findall(r'(?m)^From (\S+)', output))
            repos = {repo_from_ref(url) for url in receipts} - {None}
            if len(repos) == 1:
                transport_repo = next(iter(repos))
                transport_dir = next(iter(git_dirs))
                old_repo, _, old_state = self._location(transport_dir, session)
                transport_prior = (old_repo, self.heads.get(old_state))
                self.record_repo, self.record_session = transport_repo, session
                transport_args = next(g[2] for _, _, g in git_entries if g[1] in ('push', 'pull', 'fetch'))
                remote_args = [a for a in transport_args if not a.startswith('-')]
                if (not remote_args or remote_args[0] == 'origin') and self.repo_for_path(transport_dir, session) is None:
                    self._learn(transport_dir, session, transport_repo)

        # Read remote identity before processing commands that need repo context.
        if len(git_entries) == 1 and git_output_safe and not failed:
            directory, verb, args = git_entries[0][2]
            if verb == 'remote' and (args[:1] == ['-v'] or args[:1] == ['get-url']):
                urls = []
                if args[:1] == ['-v']:
                    urls = re.findall(r'(?m)^origin\s+(\S+)\s+\((?:fetch|push)\)', output)
                elif args[1:2] == ['origin']:
                    urls = [line.strip() for line in output.splitlines() if line.strip()]
                repos = {repo_from_ref(url) for url in urls} - {None}
                if len(repos) == 1:
                    self._learn(directory, session, next(iter(repos)))

        for argv, directory, git in entries:
            if git is None:
                if (_head(argv) in ('cp', 'mv', 'rm', 'touch', 'tee', 'rsync') or _actual_redirect(argv, raw_by_argv.get(id(argv), ''))[1]
                        or _head(argv) == 'sed' and '-i' in argv
                        or _head(argv).startswith('python') and _test_runner(argv) is None):
                    _, _, state = self._location(directory, session)
                    self.clean[state] = None
                runner = _test_runner(argv)
                if runner:
                    unambiguous = all(other is argv or (g is not None and g[1] in ('status', 'rev-parse', 'add'))
                                      for other, _, g in entries)
                    events.extend(self._tests(record, runner, directory, session, output,
                                              unambiguous,
                                              failed, rc_zero and linear))
                # An unmodelled program may edit tracked inputs. A later test
                # needs a fresh clean-status observation before citing HEAD.
                readonly = {'pwd', 'ls', 'cat', 'head', 'tail', 'grep', 'rg', 'stat', 'wc', 'sha256sum', 'md5sum', 'echo', 'printf'}
                if _head(argv) not in readonly or runner:
                    _, _, state = self._location(directory, session)
                    self.clean[state] = None
                continue
            directory, verb, args = git
            repo, root, state = self._location(directory, session)
            local_repo = repo
            if transport_repo and directory == transport_dir:
                repo = transport_repo
            unique = verb_counts[verb] == 1 and git_output_safe
            safe_mutation = unique and mutation_count == 1
            status = 'failed' if failed else 'unknown'
            prior = self.heads.get(state)
            prior_clean = self.clean.get(state)
            if verb in MUTATIONS:
                self.heads[state] = None
                self.clean[state] = None
                # A named ref observed before a mutation may have moved.
                for ref_key in [key for key in self.refs if key[0] == state]:
                    self.refs.pop(ref_key)

            if verb == 'remote' and args[:1] in (['add'], ['set-url'], ['remove'], ['rm'], ['rename']):
                if len(args) >= 2 and args[1] == 'origin':
                    self.roots.pop((session, root), None)
                    if args[0] in ('add', 'set-url') and len(args) >= 3 and rc_zero and linear and not failed:
                        self._learn(root, session, repo_from_ref(args[2]))
                continue

            if verb == 'clone':
                url = next((x for x in args if x.startswith(('https://', 'http://', 'git@'))), None)
                remote = repo_from_ref(url) if url else None
                if not remote:
                    continue
                index = args.index(url)
                target = args[index + 1] if len(args) > index + 1 and not args[index + 1].startswith('-') else remote.rsplit('/', 1)[-1]
                destination = resolve(directory, target)
                marker = re.search(r'(?:Receiving objects:.*done\.|Resolving deltas:.*done\.|Updating files:.*done\.|warning: You appear to have cloned an empty repository\.)', output)
                status = 'success' if unique and not failed and (marker or (rc_zero and linear)) else status
                if status == 'success':
                    self._learn(destination, session, remote)
                events.append(self._event(record, 'acquire', status, remote, method='clone', resolved_destination=destination))
                continue

            if verb in ('remote', 'add', 'diff', 'ls-files', 'merge-base'):
                continue

            if verb in ('apply', 'am', 'rm', 'mv', 'clean', 'restore', 'update-index'):
                self.clean[state] = None
                if verb == 'am':
                    self.heads[state] = None
                continue

            if verb == 'rev-parse' and unique and not failed:
                hashes = [line.strip().lower() for line in output.splitlines() if FULL.fullmatch(line.strip())]
                refs = [arg for arg in args if not arg.startswith('-')]
                if len(hashes) == 1 and len(refs) == 1:
                    sha = hashes[0]
                    ref = refs[0]
                    self.refs[(state, ref)] = sha
                    aliases = []
                    if ref == 'HEAD':
                        if prior and sha.startswith(prior) and prior != sha:
                            aliases.append({'source': git_key(repo, prior), 'target': git_key(repo, sha), 'kind': 'same_content'})
                        self.heads[state] = sha
                    events.append(self._event(record, 'observe', 'success', repo, sha,
                                              relations=aliases, method='rev-parse', ref=ref))
                continue

            if verb == 'status' and unique and len(git_entries) == 1 and not failed:
                if re.search(r'(?m)^nothing to commit, working (?:tree|directory) clean\s*$', output):
                    self.clean[state] = True
                elif any(x in args for x in ('--porcelain', '--porcelain=v1', '--porcelain=v2', '--short', '-s')):
                    if output.strip():
                        self.clean[state] = False
                    elif rc_zero and linear:
                        self.clean[state] = True
                elif re.search(r'Changes (?:not staged|to be committed)|Untracked files', output):
                    self.clean[state] = False
                continue

            if verb == 'commit':
                match = COMMIT.search(output)
                sha = match[1].lower() if match and safe_mutation and not failed else None
                if sha:
                    status = 'success'
                    key = git_key(repo, sha)
                    inputs = [git_key(repo, prior)] if prior else []
                    relations = [{'kind': 'revision_of' if '--amend' in args else 'derived_from',
                                  'source': key, 'target': old} for old in inputs]
                    self.heads[state] = sha
                    self.clean[state] = None
                    events.append(self._event(record, 'generate', status, repo, sha,
                                              inputs=inputs, outputs=[key], relations=relations,
                                              method='commit', evidence=_evidence(record, COMMIT)))
                else:
                    events.append(self._event(record, 'generate', status, repo, method='commit'))
                continue

            if verb in ('checkout', 'switch', 'reset'):
                values = _positionals(args, ('-b', '-B', '--orphan', '-c', '-C', '--conflict'))
                ref = values[0] if values else None
                sha = prior if ref == 'HEAD' else self._revision(ref, state)
                match = HEAD.search(output) if unique and not failed else None
                if match and safe_mutation:
                    reported = match[1].lower()
                    sha = sha if sha and sha.startswith(reported) else reported
                    status = 'success'
                elif safe_mutation and not failed and re.search(r"(?m)^(?:Switched to|Already on|HEAD is now at)\b", output):
                    status = 'success'
                elif safe_mutation and not failed and rc_zero and linear:
                    status = 'success'
                if '--' in args:
                    # File checkout/reset is a workspace edit, not HEAD adoption.
                    self.clean[state] = None
                    sha = None
                elif status == 'success':
                    self.heads[state] = sha
                    self.clean[state] = None
                events.append(self._event(record, 'acquire', status, repo, sha,
                                          inputs=[git_key(repo, sha)] if sha and status == 'success' else [],
                                          method=verb))
                continue

            if verb in ('merge', 'cherry-pick', 'pull'):
                if any(x in args for x in ('--abort', '--quit')):
                    self.clean[state] = None
                    continue
                if verb == 'pull' and any(arg.startswith('--rebase') or arg == '-r' for arg in args):
                    explicit = (len(args) == 3 and args[0] == '--rebase' and args[1] == 'origin'
                                and bool(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_./-]*', args[2])))
                    context = (safe_mutation and len(git_dirs) == 1 and len(cd_contexts) <= 1
                               and all(g[1] in ('pull', 'push', 'status', 'rev-parse') for _, _, g in git_entries))
                    incoming = re.findall(r'(?m)^\s*([a-f0-9]{7,40})\.\.([a-f0-9]{7,40})\s+(\S+)\s+->\s+(origin/\S+)[ \t]*$', output, re.I)
                    from_urls = re.findall(r'(?m)^From (\S+)', output)
                    receipt = bool(transport_repo and len(from_urls) == 1 and repo_from_ref(from_urls[0]) == transport_repo)
                    branch_match = bool(explicit and len(incoming) == 1 and incoming[0][2] == args[2]
                                        and incoming[0][3] == 'origin/' + args[2])
                    sha = incoming[0][1].lower() if branch_match and receipt else None
                    rebase_failed = failed or bool(ERROR.search(re.sub(ANSI, '', output).replace('\r', '\n')))
                    noop = bool(REBASE_NOOP.search(output) or re.search(r'Already up[ -]to[ -]date', output, re.I))
                    confirmed = bool(context and explicit and receipt and sha and REBASE_DONE.search(output) and not rebase_failed)
                    unchanged = bool(context and explicit and noop and not rebase_failed)
                    if unchanged:
                        self.heads[state], self.clean[state] = prior, prior_clean
                    status = 'success' if confirmed or unchanged else 'failed' if rebase_failed else 'unknown'
                    events.append(self._event(record, 'observe' if unchanged else 'incorporate', status,
                                              repo, sha, inputs=[git_key(repo, sha)] if confirmed and not unchanged else [],
                                              method='pull-rebase', no_change=unchanged,
                                              tracked_ref='origin/' + args[2] if explicit else None))
                    continue
                values = _positionals(args, ('-m', '--message', '-F', '--file', '-s', '--strategy', '-X', '--strategy-option', '--mainline'))
                ref = values[0] if values else None
                sha = prior if ref == 'HEAD' else self._revision(ref, state)
                noop = bool(re.search(r'Already up[ -]to[ -]date', output, re.I))
                fast = RANGE.search(output)
                made = bool(re.search(r"Merge made by|Automatic merge went well", output))
                created = COMMIT.search(output) if verb == 'cherry-pick' else None
                confirmed = safe_mutation and not failed and (noop or made or (fast and 'Fast-forward' in output) or created)
                new_sha = None
                if confirmed:
                    status = 'success'
                    if fast and not sha:
                        sha = fast[2].lower()
                    if created:
                        new_sha = created[1].lower()
                    elif fast:
                        new_sha = sha or fast[2].lower()
                    elif made and verb_counts['rev-parse'] == 1:
                        full_lines = [line.strip().lower() for line in output.splitlines() if FULL.fullmatch(line.strip())]
                        if len(full_lines) == 1:
                            new_sha = full_lines[0]
                    if not noop:
                        self.heads[state] = new_sha
                        self.clean[state] = None
                    else:
                        self.heads[state] = prior
                        self.clean[state] = prior_clean
                original = git_key(repo, sha) if sha else None
                result = git_key(repo, new_sha) if new_sha else None
                outputs = [result] if confirmed and not noop and result and (created or (made and not fast)) else []
                relations = [{'kind': 'derived_from', 'source': result, 'target': original}] if outputs and original else []
                events.append(self._event(record, 'observe' if noop else 'incorporate', status, repo,
                                          new_sha or sha, inputs=[original] if original and confirmed and not noop else [],
                                          outputs=outputs, relations=relations, method=verb, no_change=noop))
                continue

            if verb == 'fetch':
                # Acquisition of object data is distinct from checkout or incorporation.
                status = 'success' if unique and not failed and (rc_zero and linear or re.search(r'(?m)^From \S+', output)) else status
                hashes = []
                if status == 'success':
                    hashes = sorted(set(re.findall(r'\b[0-9a-f]{7,40}\.\.([0-9a-f]{7,40})\b', output, re.I)))
                if '--update-head-ok' in args:
                    self.heads[state] = None
                    self.clean[state] = None
                events.append(self._event(record, 'acquire', status, repo,
                                          inputs=[git_key(repo, h) for h in hashes], method='fetch',
                                          incorporation=False))
                continue

            if verb == 'push':
                status = 'success' if unique and not failed and (rc_zero and linear or re.search(r'(?m)^To \S+', output) and ' -> ' in output) else status
                hashes = sorted(set(re.findall(r'\b[0-9a-f]{7,40}\.\.([0-9a-f]{7,40})\b', output, re.I)))
                aliases = []
                old_repo, old_head = transport_prior or (local_repo, prior)
                if status == 'success' and transport_repo and mutation_count == 0 and old_repo.startswith('local:') and old_head:
                    if any(old_head.startswith(sha) or sha.startswith(old_head) for sha in hashes):
                        aliases.append({'kind': 'same_content', 'source': git_key(old_repo, old_head),
                                        'target': git_key(repo, old_head)})
                events.append(self._event(record, 'publish', status, repo,
                                          inputs=[git_key(repo, h) for h in hashes] if status == 'success' else [],
                                          relations=aliases, method='push'))
                continue

            if verb == 'show':
                spec = next((x for x in args if not x.startswith('-') and ':' in x), None)
                if spec:
                    ref, path = spec.split(':', 1)
                    sha = self._revision(ref, state)
                    # Successful nonempty Git show output confirms the explicit read;
                    # an empty file requires an exit code, absent from most cache rows.
                    status = 'success' if unique and len(git_entries) == 1 and not failed and (output.strip() or rc_zero and linear) else status
                    version = 'gitfile:' + repo + ':' + quote(path, safe='/._-') + '@' + sha if sha else None
                    events.append(make_event(self.actor, record, 'file', 'acquire', status,
                                             'file:' + repo + ':' + path, version=version,
                                             inputs=[git_key(repo, sha)] if sha and status == 'success' else [],
                                             evidence=_evidence(record), method='git-show', repository=repo,
                                             path=path, ref=ref, abbreviated=bool(sha and len(sha) < 40)))
                else:
                    sha = next((x.lower() for x in args if HEX.fullmatch(x)), None)
                    events.append(self._event(record, 'observe', 'success' if unique and not failed and output.strip() else status,
                                              repo, sha, method='show-listing'))
                continue

            if verb in ('log', 'reflog'):
                # These identifiers are observations, never acquisition or production.
                hashes = sorted(set(re.findall(r'(?m)^(?:[|* /\\]*)(?:commit )?([0-9a-f]{7,40})\b', output, re.I))) if unique else []
                for sha in hashes[:100]:
                    events.append(self._event(record, 'observe', 'success' if not failed else 'failed', repo, sha, method=verb))
                continue

            if verb == 'rebase':
                ref = _direct_rebase_ref(args)
                sha = ref.lower() if ref and HEX.fullmatch(ref) else prior if ref == 'HEAD' else None
                allowed = {'rebase', 'push', 'fetch', 'status', 'rev-parse'}
                context = (safe_mutation and len(git_dirs) == 1 and len(cd_contexts) <= 1
                           and all(g[1] in allowed for _, _, g in git_entries))
                rebase_failed = failed or bool(ERROR.search(re.sub(ANSI, '', output).replace('\r', '\n')))
                done = bool(REBASE_DONE.search(output))
                noop = bool(REBASE_NOOP.search(output))
                confirmed = bool(context and sha and not rebase_failed and (done or noop))
                status = 'success' if confirmed else 'failed' if rebase_failed else 'unknown'
                new_sha = None
                outputs, relations = [], []
                if confirmed and noop:
                    self.heads[state], self.clean[state] = prior, prior_clean
                elif confirmed:
                    rebasing_index = next(i for i, entry in enumerate(git_entries) if entry[2][1] == 'rebase')
                    verifier = [i for i, entry in enumerate(git_entries)
                                if entry[2][1] == 'rev-parse' and _positionals(entry[2][2]) == ['HEAD']]
                    hashes = [line.strip().lower() for line in output.splitlines() if FULL.fullmatch(line.strip())]
                    if len(verifier) == 1 and verifier[0] > rebasing_index and len(hashes) == 1:
                        new_sha = hashes[0]
                        self.heads[state] = new_sha
                        if new_sha not in (sha, prior):
                            outputs = [git_key(repo, new_sha)]
                            relations = [{'kind': 'derived_from', 'source': outputs[0], 'target': git_key(repo, sha)}]
                events.append(self._event(record, 'observe' if confirmed and noop else 'incorporate', status,
                                          repo, new_sha or sha,
                                          inputs=[git_key(repo, sha)] if confirmed and not noop else [],
                                          outputs=outputs, relations=relations, method='rebase', no_change=bool(confirmed and noop)))
                continue

            if verb == 'revert':
                # Complex history rewriting needs operation-specific attribution.
                self.heads[state] = None
                self.clean[state] = None
                events.append(self._event(record, 'incorporate', status, repo, method=verb))

        events.extend(self._files(record, entries, session, output, failed, linear, raw_by_argv))
        # Any direct local write makes previously clean-worktree evidence stale.
        for argv, directory, git in entries:
            if git is None and (_head(argv) in ('cp', 'mv', 'rm', 'touch', 'tee', 'rsync') or _redirect(argv)[1]
                                or _head(argv) == 'sed' and '-i' in argv
                                or _head(argv).startswith('python') and _test_runner(argv) is None):
                _, _, state = self._location(directory, session)
                self.clean[state] = None
        return events

    def _tests(self, record, runner, directory, session, output, unique, failed, rc_zero):
        repo, root, state = self._location(directory, session)
        kind, name = runner
        status = 'failed' if failed or FAIL.search(output) else 'unknown'
        if unique and status != 'failed' and ((PASS if kind == 'test' else BUILD).search(output) or rc_zero):
            status = 'success'
        sha = self.heads.get(state)
        exact = bool(sha and self.clean.get(state) is True)
        result = 'result:' + quote(self.actor, safe='') + ':' + str(record['i']) + ':' + name
        inputs = [git_key(repo, sha)] if exact and status == 'success' else []
        return [make_event(self.actor, record, 'test' if kind == 'test' else 'build',
                           'test' if kind == 'test' else 'transform', status,
                           repo + ':' + name, inputs=inputs,
                           outputs=[result] if status == 'success' else [],
                           relations=[{'kind': 'derived_from', 'source': result, 'target': value} for value in inputs],
                           evidence=_evidence(record), method=name, observed_head=sha,
                           relation_authority=status == 'success',
                           working_tree_confirmed_clean=self.clean.get(state) is True)]

    def _file_identity(self, path, directory, session, digest, algorithm):
        absolute = resolve(directory, path)
        if absolute is None:
            return None
        repo, root, state = self._location(directory, session)
        relative = posixpath.relpath(absolute, root)
        # A path escaping a known checkout must retain its local namespace.
        if relative == '..' or relative.startswith('../'):
            repo = 'local:' + quote(self.actor, safe='') + ':' + quote(session, safe='')
            relative = absolute
        return absolute, 'file:' + repo + ':' + relative, file_key(repo, relative, digest, algorithm)

    def _files(self, record, entries, session, output, failed, linear, raw_by_argv):
        hash_commands = [(index, argv, directory) for index, (argv, directory, _) in enumerate(entries)
                         if _head(argv) in ('sha256sum', 'md5sum')]
        if len(hash_commands) != 1 or failed:
            return []
        # No arbitrary output-producing command may impersonate checksum output.
        permitted = {'sha256sum', 'md5sum', 'cp', 'sort', 'cat', 'head', 'tail', 'cut', 'printf', 'echo'}
        if any(_head(argv) not in permitted for argv, _, _ in entries):
            return []
        if any(_head(argv) in ('printf', 'echo', 'cat', 'sort', 'head', 'tail', 'cut') and not _actual_redirect(argv, raw_by_argv.get(id(argv), ''))[1]
               for argv, _, _ in entries):
            return []
        snapshots, ambiguous = {}, set()
        for command_index, argv, directory in hash_commands:
            algorithm = 'sha256' if _head(argv) == 'sha256sum' else 'md5'
            length = 64 if algorithm == 'sha256' else 32
            requested = {resolve(directory, arg) for arg in argv[1:] if not arg.startswith('-') and _literal(arg)}
            pattern = re.compile(r'(?m)^([0-9a-f]{' + str(length) + r'}) [ *](.+)$', re.I)
            for match in pattern.finditer(output):
                digest, path = match[1].lower(), match[2]
                identity = self._file_identity(path, directory, session, digest, algorithm)
                if identity and identity[0] in requested:
                    absolute, entity, version = identity
                    if absolute in snapshots and snapshots[absolute][1] != version:
                        ambiguous.add(absolute)
                    snapshots[absolute] = (entity, version, 'content:' + algorithm + ':' + digest, algorithm, digest, pattern, command_index)
        for absolute in ambiguous:
            snapshots.pop(absolute, None)
        if not snapshots:
            return []
        generated = {}
        if linear and '&&' in (record.get('x') or ''):
            for command_index, (argv, directory, _) in enumerate(entries):
                head = _head(argv)
                source = None
                index, destination = _actual_redirect(argv, raw_by_argv.get(id(argv), ''))
                if head == 'cp' and len(argv) == 3 and all(_literal(x) and not x.startswith('-') for x in argv[1:]):
                    source, destination = argv[1:]
                elif destination and head in ('cat', 'sort', 'head', 'tail', 'cut'):
                    candidates = [x for x in argv[1:index] if _literal(x) and not x.startswith('-')]
                    if len(candidates) == 1:
                        source = candidates[0]
                elif not (destination and head in ('printf', 'echo')):
                    continue
                target = resolve(directory, destination) if destination and _literal(destination) else None
                if target in snapshots and command_index < snapshots[target][6]:
                    generated[target] = (head, resolve(directory, source) if source else None)
        events = []
        for absolute, (entity, version, content, algorithm, digest, pattern, _) in snapshots.items():
            maker = generated.get(absolute)
            inputs = []
            if maker and maker[1] in snapshots:
                inputs = [snapshots[maker[1]][1], snapshots[maker[1]][2]]
            operation = 'transform' if maker and inputs else 'generate' if maker else 'acquire'
            outputs = [version, content] if maker else []
            if not maker:
                inputs = [version, content]
            relations = [{'kind': 'same_content', 'source': version, 'target': content}]
            if maker:
                relations.extend({'kind': 'derived_from', 'source': version, 'target': value} for value in inputs)
            events.append(make_event(self.actor, record, 'file', operation, 'success', entity,
                                     version=version, inputs=inputs, outputs=outputs, relations=relations,
                                     evidence=_evidence(record, pattern), content_key=content,
                                     relation_authority=True,
                                     hash_algorithm=algorithm, method=maker[0] if maker else 'checksum-read'))
        return events
