"""Layer 1 traces shared by ordinary AI-agent computer work.

These are observable trace features, not confirmed provenance links. Standard
commands, URLs, local files and immutable pins need no knowledge of a team's
workflow. Paths without a verified shared repository remain actor/session scoped;
their basenames are emitted separately as explicitly weak candidate features.
"""
import posixpath
import re
import shlex
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from .provenance_common import command_segments, in_scope, make_event


URL_RE = re.compile(r'https?://[^\s\'"<>`\\\[\]{}]+')
SECRET_KEY = re.compile(r'(?i)(?:token|password|passwd|secret|authorization|signature|credential|api[-_]?key|access[-_]?key|^key$)')
ERROR = re.compile(r'(?mi)(?:^|\n)(?:[^\n]*: )?(?:fatal:|error:|permission denied|no such file or directory|command not found)|^curl: \([1-9]\d*\)')
PATH_SUFFIXES = {'.py', '.ipynb', '.csv', '.tsv', '.json', '.jsonl', '.parquet', '.md', '.txt', '.yaml', '.yml', '.toml', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.sh', '.sql', '.r', '.pdf', '.svg', '.png', '.zip', '.gz', '.xml', '.log'}
BORING_FILES = {'readme.md', 'requirements.txt', 'package.json', 'package-lock.json', 'pyproject.toml', 'index.html', 'index.js', 'main.py', 'config.json', '.gitignore'}
READ_COMMANDS = {'cat', 'head', 'tail', 'wc', 'stat', 'sha256sum', 'sha512sum', 'md5sum', 'less', 'more', 'diff', 'cmp'}
INTERPRETERS = {'python', 'python3', 'python3.9', 'python3.10', 'python3.11', 'python3.12', 'node', 'ruby', 'Rscript', 'bash', 'sh'}


def canonical_url(value):
    """Preserve resource-significant path/query/fragment spelling and case.

    Credential-bearing URLs are excluded instead of merging redacted identities.
    No query stripping, path lowercasing, host aliasing or tracking heuristics.
    """
    if any(char.isspace() for char in value):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in ('http', 'https') or not parsed.hostname:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        for field in parsed.query.split('&'):
            if SECRET_KEY.search(unquote(field.split('=', 1)[0])):
                return None
        if SECRET_KEY.search(unquote(parsed.fragment.split('=', 1)[0])) and '=' in parsed.fragment:
            return None
        host = parsed.hostname.lower()
        if ':' in host:
            host = '[' + host + ']'
        port = parsed.port
        if port is not None and (parsed.scheme.lower(), port) not in (('http', 80), ('https', 443)):
            host += ':' + str(port)
        return urlunsplit((parsed.scheme.lower(), host, parsed.path or '/', parsed.query, parsed.fragment))
    except ValueError:
        return None


def repository_url(value):
    """Standard remote syntax, without any provider/project naming assumptions."""
    if re.fullmatch(r'[^\s@]+@[^\s:]+:[^\s]+', value):
        value = 'https://' + value.split('@', 1)[1].replace(':', '/', 1)
    elif value.startswith('ssh://'):
        try:
            p = urlsplit(value)
            if not p.hostname or p.password or p.port not in (None, 22):
                return None
            value = 'https://' + p.hostname + p.path
        except ValueError:
            return None
    result = canonical_url(value)
    if result is None:
        return None
    parsed = urlsplit(result)
    if parsed.query or parsed.fragment:
        return None
    path = parsed.path.rstrip('/')
    if path.endswith('.git'):
        path = path[:-4]
    return urlunsplit((parsed.scheme, parsed.netloc, path, '', ''))


def _head(argv):
    return posixpath.basename(argv[0]) if argv else ''


def _path(value, cwd):
    if not isinstance(value, str) or not value or value.startswith(('-', '~')):
        return None
    if any(c in value for c in '$`*?<>\n\r\t') or '://' in value or value == '-':
        return None
    if value.startswith(('/dev/', '/proc/', '/sys/')):
        return None
    if value.startswith('/'):
        return posixpath.normpath(value)
    return posixpath.normpath(posixpath.join(cwd, value)) if cwd else None


def _positionals(args, valued=()):
    """Small documented option vocabulary; unknown flags remain options."""
    values, skip, literal = [], False, False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == '--':
            literal = True
        elif not literal and arg in valued:
            skip = True
        elif literal or not arg.startswith('-'):
            values.append(arg)
    return values


def _file_operations(argv):
    head, args = _head(argv), argv[1:]
    if head in READ_COMMANDS:
        valued = ('-n', '-c', '--lines', '--bytes') if head in ('head', 'tail') else ('--format', '-c') if head == 'stat' else ()
        return [(p, 'read') for p in _positionals(args, valued)]
    if head in ('cp', 'mv'):
        paths = _positionals(args, ('-t', '--target-directory'))
        if len(paths) == 2 and not any(a in args for a in ('-t', '--target-directory')):
            return [(paths[0], 'read'), (paths[1], 'write')]
    if head in ('touch', 'tee'):
        return [(p, 'write') for p in _positionals(args, ('-d', '-t', '-r', '--date', '--reference'))]
    if head in INTERPRETERS:
        if any(a in args for a in ('-c', '-e', '-m', '--eval', '--print')):
            return []
        paths = _positionals(args)
        if not paths:
            return []
        # Custom arguments are file references, not inferred input/output roles.
        return [(paths[0], 'execute')] + [(p, 'observe') for p in paths[1:]
                                        if posixpath.splitext(p)[1].lower() in PATH_SUFFIXES]
    if head in ('pytest', 'py.test'):
        paths = _positionals(args, ('-k', '-m', '--rootdir', '--override-ini', '-o'))
        return [(p.split('::', 1)[0], 'execute') for p in paths if '/' in p or p.endswith('.py')]
    if head in ('curl', 'wget'):
        paths = []
        options = ('-o', '--output') if head == 'curl' else ('-O', '--output-document')
        for i, arg in enumerate(args[:-1]):
            if arg in options:
                paths.append((args[i + 1], 'write'))
        if head == 'curl':
            for i, arg in enumerate(args[:-1]):
                if arg in ('-T', '--upload-file'):
                    paths.append((args[i + 1], 'read'))
                if arg in ('-d', '--data', '--data-binary', '--data-raw') and args[i + 1].startswith('@') and arg != '--data-raw':
                    paths.append((args[i + 1][1:], 'read'))
        return paths
    return []


def _redirects(command):
    """Find only operators outside quotes; never interpret printed shell code."""
    surface, quoted, escaped = [], None, False
    for char in command:
        if escaped:
            escaped = False
            surface.append(' ')
        elif char == '\\' and quoted != "'":
            escaped = True
            surface.append(' ')
        elif quoted:
            if char == quoted:
                quoted = None
            surface.append(' ')
        elif char in "\"'":
            quoted = char
            surface.append(' ')
        else:
            surface.append(char)
    result = []
    for match in re.finditer(r'(?<![<>&\d])(>>?|<)(?![<>&])', ''.join(surface)):
        try:
            tail = shlex.split(command[match.end():], comments=True)
        except ValueError:
            continue
        if tail:
            result.append((tail[0], 'read' if match[1] == '<' else 'write'))
    return result


def _without_redirects(argv):
    result, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg in ('>', '>>', '<', '2>', '2>>'):
            skip = True
        elif re.match(r'^(?:\d?)(?:>>?|<)', arg):
            continue
        else:
            result.append(arg)
    return result


def _url_operands(argv):
    head, args = _head(argv), argv[1:]
    if head not in ('curl', 'wget'):
        return []
    valued = {'-H', '--header', '-d', '--data', '--data-raw', '--data-binary', '--data-urlencode',
              '-F', '--form', '--form-string', '-X', '--request', '-o', '--output', '-O',
              '--output-document', '-T', '--upload-file', '-u', '--user', '-A', '--user-agent',
              '-e', '--referer', '-b', '--cookie', '-c', '--cookie-jar', '--proxy', '-x',
              '--connect-timeout', '--max-time', '--retry', '--request-target'}
    operation = 'read'
    for index, arg in enumerate(args):
        if arg in ('-X', '--request') and index + 1 < len(args):
            if args[index + 1].upper() in ('POST', 'PUT', 'PATCH', 'DELETE'):
                operation = 'write'
        if arg in ('-d', '--data', '--data-raw', '--data-binary', '--data-urlencode',
                   '-F', '--form', '--form-string', '-T', '--upload-file'):
            operation = 'write'
    values = _positionals(args, valued)
    return [(url, operation) for value in values if (url := canonical_url(value))]


class AgentTraceExtractor:
    """Incremental layer-1 extraction with optional verified context injection.

    ``set_context(session, cwd, repo, root)`` can reuse a provenance extractor's
    known location. ``repo`` must identify a shared remote, and ``root`` its local
    checkout. No context is assumed when these are unavailable.
    """

    def __init__(self, actor):
        self.actor = actor
        self.cwd = {}
        self.roots = {}
        self.epochs = {}

    def set_context(self, session, cwd, repo=None, root=None):
        session = str(session)
        self.cwd[session] = cwd if cwd and cwd.startswith('/') else None
        if repo and root and root.startswith('/') and not str(repo).startswith('local:'):
            self.roots[(session, root)] = repo

    def _event(self, record, family, operation, status, entity, source, rule,
               identity_strength='exact', **extra):
        return make_event(self.actor, record, family, operation, status, entity,
                          evidence=[{'field': source, 'identity': entity}],
                          research_layer=1, rule_id=rule, source=source,
                          identity_strength=identity_strength, relation_authority=False,
                          **extra)

    def _file(self, record, raw, cwd, operation, status):
        session = str(record.get('s', ''))
        absolute = _path(raw, cwd)
        # Unresolved paths remain local to the current unknown-location epoch.
        relative = raw if absolute is None and _path(raw, '/') is not None else None
        if absolute is None and relative is None:
            return []
        repo, root = None, None
        if absolute:
            candidates = [(r, p) for (s, p), r in self.roots.items()
                          if s == session and (absolute == p or absolute.startswith(p + '/'))]
            if candidates:
                repo, root = max(candidates, key=lambda pair: len(pair[1]))
        if repo:
            entity = 'file:repo:' + quote(repo, safe='') + ':' + quote(posixpath.relpath(absolute, root), safe='/._-')
            strength = 'repository_scoped'
        else:
            local = absolute or ('unresolved:' + str(self.epochs.get(session, 'initial')) + ':' + relative)
            entity = 'file:local:' + quote(self.actor, safe='') + ':' + quote(session, safe='') + ':' + quote(local, safe='/._-')
            strength = 'actor_session_scoped'
        events = [self._event(record, 'file', operation, status, entity, 'x', 'direct_file_operand', strength)]
        basename = posixpath.basename(absolute or relative)
        suffix = posixpath.splitext(basename)[1].lower()
        if len(basename) >= 5 and suffix in PATH_SUFFIXES and basename.lower() not in BORING_FILES:
            events.append(self._event(record, 'file_candidate', operation, status,
                                      'filename:' + basename, 'x', 'filename_candidate', 'candidate'))
        return events

    def extract(self, record):
        if not in_scope(record['t']):
            return []
        session = str(record.get('s', ''))
        kind = record.get('k')
        if kind == 'restart':
            self.cwd.pop(session, None)
            self.epochs[session] = str(record['i'])
            self.roots = {key: value for key, value in self.roots.items() if key[0] != session}
            return []
        command, output = record.get('x') or '', record.get('o') or ''
        error = record.get('e') or ''
        rc = record.get('exit_code', record.get('returncode'))
        failed = rc not in (None, 0) or bool(ERROR.search(error + '\n' + output))
        status = 'failed' if failed else 'success' if rc == 0 else 'attempted'
        segments = command_segments(command) if kind == 'shell' else []
        # Compound commands have no independently attributable exit status.
        direct_status = status if len(segments) == 1 else 'failed' if failed else 'attempted'
        events = []
        urls = {}
        for source, text in (('x', command), ('o', output)):
            if source == 'x' and kind not in ('shell', 'type'):
                continue
            for match in URL_RE.finditer(text):
                raw = match.group()
                # Parenthesized prose frequently surrounds an otherwise exact
                # URL; retain balanced parentheses that belong to the path.
                while raw.endswith(')') and raw.count(')') > raw.count('('):
                    raw = raw[:-1]
                url = canonical_url(raw)
                if url:
                    urls[(source, url)] = 'observe'
        for argv in segments:
            for url, operation in _url_operands(argv):
                if ('x', url) in urls:
                    urls[('x', url)] = operation
        if kind == 'type':
            url = canonical_url(command.strip())
            if url and len(command.split()) == 1 and ('x', url) in urls:
                urls[('x', url)] = 'read'
        for (source, url), operation in sorted(urls.items()):
            event_status = direct_status if operation in ('read', 'write') else 'unknown'
            events.append(self._event(record, 'url', operation, event_status,
                                      'url:' + url, source, 'exact_url'))
        if not segments:
            if kind == 'shell' and re.search(r'\b(?:cd|pushd|popd|source|eval)\b', command):
                self.cwd[session] = None
                self.epochs[session] = str(record['i'])
            return events
        cwd = self.cwd.get(session)
        simple_sequence = not re.search(r'(?:\|\||[;\n])', command)
        work = [argv for argv in segments if _head(argv) != 'cd']
        for argv in segments:
            head = _head(argv)
            if head in ('pushd', 'popd', 'source', '.', 'eval'):
                cwd = None
                self.epochs[session] = str(record['i'])
                continue
            if head == 'cd':
                target = _path(argv[1], cwd) if len(argv) == 2 else None
                if target and not failed and simple_sequence and (rc == 0 or '&&' in command):
                    cwd = target
                else:
                    cwd = None
                    self.epochs[session] = str(record['i'])
                continue
            if head == 'pwd' and len(work) == 1 and not failed and re.fullmatch(r'/[^\n\r]*\n?', output):
                cwd = posixpath.normpath(output.strip())
            if head == 'git' and argv[1:] == ['remote', '-v'] and len(work) == 1 and cwd and not failed:
                remotes = {repository_url(value) for value in re.findall(r'(?m)^\S+\s+(\S+)\s+\((?:fetch|push)\)\s*$', output)} - {None}
                if len(remotes) == 1:
                    self.roots[(session, cwd)] = remotes.pop()
            if head == 'git' and argv[1:2] == ['clone'] and len(work) == 1 and not failed:
                operands = _positionals(argv[2:], ('-b', '--branch', '--depth', '-o', '--origin', '--config', '-c'))
                if len(operands) in (1, 2):
                    repo = repository_url(operands[0])
                    target = operands[1] if len(operands) == 2 else posixpath.basename(urlsplit(repo).path) if repo else None
                    root = _path(target, cwd) if target else None
                    receipt = bool(re.search(r'(?m)^Cloning into ', output + '\n' + error))
                    complete = rc == 0 or bool(re.search(r'(?m)^(?:Receiving objects|Resolving deltas):\s+100%.*done\.', output + '\n' + error))
                    if repo and root and receipt and complete:
                        self.roots[(session, root)] = repo
            for raw, operation in _file_operations(_without_redirects(argv)):
                events.extend(self._file(record, raw, cwd, operation,
                                         'unknown' if operation == 'observe' else direct_status))
            if head in ('sha256sum', 'sha512sum', 'md5sum') and len(work) == 1 and not failed:
                algorithm = head[:-3]
                size = {'sha256': 64, 'sha512': 128, 'md5': 32}[algorithm]
                operands = {path for path, _ in _file_operations(argv)}
                for digest, filename in re.findall(r'(?mi)^([0-9a-f]{' + str(size) + r'}) [ *](.+)$', output):
                    if filename in operands and filename != '-':
                        events.append(self._event(record, 'content', 'observe', 'success',
                                                  'content:' + algorithm + ':' + digest.lower(), 'o', 'checksum_receipt'))
            events.extend(self._packages(record, argv, direct_status))
        self.cwd[session] = cwd
        # Multiple directory changes make the redirection location ambiguous.
        heads = [_head(argv) for argv in segments]
        if heads.count('cd') <= 1 and ('cd' not in heads or heads[0] == 'cd'):
            for raw, operation in _redirects(command):
                events.extend(self._file(record, raw, cwd, operation, direct_status))
        return list({event['event_id']: event for event in events}.values())

    def _packages(self, record, argv, status):
        head, args = _head(argv), argv[1:]
        events = []
        if head in ('docker', 'podman') and args[:1] in (['pull'], ['run'], ['create']):
            values = _positionals(args[1:], ('--name', '-e', '--env', '-v', '--volume', '-w', '--workdir',
                                             '--entrypoint', '--network', '-p', '--publish', '--platform',
                                             '--env-file', '--mount', '--user', '-u'))
            for arg in values[:1]:
                if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-fA-F]{64}', arg):
                    events.append(self._event(record, 'package', 'acquire', status,
                                              'container:' + arg.rsplit(':', 1)[0] + ':' + arg.rsplit(':', 1)[1].lower(),
                                              'x', 'container_digest'))
        pip_args = args if head in ('pip', 'pip3') else args[2:] if head.startswith('python') and args[:2] == ['-m', 'pip'] else []
        if pip_args[:1] == ['install'] and not any(a.startswith(('-i', '--index-url', '--extra-index-url', '-f', '--find-links')) for a in pip_args):
            for arg in pip_args[1:]:
                match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)', arg)
                if match:
                    name = re.sub(r'[-_.]+', '-', match[1]).lower()
                    events.append(self._event(record, 'package', 'acquire', status,
                                              'python-package-pin:' + name + '==' + match[2], 'x',
                                              'python_package_pin', 'candidate'))
        return events
