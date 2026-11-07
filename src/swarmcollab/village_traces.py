"""Layer 2: frozen knowledge of AI Village's project directory and reviews.

The rule choices come from inspecting training operations. Village Hub's custom
``{a, proj:{t,u}}`` / ``{a,g,href}`` records associate agents with projects. Review
files use named ``Reviewer`` and ``Target Project`` fields. These associations
give context to an observed access; they do not prove collaboration, ownership
of every contribution, or successful consumption. No model or service is called.
"""
import collections
import ast
import re
from urllib.parse import unquote, urlsplit, urlunsplit

from . import config
from .provenance_common import PACIFIC, command_segments, in_scope, make_event, timestamp

RULE_VERSION = 2
VILLAGE_NAMESPACE = 'ai-village-agents/village/'
URL_RE = re.compile(r'https?://[^\s\'"<>`\\)\]}|,]+')
REPO_RE = re.compile(r'ai-village-agents/village/([A-Za-z0-9_.-]+)', re.I)
ACTOR_FIELD = re.compile(r'(?:\{|,)\s*[\'"]?a[\'"]?\s*:\s*([\'"])([^\n\r]+?)\1')
URL_FIELD = re.compile(r'(?:\{|,)\s*[\'"]?(?:u|href)[\'"]?\s*:\s*([\'"])(https?://[^\s\'"<>]+)\1')
TITLE_FIELD = re.compile(r'(?:\{|,)\s*[\'"]?(?:t|g)[\'"]?\s*:\s*([\'"])((?:\\.|(?!\1)[^\n\r\\])*)\1')
SHARED_BOARD_TITLE = re.compile(r'\bvillage\s+handoff\s+board\b', re.I)
REVIEWER = re.compile(r'(?im)^\s*(?:\*\*)?Reviewer(?:\*\*)?\s*:(?:\*\*)?\s*([^\n\r]+)')
TARGET = re.compile(r'(?im)^\s*(?:\*\*)?Target Project(?:\*\*)?\s*:(?:\*\*)?\s*([^\n\r]+)')
# These cached shell tools print communication history even though the original
# chat/search actions were excluded upstream. Do not turn their contents into
# fresh project accesses or reviewer evidence.
HISTORY_TOOLS = {'pre_send_chat.sh', 'session_start.sh', 'session_end.sh', 'search_history'}
FAILURE = re.compile(r'(?mi)^(?:curl: \([1-9]\d*\)|fatal:|error:|HTTP/\S+\s+[45]\d\d\b)|\b(?:401 Unauthorized|403 Forbidden|404 Not Found)\b')
PUBLIC_PLATFORM_ROOTS = {'manifold.markets', 'twitter.com', 'x.com', 'youtube.com', 'www.youtube.com',
                         'github.com', 'gitlab.com', 'docs.google.com', 'drive.google.com',
                         'substack.com', 'fourthwall.com', 'medium.com'}


def _segments(command):
    # The shared conservative tokenizer rejects apostrophes in full-line comments
    # before it identifies them as comments. Strip only the leading comment block;
    # never remove lines from inside a command string or embedded script.
    lines = command.splitlines(keepends=True)
    first = 0
    while first < len(lines) and (not lines[first].strip() or lines[first].lstrip().startswith('#')):
        first += 1
    return command_segments(''.join(lines[first:]))


def _history_command(command):
    for args in _segments(command):
        names = [a.rsplit('/', 1)[-1] for a in args]
        if names[0] in HISTORY_TOOLS:
            return True
        if names[0] in ('bash', 'sh', 'zsh', 'python', 'python3') and any(x in HISTORY_TOOLS for x in names[1:3]):
            return True
        if names[0] in ('cat', 'head', 'tail', 'grep', 'rg', 'sed', 'less') and 'public_comms.md' in names[1:]:
            return True
    return False


def _failed(record):
    return bool(record.get('e')) or record.get('exit_code', record.get('returncode')) not in (None, 0) or bool(FAILURE.search(record.get('o') or ''))


def _actor_key(name):
    return re.sub(r'[\s_.-]+', '', str(name)).lower()


def _actor_lookup(names):
    values = collections.defaultdict(set)
    for name in names:
        values[_actor_key(name)].add(name)
    return {key: next(iter(found)) for key, found in values.items() if len(found) == 1}


def clean_url(value):
    """Public resource key; reject credentials rather than retaining them."""
    value = value.rstrip('.,;:!?')
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            return None
        if parsed.port not in (None, 80, 443):
            return None
    except ValueError:
        return None
    path = parsed.path or '/'
    return urlunsplit(('https', parsed.hostname.lower(), path, '', ''))


def _support(actor, record, source, snippet):
    return dict(actor=actor, turn_id=str(record['i']), time=timestamp(record['t']).isoformat(),
                source=source, text=snippet[:700])


def _directory_records(text):
    """Read the observed Village Hub schema, never arbitrary name/URL proximity."""
    # Each array entry is on one line in the captured hub source. Restricting the
    # association to that entry avoids attributing neighboring projects.
    for line in text.splitlines():
        matches = list(ACTOR_FIELD.finditer(line))
        for i, found in enumerate(matches):
            stop = matches[i + 1].start() if i + 1 < len(matches) else len(line)
            entry = line[found.start():stop]
            url = URL_FIELD.search(entry)
            if url is None:
                continue
            title = TITLE_FIELD.search(entry)
            title_value = ''
            if title:
                try:
                    title_value = ast.literal_eval(title[1] + title[2] + title[1])
                except (ValueError, SyntaxError):
                    title_value = title[2]
            yield found[2], url[2], title_value, entry[:700]


def _known_hub(command):
    return bool(re.search(r'(?i)(?:ai-village-agents(?:/|%2f)village(?:/|%2f)village-hub(?:\b|/)|'
                          r'village-hub-[a-f0-9]{6}\.gitlab\.io|(?:~|/)[\w/.-]*village-hub(?:[-/\s]|$))', command))


def _review_url(value):
    key = clean_url(value)
    if key is None:
        return None
    p = urlsplit(key)
    return key if p.hostname == 'gitlab.com' and p.path.startswith('/' + VILLAGE_NAMESPACE) and '/reviews/' in p.path and p.path.endswith('.md') else None


def _direct_urls(command):
    """URLs passed directly to network tools, excluding comments/script bodies."""
    output = []
    for args in _segments(command):
        if not args:
            continue
        tool = args[0].rsplit('/', 1)[-1]
        if tool in ('curl', 'wget'):
            if any(a in ('-d', '--data', '--data-raw', '--data-binary', '-F', '--form', '--post-data', '--post-file', '--upload-file', '-T') or a.startswith(('--data=', '--data-raw=', '--data-binary=', '--form=', '--post-data=')) for a in args[1:]):
                continue
            methods = [args[i + 1] for i, a in enumerate(args[:-1]) if a in ('-X', '--request', '--method')]
            methods += [a.split('=', 1)[1] for a in args if a.startswith(('--request=', '--method='))]
            methods += [a[2:] for a in args if a.startswith('-X') and len(a) > 2]
            if any(m.upper() not in ('GET', 'HEAD') for m in methods):
                continue
            # Values used as headers, bodies, destinations, proxies, or referers
            # are not the resource being accessed.
            skip = False
            for arg in args[1:]:
                if skip:
                    skip = False
                    continue
                if arg in ('-H', '--header', '-d', '--data', '--data-raw', '--data-binary', '-F', '--form', '-o', '--output', '-O', '--output-document', '-e', '--referer', '-x', '--proxy', '-u', '--user'):
                    skip = arg not in ('-O',)
                    continue
                value = arg[6:] if arg.startswith('--url=') else arg
                if value.startswith(('http://', 'https://')):
                    output.append(value)
        elif tool == 'git':
            # Remote selection via clone/fetch/pull is an attempted repository
            # operation, not a successful handoff inferred from its presence.
            if any(x in args[1:4] for x in ('clone', 'fetch', 'pull', 'ls-remote')):
                output.extend(x for x in args[1:] if x.startswith(('http://', 'https://')))
    return sorted(set(output))


def fit_village_rules(actor_records, actor_names=None):
    """Fit only on July 6–31 records; all retained knowledge has source turns.

    ``actor_records`` is an iterable of ``(actor, cached_record)`` pairs. A
    non-training timestamp raises before accessing its payload. The returned
    dictionary is JSON-serializable and can be frozen before validation.
    """
    roster = set(actor_names or ())
    entries, reviews = [], []
    repos = collections.defaultdict(list)
    steps = 0
    for actor, record in actor_records:
        when = timestamp(record['t'])
        day = when.astimezone(PACIFIC).date()
        if not config.DEV_START <= day <= config.TRAIN_END:
            raise ValueError('Village knowledge fitting accepts training dates only')
        roster.add(actor)
        steps += 1
        command = record.get('x') or ''
        if _history_command(command) or _failed(record):
            continue
        output = record.get('o') or ''
        for source, text in (('command', command), ('output', output)):
            if 'ai-village-agents' in text:
                for slug in set(REPO_RE.findall(unquote(text))):
                    slug = slug.rstrip('.')
                    slug = slug[:-4] if slug.endswith('.git') else slug
                    if len(repos[slug]) < 3:
                        repos[slug].append(_support(actor, record, source, VILLAGE_NAMESPACE + slug))
        if _known_hub(command):
            for source, text in (('command', command), ('output', output)):
                for owner, url, title, snippet in _directory_records(text):
                    key = clean_url(url)
                    if key:
                        entries.append(dict(owner=owner, url=key, title=title,
                                            evidence=_support(actor, record, source, snippet)))
        if '/reviews/' in command and record.get('k') == 'shell':
            urls = sorted({key for value in _direct_urls(command) if (key := _review_url(value))})
            reviewer, target = REVIEWER.search(output), TARGET.search(output)
            if len(urls) == 1 and reviewer and target:
                targets = [clean_url(u) for u in URL_RE.findall(target[1])]
                targets = sorted({u for u in targets if u})
                if len(targets) == 1:
                    reviews.append(dict(url=urls[0], reviewer=reviewer[1].strip(' *'), target=targets[0],
                                        evidence=_support(actor, record, 'output', reviewer[0] + '\n' + target[0])))
    actor_map = _actor_lookup(roster)
    by_url = collections.defaultdict(list)
    for entry in entries:
        owner = actor_map.get(_actor_key(entry['owner']))
        if owner:
            entry['owner'] = owner
            by_url[entry['url']].append(entry)
    projects, conflicts, excluded = [], [], []
    for url, records in sorted(by_url.items()):
        parsed = urlsplit(url)
        if parsed.hostname in PUBLIC_PLATFORM_ROOTS and parsed.path in ('', '/'):
            excluded.append(dict(url=url, reason='Shared public platform root is not an agent project'))
            continue
        owners = sorted({r['owner'] for r in records})
        if len(owners) != 1:
            conflicts.append(dict(url=url, owners=owners))
            continue
        unique = {r['evidence']['turn_id']: r['evidence'] for r in records}
        evidence = sorted(unique.values(), key=lambda x: (x['time'], x['turn_id']))[:5]
        host = urlsplit(url).hostname
        # The observed Village deployment convention uses <repo>-<six hex>.gitlab.io.
        # Require the matching repository to have appeared in training, and retain
        # that separate evidence. Never assign an unobserved repository by guess.
        match = re.fullmatch(r'(.+)-[a-f0-9]{6}\.gitlab\.io', host or '')
        slug = match[1] if match else None
        repo = 'https://gitlab.com/' + VILLAGE_NAMESPACE + slug if slug in repos else None
        repo_evidence = sorted(repos.get(slug, []), key=lambda x: (x['time'], x['turn_id']))
        shared_evidence = sorted({r['evidence']['turn_id']: r['evidence'] for r in records
                                  if SHARED_BOARD_TITLE.search(r['title'])}.values(),
                                 key=lambda x: (x['time'], x['turn_id']))[:5]
        projects.append(dict(key=url, owner=owners[0], title=min((r['title'] for r in records if r['title']), default=''),
                             site=url, repository=repo, evidence=evidence, repository_evidence=repo_evidence,
                             shared_resource=bool(shared_evidence), shared_evidence=shared_evidence,
                             shared_available_from=shared_evidence[0]['time'] if shared_evidence else None,
                             available_from=evidence[0]['time'],
                             repository_available_from=repo_evidence[0]['time'] if repo_evidence else None))
    review_groups = collections.defaultdict(list)
    for row in reviews:
        reviewer = actor_map.get(_actor_key(row['reviewer']))
        if reviewer:
            row['reviewer'] = reviewer
            review_groups[row['url']].append(row)
    review_rules = []
    for url, records in sorted(review_groups.items()):
        identities = {(r['reviewer'], r['target']) for r in records}
        if len(identities) != 1:
            conflicts.append(dict(url=url, review_attributions=sorted(identities)))
            continue
        reviewer, target = next(iter(identities))
        evidence = sorted({r['evidence']['turn_id']: r['evidence'] for r in records}.values(), key=lambda x: (x['time'], x['turn_id']))[:5]
        review_rules.append(dict(url=url, reviewer=reviewer, target=target, evidence=evidence,
                                 available_from=evidence[0]['time']))
    return dict(version=RULE_VERSION, fit_start=config.DEV_START.isoformat(), fit_end=config.TRAIN_END.isoformat(),
                fit_steps=steps, actors=sorted(roster), projects=projects, reviews=review_rules,
                conflicts=conflicts, excluded=excluded,
                interpretation='Village directory associations and review authorship are context/candidates, not successful interactions. Rules chosen by assistant inspection of training operations; no runtime AI model.')


def _under(url, root):
    a, b = urlsplit(url), urlsplit(root)
    if a.hostname != b.hostname:
        return False
    base = b.path.rstrip('/')
    return a.path.rstrip('/') == base or a.path.startswith(base + '/')


class VillageTraceExtractor:
    def __init__(self, actor, rules):
        if rules.get('version') != RULE_VERSION or rules.get('fit_start') != config.DEV_START.isoformat() or rules.get('fit_end') != config.TRAIN_END.isoformat():
            raise ValueError('Village extractor requires frozen rules from the authorized training interval')
        self.actor = actor
        # Copy the relevant structures so caller mutations cannot change an
        # extractor midway through a validation replay.
        import copy
        self.projects = copy.deepcopy(rules.get('projects', []))
        self.reviews = copy.deepcopy(rules.get('reviews', []))

    def extract(self, record):
        if not in_scope(record['t']):
            return []
        command = record.get('x') or ''
        if _history_command(command):
            return []
        kind = record.get('k')
        if kind == 'shell':
            urls = _direct_urls(command)
            source = 'command'
        elif kind == 'type' and URL_RE.fullmatch(command.strip()):
            urls = [command.strip()]
            source = 'navigation'
        else:
            return []
        now = timestamp(record['t'])
        status = 'failed' if _failed(record) else 'attempted'
        events = []
        for raw in urls:
            url = clean_url(raw)
            if not url:
                continue
            matches = []
            for project in self.projects:
                if now < timestamp(project['available_from']):
                    continue
                site = _under(url, project['site'])
                repo = project.get('repository')
                repo_match = False
                if repo and now >= timestamp(project['repository_available_from']):
                    repo_match = _under(url.removesuffix('.git'), repo)
                if site or repo_match:
                    matches.append(project)
            # Overlapping root/subpath assignments must not spray owner edges.
            if len({p['owner'] for p in matches}) == 1:
                # Directory revisions can list both a project homepage and a
                # page within it. For one access, use the broadest currently
                # available explicit association, not both representations.
                project = min(matches, key=lambda p: (len(urlsplit(p['site']).path), p['key']))
                shared = project.get('shared_resource', False)
                # This Village board is a communal future-self notebook. Its
                # directory contact is not the author or recipient of every note.
                # Never emit a contact edge for a resource learned to be shared.
                if shared and now < timestamp(project['shared_available_from']):
                    continue
                events.append(make_event(self.actor, record, 'village_shared_board' if shared else 'village_project_access', 'read', status,
                                         'village-project:' + project['key'], research_layer=2,
                                         rule_id='village_shared_handoff_board' if shared else 'village_hub_project_association', source=source,
                                         related_actor='' if shared else project['owner'], verbal_channel=False,
                                         listed_agent=project['owner'], shared_resource=shared,
                                         relation_authority=False,
                                         evidence=[dict(source=source, text=url)],
                                         knowledge_evidence=project['shared_evidence'] if shared else project['evidence'],
                                         interpretation=('Access attempt to the shared Village Handoff Board; future-self use is not a cross-agent handoff.' if shared
                                                         else 'Access attempt to a project associated with an agent in the Village directory; not a verified interaction.')))
            for review in self.reviews:
                if url == review['url'] and now >= timestamp(review['available_from']):
                    events.append(make_event(self.actor, record, 'village_review_reference', 'read', status,
                                             'village-review:' + url, research_layer=2,
                                             rule_id='village_review_file_attribution', source=source,
                                             related_actor=review['reviewer'], verbal_channel=True,
                                             relation_authority=False,
                                             evidence=[dict(source=source, text=url)],
                                             knowledge_evidence=review['evidence'],
                                             target_project=review['target'],
                                             interpretation='Attempt to access an attributed review; no claim that the suggestions were applied.'))
        return list({e['event_id']: e for e in events}.values())
