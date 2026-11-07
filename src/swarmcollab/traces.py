"""Stage B: turn each agent's steps into trace mentions.

A mention says: at time t, agent A touched trace T with operation `op`, seen in
source `src`. Detectors only ever see mentions and activity counts.

Trace types
  repo      a GitLab or GitHub repository: URLs, `glab -R`, API paths, and local
            clones (each folder is mapped to its remote by following cd, git clone,
            git remote and push output)
  pages     a GitLab or GitHub Pages site
  commit    a git commit, keyed by its 7-character prefix
  issue     an issue (repo#iid) or merge request (repo!iid)
  pipeline  a CI pipeline (p<id>) or job (j<id>)
  project   a numeric GitLab project ID
  gdoc      a Google Doc, Sheet, Slides deck, Form, Drive file or Apps Script project
  site      any other web host, as host + first path segment
  page      a specific page on a Pages or external site (host + path)
  name      another agent's name, kept as its own feature group
  pidmap    not a trace: a project ID / repository pair learned from API output
  pidvote   not a trace: a project ID and a repository named together in one step

Operations
  create    the step created the identifier (commit, issue, merge request, repo)
  write     the step changed the trace (push, commit, API write, file edit in a clone)
  read      the step used the trace (clone, pull, view, GET, navigation, a mention in a command)
  seen      the trace only appears in the step's output

Sources: cmd (shell command), cwd (a repository reached through the folder a
command runs in), out (command output), typed (text typed in the GUI), gui
(descriptions of GUI elements). For names: comment, code, typed, gui, out.

Run: python -I -m swarmcollab.traces [agent file names...]   (default: all agents)
Only the authorized development slice is parsed, using Pacific calendar dates.
"""
import collections
import datetime as dt
import gzip
import json
import posixpath
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

from . import config

UTC = dt.timezone.utc
PACIFIC = ZoneInfo('America/Los_Angeles')
HOME = '/home/computeruse'
MENTIONS = config.OUT / 'mentions'

WAITING = {'pause', 'wait', 'empty', 'restart'}
GUI_KINDS = {'left_click', 'right_click', 'double_click', 'triple_click', 'middle_click', 'mouse_move',
             'left_click_drag', 'scroll', 'key', 'hold_key', 'type', 'screenshot', 'cursor_position',
             'get_pixel_coords_of_element', 'gui', 'left_mouse_down', 'left_mouse_up', 'zoom'}

# ---------------------------------------------------------------- URLs
URL_RE = re.compile(r'https?://[^\s\'"<>`\\)\]}|,]+')
SSH_RE = re.compile(r'git@(gitlab\.com|github\.com):([\w.-]+(?:/[\w.-]+)+?)(?:\.git)?(?=[\s\'"]|$)', re.M)
HEX_RE = re.compile(r'[0-9a-f]{7,40}')
BORING_HOSTS = {
    'localhost', '127.0.0.1', '0.0.0.0', 'example.com', 'example.org', 'example.net', 'w3.org',
    'schema.org', 'fonts.googleapis.com', 'fonts.gstatic.com', 'cdn.jsdelivr.net', 'cdnjs.cloudflare.com',
    'unpkg.com', 'purl.org', 'xmlns.com', 'ogp.me',
}
GITLAB_RESERVED = {'users', 'explore', 'dashboard', 'help', 'api', 'oauth', 'search', 'groups', '-', 'admin',
                   'profile', 'assets', 'uploads', 'projects'}
GITHUB_RESERVED = {'settings', 'orgs', 'login', 'about', 'features', 'marketplace', 'topics', 'search',
                   'notifications', 'sponsors', 'apps', 'site', 'pricing', 'explore'}
GOOGLE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{20,}$')


def strip_git(s):
    return s[:-4] if s.endswith('.git') else s


def is_commitish(h):
    # Callers use explicit commit contexts; all-digit/all-letter hex is valid.
    return bool(HEX_RE.fullmatch(h.lower()))


def gitlab_sub(repo_key, rest):
    out = []
    if len(rest) >= 2:
        a, b = rest[0], rest[1]
        if a == 'issues' and b.isdigit():
            out.append(('issue', f'{repo_key}#{b}'))
        elif a == 'merge_requests' and b.isdigit():
            out.append(('issue', f'{repo_key}!{b}'))
        elif a in ('pipelines', 'jobs') and b.isdigit():
            out.append(('pipeline', f'{a[0]}{b}'))
        elif a == 'commit' and is_commitish(b.lower()):
            out.append(('commit', b[:7].lower()))
        elif a == 'repository' and len(rest) >= 3 and b == 'commits' and is_commitish(rest[2].lower()):
            out.append(('commit', rest[2][:7].lower()))
    return out


def gitlab_api(segs):
    """Segments after /api/v4/ (or a glab api path)."""
    if len(segs) >= 2 and segs[0] == 'projects':
        pid = unquote(segs[1])
        if pid.isdigit():
            head, key = ('project', pid), 'pid:' + pid
        elif '/' in pid:
            key = pid.lower().strip('/')
            head = ('repo', 'gitlab:' + key)
        else:
            return []
        return [head] + gitlab_sub(key, segs[2:])
    return []


LEGACY_ROUTES = {'blob', 'tree', 'raw', 'issues', 'merge_requests', 'commit', 'commits', 'pipelines', 'jobs',
                 'wikis', 'releases', 'tags', 'branches', 'edit', 'settings', 'graphs', 'network', 'compare'}


def gitlab_path(segs):
    """Project path from a gitlab.com URL path.

    GitLab groups nest (the village's projects live under ai-village-agents/village/),
    so a path is only known to be a project when it is followed by the /-/ route or
    ends in .git. Other paths are marked uncertain ('gitlab?:'), and stage C maps them
    to the projects seen elsewhere, or drops them as group pages."""
    if not segs or segs[0] in GITLAB_RESERVED:
        return []
    if '-' in segs:
        i = segs.index('-')
        repo, rest, exact = segs[:i], segs[i + 1:], True
    else:
        i = next((k for k, x in enumerate(segs) if k >= 2 and x in LEGACY_ROUTES), len(segs))
        repo, rest = segs[:i], segs[i:]
        exact = repo[-1].endswith('.git') if repo else False
    if len(repo) < 2:
        return []
    repo = [x.lower() for x in repo]
    repo[-1] = strip_git(repo[-1])
    key = '/'.join(repo)
    return [('repo', ('gitlab:' if exact else 'gitlab?:') + key)] + gitlab_sub(key, rest)


def google_id(host, segs, query):
    for i, s in enumerate(segs):
        if s in ('d', 'folders', 'projects', 'documents', 'spreadsheets', 'presentations', 'files') and i + 1 < len(segs):
            cand = segs[i + 1]
            if cand == 'e' and i + 2 < len(segs):
                cand = segs[i + 2]
            if GOOGLE_ID_RE.match(cand):
                return cand
    m = re.search(r'(?:^|&)id=([A-Za-z0-9_-]{20,})', query or '')
    return m.group(1) if m else None


def parse_url(u):
    u = u.rstrip('.,;:!?*\'"')
    try:
        parts = urlsplit(u)
        host = (parts.hostname or '').lower()
    except ValueError:
        return []
    if host.startswith('www.'):
        host = host[4:]
    if not host or '.' not in host or host in BORING_HOSTS or host.endswith('.local'):
        return []
    if re.fullmatch(r'[\d.]+', host):
        return []
    segs = [unquote(s) for s in (parts.path or '').split('/') if s]
    if host == 'gitlab.com':
        if segs[:2] == ['api', 'v4']:
            return gitlab_api(segs[2:])
        return gitlab_path(segs)
    if host.endswith('.gitlab.io') or host.endswith('.github.io'):
        label = host.split('.')[0]
        unique = re.search(r'-[0-9a-f]{6}$', label)
        first = segs[0] if segs and '.' not in segs[0] else None
        pid = host if unique or not first else f'{host}/{first.lower()}'
        out = [('pages', pid)]
        page = page_id(host, segs)
        if page and page != pid:
            out.append(('page', page))
        return out
    if host == 'github.com':
        out = []
        if len(segs) >= 2 and segs[0].lower() not in GITHUB_RESERVED:
            out.append(('repo', f'github:{segs[0].lower()}/{strip_git(segs[1]).lower()}'))
            if len(segs) >= 4 and segs[2] == 'commit' and is_commitish(segs[3].lower()):
                out.append(('commit', segs[3][:7].lower()))
        return out
    if host == 'api.github.com' and len(segs) >= 3 and segs[0] == 'repos':
        return [('repo', f'github:{segs[1].lower()}/{strip_git(segs[2]).lower()}')]
    if host == 'raw.githubusercontent.com' and len(segs) >= 2:
        return [('repo', f'github:{segs[0].lower()}/{segs[1].lower()}')]
    if host in ('docs.google.com', 'drive.google.com', 'script.google.com', 'sheets.googleapis.com',
                'docs.googleapis.com', 'slides.googleapis.com', 'www.googleapis.com', 'googleapis.com',
                'drive.googleapis.com', 'forms.gle') or host.endswith('.googleusercontent.com'):
        gid = google_id(host, segs, parts.query)
        if gid:
            return [('gdoc', gid)]
    first = segs[0].lower() if segs and '.' not in segs[0] and len(segs[0]) <= 60 else ''
    out = [('site', f'{host}/{first}' if first else host)]
    if len(segs) >= 2:
        out.append(('page', page_id(host, segs)))
    return out


def page_id(host, segs):
    """A specific page: host plus the whole path, without query, index files or trailing slash."""
    segs = [x for x in segs if x not in ('index.html', 'index.htm')]
    if not segs:
        return None
    return (host + '/' + '/'.join(segs))[:160].lower()


def repo_from_ref(ref):
    """A glab -R value or a clone URL -> repo trace id."""
    ref = ref.strip('\'"')
    if ref.startswith('http'):
        for t, i in parse_url(ref):
            if t == 'repo':
                return i.replace('gitlab?:', 'gitlab:', 1)
        return None
    m = SSH_RE.match(ref)
    if m:
        host = 'gitlab' if m.group(1).startswith('gitlab') else 'github'
        return f'{host}:{strip_git(m.group(2)).lower()}'
    if re.fullmatch(r'[\w.-]+(?:/[\w.-]+)+', ref):
        return 'gitlab:' + strip_git(ref).lower()
    return None


# ---------------------------------------------------------------- shell parsing
CD_RE = re.compile(r'(?:^|[;&|(\n{]|&&|\|\||\bthen|\bdo)\s*cd\s+("[^"\n]+"|\'[^\'\n]+\'|[^\s;&|)]+)')
GIT_C_RE = re.compile(r'\bgit\s+-C\s+("[^"\n]+"|\'[^\'\n]+\'|\S+)')
CLONE_RE = re.compile(r'\bgit\s+clone\s+((?:--?[\w-]+(?:[= ](?!https?:|git@)\S+)?\s+)*)(\S+)(?:[ \t]+([^\s;&|>]+))?')
REMOTE_SET_RE = re.compile(r'\bgit\s+remote\s+(?:add|set-url)\s+\S+\s+(\S+)')
REMOTE_OUT_RE = re.compile(r'^\S+\s+(\S+)\s+\((?:fetch|push)\)', re.M)
PUSH_TO_RE = re.compile(r'^To\s+(\S+)', re.M)
GLAB_RE = re.compile(r'\bglab\s+(\w+)(?:\s+(\w+))?([^\n;&|]*)')
GLAB_REPO_RE = re.compile(r'(?:-R|--repo)[=\s]+(\S+)')
GLAB_API_RE = re.compile(r'\bglab\s+api\s+(?:-\S+\s+(?:\S+\s+)?)*["\']?/?([^\s"\']+)')
COMMIT_OUT_RE = re.compile(r'^\[[^\]\s]+(?: \(root-commit\))? ([0-9a-f]{7,40})\]', re.M)
RANGE_RE = re.compile(r'\b([0-9a-f]{7,40})\.\.\.?([0-9a-f]{7,40})\b')
HEAD_AT_RE = re.compile(r'HEAD is now at ([0-9a-f]{7,40})\b')
LOG_LINE_RE = re.compile(r'^(?:[*|\\/ ]*)(?:commit )?([0-9a-f]{7,40})\b', re.M)
JSON_SHA_RE = re.compile(r'"(?:id|sha|short_id|commit_id|parent_id|last_commit_id|head_sha)"\s*:\s*"([0-9a-f]{7,40})"')
GIT_REF_CMD_RE = re.compile(r'\bgit\s+(?:show|checkout|revert|cherry-pick|reset(?:\s+--\w+)?|diff|log|merge|rebase|branch\s+\S+|tag\s+\S+)\s+([0-9a-f]{7,40})\b')
PROJECT_ID_OUT_RE = re.compile(r'(?:"project_id"\s*:\s*|Project ID:\s*)(\d{5,})')
PIDMAP_RE = re.compile(r'\{\s*"id"\s*:\s*(\d{5,}),[^{}]{0,3000}?"path_with_namespace"\s*:\s*"([^"]+)"')
REDIRECT_RE = re.compile(r'(?<![<>&\d])(?:\d)?>{1,2}\s*([^\s&|;)>]+)')

WRITE_CMD_RE = re.compile(
    r'\bgit\s+(?:-C\s+\S+\s+)?(?:commit|push|merge|tag|rebase|cherry-pick|revert|rm|mv)\b'
    r'|\bglab\s+(?:issue|mr|release|repo|label|ci|variable|snippet)\s+(?:create|new|note|comment|close|reopen|update|merge|approve|delete|edit|run|retry|cancel|set|fork)\b'
    r'|\bcurl\b[^\n]*?(?:-X\s*(?:POST|PUT|PATCH|DELETE)\b|--request\s+(?:POST|PUT|PATCH|DELETE)\b|\s(?:-d|--data(?:-raw|-binary|-urlencode)?|-F|--form)\s)'
    r'|\brequests\.(?:post|put|patch|delete)\('
    r'|method\s*[:=]\s*[\'"](?:POST|PUT|PATCH|DELETE)'
    r'|\bglab\s+api\b[^\n]*?(?:-X\s*(?:POST|PUT|PATCH|DELETE)\b|--method\s+(?:POST|PUT|PATCH|DELETE)\b|\s(?:-f|-F|--field|--raw-field)\s)',
    re.I)
CREATE_CMD_RE = re.compile(
    r'\bglab\s+(?:issue|mr|repo|release)\s+(?:create|new)\b'
    r'|\bcurl\b[^\n]*?(?:-X\s*POST|--request\s+POST|\s(?:-d|--data(?:-raw)?|-F|--form)\s)[^\n]*/(?:issues|merge_requests|projects)\b'
    r'|\bglab\s+api\b[^\n]*?(?:-X\s*POST|\s(?:-f|-F|--field|--raw-field)\s)[^\n]*?(?:issues|merge_requests|projects)\b',
    re.I)
FILE_WRITE_RE = re.compile(r'\b(?:tee|cp|mv|rm|mkdir|touch|rsync)\b|\bsed\s+-i|open\([^)\n]*[\'"][wa]b?\+?[\'"]|\.write_text\(|\.write\(')


def unquote_path(p):
    return p.strip('\'"')


def resolve(cur, p):
    p = unquote_path(p)
    if p in ('', '~'):
        return HOME
    if p.startswith('~/'):
        p = HOME + p[1:]
    elif p.startswith('$HOME'):
        p = HOME + p[5:]
    elif p.startswith('${HOME}'):
        p = HOME + p[7:]
    if '$' in p or '`' in p or '*' in p:
        return None
    if not p.startswith('/'):
        p = posixpath.join(cur, p)
    return posixpath.normpath(p)


def writes_files(cmd):
    if FILE_WRITE_RE.search(cmd):
        return True
    for target in REDIRECT_RE.findall(cmd):
        if not target.startswith(('/dev/', '&', '/tmp/', '/proc/')):
            return True
    return False


# ---------------------------------------------------------------- agent names
VERSION_RE = re.compile(r'^\d+(?:\.\d+)*$')


def name_variants(name):
    toks = [t for t in re.split(r'[\s\-]+', name.lower()) if t]
    variants = [toks]
    fam = toks[0]
    if fam == 'claude' and len(toks) >= 3:
        variants.append(toks[1:])                     # "opus 4.8"
    if fam == 'gemini' and toks[-1] in ('pro', 'flash') and len(toks) >= 3:
        variants.append(toks[:-1])                    # "gemini 2.5"
    if fam == 'deepseek' and len(toks) >= 2 and toks[1].startswith('v'):
        variants.append(['deepseek', toks[1][1:]] + toks[2:])   # "deepseek 3.2"
    if fam == 'glm' and len(toks) >= 3:
        variants.append(toks[:2])                     # "glm 5.3"
    if fam == 'grok':
        variants.append(['grok'])                     # the only Grok in the main period
    if fam == 'muse' and len(toks) >= 3:
        variants.append(toks[:2])                     # "muse spark"
    return variants


def variant_regex(toks):
    parts = []
    for t in toks:
        if VERSION_RE.match(t):
            parts.append(r'[._-]'.join(re.escape(x) for x in t.split('.')))
        else:
            parts.append(re.escape(t))
    return r'(?<![\w.])' + r'[\s_-]*'.join(parts) + r'(?![._-]?\d)(?![a-z])'


FAMILY_RX = re.compile(r'(?<![\w.])(?:claude|opus|sonnet|haiku|fable|gpt|gemini|deepseek|glm|kimi|grok|muse|fine)', re.I)


def build_name_regex(names):
    alts = []
    for n in names:
        for v in name_variants(n):
            alts.append((len(' '.join(v)), variant_regex(v), n))
    alts.sort(reverse=True)  # longest first, so "gpt 5.1" wins over "gpt 5"
    pattern = '|'.join(f'(?P<n{i}>{rx})' for i, (_, rx, _) in enumerate(alts))
    return re.compile(pattern, re.I), {f'n{i}': n for i, (_, _, n) in enumerate(alts)}


def find_names(rx, groups, text):
    """Names in text. The full pattern is only tried where a model family word starts."""
    found = set()
    for c in FAMILY_RX.finditer(text):
        m = rx.match(text, c.start())
        if m:
            found.add(groups[m.lastgroup])
    return found


def split_comments(cmd):
    comment, code = [], []
    for line in cmd.split('\n'):
        s = line.lstrip()
        if s.startswith('#') and not s.startswith('#!'):
            comment.append(s)
        else:
            k = line.find(' # ')
            if k >= 0:
                comment.append(line[k:])
                code.append(line[:k])
            else:
                code.append(line)
    return '\n'.join(comment), '\n'.join(code)


# ---------------------------------------------------------------- per-agent pass
class AgentParser:
    def __init__(self, agent, all_names):
        self.agent = agent
        others = [n for n in all_names if n != agent]
        self.name_rx, self.name_groups = build_name_regex(others)
        self.cwd = {}                 # session -> current directory
        self.dir_repo = {}            # clone root -> repo trace id
        self.rows = []

    def emit(self, t, day, step, typ, tid, op, src, k=0):
        if any(c in tid for c in '\t\n\r'):
            tid = re.sub(r'\s+', '_', tid)
        self.rows.append((t, day, step, typ, tid, op, src, k))

    def repo_of_dir(self, d):
        while d and d != '/':
            r = self.dir_repo.get(d)
            if r:
                return r
            d = posixpath.dirname(d)
        return None

    def learn(self, d, repo):
        if d and repo and d not in ('/', HOME, '/tmp'):
            self.dir_repo[d] = repo

    def names(self, t, day, step, text, src):
        if not text:
            return 0
        found = find_names(self.name_rx, self.name_groups, text)
        for n in found:
            self.emit(t, day, step, 'name', n, 'read', src, len(found))
        return len(found)

    def urls(self, text):
        out = []
        for u in URL_RE.findall(text):
            out += parse_url(u)
        for m in SSH_RE.finditer(text):
            host = 'gitlab' if m.group(1).startswith('gitlab') else 'github'
            out.append(('repo', f'{host}:{strip_git(m.group(2)).lower()}'))
        return out

    def shell(self, t, day, step, sess, cmd, out):
        i0 = len(self.rows)
        n = self._shell(t, day, step, sess, cmd, out)
        # a numeric project ID next to exactly one repository in the same step is a vote for that mapping
        here = self.rows[i0:]
        pids = {r[4] for r in here if r[3] == 'project'}
        repos = {r[4] for r in here if r[3] == 'repo' and r[4].startswith('gitlab') and r[6] != 'cwd'}
        if len(pids) == 1 and len(repos) == 1:
            self.emit(t, day, step, 'pidvote', f'{pids.pop()}|{repos.pop()}', 'map', 'cmd')
        return n

    def _shell(self, t, day, step, sess, cmd, out):
        n = 0
        cur = self.cwd.get(sess, HOME)
        dirs = [cur]
        for m in CD_RE.finditer(cmd):
            nd = resolve(dirs[-1], m.group(1))
            if nd:
                dirs.append(nd)
        for m in GIT_C_RE.finditer(cmd):
            nd = resolve(dirs[-1], m.group(1))
            if nd:
                dirs.append(nd)
        self.cwd[sess] = dirs[-1] if len(dirs) > 1 and CD_RE.search(cmd) else cur
        last = dirs[-1]

        # learn clone roots
        for m in CLONE_RE.finditer(cmd):
            repo = repo_from_ref(m.group(2))
            if repo:
                target = m.group(3)
                base = strip_git(m.group(2).rstrip('/').split('/')[-1].split(':')[-1])
                d = resolve(last, target if target and not target.startswith('-') else base)
                self.learn(d, repo)
        for m in REMOTE_SET_RE.finditer(cmd):
            self.learn(last, repo_from_ref(m.group(1)))
        if out:
            if 'git remote' in cmd:
                for u in REMOTE_OUT_RE.findall(out):
                    self.learn(last, repo_from_ref(u))
            if 'git push' in cmd:
                for u in PUSH_TO_RE.findall(out):
                    self.learn(last, repo_from_ref(u))

        writing = bool(WRITE_CMD_RE.search(cmd))
        creating = bool(CREATE_CMD_RE.search(cmd))
        file_write = writes_files(cmd)

        # repositories reached through local folders
        seen_repos = set()
        for d in dict.fromkeys(dirs):
            r = self.repo_of_dir(d)
            if r and r not in seen_repos:
                seen_repos.add(r)
                op = 'write' if (writing or file_write) and d != HOME else 'read'
                self.emit(t, day, step, 'repo', r, op, 'cwd')
                n += 1

        # URLs and identifiers written in the command
        api_write = writing
        for typ, tid in self.urls(cmd):
            op = 'read'
            if api_write and typ in ('repo', 'issue', 'pipeline', 'project', 'gdoc'):
                op = 'write'
            self.emit(t, day, step, typ, tid, op, 'cmd')
            n += 1
        for m in GLAB_RE.finditer(cmd):
            sub, action, rest = m.group(1), m.group(2) or '', m.group(3) or ''
            rm = GLAB_REPO_RE.search(rest)
            repo = repo_from_ref(rm.group(1)) if rm else self.repo_of_dir(last)
            if sub == 'api':
                continue
            op = 'create' if action in ('create', 'new') else 'write' if WRITE_CMD_RE.search(m.group(0)) else 'read'
            if repo:
                self.emit(t, day, step, 'repo', repo, 'write' if op != 'read' else 'read', 'cmd')
                n += 1
                num = re.match(r'\s*(\d+)\b', rest)
                if num and sub in ('issue', 'mr') and action not in ('create', 'new', 'list'):
                    key = repo.split(':', 1)[1]
                    self.emit(t, day, step, 'issue', f'{key}{"#" if sub == "issue" else "!"}{num.group(1)}', op, 'cmd')
                    n += 1
        for m in GLAB_API_RE.finditer(cmd):
            path = m.group(1).split('?')[0]
            segs = [s for s in path.split('/') if s]
            if segs[:2] == ['api', 'v4']:
                segs = segs[2:]
            if segs[:2] == ['projects', ':id'] or segs[:2] == ['projects', ':fullpath']:
                r = self.repo_of_dir(last)
                if r:
                    self.emit(t, day, step, 'repo', r, 'write' if writing else 'read', 'cmd')
                    n += 1
                continue
            for typ, tid in gitlab_api(segs):
                self.emit(t, day, step, typ, tid, 'write' if writing else 'read', 'cmd')
                n += 1
        for h in GIT_REF_CMD_RE.findall(cmd):
            if is_commitish(h):
                self.emit(t, day, step, 'commit', h[:7], 'read', 'cmd')
                n += 1

        # output
        if out:
            if 'git commit' in cmd or 'git merge' in cmd:
                for h in COMMIT_OUT_RE.findall(out):
                    if is_commitish(h):
                        self.emit(t, day, step, 'commit', h[:7], 'create', 'out')
                        n += 1
            if 'git push' in cmd:
                for a, b in RANGE_RE.findall(out):
                    for h in (a, b):
                        if is_commitish(h):
                            self.emit(t, day, step, 'commit', h[:7], 'write', 'out')
                            n += 1
                for u in PUSH_TO_RE.findall(out):
                    r = repo_from_ref(u)
                    if r:
                        self.emit(t, day, step, 'repo', r, 'write', 'out')
                        n += 1
            if 'git log' in cmd or 'git show' in cmd or 'git reflog' in cmd:
                for h in LOG_LINE_RE.findall(out):
                    if is_commitish(h):
                        self.emit(t, day, step, 'commit', h[:7], 'seen', 'out')
                        n += 1
            elif 'git pull' in cmd or 'git fetch' in cmd or 'git merge' in cmd or 'git rebase' in cmd:
                for a, b in RANGE_RE.findall(out):
                    for h in (a, b):
                        if is_commitish(h):
                            self.emit(t, day, step, 'commit', h[:7], 'seen', 'out')
                            n += 1
            for h in HEAD_AT_RE.findall(out):
                if is_commitish(h):
                    self.emit(t, day, step, 'commit', h[:7], 'seen', 'out')
                    n += 1
            for h in JSON_SHA_RE.findall(out):
                if is_commitish(h):
                    self.emit(t, day, step, 'commit', h[:7], 'seen', 'out')
                    n += 1
            for typ, tid in self.urls(out):
                op = 'create' if creating and typ in ('issue', 'repo') else 'seen'
                self.emit(t, day, step, typ, tid, op, 'out')
                n += 1
            for pid in PROJECT_ID_OUT_RE.findall(out):
                self.emit(t, day, step, 'project', pid, 'seen', 'out')
                n += 1
            for pid, path in PIDMAP_RE.findall(out):
                self.emit(t, day, step, 'pidmap', f'{pid}|{path.lower()}', 'map', 'out')

        # names
        comment, code = split_comments(cmd)
        self.names(t, day, step, comment, 'comment')
        self.names(t, day, step, code, 'code')
        if out:
            self.names(t, day, step, out, 'out')
        return n

    def typed(self, t, day, step, text):
        n = 0
        stripped = text.strip()
        nav = bool(URL_RE.fullmatch(stripped)) or bool(re.fullmatch(r'[\w.-]+\.[a-z]{2,}(/\S*)?', stripped))
        if nav and not stripped.startswith('http'):
            stripped = 'https://' + stripped
        for typ, tid in self.urls(stripped if nav else text):
            self.emit(t, day, step, typ, tid, 'read' if nav else 'write', 'typed')
            n += 1
        self.names(t, day, step, text, 'typed')
        return n


def parse_agent(path):
    agent = None
    with open(config.STEPS / 'summary.json', encoding='utf-8') as f:
        summary = json.load(f)
    names = sorted(summary['agents'])
    for n, a in summary['agents'].items():
        if a['file'] == path:
            agent = n
    p = AgentParser(agent, names)
    activity = collections.defaultdict(collections.Counter)
    t0 = time.time()
    with gzip.open(config.STEPS / path, 'rt', encoding='utf-8') as f:
        for step, line in enumerate(f):
            r = json.loads(line)
            when = config.parse_utc(r['t'])
            local_day = when.astimezone(PACIFIC).date()
            if not config.DEV_START <= local_day <= config.DEV_END:
                continue
            t = when.timestamp()
            day = local_day.isoformat()
            k = r['k']
            a = activity[day]
            a['steps'] += 1
            if k in WAITING:
                a['waiting'] += 1
                if k == 'restart':
                    p.cwd.pop(r['s'], None)
                continue
            a['working'] += 1
            n = 0
            if k == 'shell':
                a['shell'] += 1
                n = p.shell(t, day, step, r['s'], r.get('x') or '', (r.get('o') or '') + ('\n' + r['e'] if r.get('e') else ''))
            elif k == 'type':
                a['gui'] += 1
                a['typed'] += 1
                n = p.typed(t, day, step, r.get('x') or '')
            elif k in GUI_KINDS:
                a['gui'] += 1
                if k == 'get_pixel_coords_of_element':
                    p.names(t, day, step, r.get('x') or '', 'gui')
            else:
                a['other'] += 1
                if r.get('o'):
                    for typ, tid in p.urls(r['o']):
                        p.emit(t, day, step, typ, tid, 'seen', 'out')
                        n += 1
            if n:
                a['trace_steps'] += 1
    MENTIONS.mkdir(parents=True, exist_ok=True)
    stem = path.replace('.jsonl.gz', '')
    with gzip.open(MENTIONS / f'{stem}.tsv.gz', 'wt', encoding='utf-8') as f:
        f.write('agent\tt\tday\tstep\ttype\tid\top\tsrc\tk\n')
        for row in p.rows:
            f.write(agent + '\t' + '\t'.join(str(x) for x in row) + '\n')
    with open(MENTIONS / f'{stem}.activity.json', 'w', encoding='utf-8') as f:
        json.dump({'agent': agent, 'days': activity, 'clone_roots': len(p.dir_repo)}, f)
    return agent, len(p.rows), len(p.dir_repo), round(time.time() - t0, 1)


def main():
    files = sys.argv[1:]
    if not files:
        with open(config.STEPS / 'summary.json', encoding='utf-8') as f:
            summary = json.load(f)
        files = sorted((a['file'] for a in summary['agents'].values()),
                       key=lambda fn: -(config.STEPS / fn).stat().st_size)
    with ProcessPoolExecutor(max_workers=4) as ex:
        for agent, rows, roots, secs in ex.map(parse_agent, files):
            print(f'{agent:<24s} {rows:>8d} mentions  {roots:>4d} clone roots  {secs:>6.1f} s', flush=True)


if __name__ == '__main__':
    main()
