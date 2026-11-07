"""Version and workflow evidence from recorded, complete API responses.

Observing a server's job/commit/document record does not make its reader the
producer. Mutations require one direct request, a matching response and an
explicit resulting object/state. No service is contacted by this module.
"""
import base64
import binascii
import hashlib
import json
import re
from urllib.parse import unquote, urlsplit

from .provenance_common import command_segments, file_key, git_key, in_scope, json_objects, make_event

SHA = re.compile(r'^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$')
HASH256 = re.compile(r'^[0-9a-fA-F]{64}$')
FAILURE = re.compile(r'(?mi)(?:^curl: \([1-9]\d*\)|^HTTP/\S+\s+[45]\d\d\b|\b(?:401 Unauthorized|403 Forbidden|404 Not Found)\b)')
HTTP_OK = re.compile(r'(?m)^HTTP/\S+\s+2\d\d\b|^2\d\d\s*$')


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def full_sha(value):
    return isinstance(value, str) and bool(SHA.fullmatch(value))


def repo_url(url):
    """Namespace only: remove credentials, query strings and service routes."""
    if not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or '').lower()
    path = parts.path.split('/-/')[0].strip('/')
    if not host or not path or '/api/' in parts.path:
        return None
    if host == 'github.com':
        path = '/'.join(path.split('/')[:2])
    if len(path.split('/')) < 2:
        return None
    if path.endswith('.git'):
        path = path[:-4]
    service = 'gitlab' if host == 'gitlab.com' else ('github' if host == 'github.com' else host)
    return service + ':' + path.lower()


def relation(kind, source, target):
    return {'kind': kind, 'source': source, 'target': target}


def _request(argv):
    """Recognize direct curl/glab API calls, never quoted script text."""
    if not argv:
        return None
    program = argv[0].rsplit('/', 1)[-1]
    if program == 'curl':
        args = argv[1:]
        urls = [arg for arg in args if arg.startswith(('http://', 'https://'))]
        if len(urls) != 1:
            return None
        url = urls[0]
        try:
            parsed = urlsplit(url)
            path, host = parsed.path, (parsed.hostname or '').lower()
        except ValueError:
            return None
    elif program == 'glab' and 'api' in argv:
        start = argv.index('api')
        args = argv[start + 1:]
        if not args or args[0].startswith('-'):
            return None
        url = 'https://gitlab.com/api/v4/' + args[0].lstrip('/')
        path = urlsplit(url).path
        host = 'gitlab.com'
    else:
        return None
    method, payload, raw, explicit, destination = 'GET', {}, None, False, False
    status_capture, header_capture = False, False
    response_transform_safe = True
    for index, arg in enumerate(args):
        value = args[index + 1] if index + 1 < len(args) else ''
        if program == 'glab':
            if arg in ('--jq', '-q'):
                response_transform_safe &= value == '.'
            elif arg.startswith('--jq='):
                response_transform_safe &= arg.split('=', 1)[1] == '.'
            elif arg.startswith('-q') and len(arg) > 2:
                response_transform_safe &= arg[2:] == '.'
            elif arg in ('--template', '-t') or arg.startswith('--template=') or (arg.startswith('-t') and len(arg) > 2):
                response_transform_safe = False
        elif program == 'curl' and (arg in ('-w', '--write-out') or arg.startswith('--write-out=') or (arg.startswith('-w') and len(arg) > 2)):
            formatting = value if arg in ('-w', '--write-out') else (arg.split('=', 1)[1] if arg.startswith('--write-out=') else arg[2:])
            # curl's custom write-out text can fabricate a complete JSON record
            # after the real response just as a glab formatter can.
            formatting = formatting.replace(r'\n', '\n').replace(r'\r', '\r').strip()
            response_transform_safe &= formatting in ('%{http_code}', '%{response_code}')
        if arg in ('-X', '--request', '--method'):
            method, explicit = value.upper(), True
        elif arg.startswith('-X') and len(arg) > 2:
            method, explicit = arg[2:].upper(), True
        elif arg.startswith('--method=') or arg.startswith('--request='):
            method, explicit = arg.split('=', 1)[1].upper(), True
        elif arg in ('-d', '--data', '--data-raw', '--data-binary', '--json'):
            raw = value
            if not explicit:
                method = 'POST'
        elif arg in ('-f', '-F', '--field', '--raw-field') and program == 'glab':
            if '=' in value:
                key, val = value.split('=', 1)
                payload[key] = val
            if not explicit:
                method = 'POST'
        elif arg in ('-o', '--output', '-O', '--remote-name'):
            destination = True
        elif arg in ('-w', '--write-out'):
            status_capture = '%{http_code}' in value or '%{response_code}' in value
        elif arg.startswith('-w') and len(arg) > 2:
            status_capture = '%{http_code}' in arg or '%{response_code}' in arg
        elif arg in ('-i', '--include') or (arg == '-D' and value == '-'):
            header_capture = True
        elif arg in ('-I', '--head'):
            method, explicit = 'HEAD', True
    if raw:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                payload.update(decoded)
        except (ValueError, TypeError):
            pass
    # Explicit method wins even when it preceded a data flag.
    for index, arg in enumerate(args):
        if arg in ('-X', '--request', '--method') and index + 1 < len(args):
            method = args[index + 1].upper()
    match = re.search(r'/api/v4/projects/([^/]+)(/.*)?$', path)
    return {'host': host, 'path': path, 'url': host + path, 'method': method,
            'payload': payload, 'project': unquote(match.group(1)) if match else None,
            'tail': match.group(2) or '' if match else '', 'destination': destination,
            'status_capture': status_capture, 'header_capture': header_capture,
            'response_transform_safe': response_transform_safe}


class StructuredExtractor:
    def __init__(self, actor):
        self.actor = actor
        self.project_repos = {}
        self.last_state = {}
        self.last_snapshot = {}
        self.suggestions = {}
        self.context_repo = None
        self.relation_authority = False

    def set_context(self, repo=None, **unused):
        self.context_repo = repo

    def _repo(self, obj, req):
        project = obj.get('project') if isinstance(obj.get('project'), dict) else {}
        repo = next((r for r in (repo_url(obj.get('web_url')), repo_url(obj.get('http_url_to_repo')),
                                repo_url(project.get('web_url'))) if r), None)
        host = req['host'] if req else 'gitlab.com'
        prefix = 'gitlab' if host == 'gitlab.com' else host
        path = obj.get('path_with_namespace') or project.get('path_with_namespace')
        if not repo and isinstance(path, str) and '/' in path:
            repo = prefix + ':' + path.lower().strip('/')
        pid = obj.get('project_id') or project.get('id') or (req and req['project'])
        if not repo and req and req['project'] and not req['project'].isdigit():
            repo = prefix + ':' + req['project'].lower()
        if self.relation_authority and repo and pid and str(pid).isdigit():
            self.project_repos[(host, str(pid))] = repo
        if not repo and pid:
            repo = self.project_repos.get((host, str(pid))) or prefix + ':pid:' + str(pid)
        return repo or self.context_repo

    def extract(self, record):
        if not in_scope(record['t']):
            return []
        command = record.get('x') or ''
        output = record.get('o') or ''
        error = record.get('e') or ''
        segments = command_segments(command) if record.get('k') == 'shell' else []
        requests = [r for r in (_request(args) for args in segments) if r]
        req = requests[0] if len(requests) == 1 else None
        if req:
            # Output from an unrelated echo/cat/script must not be mistaken for
            # the request's response. Ordinary directory setup is harmless.
            req['response_attributable'] = req['response_transform_safe'] and all(_request(args) or args[0] == 'cd' for args in segments)
        failed = bool(FAILURE.search(output + '\n' + error))
        self.relation_authority = bool(req and req['response_attributable'] and not failed)
        events = []

        def emit(family, operation, status, entity, **kwargs):
            kwargs.setdefault('relation_authority', self.relation_authority)
            event = make_event(self.actor, record, family, operation, status, entity, **kwargs)
            events.append(event)
            return event

        if req and req['method'] != 'GET' and failed:
            emit('workflow', 'attempt', 'failed', req['url'], evidence=[{'field': 'x', 'start': 0, 'end': len(command)}])

        for obj, start, end in json_objects(output):
            values = obj if isinstance(obj, list) else [obj]
            # A list response can be observed, but cannot verify a single mutation.
            single = isinstance(obj, dict)
            for value in values:
                if not isinstance(value, dict):
                    continue
                if 'error' in value or value.get('message') in ('401 Unauthorized', '403 Forbidden', '404 Not Found'):
                    continue
                evidence = [{'field': 'o', 'start': start, 'end': end}]
                repo = self._repo(value, req)
                observed_req = req if req and req['response_attributable'] and not failed else None
                mutation = observed_req if single and observed_req and observed_req['method'] in ('POST', 'PUT', 'PATCH', 'DELETE') else None
                if mutation and mutation['project']:
                    project = str(value.get('project_id', mutation['project']))
                    expected = ('gitlab' if mutation['host'] == 'gitlab.com' else mutation['host']) + ':' + mutation['project'].lower()
                    if project.isdigit() and mutation['project'].isdigit() and project != mutation['project']:
                        mutation = None
                    elif not mutation['project'].isdigit() and repo != expected:
                        mutation = None
                self._git_record(value, repo, observed_req, mutation, evidence, emit)
                self._document(value, observed_req, mutation, evidence, emit)
                self._apply_suggestion(value, observed_req, mutation, evidence, emit)
                self._dependencies(value, repo, segments, output, failed, evidence, emit)

        self.relation_authority = False
        self._artifact_download(req, output, failed, emit)
        self._http_snapshot(req, output, failed, emit)
        self._install(segments, output, failed or bool(re.search(r'(?mi)^(?:ERROR:|Error response from daemon|npm ERR!|npm error)', output + '\n' + error)), emit)
        # Deterministic de-duplication when nested records expose the same object.
        return list({event['event_id']: event for event in events}.values())

    def _git_record(self, obj, repo, req, mutation, evidence, emit):
        if not repo:
            return
        tail = req['tail'] if req else ''
        # Commit records can be nested inside jobs, pipelines, releases and MRs.
        commit = obj.get('commit') if isinstance(obj.get('commit'), dict) else None
        if commit:
            self._commit(commit, repo, evidence, emit)
        if isinstance(obj.get('parent_ids'), list) and full_sha(obj.get('id')):
            self._commit(obj, repo, evidence, emit)
        if isinstance(obj.get('file_path'), str) and any(k in obj for k in ('content_sha256', 'blob_id', 'commit_id')):
            self._file(obj, repo, req, mutation, evidence, emit)
        elif mutation and '/repository/files/' in tail and obj.get('file_path'):
            self._file_write(obj, repo, mutation, evidence, emit)
        pipeline = obj.get('pipeline') if isinstance(obj.get('pipeline'), dict) else None
        if pipeline:
            self._pipeline(pipeline, repo, evidence, emit)
        if obj.get('id') is not None and ('stage' in obj or 'artifacts_file' in obj) and ('status' in obj or 'pipeline' in obj):
            job = 'job:' + repo + ':' + str(obj['id'])
            sha = (commit or {}).get('id') or (pipeline or {}).get('sha')
            inputs = [git_key(repo, sha)] if full_sha(sha) else []
            outputs, relations = [job], [relation('derived_from', job, key) for key in inputs]
            artifact = obj.get('artifacts_file')
            if isinstance(artifact, dict) and artifact.get('filename') and artifact.get('size', 0):
                key = 'artifact:' + repo + ':job:' + str(obj['id'])
                outputs.append(key)
                relations.append(relation('derived_from', key, job))
            emit('ci', 'observe', 'success', job, version=str(obj['id']), inputs=inputs, outputs=outputs,
                 relations=relations, evidence=evidence, workflow_status=obj.get('status'), stage=obj.get('stage'))
        elif obj.get('id') is not None and full_sha(obj.get('sha')) and 'status' in obj and ('ref' in obj or '/pipelines' in tail):
            self._pipeline(obj, repo, evidence, emit)
        if obj.get('iid') is not None and 'state' in obj:
            self._workflow(obj, repo, req, mutation, evidence, emit)
        if obj.get('id') is not None and ('body' in obj or 'suggestions' in obj) and re.search(r'/(?:merge_requests|issues)/\d+/(?:notes|discussions)', tail):
            self._review(obj, repo, req, mutation, evidence, emit)
        if isinstance(obj.get('notes'), list) and re.search(r'/(?:merge_requests|issues)/\d+/discussions', tail):
            for note in obj['notes']:
                if isinstance(note, dict) and note.get('id') is not None:
                    self._review(note, repo, req, None, evidence, emit)
        if obj.get('tag_name') and isinstance(commit, dict):
            sha = commit.get('id')
            if full_sha(sha):
                key = 'release:' + repo + ':' + str(obj['tag_name']) + '@' + sha
                emit('release', 'observe', 'success', key.split('@')[0], version=sha,
                     inputs=[git_key(repo, sha)], outputs=[key],
                     relations=[relation('derived_from', key, git_key(repo, sha))], evidence=evidence)
        if obj.get('id') is not None and isinstance(obj.get('environment'), dict) and full_sha(obj.get('sha')):
            key = 'deployment:' + repo + ':' + str(obj['id'])
            emit('deployment', 'observe', 'success', key, version=obj['sha'], outputs=[key],
                 relations=[relation('derived_from', key, git_key(repo, obj['sha']))],
                 evidence=evidence, workflow_status=obj.get('status'))
        self._packages(obj, repo, req, mutation, evidence, emit)

    def _commit(self, obj, repo, evidence, emit):
        sha = obj.get('id')
        if not full_sha(sha):
            return
        key = git_key(repo, sha)
        parent_ids = obj.get('parent_ids') if isinstance(obj.get('parent_ids'), list) else []
        parents = [git_key(repo, parent) for parent in parent_ids if full_sha(parent)]
        emit('git', 'observe', 'success', repo, version=sha.lower(), outputs=[key],
             relations=[relation('derived_from', key, parent) for parent in parents], evidence=evidence)

    def _pipeline(self, obj, repo, evidence, emit):
        if obj.get('id') is None:
            return
        key = 'pipeline:' + repo + ':' + str(obj['id'])
        sha = obj.get('sha')
        inputs = [git_key(repo, sha)] if full_sha(sha) else []
        emit('ci', 'observe', 'success', key, version=str(obj['id']), outputs=[key], inputs=inputs,
             relations=[relation('derived_from', key, parent) for parent in inputs],
             evidence=evidence, workflow_status=obj.get('status'))

    def _file(self, obj, repo, req, mutation, evidence, emit):
        path = obj['file_path']
        entity = 'file:' + repo + ':' + path
        declared = obj.get('content_sha256')
        blob = obj.get('blob_id')
        sha = obj.get('commit_id')
        # Provider content checksums are distinguished from raw-byte checksums.
        if isinstance(declared, str) and HASH256.fullmatch(declared):
            key = file_key(repo, path, declared, 'gitlab_content_sha256')
            version = declared.lower()
        elif full_sha(blob):
            key = file_key(repo, path, blob, 'git_blob')
            version = blob.lower()
        else:
            return
        keys, relations = [key], []
        if full_sha(sha):
            relations.append(relation('derived_from', key, git_key(repo, sha)))
        content = None
        if obj.get('encoding') == 'base64' and isinstance(obj.get('content'), str):
            try:
                content = base64.b64decode(obj['content'], validate=True)
            except (ValueError, binascii.Error):
                pass
        elif obj.get('encoding') in ('text', 'utf-8') and isinstance(obj.get('content'), str):
            content = obj['content'].encode()
        if content is not None:
            raw_hash = digest(content)
            # This explicitly records the representation rather than assuming the
            # server checksum always hashes the raw Git bytes.
            raw_key = file_key(repo, path, raw_hash)
            content_key = 'content:sha256:' + raw_hash
            keys += [raw_key, content_key]
            relations.append(relation('same_content', raw_key, content_key))
            if isinstance(declared, str) and declared.lower() == raw_hash:
                relations.append(relation('same_content', key, raw_key))
        matched = req and '/repository/files/' in req['tail'] and unquote(req['tail'].split('/repository/files/', 1)[1]) == path
        operation = 'acquire' if matched and req['method'] == 'GET' and content is not None else 'observe'
        if matched and mutation and mutation['method'] in ('PUT', 'POST'):
            operation = 'generate'
        emit('file', operation, 'success', entity, version=version,
             inputs=keys if operation == 'acquire' else [], outputs=keys if operation != 'acquire' else [],
             relations=relations, evidence=evidence, content_length=len(content) if content is not None else None)

    def _file_write(self, obj, repo, req, evidence, emit):
        path = obj['file_path']
        requested = unquote(req['tail'].split('/repository/files/', 1)[1])
        if path != requested or req['method'] not in ('PUT', 'POST') or not obj.get('branch'):
            return
        payload = req['payload']
        content = payload.get('content')
        if not isinstance(content, str):
            return
        try:
            data = base64.b64decode(content, validate=True) if payload.get('encoding') == 'base64' else content.encode()
        except (ValueError, binascii.Error):
            return
        checksum = digest(data)
        key = file_key(repo, path, checksum)
        content_key = 'content:sha256:' + checksum
        emit('file', 'generate', 'success', 'file:' + repo + ':' + path, version=checksum,
             outputs=[key, content_key], relations=[relation('same_content', key, content_key)],
             evidence=evidence, content_length=len(data))

    def _workflow(self, obj, repo, req, mutation, evidence, emit):
        is_mr = 'merge_status' in obj or 'source_branch' in obj or 'merge_requests' in (req or {}).get('tail', '')
        kind, route = ('mr', 'merge_requests') if is_mr else ('issue', 'issues')
        entity = kind + ':' + repo + ':' + str(obj['iid'])
        state = {k: obj[k] for k in ('state', 'sha', 'merge_commit_sha', 'squash_commit_sha', 'resolved') if k in obj}
        if isinstance(obj.get('assignees'), list):
            state['assignees'] = sorted(str(a.get('id')) for a in obj['assignees'] if isinstance(a, dict))
        previous = self.last_state.get(entity) if self.relation_authority else None
        version = digest(stable(state))
        key = entity + '@state:' + version
        old_state = previous[0] if previous else None
        changed = previous is not None and old_state != state
        inputs, outputs, relations = [], [key], []
        sha = obj.get('sha')
        if full_sha(sha):
            relations.append(relation('derived_from', key, git_key(repo, sha)))
        operation = 'observe'
        tail = (mutation or {}).get('tail', '')
        if mutation and re.fullmatch(r'/' + route, tail) and mutation['method'] == 'POST':
            operation = 'generate'
        elif mutation and tail == '/' + route + '/' + str(obj['iid']) and mutation['method'] in ('PUT', 'PATCH'):
            payload = mutation['payload']
            expected = {'close': 'closed', 'reopen': 'opened'}.get(payload.get('state_event'))
            if expected == obj.get('state'):
                operation = 'transform'
            elif 'assignee_ids' in payload and 'assignees' in state:
                requested = payload['assignee_ids']
                if isinstance(requested, list) and sorted(map(str, requested)) == state['assignees']:
                    operation = 'assign'
        elif mutation and tail == '/merge_requests/' + str(obj['iid']) + '/merge' and obj.get('state') == 'merged':
            result_sha = obj.get('merge_commit_sha') or obj.get('squash_commit_sha')
            if full_sha(sha) and full_sha(result_sha):
                operation = 'incorporate'
                inputs.append(git_key(repo, sha))
                outputs.append(git_key(repo, result_sha))
                relations.append(relation('derived_from', git_key(repo, result_sha), git_key(repo, sha)))
        event = emit('workflow', operation, 'success', entity, version=version, inputs=inputs,
                     outputs=outputs, relations=relations, evidence=evidence, state=state,
                     state_changed=changed, previous_event_id=previous[1] if changed else None)
        if self.relation_authority:
            self.last_state[entity] = (state, event['event_id'])

    def _review(self, obj, repo, req, mutation, evidence, emit):
        match = re.search(r'/(merge_requests|issues)/(\d+)/(notes|discussions)', req['tail'])
        if not match:
            return
        parent = ('mr' if match.group(1) == 'merge_requests' else 'issue') + ':' + repo + ':' + match.group(2)
        entity = 'review:' + parent + ':' + str(obj['id'])
        body = obj.get('body') if isinstance(obj.get('body'), str) else ''
        version = digest(body)
        key = entity + '@sha256:' + version
        position = obj.get('position') if isinstance(obj.get('position'), dict) else {}
        inputs = [git_key(repo, position['head_sha'])] if full_sha(position.get('head_sha')) else []
        operation = 'review' if mutation and mutation['method'] == 'POST' and re.search(r'/(?:notes|discussions)$', mutation['tail']) and isinstance(obj.get('body'), str) and mutation['payload'].get('body') == body else 'observe'
        suggestion_items = obj.get('suggestions') if isinstance(obj.get('suggestions'), list) else []
        suggestions = [{'id': str(item['id']), 'applied': item.get('applied')} for item in suggestion_items if isinstance(item, dict) and item.get('id') is not None]
        if self.relation_authority:
            for suggestion in suggestions:
                self.suggestions[(req['host'], suggestion['id'])] = (repo, key)
        emit('review', operation, 'success', entity, version=version, inputs=inputs, outputs=[key],
             evidence=evidence, verbal_channel=True, resolved=obj.get('resolved'), suggestions=suggestions)

    def _apply_suggestion(self, obj, req, mutation, evidence, emit):
        if not mutation or mutation['method'] != 'PUT' or obj.get('applied') is not True:
            return
        match = re.fullmatch(r'/api/v4/suggestions/(\d+)/apply', mutation['path'])
        if not match or str(obj.get('id')) != match.group(1):
            return
        previous = self.suggestions.get((mutation['host'], match.group(1)))
        if not previous:
            return
        repo, note_key = previous
        entity = 'suggestion:' + repo + ':' + match.group(1)
        key = entity + '@applied'
        emit('review', 'incorporate', 'success', entity, version='applied', inputs=[note_key], outputs=[key],
             relations=[relation('derived_from', key, note_key)], evidence=evidence, verbal_channel=True)

    def _packages(self, obj, repo, req, mutation, evidence, emit):
        files = obj.get('package_files')
        if not isinstance(files, list):
            files = [obj] if any(k in obj for k in ('file_sha256', 'file_md5')) else []
        for value in files:
            if not isinstance(value, dict) or not isinstance(value.get('file_name'), str) or not value['file_name']:
                continue
            checksum = value.get('file_sha256')
            if not isinstance(checksum, str) or not HASH256.fullmatch(checksum):
                continue
            entity = 'package:' + repo + ':' + str(obj.get('name', '')) + ':' + str(obj.get('version', '')) + ':' + value['file_name']
            key = entity + '@sha256:' + checksum.lower()
            content_key = 'content:sha256:' + checksum.lower()
            pipelines = obj.get('pipelines') if isinstance(obj.get('pipelines'), list) else []
            if isinstance(obj.get('pipeline'), dict):
                pipelines = list(pipelines) + [obj['pipeline']]
            parents = []
            for pipeline in pipelines:
                if not isinstance(pipeline, dict):
                    continue
                if full_sha(pipeline.get('sha')):
                    parents.append(git_key(repo, pipeline['sha']))
                elif pipeline.get('id') is not None:
                    parents.append('pipeline:' + repo + ':' + str(pipeline['id']))
            emit('package', 'observe', 'success', entity, version=checksum.lower(), outputs=[key, content_key], inputs=parents,
                 relations=[relation('same_content', key, content_key)] + [relation('derived_from', key, parent) for parent in parents], evidence=evidence,
                 publication_observed=True, producer_unknown=True)

    def _dependencies(self, obj, repo, segments, output, failed, evidence, emit):
        """Captured npm lock dependencies; installation needs the same-step lock."""
        if not repo or not isinstance(obj.get('packages'), dict) or not obj.get('lockfileVersion'):
            return
        installing = any(args[:2] == ['npm', 'ci'] for args in segments)
        lock_read = any(args == ['cat', 'package-lock.json'] for args in segments)
        only_expected = all(args[0] == 'cd' or args == ['cat', 'package-lock.json'] or args[:2] == ['npm', 'ci'] for args in segments)
        installed = bool(installing and lock_read and only_expected and not failed and re.search(r'\badded \d+ packages?\b', output) and not re.search(r'(?i)npm (?:ERR!|error)', output))
        for name, package in obj['packages'].items():
            if not name or not isinstance(package, dict):
                continue
            integrity = package.get('integrity')
            if not isinstance(integrity, str):
                continue
            match = re.fullmatch(r'(sha256|sha512)-([A-Za-z0-9+/]+=*)', integrity)
            if not match:
                continue
            try:
                value = base64.b64decode(match.group(2), validate=True)
            except (ValueError, binascii.Error):
                continue
            if len(value) != {'sha256': 32, 'sha512': 64}[match.group(1)]:
                continue
            key = 'content:' + match.group(1) + ':' + value.hex()
            emit('dependency', 'install' if installed else 'observe', 'success',
                 'dependency:' + repo + ':' + name, version=match.group(1) + ':' + value.hex(),
                 inputs=[key], evidence=evidence, declared_dependency=True, installation_observed=installed,
                 relation_authority=installed)

    def _install(self, segments, output, failed, emit):
        commands = [args for args in segments if args[0] != 'cd']
        if len(commands) != 1:
            return
        args = commands[0]
        pip = args[1:] if args[0].rsplit('/', 1)[-1] in ('pip', 'pip3') else (args[3:] if len(args) > 3 and args[0].rsplit('/', 1)[-1] in ('python', 'python3') and args[1:3] == ['-m', 'pip'] else [])
        if pip and pip[0] == 'install':
            hashes = [match.group(1).lower() for arg in pip[1:] for match in re.finditer(r'#sha256=([0-9a-fA-F]{64})(?:\s|$)', arg)]
            if hashes:
                successful = not failed and 'Successfully installed ' in output and not re.search(r'(?mi)^ERROR:', output)
                emit('package', 'install', 'success' if successful else ('failed' if failed else 'attempted'),
                     'package-install:' + self.actor, inputs=['content:sha256:' + h for h in hashes],
                     evidence=[{'field': 'o', 'start': 0, 'end': len(output)}], relation_authority=successful)
        if len(args) >= 3 and args[0].rsplit('/', 1)[-1] in ('docker', 'podman') and args[1] == 'pull':
            images = [arg for arg in args[2:] if re.fullmatch(r'[^\s]+@sha256:[0-9a-fA-F]{64}', arg)]
            if len(images) == 1:
                image = images[0]
                checksum = image.rsplit(':', 1)[1].lower()
                confirmed = not failed and ('Digest: sha256:' + checksum) in output and bool(re.search(r'(?:Downloaded newer image|Image is up to date)', output))
                emit('package', 'acquire', 'success' if confirmed else ('failed' if failed else 'attempted'),
                     'container:' + image.split('@', 1)[0], version='sha256:' + checksum,
                     inputs=['container:' + image.lower()], evidence=[{'field': 'o', 'start': 0, 'end': len(output)}], relation_authority=confirmed)

    def _document(self, obj, req, mutation, evidence, emit):
        docid = obj.get('documentId')
        if isinstance(docid, str):
            entity = 'gdoc:' + docid
            matched = bool(req and re.search(r'/documents/' + re.escape(docid) + r'(?::batchUpdate)?$', req['path']))
            body = obj.get('body')
            if isinstance(body, dict) and isinstance(body.get('content'), list):
                pieces = []

                def text_runs(value):
                    if isinstance(value, dict):
                        if isinstance(value.get('textRun'), dict) and isinstance(value['textRun'].get('content'), str):
                            pieces.append(value['textRun']['content'])
                        for child in value.values():
                            if isinstance(child, (dict, list)):
                                text_runs(child)
                    elif isinstance(value, list):
                        for child in value:
                            text_runs(child)

                text_runs(body['content'])
                content = ''.join(pieces)
                checksum = digest(content)
                key = entity + '@plaintext-sha256:' + checksum
                content_key = 'content:text-sha256:' + checksum
                operation = 'acquire' if matched and req['method'] == 'GET' else 'observe'
                keys = [key, content_key] if len(content) >= 80 else [key]
                emit('document', operation, 'success', entity, version=checksum,
                     inputs=keys if operation == 'acquire' else [], outputs=keys if operation == 'observe' else [],
                     relations=[relation('same_content', key, content_key)] if len(content) >= 80 else [],
                     evidence=evidence, content_key=content_key,
                     content_length=len(content), representation='google-docs-plaintext-v1',
                     provider_revision_scope=self.actor if 'revisionId' in obj else None)
                if self.relation_authority:
                    self.last_snapshot[entity] = key
            elif matched and mutation and isinstance(obj.get('replies'), list) and isinstance(mutation['payload'].get('requests'), list):
                changes = sorted({name for item in mutation['payload']['requests'] if isinstance(item, dict) for name in item})
                if changes:
                    emit('document', 'transform', 'success', entity,
                         # A prior snapshot may already be stale. Without a
                         # verified version precondition it is not this edit's
                         # proven immutable input.
                         inputs=[],
                         evidence=evidence, changed_operations=changes, version_unobserved=True,
                         provider_revision_scope=self.actor if 'writeControl' in obj else None)
        self._sheet(obj, req, mutation, evidence, emit)

    def _sheet(self, obj, req, mutation, evidence, emit):
        match = re.search(r'/spreadsheets/([^/:]+)(?:/values/([^?]+))?', req['path']) if req else None
        sheet = obj.get('spreadsheetId') or (match.group(1) if match else None)
        if not sheet:
            return
        value = obj.get('updatedData') if isinstance(obj.get('updatedData'), dict) else obj
        updates = obj.get('updates') if isinstance(obj.get('updates'), dict) else {}
        if updates:
            value = updates.get('updatedData', updates)
        if not isinstance(value, dict):
            return
        cells = value.get('values')
        cell_range = value.get('range')
        if not isinstance(cells, list) or not isinstance(cell_range, str):
            return
        entity = 'gsheet:' + str(sheet) + ':' + cell_range
        checksum = digest(stable(cells))
        key = entity + '@cells-sha256:' + checksum
        content_key = 'content:cells-sha256:' + checksum
        keys = [key, content_key] if len(stable(cells)) >= 80 else [key]
        operation = 'observe'
        if match and match.group(1) == str(sheet):
            if req['method'] == 'GET':
                operation = 'acquire'
            elif mutation and mutation['payload'].get('values') == cells and isinstance(obj.get('updatedCells', updates.get('updatedCells', 0)), (int, float)) and obj.get('updatedCells', updates.get('updatedCells', 0)) > 0:
                operation = 'generate'
        emit('document', operation, 'success', entity, version=checksum,
             inputs=keys if operation == 'acquire' else [], outputs=keys if operation != 'acquire' else [],
             relations=[relation('same_content', key, content_key)] if len(stable(cells)) >= 80 else [],
             evidence=evidence, content_key=content_key,
             content_length=len(stable(cells)), representation='google-sheets-values-json-v1')

    def _artifact_download(self, req, output, failed, emit):
        if not req or req['method'] != 'GET':
            return
        match = re.fullmatch(r'/jobs/(\d+)/artifacts(?:/(.*))?', req['tail'])
        if not match:
            return
        repo = self._repo({}, req)
        if not repo:
            return
        key = 'artifact:' + repo + ':job:' + match.group(1)
        relations = []
        if match.group(2):
            archive = key
            key += ':path:' + unquote(match.group(2))
            relations.append(relation('derived_from', key, archive))
        confirmed_status = bool(req['response_attributable'] and (
            (req['header_capture'] and re.search(r'(?m)^HTTP/\S+\s+2\d\d\b', output)) or
            (req['status_capture'] and re.search(r'(?m)^2\d\d\s*$', output))))
        status = 'failed' if failed else ('success' if req['destination'] and confirmed_status else 'attempted')
        emit('ci', 'acquire', status, key, inputs=[key], relations=relations,
             evidence=[{'field': 'o', 'start': 0, 'end': len(output)}], relation_authority=status == 'success')

    def _http_snapshot(self, req, output, failed, emit):
        if not req or req['method'] != 'GET' or failed or req['destination'] or not req['response_attributable'] or not req['header_capture']:
            return
        # Require an actual HTTP response header/body. Plain JSON or an echoed
        # URL alone cannot establish successful content retrieval here.
        match = re.search(r'(?ms)^HTTP/\S+\s+200[^\n]*\n(.*?)\r?\n\r?\n(.+)\Z', output)
        if not match or '[... ' in match.group(2) or '\x00' in match.group(2):
            return
        content = match.group(2)
        checksum = digest(content)
        entity = 'http:' + req['url']
        key = entity + '@sha256:' + checksum
        emit('http', 'acquire', 'success', entity, version=checksum, inputs=[key, 'content:sha256:' + checksum],
             evidence=[{'field': 'o', 'start': match.start(2), 'end': match.end(2)}], content_length=len(content), relation_authority=True)
