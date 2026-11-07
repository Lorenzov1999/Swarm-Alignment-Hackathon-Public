"""Shared, conservative primitives for provenance in the authorized first slice.

These records describe observed evidence, never counterfactual causal effects.
No commands are executed and no remote services are queried by these parsers.
"""
import datetime as dt
import hashlib
import json
import re
import shlex
from urllib.parse import quote
from zoneinfo import ZoneInfo

from . import config

PACIFIC = ZoneInfo('America/Los_Angeles')
STATUSES = {'attempted', 'success', 'failed', 'unknown'}


def timestamp(value):
    value = re.sub(r'(\.)(\d+)', lambda m: '.' + (m[2] + '000000')[:6], value, count=1)
    when = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return when.astimezone(dt.timezone.utc)


def in_scope(value):
    day = timestamp(value).astimezone(PACIFIC).date()
    return config.DEV_START <= day <= config.DEV_END


def git_key(repo, sha):
    return 'git:' + str(repo) + '@' + str(sha).lower()


def file_key(repo, path, digest, algorithm='sha256'):
    return 'file:' + str(repo) + ':' + quote(str(path), safe='/._-') + '@' + algorithm + ':' + str(digest).lower()


def make_event(actor, record, family, operation, status, entity, version=None,
               inputs=None, outputs=None, relations=None, evidence=None, **extra):
    if not in_scope(record['t']):
        raise ValueError('Provenance events must remain in the authorized development slice')
    if status not in STATUSES:
        raise ValueError('Unknown operation status: ' + str(status))
    when = timestamp(record['t'])
    event = dict(actor=actor, time=when.isoformat(),
                 day=when.astimezone(PACIFIC).date().isoformat(),
                 turn_id=str(record['i']), session=str(record.get('s', '')),
                 family=family, operation=operation, status=status,
                 entity=str(entity), version=version,
                 inputs=list(dict.fromkeys(inputs or [])),
                 outputs=list(dict.fromkeys(outputs or [])),
                 relations=relations or [], evidence=evidence or [])
    event.update(extra)
    digest = hashlib.sha256(json.dumps(event, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    event['event_id'] = digest[:24]
    return event


def command_segments(command):
    """Return direct argv lists; embedded script bodies are never shell actions.

    This deliberately rejects complex shell syntax rather than attributing quoted
    code, conditional branches or command substitutions to executed operations.
    The return value does not prove that any segment succeeded. Callers must
    disambiguate shared output and inspect explicit operation outcomes.
    """
    if not isinstance(command, str):
        return []
    # Shell comments ignore quoting: an apostrophe in a leading comment must not
    # open a quoted string for shlex. Strip only the prefix before any code so
    # comment-looking lines inside quoted programs or strings remain intact.
    lines = command.splitlines(keepends=True)
    first_code = 0
    while first_code < len(lines):
        stripped = lines[first_code].lstrip(' \t')
        if stripped.strip() and not stripped.startswith('#'):
            break
        first_code += 1
    command = ''.join(lines[first_code:])
    if any(s in command for s in ('<<', '$(', '`')):
        return []
    if re.search(r'(?m)(?:^|[;\n])\s*(?:if|for|while|until|case|function|select)\b', command):
        return []
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|()\n')
        lexer.whitespace = ' \t\r'
        lexer.whitespace_split = True
        lexer.commenters = ''  # '#' inside URLs/words is not a shell comment.
        tokens = list(lexer)
    except ValueError:
        return []
    if any(t in ('(', ')', '{', '}') for t in tokens):
        return []
    result, current, commenting = [], [], False
    for token in tokens + [';']:
        if token.startswith('#'):
            commenting = True
        if commenting:
            if '\n' not in token:
                continue
            commenting = False
        if token and all(c in ';&|\n' for c in token):
            if current:
                while current and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*=.*', current[0], re.S):
                    current.pop(0)
                if current and current[0] in ('command', 'env'):
                    current.pop(0)
                    while current and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*=.*', current[0], re.S):
                        current.pop(0)
                if current:
                    result.append(current)
                current = []
        else:
            current.append(token)
    if current and not commenting:
        result.append(current)
    return result


def json_objects(text):
    """Yield (complete top-level object/array, start, end), retaining spans.

    Accept JSON beginning at a line boundary, including after plain log lines.
    Once a JSON container is incomplete, stop: scanning into it would mistake
    nested fragments for complete API responses.
    """
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        match = re.search(r'(?m)^[ \t]*(?=[{\[])', text[offset:])
        if match is None:
            return
        start = offset + match.end()
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            return
        if isinstance(value, (dict, list)):
            yield value, start, end
        offset = end


def iter_scoped_steps(path):
    """Filter cached records before exposing payloads to extractors."""
    import gzip
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if in_scope(record['t']):
                yield record
