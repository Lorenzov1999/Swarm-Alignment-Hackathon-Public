"""Create new step caches for the two authorized transfer windows only.

The whole turns stream is scanned: ordering is not assumed. A metadata-only
top-level timestamp gate precedes JSON decoding of actions or outputs. Existing
extractor semantics and truncation limits are reused unchanged. This module
never reads agent memories, transcript text, goals, or investigation outcomes.
"""
import argparse
import collections
from contextlib import contextmanager
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from zoneinfo import ZoneInfo

from . import config, extract
from .provenance_common import timestamp


PERIODS = {'temporal': ('2026-08-17', '2026-09-15'),
           'shared_goals': ('2026-06-01', '2026-07-05')}
PACIFIC = ZoneInfo('America/Los_Angeles')
DEFAULT_FREEZE = config.OUT / 'generalization-20261106' / 'freeze-manifest.json'
_STRING = rb'"[^"\\]*(?:\\.[^"\\]*)*"'
_STRING_RE = re.compile(_STRING, re.S)
_TOKENS = re.compile(_STRING + rb'|[{}\[\]]', re.S)
_LAST_TIMESTAMP = re.compile(rb'"created_at"\s*:\s*(' + _STRING + rb')\s*}\s*$', re.S)
_CLOCK_PREFIX = re.compile(r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}')


class MetadataError(ValueError):
    """A timestamp cannot safely be obtained without payload interpretation."""


class _Object(dict):
    def __init__(self, pairs):
        self.duplicate_keys = {key for key, count in collections.Counter(key for key, _ in pairs).items()
                               if count > 1}
        super().__init__(pairs)


def _decode_record(line):
    # Called only after the metadata timestamp has passed the exact window gate.
    return json.loads(line, object_pairs_hook=_Object)


def _skip_space(line, index):
    while index < len(line) and line[index] in b' \t\r\n':
        index += 1
    return index


def top_level_created_at(line):
    """Read only the top-level timestamp; skip payload strings as lexical spans.

    A final created_at field has a fast suffix path using C-level byte search and
    regex matching. Otherwise the regex scans entire string tokens, so braces
    and timestamp-looking text inside strings cannot become metadata. Nested
    objects are counted but never decoded. Duplicate top-level keys follow JSON
    decoder last-value semantics for gating; included duplicate-key rows are
    rejected before interpreting any action or output.
    """
    if not isinstance(line, bytes):
        raise TypeError('The turns stream must yield bytes')
    first = _skip_space(line, 0)
    if first == len(line) or line[first] != ord('{'):
        raise MetadataError('non_object_or_empty_row')
    position = line.rfind(b'"created_at"')
    if position >= 0 and (position == 0 or line[position - 1] != ord('\\')):
        final = _LAST_TIMESTAMP.match(line, position)
        if final:
            try:
                return json.loads(final.group(1)), 'suffix'
            except (ValueError, UnicodeDecodeError) as error:
                raise MetadataError('invalid_created_at_encoding') from error
    depth, found = 0, []
    for match in _TOKENS.finditer(line):
        initial = line[match.start()]
        if initial in (ord('{'), ord('[')):
            depth += 1
        elif initial in (ord('}'), ord(']')):
            depth -= 1
            if depth < 0:
                raise MetadataError('invalid_container_nesting')
        elif depth == 1:
            after = _skip_space(line, match.end())
            if after >= len(line) or line[after] != ord(':'):
                continue
            key = match.group()
            if key != b'"created_at"':
                if b'\\' not in key:
                    continue
                try:
                    if json.loads(key) != 'created_at':
                        continue
                except (ValueError, UnicodeDecodeError) as error:
                    raise MetadataError('invalid_top_level_key_encoding') from error
            value_start = _skip_space(line, after + 1)
            value = _STRING_RE.match(line, value_start)
            if value is None:
                # A later duplicate timestamp may replace this value. Mark it
                # without interpreting any non-string object or scalar value.
                found.append(None)
            else:
                try:
                    found.append(json.loads(value.group()))
                except (ValueError, UnicodeDecodeError) as error:
                    raise MetadataError('invalid_created_at_encoding') from error
    if depth != 0:
        raise MetadataError('invalid_container_nesting')
    if not found:
        raise MetadataError('missing_top_level_created_at')
    if found[-1] is None:
        raise MetadataError('created_at_not_string')
    return found[-1], 'structural_scan'


def _parse_timestamp(value):
    if not isinstance(value, str) or not _CLOCK_PREFIX.match(value):
        raise ValueError('created_at must include an ISO calendar date and clock time')
    return timestamp(value)


def _bounds(period):
    if period not in PERIODS:
        raise ValueError('Unknown authorized transfer period: ' + str(period))
    lo, hi = [dt.date.fromisoformat(value) for value in PERIODS[period]]
    lower = dt.datetime.combine(lo, dt.time(), PACIFIC).astimezone(dt.timezone.utc)
    upper = dt.datetime.combine(hi + dt.timedelta(days=1), dt.time(), PACIFIC).astimezone(dt.timezone.utc)
    return lower, upper


@contextmanager
def _turn_stream(path):
    if path.suffix != '.gz':
        with path.open('rb') as stream:
            yield stream
        return
    executable = shutil.which('pigz')
    if executable is None:
        raise RuntimeError('pigz is required to stream compressed turns; no dependencies are installed automatically')
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([executable, '-dc', '--', str(path)], stdout=subprocess.PIPE,
                                   stderr=errors, bufsize=1 << 22)
        try:
            yield process.stdout
            returncode = process.wait()
            if returncode:
                # Do not reproduce decompressor diagnostics that could include
                # data bytes; a nonzero status invalidates the entire output.
                raise RuntimeError('pigz failed with exit status ' + str(returncode))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()


def _identity(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _metadata_maps(agents_path, sessions_path):
    result, diagnostics = {}, {}
    for kind, path, value_key in [('agents', agents_path, 'name'), ('sessions', sessions_path, 'agent_id')]:
        mapping, anomalies, records = {}, collections.Counter(), 0
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                records += 1
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    anomalies['invalid_json'] += 1
                    continue
                if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']
                        or not isinstance(row.get(value_key), str) or not row[value_key]):
                    anomalies['missing_or_invalid_identifier'] += 1
                    continue
                if row['id'] in mapping:
                    if mapping[row['id']] != row[value_key]:
                        raise ValueError('Conflicting duplicate identifiers in ' + kind + ' metadata')
                    anomalies['repeated_identical_identifier'] += 1
                mapping[row['id']] = row[value_key]
        result[kind] = mapping
        diagnostics[kind] = {'records': records, 'mapped': len(mapping), 'anomalies': dict(anomalies)}
    counts = collections.Counter(result['agents'].values())
    diagnostics['agent_names_with_multiple_ids'] = {name: count for name, count in counts.items() if count > 1}
    return result['agents'], result['sessions'], diagnostics


def _serialized_length(value):
    if value is None:
        return 0
    return len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))


def _removed_characters(value, head, tail=0):
    return max(0, _serialized_length(value) - head - tail)


def _action_removed(action, kind):
    if kind == 'shell' and isinstance(action, dict):
        return _removed_characters(action.get('command'), extract.CMD_HEAD, extract.CMD_TAIL)
    if action is not None and (not isinstance(action, dict) or kind == 'other'):
        return _removed_characters(action, 2000)
    return 0


def _resort(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    rows.sort(key=lambda row: _parse_timestamp(row['t']))
    digest, byte_count = hashlib.sha256(), 0
    with gzip.open(path, 'wt', compresslevel=4, encoding='utf-8') as stream:
        for row in rows:
            text = json.dumps(row, ensure_ascii=False) + '\n'
            encoded = text.encode('utf-8')
            digest.update(encoded)
            byte_count += len(encoded)
            stream.write(text)
    return digest.hexdigest(), byte_count


def _resume_summary(output, period, inputs, freeze_hash, extractor_hash, parser_hash):
    summary = json.loads((output / 'summary.json').read_text())
    if (summary.get('period_name') != period or summary.get('period') != list(PERIODS[period])
            or summary.get('timezone') != 'America/Los_Angeles'
            or summary.get('complete_source_scan') is not True
            or summary.get('inputs') != inputs or summary.get('freeze_manifest_sha256') != freeze_hash
            or summary.get('extractor_sha256') != extractor_hash
            or summary.get('original_parser_sha256') != parser_hash):
        raise ValueError('Existing extraction summary does not match the requested frozen inputs and scope')
    for values in summary.get('agents', {}).values():
        name = values.get('file', '')
        if not name or Path(name).name != name:
            raise ValueError('Invalid cache filename in existing summary')
        path = output / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != values.get('compressed_bytes'):
            raise ValueError('Existing extraction cache is missing or has changed size: ' + name)
        digest, byte_count = hashlib.sha256(), 0
        try:
            with gzip.open(path, 'rb') as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(chunk)
                    byte_count += len(chunk)
        except (OSError, EOFError) as error:
            raise ValueError('Existing extraction cache is corrupt: ' + name) from error
        if digest.hexdigest() != values.get('uncompressed_sha256') or byte_count != values.get('uncompressed_bytes'):
            raise ValueError('Existing extraction cache content has changed: ' + name)
    return summary


def extract_period(period, output_dir, *, turns_path=None, agents_path=None, sessions_path=None,
                   freeze_manifest=None, resume=False, progress_every=500000):
    """Extract one authorized window to a fresh, atomically published cache.

    ``resume=True`` only skips a previously completed matching extraction; it
    never appends to a partial cache. No target extraction is run at import time.
    ``summary['agents'][actor]['file']`` matches the original cache interface.
    """
    lower, upper = _bounds(period)
    if not isinstance(progress_every, int) or isinstance(progress_every, bool) or progress_every < 0:
        raise ValueError('progress_every must be a nonnegative integer')
    output = Path(output_dir).resolve()
    original_steps = config.STEPS.resolve()
    if output == original_steps or original_steps in output.parents:
        raise ValueError('Transfer extraction cannot write inside the original step cache')
    paths = {'turns': Path(turns_path or config.TURNS).resolve(),
             'agents': Path(agents_path or config.AGENTS).resolve(),
             'sessions': Path(sessions_path or config.SESSIONS).resolve()}
    inputs = {name: _identity(path) for name, path in paths.items()}
    manifest = Path(freeze_manifest or DEFAULT_FREEZE).resolve()
    freeze_bytes = manifest.read_bytes()
    json.loads(freeze_bytes)  # A missing or malformed freeze manifest is a hard failure.
    freeze_hash = hashlib.sha256(freeze_bytes).hexdigest()
    extractor_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parser_hash = hashlib.sha256(Path(extract.__file__).read_bytes()).hexdigest()
    if output.exists():
        if (output / 'summary.json').exists():
            if not resume:
                raise FileExistsError('A completed extraction already exists; use explicit resume to skip it')
            return _resume_summary(output, period, inputs, freeze_hash, extractor_hash, parser_hash)
        if any(output.iterdir()):
            raise FileExistsError('Refusing to overwrite a nonempty partial output directory')
    agents, sessions, metadata = _metadata_maps(paths['agents'], paths['sessions'])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '-partial-', dir=output.parent))
    started = time.monotonic()
    writers, files, hashes = {}, {}, {}
    raw_bytes, kept, dropped = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    first, last, previous = {}, {}, {}
    disorder, system_seen, unknown_agents = collections.Counter(), collections.Counter(), collections.Counter()
    truncated, removed = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    anomalies, gates, signatures, days = [collections.Counter() for _ in range(4)]
    lines_read = in_period = unknown_session = 0
    published = False
    try:
        with _turn_stream(paths['turns']) as stream:
            for line in stream:
                lines_read += 1
                if progress_every and lines_read % progress_every == 0:
                    extract.log(f'{lines_read} metadata rows scanned; {in_period} rows inside {period}')
                try:
                    raw_stamp, route = top_level_created_at(line)
                    when = _parse_timestamp(raw_stamp)
                except (MetadataError, ValueError, OverflowError) as error:
                    key = str(error) if isinstance(error, MetadataError) else 'invalid_created_at'
                    anomalies[key] += 1
                    continue
                gates[route] += 1
                if not lower <= when < upper:
                    gates['outside_period'] += 1
                    continue
                try:
                    row = _decode_record(line)
                except (ValueError, UnicodeDecodeError, RecursionError):
                    anomalies['invalid_json_inside_period'] += 1
                    continue
                if not isinstance(row, dict):
                    anomalies['decoded_non_object'] += 1
                    continue
                if row.duplicate_keys:
                    anomalies['duplicate_top_level_keys'] += 1
                    continue
                verified_stamp = row.get('created_at')
                try:
                    verified_when = _parse_timestamp(verified_stamp)
                except (ValueError, TypeError, OverflowError):
                    anomalies['invalid_decoded_created_at'] += 1
                    continue
                if verified_stamp != raw_stamp or verified_when != when or not lower <= verified_when < upper:
                    anomalies['timestamp_verification_mismatch'] += 1
                    continue
                in_period += 1
                day = when.astimezone(PACIFIC).date().isoformat()
                days[day] += 1
                session = row.get('session_id')
                if not isinstance(session, str) or session not in sessions:
                    unknown_session += 1
                    continue
                aid = sessions[session]
                name = agents.get(aid, aid)
                if aid not in agents:
                    unknown_agents[aid] += 1
                if row.get('id') is None or isinstance(row.get('id'), (dict, list, bool)):
                    anomalies['missing_or_invalid_turn_id'] += 1
                    continue
                action = row.get('agent_action')
                if 'agent_action' not in row:
                    anomalies['absent_agent_action_field'] += 1
                try:
                    leak = extract.is_leak(action)
                except (TypeError, ValueError):
                    anomalies['invalid_action_schema'] += 1
                    continue
                if leak:
                    dropped[name][action.get('action') or 'session-goal keys'] += 1
                    continue
                try:
                    kind, payload = extract.kind_and_payload(action)
                except (TypeError, ValueError):
                    anomalies['invalid_action_schema'] += 1
                    continue
                if isinstance(action, dict):
                    signatures[f'{kind} {sorted(action)}'] += 1
                record = {'i': row.get('id'), 't': verified_stamp, 's': session, 'k': kind}
                if payload:
                    record['x'] = payload
                counts = {'action': _action_removed(action, kind)}
                value = row.get('output')
                if value not in (None, ''):
                    record['o'] = extract.cap(value, extract.OUT_HEAD, extract.OUT_TAIL)
                    counts['output'] = _removed_characters(value, extract.OUT_HEAD, extract.OUT_TAIL)
                value = row.get('error')
                if value:
                    record['e'] = extract.cap(value, extract.ERR_HEAD)
                    counts['error'] = _removed_characters(value, extract.ERR_HEAD)
                for field, count in counts.items():
                    if count:
                        truncated[name][field] += 1
                        removed[name][field] += count
                if row.get('system'):
                    system_seen[name] += 1
                if name not in writers:
                    slug = extract.slug(name) or 'agent'
                    filename = slug + '.jsonl.gz'
                    if filename in files.values():
                        filename = slug + '--' + hashlib.sha256(name.encode()).hexdigest()[:12] + '.jsonl.gz'
                    files[name] = filename
                    writers[name] = gzip.open(staging / filename, 'wt', compresslevel=4, encoding='utf-8')
                    hashes[name] = hashlib.sha256()
                text = json.dumps(record, ensure_ascii=False) + '\n'
                encoded = text.encode('utf-8')
                writers[name].write(text)
                hashes[name].update(encoded)
                raw_bytes[name] += len(encoded)
                kept[name][kind] += 1
                if name in previous and when < previous[name]:
                    disorder[name] += 1
                previous[name] = when
                if name not in first or when < first[name][0]:
                    first[name] = (when, verified_stamp)
                if name not in last or when > last[name][0]:
                    last[name] = (when, verified_stamp)
        if {name: _identity(path) for name, path in paths.items()} != inputs:
            raise RuntimeError('An input changed during extraction; refusing to publish the cache')
        for writer in writers.values():
            writer.close()
        digests = {name: digest.hexdigest() for name, digest in hashes.items()}
        for name in disorder:
            extract.log(f'Resorting {name}: {disorder[name]} adjacent timestamp inversions')
            digests[name], raw_bytes[name] = _resort(staging / files[name])
        summary = dict(
            schema_version=1, period_name=period, period=list(PERIODS[period]),
            timezone='America/Los_Angeles',
            utc_bounds={'start_inclusive': lower.isoformat(), 'end_exclusive': upper.isoformat()},
            inputs=inputs, freeze_manifest=str(manifest), freeze_manifest_sha256=freeze_hash,
            extractor_sha256=extractor_hash,
            original_parser_sha256=parser_hash,
            lines_read=lines_read, steps_in_period=in_period, unknown_session=unknown_session,
            unknown_agent_ids=dict(unknown_agents), schema_anomalies=dict(anomalies),
            metadata=metadata, timestamp_gate=dict(gates), steps_in_period_by_day=dict(sorted(days.items())),
            stopped_early_at_date=None, complete_source_scan=True,
            leak_actions_dropped=sorted(extract.LEAK_ACTIONS),
            truncation_policy={'command_head': extract.CMD_HEAD, 'command_tail': extract.CMD_TAIL,
                               'output_head': extract.OUT_HEAD, 'output_tail': extract.OUT_TAIL,
                               'error_head': extract.ERR_HEAD},
            agents={name: dict(file=files[name], kept=sum(kept[name].values()),
                              dropped=sum(dropped[name].values()), kept_by_kind=dict(kept[name].most_common()),
                              dropped_by_kind=dict(dropped[name].most_common()),
                              first=first[name][1], last=last[name][1], out_of_order=disorder[name],
                              system_field_present=system_seen[name],
                              truncated_fields=dict(truncated[name]), removed_characters=dict(removed[name]),
                              uncompressed_sha256=digests[name], uncompressed_bytes=raw_bytes[name],
                              compressed_bytes=(staging / files[name]).stat().st_size)
                    for name in sorted(kept)},
            agents_with_only_dropped_steps=sorted(set(dropped) - set(kept)),
            dropped_only_counts={name: dict(dropped[name]) for name in sorted(set(dropped) - set(kept))},
            action_signatures=dict(signatures.most_common(200)),
            seconds=round(time.monotonic() - started, 3))
        (staging / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        if output.exists():
            output.rmdir()  # Only the previously verified empty directory can be replaced.
        os.replace(staging, output)
        published = True
        extract.log(f"Completed {period}: {sum(item['kept'] for item in summary['agents'].values())} "
                    f"steps kept; {lines_read} metadata rows scanned")
        return summary
    finally:
        for writer in writers.values():
            writer.close()
        if not published:
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period', required=True, choices=sorted(PERIODS))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--turns', type=Path, default=config.TURNS)
    parser.add_argument('--agents', type=Path, default=config.AGENTS)
    parser.add_argument('--sessions', type=Path, default=config.SESSIONS)
    parser.add_argument('--freeze-manifest', type=Path, default=DEFAULT_FREEZE)
    parser.add_argument('--resume', action='store_true', help='Validate and skip an already completed matching cache')
    args = parser.parse_args()
    result = extract_period(args.period, args.output, turns_path=args.turns,
                            agents_path=args.agents, sessions_path=args.sessions,
                            freeze_manifest=args.freeze_manifest, resume=args.resume)
    print(json.dumps({'period': args.period, 'output': str(args.output),
                      'kept': sum(item['kept'] for item in result['agents'].values()),
                      'complete_source_scan': result['complete_source_scan']}))


if __name__ == '__main__':
    main()
