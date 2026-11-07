"""Static reference incidence in a blinded, unordered extraction-record export.

Payload records are not asserted to be executed tool calls. Parent-linked
responses are structural extraction associations, not successful operations.
No code is executed, network is accessed, identity is inferred, or event order
is constructed. Only exact references and administrative whole-payload hashes
are used; there is no passage, fuzzy, or near-copy matching.
"""
import argparse
from collections import Counter
from collections.abc import Mapping
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import unquote

import pandas as pd

from .agent_traces import PATH_SUFFIXES, _path, canonical_url


FAMILIES = ('url', 'opaque_alias', 'local_path')
ALIAS = re.compile(r'\[(?:SERVICE [0-9]+ URL [0-9]+|REDACTED:destination:[0-9]+)\]')
# Bracketed tokens are consumed whole, including internal spaces. This prevents
# a redacted path/host/query from producing an apparently valid URL prefix.
URL_SPAN = re.compile(r'https?://(?:\[[^\]\r\n]*\]|\{[^}\r\n]*\}|[^\s\x22\x27<>`\\\[\]{}])+')
IPV6_HEAD = re.compile(r'^https?://\[[0-9a-fA-F:.]+\]')
REDACTION_MARKER = re.compile(r'\[(?:REDACTED(?::|\])|SERVICE\s|CREDENTIAL\s|SOURCE(?:\s|:))',re.I)
QUOTED_LITERAL = re.compile(r'(?P<quote>[\x22\x27])(?P<value>[^\x22\x27\\\n\r]*)\1')
SIMPLE_PATH = re.compile(r'[A-Za-z0-9_./:@%+=-]+\Z')
GATED_METRICS = ('eligible_tool_calls', 'tool_coverage', 'cross_agent_interaction_rate',
                 'before_after_changes', 'message_anchor_comparison', 'unsupervised_change_points',
                 'version_acquisition_rate', 'downstream_use_rate')


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _check_config(config):
    if not isinstance(config, Mapping) or not re.fullmatch(r'[0-9a-f]{64}', str(config.get('source_sha256', ''))):
        raise ValueError('An exact lowercase SHA-256 source identity is required')
    fields = config.get('fields')
    if (not isinstance(fields, Mapping) or set(fields) != {'id', 'kind', 'parent_id', 'text'}
            or any(not isinstance(value, str) or not value for value in fields.values())
            or len(set(fields.values())) != 4):
        raise ValueError('Explicit distinct id/kind/parent_id/text field names are required')
    payload, response = config.get('payload_kind'), config.get('response_kind')
    excluded = config.get('excluded_kinds')
    if (not isinstance(payload, str) or not payload or not isinstance(response, str) or not response
            or payload == response or not isinstance(excluded, list)
            or any(not isinstance(kind, str) or not kind for kind in excluded)
            or payload in excluded or response in excluded):
        raise ValueError('Payload, response, and excluded kinds must be explicitly distinct')
    if config.get('format') != 'jsonl' or config.get('compression') not in ('gzip', 'none'):
        raise ValueError('Only explicitly declared JSONL with gzip or no compression is supported')
    return fields


def _dynamic_literal_context(text, start, end):
    """Reject identifiable URL/path literal concatenation, without evaluating it."""
    if start and text[start-1] in '\"\x27' and text[:start-1].rstrip().endswith('+'):
        return True
    if end < len(text) and text[end] in '\"\x27' and text[end+1:].lstrip().startswith('+'):
        return True
    return False


def extract_references(text):
    """Return literal sets and exclusions, with no operation or success labels.

    Numbered opaque aliases preserve exact case and leading zeros. Credential,
    runtime, and source-identifier placeholders cannot enter these families.
    Path-looking literals may also be URL routes; they remain weak references.
    """
    if not isinstance(text, str):
        raise ValueError('Analyzed text must be a string')
    found = {'url': set(), 'opaque_alias': set(ALIAS.findall(text)), 'local_path': set()}
    exclusions = Counter()
    for match in URL_SPAN.finditer(text):
        raw = match.group()
        rest = IPV6_HEAD.sub('', raw, count=1)
        if (any(char in rest for char in '[]{}')
                or (match.end() < len(text) and text[match.end()] in '[]{}')):
            exclusions['redacted_or_bracketed_url_spans'] += 1
            continue
        if match.end() < len(text) and text[match.end()] == '\\':
            exclusions['escaped_url_spans'] += 1
            continue
        # Decode only to recognize an exclusion. Never decode or reconstruct an
        # accepted identity, and never convert encoded placeholders into aliases.
        if '%' in raw and REDACTION_MARKER.search(unquote(raw)):
            exclusions['encoded_redaction_url_spans'] += 1
            continue
        if (raw.endswith('$') and text[match.end():].startswith('{')) or _dynamic_literal_context(text, match.start(), match.end()):
            exclusions['dynamic_url_literals'] += 1
            continue
        while raw.endswith(')') and raw.count(')') > raw.count('('):
            raw = raw[:-1]
        value = canonical_url(raw)
        if value is None:
            exclusions['unusable_or_credential_url_spans'] += 1
        else:
            found['url'].add(value)
    for match in QUOTED_LITERAL.finditer(text):
        raw = match['value']
        if (not raw or not SIMPLE_PATH.fullmatch(raw) or '://' in raw
                or Path(raw).suffix.lower() not in PATH_SUFFIXES
                or _dynamic_literal_context(text, match.start('value'), match.end('value'))):
            continue
        if _path(raw, '/') is not None:
            # _path validates a candidate, but no cwd or filesystem is invented.
            # Keep exact literal spelling: a relative and absolute path differ.
            found['local_path'].add(raw)
    return found, dict(exclusions)


def _identity(source_sha256, family, value, root_id=None):
    fields = [source_sha256, family, value]
    if family == 'local_path':
        fields.append(root_id)
    return family + ':' + _sha_bytes(json.dumps(fields, ensure_ascii=False, separators=(',', ':')).encode())


def _validate_graph(metadata):
    missing = sorted({parent for _, parent in metadata.values() if parent is not None and parent not in metadata})
    if missing:
        raise ValueError('Dangling parent IDs: ' + ', '.join(missing[:5]))
    # Iterative first-parent validation handles long extraction chains without
    # recursion and does not interpret the graph as temporal order.
    complete = set()
    for start in metadata:
        node, active = start, set()
        while node is not None and node not in complete:
            if node in active:
                raise ValueError('Cycle in extraction-parent graph')
            active.add(node)
            node = metadata[node][1]
        complete.update(active)


def _rate(count, denominator):
    return 1000 * count / denominator if denominator else None


def _payload_summary(payloads, resource_sets):
    denominator = len(payloads)
    result = dict(payload_records=denominator, families={})
    counters = {family: Counter(identity for record_id in payloads for identity in resource_sets[record_id][family])
                for family in FAMILIES}
    for family in FAMILIES:
        covered = sum(bool(resource_sets[record_id][family]) for record_id in payloads)
        recurrent = (sum(any(counters[family][identity] >= 2 for identity in resource_sets[record_id][family])
                         for record_id in payloads) if family != 'local_path' else None)
        result['families'][family] = dict(
            records_with_reference=covered, reference_coverage_per_1000_payload_records=_rate(covered, denominator),
            distinct_reference_identities=len(counters[family]),
            recurrent_reference_participating_records=recurrent,
            recurrent_reference_participation_per_1000_payload_records=_rate(recurrent, denominator) if recurrent is not None else None,
            identity_interpretation=('Exact canonical URL reference, not verified resource use.' if family == 'url'
                else 'Exact opaque alias equality; unique original resource identity is not established.' if family == 'opaque_alias'
                else 'Payload-root-scoped path-looking literal; no shared-file or operation inference.'))
    return result, counters


def analyze_records(records, config):
    """Analyze supplied extraction mappings; never read excluded text fields.

    Returns ``summary`` plus ``payload_features``, ``resource_counts``, and
    ``parent_associations`` DataFrames. Source byte validation is performed by
    ``run_file``; callers supplying records directly must retain that distinction.
    """
    fields = _check_config(config)
    metadata, references, payload_hashes = {}, {}, {}
    kind_counts, exclusions = Counter(), Counter()
    payload_kind, response_kind = config['payload_kind'], config['response_kind']
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError('Each extraction record must be a mapping')
        record_id, kind, parent = (record[fields[key]] for key in ('id', 'kind', 'parent_id'))
        if (not isinstance(record_id, str) or not record_id or not isinstance(kind, str) or not kind
                or (parent is not None and (not isinstance(parent, str) or not parent))):
            raise ValueError('Record IDs/kinds must be strings and parent IDs must be strings or null')
        if record_id in metadata:
            raise ValueError('Duplicate extraction record ID: ' + record_id)
        metadata[record_id] = (kind, parent)
        kind_counts[kind] += 1
        # Metadata is validated for every row, but recovered/unknown kinds do
        # not expose text, tags, or citations to any content extractor.
        if kind not in (payload_kind, response_kind):
            continue
        value = record[fields['text']]
        found, dropped = extract_references(value)
        references[record_id] = found
        exclusions.update(dropped)
        if kind == payload_kind:
            payload_hashes[record_id] = _sha_bytes(value.encode('utf-8'))
    _validate_graph(metadata)
    source_sha = config['source_sha256']
    payload_ids, response_ids = sorted(payload_hashes), sorted(record_id for record_id in references
                                                             if metadata[record_id][0] == response_kind)
    identities, labels = {}, {}
    for record_id, families in references.items():
        kind, parent = metadata[record_id]
        root = parent if kind == response_kind and parent is not None and metadata[parent][0] == payload_kind else record_id
        identities[record_id] = {}
        for family, values in families.items():
            values_by_id = {_identity(source_sha, family, value, root): value for value in values}
            identities[record_id][family] = set(values_by_id)
            labels.update({identity: value for identity, value in values_by_id.items() if family == 'opaque_alias'})
    groups = {}
    for record_id in payload_ids:
        groups.setdefault(payload_hashes[record_id], []).append(record_id)
    representative = {record_id: ids[0] for ids in groups.values() for record_id in ids}
    dedup_ids = sorted(ids[0] for ids in groups.values())
    primary, counters = _payload_summary(payload_ids, identities)
    sensitivity, dedup_counters = _payload_summary(dedup_ids, identities)
    response_counters = {family: Counter(identity for record_id in response_ids for identity in identities[record_id][family])
                         for family in FAMILIES}
    associations, responses_per_payload, intersections_per_payload = [], Counter(), {family: Counter() for family in FAMILIES}
    excluded_response_parents = Counter()
    for response_id in response_ids:
        parent = metadata[response_id][1]
        if parent is None:
            excluded_response_parents['no_parent'] += 1
            continue
        if metadata[parent][0] != payload_kind:
            excluded_response_parents['nonpayload_parent'] += 1
            continue
        responses_per_payload[parent] += 1
        row = dict(payload_id=parent, response_id=response_id)
        for family in FAMILIES:
            shared = identities[parent][family] & identities[response_id][family]
            row[family+'_intersection_count'] = len(shared)
            row[family+'_shared_identity_ids'] = json.dumps(sorted(shared), separators=(',', ':'))
            intersections_per_payload[family][parent] += bool(shared)
        row['any_reference_intersection'] = any(row[family+'_intersection_count'] for family in FAMILIES)
        associations.append(row)
    associations.sort(key=lambda row: (row['payload_id'], row['response_id']))
    payload_rows = []
    for record_id in payload_ids:
        row = dict(record_id=record_id, payload_sha256=payload_hashes[record_id],
                   duplicate_group_size=len(groups[payload_hashes[record_id]]),
                   duplicate_representative_id=representative[record_id],
                   included_in_duplicate_sensitivity=representative[record_id] == record_id,
                   associated_response_records=responses_per_payload[record_id])
        for family in FAMILIES:
            row[family+'_reference_count'] = len(identities[record_id][family])
            row[family+'_reference_present'] = bool(identities[record_id][family])
            row[family+'_parent_linked_responses_with_intersection'] = intersections_per_payload[family][record_id]
            if family != 'local_path':
                row[family+'_recurrent_reference_participation'] = any(counters[family][identity] >= 2 for identity in identities[record_id][family])
        payload_rows.append(row)
    resource_rows = []
    for family in FAMILIES:
        all_ids = set(counters[family]) | set(response_counters[family])
        for identity in sorted(all_ids):
            resource_rows.append(dict(family=family, identity_id=identity,
                opaque_alias=labels.get(identity), payload_record_count=counters[family][identity],
                deduplicated_payload_record_count=dedup_counters[family][identity],
                response_record_count=response_counters[family][identity]))
    paired = len(associations)
    parent_summary = dict(response_records=len(response_ids), direct_payload_parent_associations=paired,
        response_records_with_payload_parent_per_1000_responses=_rate(paired, len(response_ids)),
        payload_records_with_associated_response=len(responses_per_payload),
        payload_records_with_associated_response_per_1000_payload_records=_rate(len(responses_per_payload),len(payload_ids)),
        excluded_response_parents=dict(excluded_response_parents), families={},
        interpretation='Direct extraction-parent associations only; no chronology, success, handoff, or execution claim.')
    for family in FAMILIES:
        matching = sum(row[family+'_intersection_count'] > 0 for row in associations)
        parent_summary['families'][family] = dict(associations_with_reference_intersection=matching,
            shared_reference_identity_occurrences=sum(row[family+'_intersection_count'] for row in associations),
            intersection_per_1000_direct_payload_parent_associations=_rate(matching, paired),
            payload_records_with_reference_intersection=sum(bool(value) for value in intersections_per_payload[family].values()),
            intersection_per_1000_all_response_records=_rate(matching, len(response_ids)))
    summary = dict(schema_version='swarm-static-summary-v1', source_sha256=source_sha,
        input_byte_hash_validated=False, unit='Exported payload record, not an established executed tool call.',
        total_records=len(metadata), record_kind_counts=dict(sorted(kind_counts.items())),
        analyzed_payload_records=len(payload_ids), analyzed_response_records=len(response_ids),
        excluded_content_records=len(metadata)-len(references),
        graph_validation=dict(unique_ids=len(metadata), dangling_parent_ids=0, cycles=0,
                              interpretation='Structural first-parent graph; not execution chronology.'),
        source_field_mapping=dict(fields), content_fields_ignored=['tags','cite'],
        reference_exclusions=dict(sorted(exclusions.items())), primary=primary,
        duplicate_sensitivity=dict(method='Exact full text-field UTF-8 byte hash; lexicographically first record ID represents each duplicate group.',
            duplicate_payload_records=len(payload_ids)-len(dedup_ids), exact_unique_payloads=len(dedup_ids),
            duplicate_groups=sum(len(ids)>1 for ids in groups.values()),
            scope='Payload-reference summaries only. Response-parent associations retain their primary record denominators.',
            summary=sensitivity), parent_associations=parent_summary,
        top_reference_identities={family: sorted((row for row in resource_rows if row['family']==family and row['payload_record_count']>0),
                                     key=lambda row: (-row['payload_record_count'],row['identity_id']))[:20] for family in FAMILIES},
        unavailable_metrics={metric: None for metric in GATED_METRICS},
        limitations=['No actor identity, executed tool identity, observation time, run boundary, or execution order is inferred.',
            'Literal references do not establish operations, success, resource use, or cross-agent interaction.',
            'Opaque alias equality does not establish one-to-one original resource identity; service groups are never merged.',
            'Path-looking literals may be URL routes or other strings. They are scoped to the payload root, with direct response inheritance only.',
            'Exact whole-payload hashes are administrative duplicate accounting, not immutable artifact versions or a near-copy detector.',
            'No independent-replicate uncertainty, p-values, or temporal windows are computed.'])
    payload_columns = ['record_id','payload_sha256','duplicate_group_size','duplicate_representative_id',
                       'included_in_duplicate_sensitivity','associated_response_records']
    for family in FAMILIES:
        payload_columns += [family+'_reference_count',family+'_reference_present',family+'_parent_linked_responses_with_intersection']
        if family != 'local_path':
            payload_columns.append(family+'_recurrent_reference_participation')
    association_columns = ['payload_id','response_id']
    for family in FAMILIES:
        association_columns += [family+'_intersection_count',family+'_shared_identity_ids']
    association_columns.append('any_reference_intersection')
    return dict(summary=summary,
        payload_features=pd.DataFrame(payload_rows,columns=payload_columns),
        resource_counts=pd.DataFrame(resource_rows,columns=['family','identity_id','opaque_alias','payload_record_count',
                                                          'deduplicated_payload_record_count','response_record_count']),
        parent_associations=pd.DataFrame(associations,columns=association_columns))


def _unique_object(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key: ' + key)
        result[key] = value
    return result


def _code_hashes():
    package = Path(__file__).resolve().parent
    return {path.name: _sha_file(path) for path in [package/'__init__.py',package/'config.py',
             package/'provenance_common.py',package/'agent_traces.py',Path(__file__).resolve()]}


def run_file(source, config_path, output):
    """Validate the frozen source bytes, analyze once, and atomically publish."""
    source, config_path, output = Path(source), Path(config_path), Path(output)
    config_bytes = config_path.read_bytes()
    config_hash = _sha_bytes(config_bytes)
    config = json.loads(config_bytes,object_pairs_hook=_unique_object)
    _check_config(config)
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing static-analysis output directory')
    source_stat = source.stat()
    code_hashes = _code_hashes()
    observed_hash = _sha_file(source)
    if observed_hash != config['source_sha256']:
        raise ValueError('Source SHA-256 differs from the frozen adapter configuration')
    opener = gzip.open if config['compression'] == 'gzip' else open
    def records():
        with opener(source,'rt',encoding='utf-8') as stream:
            for line_number,line in enumerate(stream,1):
                if not line.strip():
                    raise ValueError('Blank extraction row at line ' + str(line_number))
                yield json.loads(line,object_pairs_hook=_unique_object)
    result = analyze_records(records(),config)
    if (source.stat().st_size,source.stat().st_mtime_ns) != (source_stat.st_size,source_stat.st_mtime_ns):
        raise ValueError('Source file changed during analysis')
    if _sha_file(config_path) != config_hash or _code_hashes() != code_hashes:
        raise ValueError('Adapter configuration or implementation changed during analysis')
    result['summary']['input_byte_hash_validated'] = True
    output.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'.tmp-',dir=output.parent))
    try:
        (staging/'summary.json').write_text(json.dumps(result['summary'],indent=2,allow_nan=False)+'\n')
        result['payload_features'].to_csv(staging/'payload-features.csv.gz',index=False)
        result['resource_counts'].to_csv(staging/'resource-counts.csv',index=False)
        result['parent_associations'].to_csv(staging/'parent-associations.csv.gz',index=False)
        manifest = dict(schema_version='swarm-static-output-manifest-v1', source_sha256=observed_hash,
            source_bytes=source_stat.st_size, adapter_config_sha256=config_hash,
            code_sha256=code_hashes, python=sys.version, pandas=pd.__version__,
            outputs_sha256={path.name:_sha_file(path) for path in sorted(staging.iterdir())},
            no_temporal_analysis=True, no_metr_findings_used=True,
            interpretation='Hash inventory of static reference artifacts; output locking is a separate research step.')
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    manifest = run_file(args.source,args.config,args.output)
    print(json.dumps({'output':str(args.output),'source_sha256':manifest['source_sha256'],
                      'outputs':sorted(manifest['outputs_sha256'])}))


if __name__ == '__main__':
    main()
