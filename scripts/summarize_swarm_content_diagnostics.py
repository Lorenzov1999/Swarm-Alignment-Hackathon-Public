"""Report-only audit of a completed, immutable Swarm content census.

No detector is applied or changed. Counts are reconstructed from saved evidence,
source views, and payload flags; raw source content and external reports are not
decoded. All hashes are checked before the derived report is written elsewhere.
"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/swarm-improvement-20261107'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def json_lines(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def verify(census):
    census = Path(census).resolve()
    manifest = json.loads((census / 'manifest.json').read_text())
    lock = json.loads((census / 'pre-census-lock.json').read_text())
    summary = json.loads((census / 'summary.json').read_text())
    if len({manifest['source_sha256'], lock['source_sha256'], summary['source_sha256']}) != 1:
        raise ValueError('Census source identities disagree')
    for relative, expected in manifest['outputs_sha256'].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ValueError('Manifest contains an invalid relative path')
        if sha(census / relative) != expected:
            raise ValueError('Locked census output changed: ' + relative)
    if summary['pre_census_lock_sha256'] != sha(census / 'pre-census-lock.json'):
        raise ValueError('Summary does not bind the pre-census lock')
    if manifest['code_sha256'] != lock['code_sha256']:
        raise ValueError('Manifest and pre-census code identities differ')
    current = {}
    for relative, expected in lock['code_sha256'].items():
        if sha(census / 'code-snapshot' / relative) != expected:
            raise ValueError('Archived code differs from the pre-census lock')
        current[relative] = sha(ROOT / relative) == expected
    if sha(census / 'code-snapshot/protocol.md') != lock['inputs']['protocol']['sha256']:
        raise ValueError('Archived protocol differs from the pre-census lock')
    inputs = {}
    for name, item in lock['inputs'].items():
        path = Path(item['path'])
        if not path.is_file():
            inputs[name] = dict(present=False, matches=None, expected_sha256=item['sha256'])
        else:
            inputs[name] = dict(present=True, matches=sha(path) == item['sha256'], expected_sha256=item['sha256'])
            if name != 'protocol' and not inputs[name]['matches']:
                raise ValueError('Original census input changed: ' + name)
    return lock, summary, dict(census_manifest_sha256=sha(census / 'manifest.json'),
        verified_census_files=len(manifest['outputs_sha256']), archived_code_matches=True,
        archived_protocol_matches=True, current_code_matches=current, original_inputs=inputs,
        code_sha256=lock['code_sha256'])


def load_payloads(path):
    fields = ('operation_bearing_payload', 'artifact_view_available', 'claim_candidate',
              'root_structured_artifact_available', 'direct_child_available', 'text_available',
              'role_or_task_allocation', 'publication_intent', 'result_request_intent')
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            for field in fields:
                if row[field] not in ('True', 'False'):
                    raise ValueError('Invalid saved boolean: ' + field)
                row[field] = row[field] == 'True'
            rows.append(row)
    return rows


def calculate(payloads, evidence, association_evidence, artifact_signals, view_metadata, old_hashes, categories):
    indexed = {row['record_id']: row for row in payloads}
    if len(indexed) != len(payloads):
        raise ValueError('Duplicate saved payload IDs')
    old_hashes = set(old_hashes)
    bodies = {row['payload_sha256'] for row in payloads}
    def counts(identifiers):
        identifiers = set(identifiers)
        if identifiers - set(indexed):
            raise ValueError('Evidence refers to missing payload')
        hashes = {indexed[key]['payload_sha256'] for key in identifiers}
        return dict(payload_records=len(identifiers), exact_unique_bodies=len(hashes),
                    operation_bearing_payload_records=sum(indexed[key]['operation_bearing_payload'] for key in identifiers),
                    prior_development_bodies=len(hashes & old_hashes), bodies_outside_prior_development=len(hashes - old_hashes))
    eligible = {row['record_id'] for row in evidence}
    raw = {row['record_id'] for row in evidence if row['view_kind'] == 'root_source' and row['origin_kind'] == 'payload'}
    child = {row['record_id'] for row in evidence if row['origin_kind'] != 'payload'}
    structured = {row['record_id'] for row in evidence if row['view_kind'] == 'structured_field'}
    transformed = {row['record_id'] for row in evidence if not row['text_is_exact_source_substring']}
    if eligible != {row['record_id'] for row in payloads if row['claim_candidate']}:
        raise ValueError('Saved candidate flags and eligible evidence disagree')
    for item in evidence:
        if not item['counts_toward_payload_candidate'] or item['containment'] != 'unique':
            raise ValueError('Eligible evidence includes an association-only record')
    physical = set()
    for row in evidence:
        span = row['raw_payload_match_span']
        if span is not None:
            key = (row['record_id'], row['category'], row['rule'], tuple(span))
        else:
            key = (row['record_id'], row['category'], row['rule'], row['physical_evidence_id'],
                   row['text_sha256'], row['start'], row['end'])
        physical.add(key)
    syntax_denominator = sum(row['operation_bearing_payload'] for row in payloads)
    category_rows = {}
    for category in categories:
        ids = {row['record_id'] for row in evidence if row['category'] == category}
        raw_category = {row['record_id'] for row in evidence if row['category'] == category
                        and row['view_kind'] == 'root_source' and row['origin_kind'] == 'payload'}
        if ids != {row['record_id'] for row in payloads if row[category]}:
            raise ValueError('Saved category flags and evidence disagree')
        values = counts(ids)
        supported = values['operation_bearing_payload_records']
        values.update(all_payload_denominator=len(payloads), operation_bearing_denominator=syntax_denominator,
                      candidate_fraction_of_all_payloads=len(ids) / len(payloads) if payloads else None,
                      candidate_fraction_of_operation_bearing_payloads=supported / syntax_denominator if syntax_denominator else None,
                      syntax_coverage_among_category_candidates=supported / len(ids) if ids else None,
                      raw_root_match_records=len(raw_category), extra_view_only_records=len(ids - raw_category))
        category_rows[category] = values
    artifact_roles = {}
    for role in ('summary', 'description', 'path', 'version'):
        items = [row for row in artifact_signals if row['artifact_signal'] == role]
        supported = [row for row in items if row['counts_toward_payload_candidate']]
        root_ids = {row['record_id'] for row in supported if row['origin_kind'] == 'payload'}
        child_ids = {row['record_id'] for row in supported if row['origin_kind'] != 'payload'}
        artifact_roles[role] = dict(view_records=len(items), eligible_view_records=len(supported),
            association_only_view_records=len(items) - len(supported), root_origin=counts(root_ids),
            uniquely_contained_child_origin=counts(child_ids),
            extra_child_only_roots=counts(child_ids - root_ids),
            transformed_eligible_view_records=sum(not row['text_is_exact_source_substring'] for row in supported))
    artifact_root_ids = {row['record_id'] for row in artifact_signals
                        if row['counts_toward_payload_candidate'] and row['origin_kind'] == 'payload'}
    artifact_child_ids = {row['record_id'] for row in artifact_signals
                         if row['counts_toward_payload_candidate'] and row['origin_kind'] != 'payload'}
    if artifact_root_ids | artifact_child_ids != {row['record_id'] for row in payloads if row['artifact_view_available']}:
        raise ValueError('Artifact evidence and structural availability disagree')
    origin_documents = [row for row in view_metadata if row['view_kind'] != 'structured_field']
    source_classes = {kind: dict(Counter(row['source_class'] for row in origin_documents if row['origin_kind'] == kind))
                      for kind in ('payload', 'response', 'recovered_text')}
    structured_views = [row for row in view_metadata if row['view_kind'] == 'structured_field']
    return dict(payload_population=dict(records=len(payloads), exact_unique_bodies=len(bodies),
                    operation_bearing_records=syntax_denominator,
                    prior_development_bodies=len(bodies & old_hashes),
                    bodies_outside_prior_development=len(bodies - old_hashes)),
        claim_routes=dict(eligible_any_view=counts(eligible), raw_full_root=counts(raw),
            extra_view_only=counts(eligible - raw), any_direct_child_evidence=counts(child),
            any_structured_field_evidence=counts(structured), any_transformed_text_evidence=counts(transformed),
            association_only=counts(row['record_id'] for row in association_evidence),
            evidence_view_records=len(evidence), distinct_rule_raw_span_or_transformed_container_matches=len(physical),
            categories=category_rows),
        artifacts=dict(root_origin=counts(artifact_root_ids),
            uniquely_contained_child_origin=counts(artifact_child_ids),
            extra_child_only_roots=counts(artifact_child_ids - artifact_root_ids), roles=artifact_roles,
            structurally_available_but_without_claim_candidate=counts((artifact_root_ids | artifact_child_ids) - eligible)),
        source_formats_by_origin_document=source_classes,
        structured_fields=dict(total=len(structured_views),
            by_nested_decode_depth=dict(Counter(str(row['nested_decode_depth']) for row in structured_views)),
            transformed_text_views=sum(not row['text_is_exact_source_substring'] for row in structured_views)),
        interpretation='Saved claim candidates are lexical/structured text observations. Additional views, matching spans, '
            'syntax support and apparent artifact fields do not establish intent, execution, communication between actors, '
            'independent episodes or actual cooperation. This is reporting only, with no detector rerun or retuning.')


def run(census, output):
    census, output = Path(census).resolve(), Path(output).resolve()
    if output == census or census in output.parents:
        raise ValueError('Diagnostics must be written outside the locked census')
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing diagnostics')
    lock, summary, verification = verify(census)
    metadata = []
    for row in json_lines(census / 'views.jsonl.gz'):
        metadata.append({key: row[key] for key in ('view_id', 'root_id', 'view_kind', 'origin_kind',
            'source_class', 'nested_decode_depth', 'text_is_exact_source_substring')})
    result = calculate(load_payloads(census / 'payloads.csv.gz'),
        list(json_lines(census / 'claim-evidence.jsonl.gz')),
        list(json_lines(census / 'association-claim-evidence.jsonl.gz')),
        list(json_lines(census / 'artifact-signals.jsonl.gz')), metadata,
        lock['excluded_development_body_hashes'], lock['detector']['categories'])
    if result['payload_population']['records'] != summary['views']['payload_records']:
        raise ValueError('Reconstructed denominator differs from locked summary')
    if result['claim_routes']['eligible_any_view']['payload_records'] != summary['claims']['cohorts']['all_payload_records']['claim_candidate']['count']:
        raise ValueError('Reconstructed candidate count differs from locked summary')
    result.update(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), source_sha256=lock['source_sha256'],
                  reporting_script_sha256=sha(__file__), census_verification=verification,
                  limits=dict(declared=lock['limits'], observed_diagnostics=summary['views']['diagnostics'],
                              json_size_limited_origin_documents=sum(counts.get('json_size_limit', 0)
                                  for counts in result['source_formats_by_origin_document'].values()),
                              structural_parse_rejection_reasons_separately_counted=False,
                              note='Malformed, non-JSON, duplicate-key and structural-limit rejections share the '
                                   'unstructured-source class; absence of a separate counter does not prove no such exclusions.'),
                  source_payloads_redecoded=False, external_reports_read=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census', type=Path, default=DEFAULT / 'census')
    parser.add_argument('--output', type=Path, default=DEFAULT / 'census-diagnostics.json')
    args = parser.parse_args()
    result = run(args.census, args.output)
    print(json.dumps(dict(output=str(args.output), population=result['payload_population'],
                          claims=result['claim_routes'], artifacts=result['artifacts']), indent=2))


if __name__ == '__main__':
    main()
