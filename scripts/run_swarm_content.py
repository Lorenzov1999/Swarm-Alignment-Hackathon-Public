"""Freeze and census bounded root-associated artifact text, without execution.

This runner reads no investigation reports. Source rows are never chronological
events. Original operation coverage is an independently verified payload-level
covariate; its sites are never counted as cooperation or assigned claim labels.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from dataclasses import asdict
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sqlite3
import sys
import tempfile
import time

from swarmcollab import swarm_claims as claims, swarm_views as views
from swarmcollab.swarm_static import _validate_graph


ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = ('scripts/run_swarm_content.py', 'src/swarmcollab/swarm_views.py',
              'src/swarmcollab/swarm_claims.py', 'src/swarmcollab/swarm_static.py',
              'tests/test_swarm_views.py', 'tests/test_swarm_claims.py',
              'tests/test_run_swarm_content.py')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line, object_pairs_hook=_strict_object)


def load_operations(path, manifest_path, expected_source):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['source_sha256'] != expected_source:
        raise ValueError('Operation census source identity differs')
    if manifest['outputs_sha256'].get(Path(path).name) != sha(path):
        raise ValueError('Operation coverage CSV does not match its original manifest')
    result = {}
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key, value = row['record_id'], row['operation_bearing_payload']
            if not key or key in result or value not in ('True', 'False'):
                raise ValueError('Invalid/duplicate operation coverage row')
            result[key] = dict(record_id=key, payload_sha256=row['payload_sha256'],
                               operation_bearing_payload=value == 'True')
    return result


def development_hashes(path, expected_source, operation_rows):
    value = json.loads(Path(path).read_text())
    if value['source_sha256'] != expected_source:
        raise ValueError('Development candidate source identity differs')
    result, identifiers = set(), set()
    for row in value['candidates']:
        key = row['record_id']
        if (key in identifiers or key not in operation_rows
                or row['text_sha256'] != operation_rows[key]['payload_sha256']):
            raise ValueError('Development candidate ID/hash does not bind a unique original payload')
        identifiers.add(key); result.add(row['text_sha256'])
    for digest in result:
        claims._digest(digest, 'development body hash')
    return result


def _metadata(source):
    metadata, kinds = {}, Counter()
    for record in records(source):
        key, kind, parent = record['id'], record['kind'], record['parent_id']
        if (not isinstance(key, str) or not key or key in metadata or not isinstance(kind, str) or not kind
                or (parent is not None and (not isinstance(parent, str) or not parent))):
            raise ValueError('Invalid or duplicate extraction metadata')
        metadata[key] = (kind, parent); kinds[kind] += 1
    _validate_graph(metadata)
    return metadata, kinds


def _store_text(source, database, metadata):
    database.execute('CREATE TABLE source (id TEXT PRIMARY KEY, kind TEXT NOT NULL, parent TEXT, text TEXT NOT NULL)')
    database.execute('CREATE INDEX source_parent ON source(parent)')
    inserted, seen, excluded = 0, set(), Counter()
    for record in records(source):
        key = record['id']; kind, parent = metadata[key]
        if key in seen or (record['kind'], record['parent_id']) != (kind, parent):
            raise ValueError('Source metadata changed between passes')
        seen.add(key)
        if kind != 'payload' and not (kind in views.DIRECT_CHILD_KINDS and parent is not None
                                      and metadata[parent][0] == 'payload'):
            if kind in views.DIRECT_CHILD_KINDS:
                excluded['no_direct_payload_parent'] += 1
            else:
                excluded['unsupported_record_kind'] += 1
            continue
        text = record['text']
        if not isinstance(text, str):
            raise ValueError('Eligible source text must be a string')
        database.execute('INSERT INTO source VALUES (?, ?, ?, ?)', (key, kind, parent, text))
        inserted += 1
        if inserted % 5000 == 0:
            database.commit()
    if seen != set(metadata):
        raise ValueError('Source records changed between passes')
    database.commit()
    return dict(excluded)


def _group(database, key, source_hash, limits):
    row = database.execute('SELECT id,kind,parent,text FROM source WHERE id=?', (key,)).fetchone()
    if row is None or row[1] != 'payload':
        raise ValueError('Sample/root payload missing from source store')
    payload = dict(zip(('id', 'kind', 'parent_id', 'text'), row))
    children = [dict(zip(('id', 'kind', 'parent_id', 'text'), child)) for child in
                database.execute('SELECT id,kind,parent,text FROM source WHERE parent=? AND kind IN (?,?) ORDER BY id',
                                 (key, 'response', 'recovered_text'))]
    return views.payload_views(payload, children, source_sha256=source_hash, limits=limits)


def _write_lines(stream, rows):
    for row in rows:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def write_audit_context(path, selection, database, source_hash, limits):
    """Full source context without automatic claim/category/coverage flags."""
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for selected in selection:
            key = selected['record_id']
            body = database.execute('SELECT text FROM source WHERE id=?', (key,)).fetchone()[0]
            if hashlib.sha256(body.encode()).hexdigest() != selected['payload_sha256']:
                raise ValueError('Audit body differs from its selected identity')
            related = {}
            for member in selected['body_member_ids']:
                group = _group(database, member, source_hash, limits)
                if group['payload']['payload_sha256'] != selected['payload_sha256']:
                    raise ValueError('Audit body member differs from canonical body')
                for view in group['views']:
                    if view['view_kind'] == 'root_source' or view['root_span'] is None:
                        continue
                    identity = (tuple(view['root_span']), view['source_text_sha256'], view['text_sha256'],
                                view['view_kind'], view['field_role'])
                    if identity not in related:
                        # Only source/provenance metadata. No detector outputs,
                        # strata, automatic categories, or candidate flags.
                        allowed = ('view_id', 'root_id', 'origin_record_id', 'origin_kind', 'view_kind',
                                   'field_role', 'text', 'source_text', 'text_sha256', 'source_text_sha256',
                                   'source_span', 'root_span', 'source_literal_span', 'json_pointer',
                                   'json_pointer_chain', 'ndjson_line_index', 'nested_decode_depth',
                                   'text_is_exact_source_substring', 'source_class')
                        related[identity] = {name: view[name] for name in allowed if name in view}
                        related[identity]['origin_members'] = []
                    related[identity]['origin_members'].append(dict(root_id=member,
                        origin_record_id=view['origin_record_id'], view_id=view['view_id']))
            _write_lines(stream, [dict(record_id=key, payload_sha256=selected['payload_sha256'],
                source_sha256=source_hash, full_raw_body=body, body_characters=len(body),
                exact_body_member_ids=selected['body_member_ids'], related_contained_views=list(related.values()),
                interpretation='Source text and uniquely contained views only; no automatic detector decisions supplied.')])


def run(source, operations, operation_manifest, development_candidates, protocol, output,
        *, expected_source=views.SOURCE_SHA256, limits=None):
    paths = {name: Path(value).resolve() for name, value in dict(source=source, operations=operations,
             operation_manifest=operation_manifest, development_candidates=development_candidates,
             protocol=protocol).items()}
    output = Path(output).resolve(); limits = limits or views.Limits()
    if output.exists():
        raise FileExistsError('Refusing to overwrite a census, failed phase, or pre-census lock')
    input_hashes = {name: sha(path) for name, path in paths.items()}
    source_hash = input_hashes['source']
    if source_hash != expected_source:
        raise ValueError('Source bytes differ from the explicitly expected identity')
    operation_rows = load_operations(paths['operations'], paths['operation_manifest'], source_hash)
    prior_hashes = development_hashes(paths['development_candidates'], source_hash, operation_rows)
    if {name: sha(path) for name, path in paths.items()} != input_hashes:
        raise ValueError('An input changed while loading frozen metadata')
    code_hashes = {name: sha(ROOT / name) for name in CODE_PATHS}
    output.mkdir(parents=True)
    started = time.monotonic()
    try:
        for relative, digest in code_hashes.items():
            target = output / 'code-snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
            if sha(target) != digest:
                raise ValueError('Code changed while creating the frozen snapshot')
        protocol_copy = output / 'code-snapshot/protocol.md'
        protocol_copy.write_bytes(paths['protocol'].read_bytes())
        if sha(protocol_copy) != input_hashes['protocol']:
            raise ValueError('Protocol changed while creating the frozen snapshot')
        if {name: sha(path) for name, path in paths.items()} != input_hashes:
            raise ValueError('An input changed before the pre-census lock')
        lock = dict(locked_before_census_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    source_sha256=source_hash, inputs={name: dict(path=str(path), sha256=input_hashes[name])
                        for name, path in paths.items()}, code_sha256=code_hashes,
                    limits=asdict(limits), detector=claims.detector_specification(), audit_plan=claims.audit_plan(),
                    runtime=dict(python=sys.version, sqlite=sqlite3.sqlite_version),
                    review_rubric=claims.review_rubric(), excluded_development_body_hashes=sorted(prior_hashes),
                    artifact_view_availability='Summary/description/path/version field in a complete structured root '
                        'or uniquely exactly contained direct response/recovered child; structural, not a claim label.',
                    no_execution=True, no_source_subtraction=True, no_external_comparison=True,
                    source_coordinate_system='Unicode character offsets; decoded views explicitly distinguished.',
                    interpretation='Exploratory frozen rerun with prior project exposure; not fresh outcome-naive validation.')
        save(output / 'pre-census-lock.json', lock)
        metadata, kinds = _metadata(paths['source'])
        payload_ids = sorted(key for key, (kind, _) in metadata.items() if kind == 'payload')
        if set(payload_ids) != set(operation_rows):
            raise ValueError('Operation denominator does not equal the complete source payload population')
        source_counts, source_classes, containment, diagnostic = [Counter() for _ in range(4)]
        total_views = total_fields = total_associations = physical_spans = parent_text_groups = 0
        evidence_count = association_claim_count = artifact_count = 0
        payloads, claim_rows = [], []
        with tempfile.TemporaryDirectory(prefix='swarm-content-') as scratch, \
                sqlite3.connect(str(Path(scratch) / 'source.sqlite')) as database, ExitStack() as stack:
            excluded = _store_text(paths['source'], database, metadata)
            streams = {name: stack.enter_context(gzip.open(output / (name + '.jsonl.gz'), 'wt', encoding='utf-8'))
                       for name in ('views', 'associations', 'claim-evidence', 'association-claim-evidence', 'artifact-signals')}
            for index, key in enumerate(payload_ids):
                group = _group(database, key, source_hash, limits)
                classified = claims.classify_payload(group, operation_rows[key])
                payloads.append(group['payload']); claim_rows.append({**group['payload'], **classified['payload']})
                _write_lines(streams['views'], group['views'])
                _write_lines(streams['associations'], group['associations'])
                _write_lines(streams['claim-evidence'], classified['evidence'])
                _write_lines(streams['association-claim-evidence'], classified['association_evidence'])
                _write_lines(streams['artifact-signals'], classified['artifact_signals'])
                source_counts.update(v['origin_kind'] for v in group['views'])
                source_classes.update(v['source_class'] for v in group['views'])
                containment.update(a['containment'] for a in group['associations'])
                diagnostic.update(group['diagnostics'])
                total_views += sum(v['view_kind'] != 'structured_field' for v in group['views'])
                total_fields += sum(v['view_kind'] == 'structured_field' for v in group['views'])
                total_associations += len(group['associations'])
                physical_spans += len({v['physical_evidence_id'] for v in group['views'] if v['physical_evidence_id']})
                parent_text_groups += len({v['parent_text_duplicate_id'] for v in group['views']})
                evidence_count += len(classified['evidence']); association_claim_count += len(classified['association_evidence'])
                artifact_count += len(classified['artifact_signals'])
                if (index + 1) % 10000 == 0:
                    print('Processed', index + 1, 'payload roots', flush=True)
            with gzip.open(output / 'payloads.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
                columns = list(claim_rows[0]) if claim_rows else ['record_id', 'payload_sha256']
                writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(claim_rows)
            for name, selection in [('primary', claims.select_unmarked_audit(claim_rows, prior_hashes)),
                                    ('targeted', claims.select_candidate_audit(claim_rows, prior_hashes))]:
                annotations = selection.pop('annotation_template')
                save(output / ('audit-' + name + '-selection.json'), selection)
                save(output / ('audit-' + name + '-annotations.json'),
                     dict(rubric=claims.review_rubric(), annotations=annotations))
                write_audit_context(output / ('audit-' + name + '-context.jsonl.gz'), selection['selection'],
                                    database, source_hash, limits)
        view_summary = dict(payload_records=len(payloads), exact_unique_payload_bodies=len({p['payload_sha256'] for p in payloads}),
            payload_roots_with_artifact_fields=sum(p['root_structured_artifact_available'] for p in payloads),
            payload_roots_with_contained_artifact_fields=sum(p['artifact_view_available'] for p in payloads),
            payload_roots_with_direct_children=sum(p['direct_child_available'] for p in payloads),
            source_view_records=total_views, structured_field_views=total_fields,
            direct_child_records=total_associations, child_containment_counts=dict(containment),
            view_counts_by_origin=dict(source_counts), view_counts_by_source_class=dict(source_classes),
            unique_physical_evidence_spans=physical_spans, exact_parent_text_groups=parent_text_groups,
            diagnostics=dict(diagnostic), excluded_record_counts=excluded, source_record_kind_counts=dict(kinds))
        summary = dict(source_sha256=source_hash, source_file_bytes_verified=True,
                       pre_census_lock_sha256=sha(output / 'pre-census-lock.json'), views=view_summary,
                       claims=claims.summarize_claims(claim_rows, evidence_records=evidence_count,
                                                     association_records=association_claim_count),
                       artifact_signal_view_records=artifact_count,
                       resources=dict(elapsed_seconds=time.monotonic() - started,
                                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                       actual_cooperation_rate=None, execution_count=None, chronology=None,
                       interpretation='Root-associated text/artifact census and fixed diagnostic samples; '
                           'no executed-call, actor, interaction, chronology, or cooperation prevalence claim.')
        if {name: sha(path) for name, path in paths.items()} != input_hashes:
            raise ValueError('An input changed after the pre-census lock')
        if {name: sha(ROOT / name) for name in CODE_PATHS} != code_hashes:
            raise ValueError('Analysis code changed after the pre-census lock')
        save(output / 'summary.json', summary)
        save(output / 'manifest.json', dict(source_sha256=source_hash, code_sha256=code_hashes,
             inputs_sha256=input_hashes, outputs_sha256={str(path.relative_to(output)): sha(path)
                for path in sorted(output.rglob('*')) if path.is_file()}))
        return summary
    except BaseException as error:
        save(output / 'failed.json', dict(status='failed_closed', error_type=type(error).__name__,
                                        elapsed_seconds=time.monotonic() - started,
                                        preserved_for_audit=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('source', 'operations', 'development-candidates', 'output'):
        parser.add_argument('--' + argument, type=Path, required=True)
    parser.add_argument('--operation-manifest', type=Path)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'docs/swarm-content-protocol.md')
    args = parser.parse_args()
    result = run(args.source, args.operations, args.operation_manifest or args.operations.parent / 'manifest.json',
                 args.development_candidates, args.protocol, args.output)
    print(json.dumps(dict(output=str(args.output), payload_records=result['views']['payload_records'],
                          elapsed_seconds=result['resources']['elapsed_seconds'])))


if __name__ == '__main__':
    main()
