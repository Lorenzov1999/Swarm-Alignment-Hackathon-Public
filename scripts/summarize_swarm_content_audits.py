"""Validate locked contextual audits, then join frozen candidate flags.

Reporting only: no source extraction, detector fitting, rule changes, execution,
external access, cooperation prevalence, absence bound, or recall estimate.
"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path


FIELDS = ('role_or_task_allocation', 'publication_intent', 'result_request_intent', 'artifact_evidence')
DECISIONS = ('yes', 'no_visible_evidence', 'unknown')
CONTEXTS = ('explicit_claim_or_intent', 'generic_or_conditional_or_quoted_artifact_wording',
            'no_visible_explicit_claim', 'unresolved_or_truncated')
RAW_COORDINATES = {'raw_payload_unicode_characters', 'raw_body_unicode_characters',
                   'full_raw_body_unicode_characters'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def bound_file(census, manifest, relative):
    """Only explicitly manifest-bound census artifacts can enter a report."""
    path = Path(census) / relative
    expected = manifest['outputs_sha256'].get(relative)
    if expected is None or sha(path) != expected:
        raise ValueError('Census manifest hash mismatch: ' + relative)
    return path


def load_packets(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def validate_selection_context(selection, packets, source_sha256):
    selected, contexts, bodies = {}, {}, set()
    for row in selection['selection']:
        identifier, body = row['record_id'], row['payload_sha256']
        if not isinstance(identifier, str) or not identifier or identifier in selected or body in bodies:
            raise ValueError('Audit selection must contain unique canonical IDs and body hashes')
        if identifier != min(row['body_member_ids']) or len(set(row['body_member_ids'])) != len(row['body_member_ids']):
            raise ValueError('Invalid canonical body membership')
        selected[identifier] = row; bodies.add(body)
    for packet in packets:
        identifier = packet['record_id']
        if identifier in contexts or identifier not in selected:
            raise ValueError('Duplicate or unselected audit context')
        row = selected[identifier]
        if (packet['source_sha256'] != source_sha256 or packet['payload_sha256'] != row['payload_sha256']
                or text_sha(packet['full_raw_body']) != row['payload_sha256']
                or packet['body_characters'] != len(packet['full_raw_body'])
                or sorted(packet['exact_body_member_ids']) != sorted(row['body_member_ids'])):
            raise ValueError('Context body/source identity differs from selection')
        for view in packet['related_contained_views']:
            start, end = _span(view['root_span'], len(packet['full_raw_body']))
            if (packet['full_raw_body'][start:end] != view['source_text']
                    or text_sha(view['source_text']) != view['source_text_sha256']
                    or text_sha(view['text']) != view['text_sha256']):
                raise ValueError('Context view source span or hash is inconsistent')
        contexts[identifier] = packet
    if set(contexts) != set(selected):
        raise ValueError('Audit selection has missing context bodies')
    return selected, contexts


def _span(span, length):
    if (not isinstance(span, list) or len(span) != 2 or any(type(value) is not int for value in span)
            or not 0 <= span[0] <= span[1] <= length):
        raise ValueError('Invalid evidence/review span')
    return span


def _coordinates(entry, packet):
    """Resolve the two declared annotation schemas without inventing offsets."""
    view_id = entry.get('view_id')
    coordinate = entry.get('coordinate_system')
    raw = (coordinate in RAW_COORDINATES or
           (coordinate is None and entry.get('source', entry.get('view')) == 'full_raw_body'))
    if raw:
        if view_id is not None:
            raise ValueError('Raw body coordinates cannot silently refer to a decoded view')
        record_id = entry.get('record_id', entry.get('source_record_id', packet['record_id']))
        if record_id not in packet['exact_body_member_ids']:
            raise ValueError('Raw evidence points outside the exact body group')
        text, identity = packet['full_raw_body'], 'full_raw_body'
    elif coordinate == 'view_text_unicode_characters' and view_id is not None:
        matches = [view for view in packet['related_contained_views']
                   if view['view_id'] == view_id or view_id in
                   {item['view_id'] for item in view.get('origin_members', [])}]
        if len(matches) != 1:
            raise ValueError('Evidence view identity is missing or ambiguous')
        view = matches[0]
        allowed = {view.get('root_id'), view.get('origin_record_id')}
        for member in view.get('origin_members', []):
            allowed.update([member['root_id'], member['origin_record_id']])
        record_id = entry.get('record_id', entry.get('source_record_id'))
        if record_id is not None and record_id not in allowed:
            raise ValueError('Evidence record does not belong to its view')
        text, identity = view['text'], view['view_id']
    else:
        raise ValueError('Unsupported or missing evidence coordinate system')
    span = entry.get('span', [entry.get('start'), entry.get('end')])
    start, end = _span(span, len(text))
    return text, identity, start, end


def _covers_body(extents, packet):
    spans = []
    for extent in extents:
        _, identity, start, end = _coordinates(extent, packet)
        if identity == 'full_raw_body':
            spans.append((start, end))
    reached = 0
    for start, end in sorted(spans):
        if start > reached:
            return False
        reached = max(reached, end)
    return reached == len(packet['full_raw_body']) and bool(spans)


def validate_annotations(document, selection, contexts):
    """Reject incomplete labels, invented spans and negative partial reviews."""
    indexed = {}
    for row in document['annotations']:
        identifier = row['record_id']
        if identifier in indexed or identifier not in selection:
            raise ValueError('Duplicate or unselected annotation ID')
        packet = contexts[identifier]
        if row['payload_sha256'] != selection[identifier]['payload_sha256']:
            raise ValueError('Annotation payload hash differs from selected body')
        if any(row.get(field) not in DECISIONS for field in FIELDS):
            raise ValueError('Every field needs a completed yes/no-visible-evidence/unknown decision')
        if row.get('context_status') not in CONTEXTS or type(row.get('body_fully_reviewed')) is not bool:
            raise ValueError('Annotation requires completed context and review-extent status')
        if not isinstance(row.get('reviewed_extent'), list) or not row['reviewed_extent']:
            raise ValueError('Annotation requires explicit inspected extents')
        full = _covers_body(row['reviewed_extent'], packet)
        if row['body_fully_reviewed'] and not full:
            raise ValueError('Full-body review assertion lacks complete raw-body extents')
        if not row['body_fully_reviewed'] and (
                any(row[field] == 'no_visible_evidence' for field in FIELDS)
                or row['context_status'] == 'no_visible_explicit_claim'):
            raise ValueError('Partial reviews cannot receive whole-body negative decisions')
        if not isinstance(row.get('rationale'), str) or not row['rationale'].strip():
            raise ValueError('Annotation requires a contextual rationale')
        supported = set()
        inspected = [_coordinates(extent, packet) for extent in row['reviewed_extent']]
        if not isinstance(row.get('evidence'), list):
            raise ValueError('Evidence must be an explicit list')
        for evidence in row['evidence']:
            text, identity, start, end = _coordinates(evidence, packet)
            if not any(other_identity == identity and inspected_start <= start <= end <= inspected_end
                       for _, other_identity, inspected_start, inspected_end in inspected):
                raise ValueError('Evidence lies outside declared inspected extents')
            if start == end or evidence.get('literal') != text[start:end]:
                raise ValueError('Evidence literal differs from its exact source span')
            field = evidence.get('field', evidence.get('supports'))
            if field not in FIELDS + ('context_status',):
                raise ValueError('Evidence must name a reviewed field or context status')
            if not isinstance(evidence.get('rationale'), str) or not evidence['rationale'].strip():
                raise ValueError('Evidence requires a rationale')
            supported.add(field)
        if any(row[field] == 'yes' and field not in supported for field in FIELDS):
            raise ValueError('Positive annotation lacks field-specific supporting evidence')
        indexed[identifier] = row
    if set(indexed) != set(selection):
        raise ValueError('Audit annotations are missing selected bodies')
    return indexed


def load_locked_annotations(annotation_path, lock_path, context_path):
    """Accept the two existing lock formats; never mutate original audit files."""
    annotation_path, lock_path = Path(annotation_path), Path(lock_path)
    lock = load_json(lock_path)
    expected = lock.get('annotation_sha256', lock.get('sha256'))
    name = lock.get('annotation_file', lock.get('path'))
    stamp = lock.get('locked_at_utc', lock.get('locked_utc'))
    if not isinstance(name, str) or Path(name).name != annotation_path.name or expected != sha(annotation_path):
        raise ValueError('Annotation lock does not match the supplied annotation file')
    try:
        timestamp = dt.datetime.fromisoformat(stamp)
    except (TypeError, ValueError) as error:
        raise ValueError('Annotation lock needs an explicit timestamp') from error
    if timestamp.tzinfo is None or timestamp > dt.datetime.now(dt.timezone.utc):
        raise ValueError('Annotation lock timestamp must be an existing timezone-aware time')
    document = load_json(annotation_path)
    context_digest = sha(context_path)
    bindings = document.get('input_sha256', {})
    bound = document.get('context_sha256') == context_digest or any(
        digest == context_digest and (Path(key).name == Path(context_path).name or key == 'context')
        for key, digest in bindings.items())
    if not bound:
        raise ValueError('Annotations do not bind the supplied context bytes')
    if 'input_sha256' in lock and lock['input_sha256'] != bindings:
        raise ValueError('Annotation lock and document input bindings differ')
    return document, dict(annotation_file=str(annotation_path), annotation_sha256=expected,
                          lock_file=str(lock_path), lock_sha256=sha(lock_path), locked_at_utc=timestamp.isoformat())


def annotation_counts(rows):
    rows = list(rows)
    return dict(bodies=len(rows), full_body_reviews=sum(row['body_fully_reviewed'] for row in rows),
                partial_body_reviews=sum(not row['body_fully_reviewed'] for row in rows),
                fields={field: {decision: sum(row[field] == decision for row in rows)
                                for decision in DECISIONS} for field in FIELDS},
                context_status=dict(Counter(row['context_status'] for row in rows)))


def summarize_reviews(primary, primary_selection, targeted_a, targeted_b, *, prediction_rows):
    """Pure summary; caller must validate locks before loading predictions."""
    if set(targeted_a) != set(targeted_b):
        raise ValueError('Targeted reviewers cover different bodies')
    strata = {}
    for name, population in primary_selection['strata'].items():
        rows = [primary[item['record_id']] for item in primary_selection['selection'] if item['stratum'] == name]
        if len(rows) != population['selected']:
            raise ValueError('Primary stratum selected count differs from annotations')
        strata[name] = dict(**population, annotations=annotation_counts(rows))
    agreements = {}
    for field in FIELDS + ('context_status',):
        pairs = Counter((targeted_a[key][field], targeted_b[key][field]) for key in sorted(targeted_a))
        agreements[field] = dict(agree=sum(count for (a,b), count in pairs.items() if a == b),
                                 disagree=sum(count for (a,b), count in pairs.items() if a != b),
                                 label_pairs=[dict(reviewer_a=a,reviewer_b=b,bodies=count)
                                              for (a,b),count in sorted(pairs.items())])
    groups, identifiers = defaultdict(list), set()
    for row in prediction_rows:
        if row['record_id'] in identifiers:
            raise ValueError('Duplicate frozen prediction record ID')
        identifiers.add(row['record_id'])
        if type(row['claim_candidate']) is not bool:
            raise ValueError('Frozen candidate flag must be an explicit boolean')
        groups[row['payload_sha256']].append(row)
    joined = []
    for item in primary_selection['selection']:
        rows = groups.get(item['payload_sha256'], [])
        if {row['record_id'] for row in rows} != set(item['body_member_ids']):
            raise ValueError('Prediction body membership differs from audit selection')
        flag = any(row['claim_candidate'] for row in rows)
        annotation = primary[item['record_id']]
        joined.append(dict(record_id=item['record_id'],payload_sha256=item['payload_sha256'],
                           frozen_candidate=flag,stratum=item['stratum'],
                           **{field:annotation[field] for field in FIELDS}))
    cross_tabs = {field:[dict(frozen_candidate=flag, annotation=decision,
                              bodies=sum(row['frozen_candidate'] == flag and row[field] == decision for row in joined))
                         for flag in (False,True) for decision in DECISIONS] for field in FIELDS}
    primary_hashes = {row['payload_sha256'] for row in primary.values()}
    targeted_hashes = {row['payload_sha256'] for row in targeted_a.values()}
    overlap = sorted(primary_hashes & targeted_hashes)
    return dict(primary=annotation_counts(primary.values()),primary_by_stratum=strata,
                targeted_reviewer_a=annotation_counts(targeted_a.values()),
                targeted_reviewer_b=annotation_counts(targeted_b.values()),targeted_agreement=agreements,
                sample_overlap=dict(exact_bodies=len(overlap),payload_sha256=overlap),
                primary_after_lock_prediction_join=joined,primary_prediction_annotation_counts=cross_tabs,
                actual_cooperation_rate=None,recall=None,absence_bound=None,
                interpretation='Counts describe reviewed sampled text and automatic claim candidates only. '
                'No visible evidence is a sampled-text judgment, not a negative cooperation label. '
                'Unknown and partial reviews remain explicit. Deterministic stratified/hash samples, '
                'assistant annotations, redaction and content dependence do not establish prevalence, recall, '
                'independent episodes or absence; artifact evidence is not actual coordination.')


def load_predictions(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as stream:
        rows = []
        for row in csv.DictReader(stream):
            value = row['claim_candidate']
            if value not in ('True','False'):
                raise ValueError('Invalid frozen prediction flag')
            rows.append(dict(record_id=row['record_id'],payload_sha256=row['payload_sha256'],
                             claim_candidate=value=='True'))
        return rows


def build(census, primary_annotation, primary_lock, targeted_annotation_a, targeted_lock_a,
          targeted_annotation_b, targeted_lock_b, output):
    census, output = Path(census).resolve(), Path(output).resolve()
    if output == census or census in output.parents:
        raise ValueError('Audit summary must be outside the frozen census')
    if output.exists():
        raise FileExistsError('Refusing to overwrite a completed audit summary')
    manifest_path = census/'manifest.json'; manifest=load_json(manifest_path)
    inputs = {str(manifest_path):sha(manifest_path)}
    selections, contexts, context_paths = {}, {}, {}
    for name in ('primary','targeted'):
        selection_path=bound_file(census,manifest,'audit-'+name+'-selection.json')
        context_path=bound_file(census,manifest,'audit-'+name+'-context.jsonl.gz')
        selection=load_json(selection_path)
        selected, packets=validate_selection_context(selection,load_packets(context_path),manifest['source_sha256'])
        selections[name]=(selection,selected); contexts[name]=packets; context_paths[name]=context_path
        inputs.update({str(path):sha(path) for path in (selection_path,context_path)})
    reviews, locks = [], []
    for name, annotation, lock in [('primary',primary_annotation,primary_lock),
                                 ('targeted',targeted_annotation_a,targeted_lock_a),
                                 ('targeted',targeted_annotation_b,targeted_lock_b)]:
        document, metadata=load_locked_annotations(annotation,lock,context_paths[name])
        reviews.append(validate_annotations(document,selections[name][1],contexts[name])); locks.append(metadata)
        inputs.update({str(Path(annotation)):sha(annotation),str(Path(lock)):sha(lock)})
    # This is deliberately after ALL annotation locks and completed decisions.
    prediction_path=bound_file(census,manifest,'payloads.csv.gz')
    predictions=load_predictions(prediction_path); inputs[str(prediction_path)]=sha(prediction_path)
    result=summarize_reviews(reviews[0],selections['primary'][0],reviews[1],reviews[2],prediction_rows=predictions)
    if any(sha(path)!=digest for path,digest in inputs.items()):
        raise ValueError('An audit input changed during reporting')
    result.update(source_sha256=manifest['source_sha256'],input_sha256=inputs,annotation_locks=locks,
                  primary_sampling_design={key:selections['primary'][0].get(key) for key in
                      ('plan','excluded_development_bodies','eligible_unmarked_bodies','sampling_unit')},
                  targeted_sampling_design={key:selections['targeted'][0].get(key) for key in
                      ('population','selected','seed','cap','sampling','interpretation')},
                  reporting_code_sha256=sha(__file__),created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  prediction_join_performed_after_all_annotation_locks_validated=True,
                  detector_or_annotation_changes=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp')
    if temporary.exists():
        raise FileExistsError('Refusing to overwrite an incomplete reporting artifact')
    temporary.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(output)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('census','primary-annotation','primary-lock','targeted-annotation-a','targeted-lock-a',
                 'targeted-annotation-b','targeted-lock-b','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=build(args.census,args.primary_annotation,args.primary_lock,args.targeted_annotation_a,args.targeted_lock_a,
                 args.targeted_annotation_b,args.targeted_lock_b,args.output)
    print(json.dumps(dict(primary_bodies=result['primary']['bodies'],
                         targeted_bodies=result['targeted_reviewer_a']['bodies'],output=str(args.output))))


if __name__=='__main__':
    main()
