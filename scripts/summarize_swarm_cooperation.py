"""Join reviewed wording to static payload coverage, never to executed actions."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_operation_manifest(operations, manifest_path, expected_source):
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest['source_sha256'] != expected_source:
        raise ValueError('Operation census and marker review source identities differ')
    expected=manifest['outputs_sha256'].get(Path(operations).name)
    if expected is None or sha(operations) != expected:
        raise ValueError('Operation denominator does not match its census manifest')


def summarize(operations, review):
    frame = operations.copy()
    if frame.record_id.duplicated().any():
        raise ValueError('Duplicate operation record ID')
    if not frame.operation_bearing_payload.isin([True, False]).all():
        raise ValueError('Missing operation coverage status')
    indexed = frame.set_index('record_id')
    records = review['candidates']
    if len({r['record_id'] for r in records}) != len(records):
        raise ValueError('Duplicate reviewed record ID')
    for row in records:
        if row['record_id'] not in indexed.index:
            raise ValueError('Reviewed payload missing from denominator')
        if indexed.loc[row['record_id'], 'payload_sha256'] != row['text_sha256']:
            raise ValueError('Reviewed payload hash differs from denominator')
        if row['confirmed_cooperation']:
            raise ValueError('This report supports claims only, not confirmed cooperation')
    mapping = {r['record_id']:r for r in records}
    frame['lexical_candidate'] = frame.record_id.isin(mapping)
    frame['explicit_role_or_allocation_claim'] = frame.record_id.isin(
        {r['record_id'] for r in records if r['primary_explicit_work_sharing_claim']})
    frame['publication_intent'] = frame.record_id.isin(
        {r['record_id'] for r in records if r['category'] == 'publication_intent'})
    # Exact duplicate bodies must receive the same syntax/wording decisions.
    fields = ['operation_bearing_payload', 'lexical_candidate',
              'explicit_role_or_allocation_claim', 'publication_intent']
    if (frame.groupby('payload_sha256')[fields].nunique() > 1).any().any():
        raise ValueError('Identical payload bodies have inconsistent decisions')
    cohorts = {'all_payload_records':frame,
               'exact_unique_payload_bodies':frame.sort_values('record_id').drop_duplicates('payload_sha256')}
    summaries = {}
    for name, group in cohorts.items():
        summaries[name] = {}
        for subset, selected in [('all', group),
                                 ('operation_bearing', group[group.operation_bearing_payload])]:
            n = len(selected)
            summaries[name][subset] = dict(denominator=n, **{
                field:dict(count=int(selected[field].sum()),
                           fraction=float(selected[field].sum()/n) if n else None)
                for field in fields[1:]})
    return dict(cohorts=summaries,
                reviewed_candidate_coverage=[dict(record_id=r['record_id'], category=r['category'],
                    explicit_role_or_allocation_claim=r['primary_explicit_work_sharing_claim'],
                    operation_bearing_payload=bool(indexed.loc[r['record_id'],'operation_bearing_payload']),
                    operation_site_count=int(indexed.loc[r['record_id'],'operation_site_count'])) for r in records],
                true_cooperation_rate=None, executed_action_rate=None, timeline=None,
                interpretation='Fractions of payload records containing reviewed wording, with optional supported syntax. '
                    'A marker is never assigned to every site within its payload. No action, success, actor, time, '
                    'independent episode, recall or actual cooperation prevalence is inferred. Unmarked status is unknown.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operations',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--time-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a coverage join')
    review=json.loads(args.review.read_text())
    time_audit=json.loads(args.time_audit.read_text())
    if time_audit['accepted_original_action_anchors'] != 0:
        raise ValueError('This no-timeline report requires a separate review for any accepted clock')
    if review['source_sha256'] != time_audit['source_sha256']:
        raise ValueError('Review and clock audit source identities differ')
    operation_manifest=args.operations.parent/'manifest.json'
    validate_operation_manifest(args.operations,operation_manifest,review['source_sha256'])
    args.output.mkdir(parents=True)
    lock=dict(created_before_coverage_join_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
              source_sha256=review['source_sha256'], code_sha256=sha(__file__),
              input_sha256={str(p):sha(p) for p in [args.operations,operation_manifest,args.review,args.time_audit]},
              numerator='Reviewed explicit task-allocation/coordinator-role claims; publication intent and any lexical candidates separate.',
              denominators='All payloads and static operation-bearing payloads, each with exact whole-body duplicate sensitivity.',
              unit='Payload record or exact-unique payload body, not an executed action or individual operation site.',
              time_join=False, significance_test=False, post_exposure_exploratory=True)
    (args.output/'signal-definition-lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    result=summarize(pd.read_csv(args.operations),review)
    result.update(source_sha256=review['source_sha256'],
                  signal_lock_sha256=sha(args.output/'signal-definition-lock.json'))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['cohorts'],indent=2))


if __name__=='__main__':
    main()
