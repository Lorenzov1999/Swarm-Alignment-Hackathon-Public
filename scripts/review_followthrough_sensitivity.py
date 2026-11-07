"""Request-intensity sensitivity reusing the tested, frozen-score summarizer.

Reporting only, after the initial descriptive result: restrict to exactly one
recorded request per eligible pair-day. Preserve every original output.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
REPORTER=ROOT/'scripts/review_label_visibility.py'
SPEC=importlib.util.spec_from_file_location('label_visibility_report',REPORTER)
report=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(report)


def run(folder):
    folder=Path(folder)
    output=folder/'single-request-sensitivity.json'
    if output.exists():raise FileExistsError('Refusing to overwrite completed sensitivity')
    summary=json.loads((folder/'summary.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,digest in manifest['outputs_sha256'].items():
        if report.sha(folder/name)!=digest:raise ValueError('Original report output changed: '+name)
    if summary['code_sha256']!=report.sha(REPORTER):
        raise ValueError('Original summary implementation changed')
    source=folder/'request-positive-pair-days.csv.gz'
    frame=pd.read_csv(source)
    if frame.duplicated(report.KEYS).any():raise ValueError('Pair-day source is not unique')
    selected=frame.loc[frame.requests==1].copy()
    if not selected.followed_group.isin(report.STATUS).all():
        raise ValueError('One request cannot have a mixed follow-through group')
    result=dict(schema='single-request-pair-day-sensitivity-v1',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_pair_days=len(frame),single_request_pair_days=len(selected),
        followthrough=report.followthrough_summary(selected),
        interpretation='Exploratory request-intensity sensitivity proposed after observing unequal request counts. '
        'One request per pair-day removes within-day request multiplicity from this contrast; roster, work type, '
        'visibility, activity and annotation dependence remain uncontrolled. No new detector fitting or labels.',
        input_sha256={str(source.relative_to(ROOT)):report.sha(source),
            str((folder/'summary.json').relative_to(ROOT)):report.sha(folder/'summary.json'),
            str((folder/'manifest.json').relative_to(ROOT)):report.sha(folder/'manifest.json'),
            str(REPORTER.relative_to(ROOT)):report.sha(REPORTER)},
        code_sha256=report.sha(__file__))
    report.save(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ROOT/'outputs/claude-review-20261107/labels-visibility')
    args=parser.parse_args();r=run(args.folder)
    print(json.dumps(dict(single_request_pair_days=r['single_request_pair_days'],
                         contrast=r['followthrough']['clean_contrast'])))
