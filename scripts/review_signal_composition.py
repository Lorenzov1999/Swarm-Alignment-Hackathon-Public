#!/usr/bin/env python3
"""Which kinds of shared resource carry the shared-resource signal? July only.

Reporting only, on the training period (6-31 July): no validation or test day is
read, and no model, weight or threshold changes. For every pair-day and every
resource that both agents touched that day, the pair-day receives the
resource's July rarity weight (the frozen IDF used by `pairs`). Summing by
resource type shows what the rarity-weighted overlap is made of on pair-days
with a recorded request, and how much denser each type is there than on other
pair-days. The weighting matches the shared-resource feature; the cosine's
per-agent normalization and repeat-count damping are left out, so these are
descriptive shares of overlap, not a decomposition of the cosine.

Usage: python scripts/review_signal_composition.py [--output DIR]
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

from swarmcollab import config, pairs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
TOP = 10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(series):
    return series.astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy()


def mentions_digest():
    digest = hashlib.sha256()
    for path in sorted((OUT / 'mentions').iterdir()):
        digest.update(path.name.encode() + b'\0' + sha256(path).encode() + b'\n')
    return digest.hexdigest()


def july_days():
    first, last = config.DEV_START, config.TRAIN_END
    return {(first + dt.timedelta(days=i)).isoformat() for i in range((last - first).days + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=OUT / 'signal-composition')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; refusing to overwrite a reporting run.')

    days = july_days()
    mentions, mapping = pairs.prepare_mentions(days)
    traces = mentions[mentions.type.isin(pairs.TRACE_TYPES)].copy()
    traces['tid'] = traces.tid.astype(str).astype('category')
    activity = pairs.load_activity(days)
    training_agents = activity.agent.unique()   # as in pairs.build
    idf = pairs.trace_weights(traces, training_agents, days)

    labels_path = OUT / 'research-layers' / 'features_dev.csv.gz'
    labelled = pd.read_csv(labels_path, usecols=['a', 'b', 'day', 'y'])
    labelled = labelled[labelled.day.isin(days)]
    labelled['y'] = boolean(labelled.y)
    label = dict(zip(zip(labelled.a, labelled.b, labelled.day), labelled.y))
    counts = collections.Counter(labelled.y)

    kinds = traces.drop_duplicates('tid').set_index('tid')['type'].astype(str)
    users = (traces.assign(agent=traces.agent.astype(str), day=traces.day.astype(str))
             .groupby(['day', 'tid'], observed=True).agent.agg(lambda s: sorted(set(s))))
    by_kind = {True: collections.Counter(), False: collections.Counter()}
    by_resource = collections.Counter()
    for (day, tid), agents in users.items():
        if len(agents) < 2:
            continue
        weight = idf.get(tid, 0.0)
        for i, first in enumerate(agents):
            for second in agents[i + 1:]:
                y = label.get((first, second, day))
                if y is None:
                    continue
                by_kind[y][kinds[tid]] += weight
                if y:
                    by_resource[tid] += weight

    total = sum(by_kind[True].values())
    rows = []
    for kind in sorted(set(by_kind[True]) | set(by_kind[False]), key=lambda k: -by_kind[True][k]):
        request = by_kind[True][kind] / counts[True]
        other = by_kind[False][kind] / counts[False]
        rows.append(dict(type=kind, share_of_request_day_overlap=by_kind[True][kind] / total,
                         per_request_pair_day=request, per_other_pair_day=other,
                         ratio=request / other if other else None))
    top = [dict(resource=tid, type=kinds[tid], share_of_request_day_overlap=w / total)
           for tid, w in by_resource.most_common(TOP)]

    args.output.mkdir(parents=True)
    summary = dict(
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Training period 6-31 July only; descriptive; no fitting or threshold change.',
        definition=__doc__.split('\n\n')[1].replace('\n', ' '),
        pair_days=dict(request=counts[True], other=counts[False]),
        training_agents=len(training_agents), mapping=mapping,
        by_type=rows, largest_single_resources=top,
        inputs_sha256={str(labels_path.relative_to(ROOT)): sha256(labels_path),
                       'outputs/mentions/* (name and hash of every file)': mentions_digest()},
        code_sha256=sha256(__file__))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('largest single resource share of request-day overlap:', round(top[0]['share_of_request_day_overlap'], 4))


if __name__ == '__main__':
    main()
