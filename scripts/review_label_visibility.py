"""Reporting-only diagnostics for saved temporal predictions and site labels.

No model is fitted, no threshold is selected, and no trace source is reopened.
The request snapshot is date-scoped with the existing loader; the later re-fetch
is never read. Follow-through is a site annotation, not independent ground truth.
"""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from swarmcollab import labels
from swarmcollab.metrics import auc


ROOT = Path(__file__).resolve().parents[1]
FIRST, LAST = '2026-08-17', '2026-09-15'
MODELS = ('layer1_behavior', 'layer1_with_attribution')
STATUS = ('yes', 'no', 'ambiguous')
KEYS = ['a', 'b', 'day']
SEED, DRAWS = 20261107, 2000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def bounded_predictions(frame, model_specs, first=FIRST, last=LAST):
    frame = frame.copy()
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError('Prediction keys must be complete and unique')
    days = frame.day.map(lambda x: dt.date.fromisoformat(x))
    if not days.between(dt.date.fromisoformat(first), dt.date.fromisoformat(last)).all():
        raise ValueError('Prediction file includes dates outside the authorized diagnostic window')
    if not (frame.a < frame.b).all():
        raise ValueError('Predictions must use canonical distinct-agent pairs')
    if frame.y.dtype != bool:
        raise ValueError('Saved request labels must be explicit booleans')
    for name in MODELS:
        if not np.isfinite(frame[name].to_numpy(float)).all() or frame[name + '__alert'].dtype != bool:
            raise ValueError('Saved scores/alerts must be finite and explicit')
        expected = frame[name].to_numpy() > model_specs[name]['threshold']
        if not np.array_equal(expected, frame[name + '__alert'].to_numpy()):
            raise ValueError('Saved alerts differ from frozen training threshold')
    return frame


def add_visibility(frame, activity):
    activity = activity.copy()
    if activity.duplicated(['agent', 'day']).any():
        raise ValueError('Activity agent-days must be unique')
    counts = activity[['working', 'trace_steps']].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts[:, 1] > counts[:, 0]).any():
        raise ValueError('Activity and trace coverage counts are invalid')
    indexed = activity.set_index(['agent', 'day'])
    ka, kb = [list(zip(frame[x], frame.day)) for x in ('a', 'b')]
    a, b = indexed.reindex(ka), indexed.reindex(kb)
    if a[['working', 'trace_steps']].isna().any().any() or b[['working', 'trace_steps']].isna().any().any():
        raise ValueError('Missing activity cannot be silently treated as zero visibility')
    if (a.working <= 0).any() or (b.working <= 0).any():
        raise ValueError('Eligible pair-day requires both actors to work')
    result = frame.copy()
    result['visibility_count'] = np.minimum(a.trace_steps.to_numpy(), b.trace_steps.to_numpy())
    result['visibility_share'] = np.minimum((a.trace_steps / a.working).to_numpy(),
                                          (b.trace_steps / b.working).to_numpy())
    return result


def visibility_table(frame, name):
    values = frame[name].to_numpy(float)
    cuts = np.quantile(values, [1 / 3, 2 / 3])
    group = np.digitize(values, cuts, right=True)
    rows = []
    for model in MODELS:
        for index, label in enumerate(('lowest', 'middle', 'highest')):
            selected = frame.loc[group == index]
            y = selected.y.to_numpy(bool); alerts = selected[model + '__alert'].to_numpy(bool)
            positives, negatives = int(y.sum()), int((~y).sum())
            rows.append(dict(model=model, visibility=name, group=label, pair_days=len(selected),
                positive_pair_days=positives, negative_pair_days=negatives,
                true_positives=int((y & alerts).sum()), false_positives=int((~y & alerts).sum()),
                recall=float(alerts[y].mean()) if positives else None,
                fpr=float(alerts[~y].mean()) if negatives else None,
                request_prevalence=positives / len(selected) if len(selected) else None,
                min_visibility=float(selected[name].min()) if len(selected) else None,
                max_visibility=float(selected[name].max()) if len(selected) else None))
    return rows, list(map(float, cuts))


def join_request_annotations(frame, request_rows):
    """One score per pair-day; repeated requests are explicit counts, not units."""
    indexed = {tuple(row): index for index, row in enumerate(frame[KEYS].itertuples(index=False, name=None))}
    groups = defaultdict(list); seen = set(); unique_requests = {}; duplicates = 0; outside = []
    for original in request_rows:
        row = dict(original); request = row['request']; recipient = row['to']
        if row['accepted'] not in STATUS or row['followed_through'] not in STATUS:
            raise ValueError('Unknown site annotation value')
        day = row['day'].isoformat() if isinstance(row['day'], dt.date) else row['day']
        if not FIRST <= day <= LAST:
            raise ValueError('Request row is outside the diagnostic window')
        invariant = (row['from'], day, row['accepted'], row['followed_through'])
        if request in unique_requests and unique_requests[request] != invariant:
            raise ValueError('One request has inconsistent annotations')
        unique_requests[request] = invariant
        identity = request, recipient
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        key = (*sorted((row['from'], recipient)), day)
        if key not in indexed:
            outside.append(dict(request=request, a=key[0], b=key[1], day=day))
            continue
        groups[key].append(row)
    expected = np.array([tuple(key) in groups for key in frame[KEYS].itertuples(index=False, name=None)])
    if not np.array_equal(expected, frame.y.to_numpy(bool)):
        raise ValueError('Scoped request rows do not reproduce frozen pair-day labels')
    out = []
    for key, rows in groups.items():
        record = frame.iloc[indexed[key]].to_dict()
        record['requests'] = len({r['request'] for r in rows})
        for name, source in [('accepted', 'accepted'), ('followed', 'followed_through')]:
            values = {r[source] for r in rows}
            record[name + '_group'] = '_and_'.join(value for value in STATUS if value in values)
            for value in STATUS:
                record[name + '_' + value + '_requests'] = len({r['request'] for r in rows if r[source] == value})
        out.append(record)
    requests = list(unique_requests.values())
    coverage = dict(unique_requests=len(requests), recipient_rows_before_dedup=len(request_rows),
        duplicate_request_recipient_rows=duplicates, recipient_rows_outside_eligible_pairs=len(outside),
        requests_outside_eligible_pairs=len({r['request'] for r in outside}),
        request_pair_days=len(groups), accepted=dict(Counter(r[2] for r in requests)),
        followed=dict(Counter(r[3] for r in requests)),
        accepted_by_followed=[dict(accepted=a, followed=f, requests=n)
                             for (a, f), n in sorted(Counter((r[2], r[3]) for r in requests).items())],
        outside_pair_days=sorted({tuple(r[k] for k in ('a', 'b', 'day')) for r in outside}))
    return pd.DataFrame(out), coverage


def followthrough_summary(pair_days, draws=DRAWS):
    groups = []
    for group, frame in pair_days.groupby('followed_group', sort=True):
        groups.append(dict(group=group, pair_days=len(frame), requests=int(frame.requests.sum()),
            days=frame.day.nunique(), distinct_pairs=frame[['a', 'b']].drop_duplicates().shape[0],
            model_results={model:dict(median_score=float(frame[model].median()),
                mean_score=float(frame[model].mean()), alerts=int(frame[model+'__alert'].sum()),
                alert_rate=float(frame[model+'__alert'].mean())) for model in MODELS}))
    clean = pair_days.loc[pair_days.followed_group.isin(['yes', 'no'])].copy()
    y = clean.followed_group.eq('yes').to_numpy()
    by_day = [np.flatnonzero(clean.day.to_numpy() == day) for day in sorted(clean.day.unique())]
    result = {}
    for model in MODELS:
        score = clean[model].to_numpy(float); alert = clean[model+'__alert'].to_numpy(bool)
        point = auc(score, y)
        delta = float(alert[y].mean()-alert[~y].mean()) if y.any() and (~y).any() else None
        rng = np.random.default_rng(SEED); samples, deltas = [], []
        for _ in range(draws):
            indices = np.concatenate([by_day[i] for i in rng.integers(len(by_day), size=len(by_day))]) if by_day else np.array([], dtype=int)
            yy = y[indices]
            if not yy.any() or yy.all():
                continue
            samples.append(auc(score[indices], yy))
            deltas.append(float(alert[indices][yy].mean()-alert[indices][~yy].mean()))
        result[model] = dict(auroc=float(point) if np.isfinite(point) else None,
            alert_rate_difference_yes_minus_no=delta,
            day_block_descriptive_95pct_auroc=list(map(float,np.quantile(samples,[.025,.975]))) if samples else None,
            day_block_descriptive_95pct_alert_difference=list(map(float,np.quantile(deltas,[.025,.975]))) if deltas else None,
            successful_resamples=len(samples))
    return dict(groups=groups, clean_contrast=dict(only_yes_pair_days=int(y.sum()),
        only_no_pair_days=int((~y).sum()), excluded_mixed_or_ambiguous_pair_days=len(pair_days)-len(clean),
        models=result), bootstrap=dict(draws=draws, seed=SEED, unit='Pacific day',
        interpretation='Descriptive only; serial/actor dependence and upstream annotation errors remain. No significance claim.'),
        interpretation='Exploratory whole-day score association among request-positive pair-days. All-yes versus all-no '
        'uses only internally unanimous site follow-through annotations. Mixed or ambiguous pair-days are separate. '
        'Repeated requests do not become independent scored units; no validation of actual consumption follows.')


def run(output, allow_label_snapshot_mismatch=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing reporting run')
    study = ROOT/'outputs/generalization-20261106'
    paths = dict(predictions=study/'temporal/predictions.csv.gz',
                 predictions_before_labels=study/'temporal/predictions-before-labels.csv.gz',
                 activity=study/'temporal/activity.csv.gz',
                 fit=study/'frozen/outputs/research-layers/scores.json',
                 labels=ROOT/'data/labels/ai-village-chart-props.json',
                 dataset_manifest=ROOT/'data/manifest.json',
                 village_lock=study/'village-results-lock.json')
    hashes = {name:sha(path) for name,path in paths.items()}
    bound = json.loads(paths['village_lock'].read_text())['files_sha256']
    for name in ('predictions','predictions_before_labels','activity'):
        if bound[str(paths[name].relative_to(study))] != hashes[name]:
            raise ValueError('Original frozen output identity changed: '+name)
    dataset = next(item for item in json.loads(paths['dataset_manifest'].read_text())['datasets']
                   if item['name']=='AI Village request records (validation labels)')
    label_snapshot_matches = dataset['files'][0]['sha256'] == hashes['labels']
    if not label_snapshot_matches and not allow_label_snapshot_mismatch:
        raise ValueError('Request snapshot differs from its retained source manifest '
                         '(pass --allow-label-snapshot-mismatch to use a newer snapshot)')
    fitted = json.loads(paths['fit'].read_text())['models']
    frame = bounded_predictions(pd.read_csv(paths['predictions']), fitted)
    before = pd.read_csv(paths['predictions_before_labels'])
    pd.testing.assert_frame_equal(frame.drop(columns='y'), before)
    frame = add_visibility(frame, pd.read_csv(paths['activity']))
    visibility, cuts = [], {}
    for name in ('visibility_count','visibility_share'):
        table, values = visibility_table(frame,name); visibility.extend(table); cuts[name]=values
    request_rows, requests, orphans = labels.load_requests(paths['labels'], start=FIRST, end=LAST)
    positive, coverage = join_request_annotations(frame, request_rows)
    coverage.update(selected_requests_including_orphans=requests, orphan_requests=orphans)
    follow = followthrough_summary(positive)
    result=dict(schema='saved-temporal-label-visibility-diagnostics-v1', created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        window=dict(start=FIRST,end=LAST,timezone='America/Los_Angeles'),
        request_source=dict(url='https://theaidigest.org/village/agent/gemini-2-5-pro',snapshot_date='2026-11-05',
            sha256=hashes['labels'], procedure='Requests extracted from the public chart props, recipients joined through '
            'edge instanceIds. Timestamp is interpreted as Pacific time; both directions collapse into an active pair-day.',
            annotation_method='The retained snapshot does not document how accepted/followedThrough were judged or independently validated.',
            label_target='Existence of at least one request, regardless of accepted or followedThrough. Those fields are not used by the frozen classifier.'),
        visibility=dict(definition='Minimum of the two agents\' trace-bearing working-step counts/shares on that Pacific day. '
            'This measures extractor coverage, not all observable activity or a randomized hiding intervention.',
            grouping='Target-window visibility tertiles computed without request labels; ties stay together. No score threshold retuning.',
            cuts=cuts,rows=visibility),
        request_coverage=coverage,followthrough=follow,
        frozen_thresholds={name:fitted[name]['threshold'] for name in MODELS},
        interpretation='Reporting-only post-review diagnostics. No models, primary labels, feature rules or original outputs changed. '
            'Site follow-through labels may share textual evidence with features and are not independent human consumption annotations. '
            'Whole-day scores are retrospective; direct chat exclusion does not remove conversations embedded in tool outputs.',
        input_files={name:dict(path=str(path.relative_to(ROOT)),sha256=hashes[name]) for name,path in paths.items()},
        code_sha256=sha(__file__), reserved_dates_opened=False, new_label_fetch=False,
        label_snapshot_matches_manifest=label_snapshot_matches)
    if hashes != {name:sha(path) for name,path in paths.items()}:
        raise ValueError('An input changed during reporting')
    output.mkdir(parents=True)
    save(output/'summary.json',result)
    pd.DataFrame(visibility).to_csv(output/'visibility.csv',index=False)
    positive.to_csv(output/'request-positive-pair-days.csv.gz',index=False)
    save(output/'manifest.json',dict(inputs=result['input_files'],code_sha256=result['code_sha256'],
        outputs_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/claude-review-20261107/labels-visibility')
    parser.add_argument('--allow-label-snapshot-mismatch',action='store_true',
                        help='accept a request snapshot that differs from the one recorded in data/manifest.json')
    args=parser.parse_args();r=run(args.output,allow_label_snapshot_mismatch=args.allow_label_snapshot_mismatch)
    print(json.dumps(dict(output=str(args.output),request_pair_days=r['request_coverage']['request_pair_days'],
                         clean_followthrough=r['followthrough']['clean_contrast'])))
