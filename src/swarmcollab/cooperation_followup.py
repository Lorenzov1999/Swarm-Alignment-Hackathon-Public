"""Reporting-only follow-up of frozen Village predictions and provenance.

No extraction, fitting, threshold selection, raw-data reads, or external-source
access. Day resampling is descriptive and preserves dependence within a day;
it does not remove serial dependence or identify a causal effect of goals.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .transfer_scoring import PERIODS, average_precision, _validate_rows


SEED = 20261106
REPETITIONS = 5000
GOAL_HASH = '48c8961ccc0350781957fb0b4ea2e553db82d42d7dd4057eebee472db9cc753b'
MODELS = ('layer1_behavior', 'layer1_plus_layer2_behavior',
          'layer1_with_attribution', 'layer1_plus_layer2_with_attribution')
BANDS = {'exposure': (1,), 'candidate': (2,), 'acquisition': (3,),
         'incorporation': (4,), 'strong': (3, 4)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def rate_comparison(left, right, numerator, denominator, repetitions=REPETITIONS, seed=SEED):
    """Independent day-block draws, retaining positive/zero infinite ratios.

    A zero/zero ratio is undefined and explicitly counted. For the reported
    empirical bounds, undefined draws range from zero to infinity. Finite-only
    draws are never substituted for an interval with an unbounded upper end.
    """
    arrays = [frame[[numerator, denominator]].to_numpy(dtype=float) for frame in (left, right)]
    if any(not len(a) or not np.isfinite(a).all() or (a < 0).any() for a in arrays):
        raise ValueError('Comparison requires nonempty finite nonnegative day blocks')
    if any(a[:, 1].sum() <= 0 for a in arrays):
        raise ValueError('Both groups require positive total exposure')
    rng = np.random.default_rng(seed)
    totals = [a.sum(axis=0) for a in arrays]
    rates = [x[0] / x[1] for x in totals]
    draws = []
    for a in arrays:
        sampled = a[rng.integers(0, len(a), size=(repetitions, len(a)))].sum(axis=1)
        draws.append(np.divide(sampled[:, 0], sampled[:, 1], out=np.full(repetitions, np.nan),
                               where=sampled[:, 1] > 0))
    usable = np.isfinite(draws[0]) & np.isfinite(draws[1])
    l, r = draws[0][usable], draws[1][usable]
    differences = l - r
    undefined = (l == 0) & (r == 0)
    extended = np.divide(l[~undefined], r[~undefined], out=np.full((~undefined).sum(), np.inf),
                         where=r[~undefined] > 0)
    conditional = np.quantile(extended, [.025, .975], method='inverted_cdf') if len(extended) else [np.nan, np.nan]
    status = 'finite' if rates[1] else ('undefined_both_zero' if not rates[0] else 'unbounded_zero_baseline')
    sparse = any(x[0] == 0 for x in totals)
    if sparse:
        bounds = [np.nan, np.nan]
    else:
        bounds = [np.quantile(np.r_[extended, np.zeros(undefined.sum())], .025, method='inverted_cdf'),
                  np.quantile(np.r_[extended, np.full(undefined.sum(), np.inf)], .975, method='inverted_cdf')]
    return dict(numerator=numerator, denominator=denominator, left_days=len(left), right_days=len(right),
                left_count=int(totals[0][0]), right_count=int(totals[1][0]),
                left_exposure=float(totals[0][1]), right_exposure=float(totals[1][1]),
                left_rate=float(rates[0]), right_rate=float(rates[1]),
                rate_difference=float(rates[0]-rates[1]),
                difference_interval_95=np.quantile(differences, [.025, .975]).tolist() if len(differences) else None,
                rate_ratio=ratio(rates[0], rates[1]), rate_ratio_status=status,
                ratio_interval_95=[float(v) if np.isfinite(v) else None for v in bounds],
                ratio_bound_status=['finite' if np.isfinite(v) else ('unbounded' if np.isinf(v) else 'undefined') for v in bounds],
                ratio_defined_draw_interval_95=[float(v) if np.isfinite(v) else None for v in conditional],
                ratio_interval_policy='Undefined resampled 0/0 ratios contribute zero to the lower-bound '
                                      'distribution and infinity to the upper-bound distribution. No ratio '
                                      'interval is reported when either observed group has zero events.',
                resamples=repetitions, usable_exposure_draws=int(usable.sum()),
                undefined_zero_over_zero_draws=int(undefined.sum()),
                positive_over_zero_draws=int(np.isinf(extended).sum()),
                zero_observed_group=sparse,
                interpretation='Descriptive day-block bootstrap, not a causal or independent-day significance test. '
                               'A zero-event group gives degenerate empirical resampling and cannot establish absence. '
                               'Undefined 0/0 draws are retained as an unknown range in the reported ratio bounds.')


def prepare_chains(chains, predictions):
    frame = chains.copy()
    if frame.chain_id.duplicated().any() or not frame.level.isin([1, 2, 3, 4]).all():
        raise ValueError('Invalid chain identity or level')
    frame['repository'] = frame.version_key.str.rsplit('@', n=1).str[0]
    universe = set(predictions[['a', 'b', 'day']].itertuples(index=False, name=None))
    frame['supported_pair'] = [(min(a, b), max(a, b), day) in universe
                              for a, b, day in frame[['source_actor', 'target_actor', 'day']].itertuples(index=False, name=None)]
    return frame


def daily_table(activity, predictions, chains, assignments=None, git_events=None):
    """Keep working-day exposure independent of the number of detected chains."""
    if activity.duplicated(['agent', 'day']).any() or (activity.working < 0).any():
        raise ValueError('Invalid activity table')
    days = sorted(activity.loc[activity.working > 0, 'day'].unique())
    if not set(chains.day) <= set(days) or not set(predictions.day) <= set(days):
        raise ValueError('Observation day lacks working exposure')
    labels = {} if assignments is None else {r['day']: r for r in assignments}
    rows = []
    for day in days:
        a, p, c = activity[activity.day == day], predictions[predictions.day == day], chains[chains.day == day]
        label = labels.get(day, {})
        row = dict(day=day, category=label.get('primary_category', 'temporal'),
                   goal_id=label.get('primary_goal_id'), transition=label.get('transition_day', False),
                   working_steps=int(a.working.sum()), active_agent_days=int((a.working > 0).sum()),
                   active_pair_days=len(p), request_pair_days=int(p.y.sum()),
                   shell_steps=int(a.shell.sum()), gui_steps=int(a.gui.sum()))
        for name, levels in BANDS.items():
            subset = c[c.level.isin(levels)]
            supported = subset[subset.supported_pair]
            row[name+'_chains'] = len(subset)
            row[name+'_target_turns'] = len(subset[['target_actor', 'target_turn']].drop_duplicates())
            row[name+'_eligible_chains'] = len(supported)
            row[name+'_supported_pair_days'] = len({(min(x, y), max(x, y)) for x, y in
                                                    supported[['source_actor', 'target_actor']].itertuples(index=False, name=None)})
        for model in MODELS:
            row[model+'_alerts'] = int(p[model+'__alert'].sum())
        if git_events is not None:
            e=git_events[git_events.day==day]
            row['git_observed_turns']=len(e[['actor','turn_id']].drop_duplicates())
            consumer=e[e.status.eq('success') & e.operation.isin(['acquire','incorporate'])]
            producer=e[e.status.eq('success') & e.operation.eq('generate')]
            row['git_success_consumer_turns']=len(consumer[['actor','turn_id']].drop_duplicates())
            row['git_success_producer_turns']=len(producer[['actor','turn_id']].drop_duplicates())
        rows.append(row)
    return pd.DataFrame(rows)


def chain_summary(chains):
    """Several explicit cluster definitions; none counts true collaborations."""
    strong = chains[chains.level >= 3]
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    for r in strong.to_dict('records'):
        nodes = [('turn', r['source_actor'], r['source_turn']),
                 ('turn', r['target_actor'], r['target_turn']), ('version', r['version_key'])]
        root = find(nodes[0])
        for node in nodes[1:]:
            parents[find(node)] = root
    pair_counts = strong.groupby(['source_actor', 'target_actor']).size().sort_values(ascending=False)
    repo_counts = strong.repository.value_counts()
    return dict(strong_chains=len(strong),
                unique_target_turns=len(strong[['target_actor', 'target_turn']].drop_duplicates()),
                directed_actor_pair_days=len(strong[['source_actor', 'target_actor', 'day']].drop_duplicates()),
                repository_days=len(strong[['repository', 'day']].drop_duplicates()),
                repository_actor_pair_groups=len(strong[['repository', 'source_actor', 'target_actor']].drop_duplicates()),
                version_or_turn_linked_components=len({find(x) for x in parents}),
                unique_versions=int(strong.version_key.nunique()),
                event_days=int(strong.day.nunique()),
                repositories=[dict(repository=k, chains=int(v)) for k, v in repo_counts.items()],
                actor_pairs=[dict(source=a, target=b, chains=int(v)) for (a, b), v in pair_counts.items()],
                target_actors={k:int(v) for k, v in strong.target_actor.value_counts().items()},
                eligible_strong_chains=int(strong.supported_pair.sum()),
                interpretation='Components join shared actor-turn nodes or exact frozen version keys; '
                               'repository/actor groups are coarser concentration summaries, not independent episodes.')


def monitoring(predictions, model, repetitions=REPETITIONS):
    p = predictions
    y, alert = p.y.to_numpy(dtype=bool), p[model+'__alert'].to_numpy(dtype=bool)
    tp, fp = int((y & alert).sum()), int((~y & alert).sum())
    days = sorted(p.day.unique())
    counts = np.array([[int((g.y & g[model+'__alert']).sum()),
                        int((~g.y & g[model+'__alert']).sum()), int(g.y.sum()), int((~g.y).sum())]
                       for _, g in p.groupby('day', sort=True)], dtype=float)
    rng = np.random.default_rng(SEED)
    sampled = counts[rng.integers(0, len(days), size=(repetitions, len(days)))].sum(axis=1)
    terms = {'precision': (sampled[:, 0], sampled[:, 0]+sampled[:, 1]),
             'recall': (sampled[:, 0], sampled[:, 2]), 'fpr': (sampled[:, 1], sampled[:, 3]),
             'alerts_per_day': (sampled[:, 0]+sampled[:, 1], np.full(repetitions, len(days)))}
    intervals = {}
    for name, (n, d) in terms.items():
        values = n[d > 0]/d[d > 0]
        intervals[name] = dict(interval_95=np.quantile(values, [.025, .975]).tolist() if len(values) else None,
                               defined_draws=len(values), omitted_zero_denominator_draws=int((d == 0).sum()))
    grouped = p.groupby(['a', 'b'])
    any_request = grouped.y.any()
    alert_days = grouped[model+'__alert'].sum()
    selected = alert_days[alert_days > 0]
    auc = metrics.auc(p[model].to_numpy(), y)
    return dict(model=model, rows=len(p), days=len(days), positives=int(y.sum()), alerts=int(alert.sum()),
                true_positives=tp, request_negative_alerts=fp, prevalence=float(y.mean()),
                auc=float(auc) if np.isfinite(auc) else None,
                average_precision=average_precision(p[model].to_numpy(), y),
                precision=ratio(tp,tp+fp), recall=ratio(tp,y.sum()), fpr=ratio(fp,(~y).sum()),
                alerts_per_day=ratio(alert.sum(),len(days)), request_negative_alerts_per_day=ratio(fp,len(days)),
                alerts_per_detected_request_pair_day=ratio(alert.sum(),tp),
                unique_alerted_pairs=len(selected), median_alert_days_per_alerted_pair=float(selected.median()) if len(selected) else None,
                max_alert_days_per_pair=int(selected.max()) if len(selected) else 0,
                alerted_pairs_without_any_request_in_period=int((~any_request.loc[selected.index]).sum()),
                descriptive_day_bootstrap=intervals,
                interpretation='Retrospective complete-day request-proxy detection at the saved threshold; '
                               'request-negative alerts are not verified non-interactions. No latency, lead-time, '
                               'prospective replay, calibrated interaction probability, or reviewer-time claim.')


def build(study, destination):
    if destination.resolve() == study.resolve() or study.resolve() in destination.resolve().parents:
        raise ValueError('Follow-up output must be separate from the frozen study')
    inputs = [study/'goal-regime-classification.json']
    if sha(inputs[0]) != GOAL_HASH:
        raise ValueError('Frozen goal classification hash mismatch')
    classification = json.loads(inputs[0].read_text())
    bundles, tables, monitor = {}, [], {}
    for period in ('shared_goals', 'temporal'):
        paths = {name:study/period/name for name in ('activity.csv.gz','predictions.csv.gz','chains.jsonl.gz','scores.json')}
        inputs += list(paths.values())
        a, p = pd.read_csv(paths['activity.csv.gz']), pd.read_csv(paths['predictions.csv.gz'])
        _validate_rows(p, period, require_labels=True)
        lo, hi = PERIODS[period]
        if not a.day.between(lo,hi).all():
            raise ValueError('Activity outside authorized period')
        with gzip.open(paths['chains.jsonl.gz'], 'rt') as stream:
            c = prepare_chains(pd.DataFrame(json.loads(line) for line in stream), p)
        if not c.day.between(lo,hi).all():
            raise ValueError('Chain target outside authorized period')
        saved = json.loads(paths['scores.json'].read_text())
        assert int(a.working.sum()) == saved['measurement']['working_steps']
        assert len(c) == saved['measurement']['provenance']['target_chains']
        for model in MODELS:
            assert np.array_equal(p[model+'__alert'], p[model] > saved['models'][model]['threshold'])
        assignments = classification['day_assignments'] if period == 'shared_goals' else None
        event_paths=sorted((study/period/'provenance-events').glob('*.jsonl.gz'))
        if not event_paths:
            raise ValueError('Missing frozen provenance event metadata')
        inputs+=event_paths
        event_rows=[]
        for event_path in event_paths:
            with gzip.open(event_path,'rt') as stream:
                for line in stream:
                    event=json.loads(line)
                    if not lo <= event['day'] <= hi:
                        raise ValueError('Provenance event outside authorized period')
                    if event['family']=='git':
                        event_rows.append({key:event[key] for key in ('actor','turn_id','day','operation','status')})
        git=pd.DataFrame(event_rows)
        consumers=git[git.status.eq('success') & git.operation.isin(['acquire','incorporate'])]
        strong_turns=set(c.loc[c.level>=3,['target_actor','target_turn','day']].itertuples(index=False,name=None))
        consumer_turns=set(consumers[['actor','turn_id','day']].itertuples(index=False,name=None))
        if not strong_turns <= consumer_turns:
            raise ValueError('Strong target turns not included in Git-opportunity denominator')
        daily = daily_table(a,p,c,assignments,git)
        daily['period'] = period
        tables.append(daily)
        bundles[period] = (a,p,c,daily)
        monitor[period] = {model: monitoring(p,model) for model in MODELS}
        monitor[period]['strata'] = {stratum:{model:monitoring(g,model) for model in MODELS}
                                     for stratum,g in p.groupby('agent_stratum')}
    summaries, comparisons, concentrations = [], [], {}
    groups = {}
    for sensitivity in ('all_days','exclude_transitions'):
        for period, (a,p,c,d) in bundles.items():
            retained = d if sensitivity == 'all_days' else d[~d.transition]
            groups[(sensitivity,period)] = retained
            selections = [('all',retained)] + (list(retained.groupby('category',sort=True)) if period=='shared_goals' else [])
            for name,g in selections:
                days=set(g.day); selected=c[c.day.isin(days)]; key='|'.join((sensitivity,period,name))
                row=dict(sensitivity=sensitivity,period=period,group=name,observed_days=len(g))
                for column in g.select_dtypes(include='number'):
                    row[column]=int(g[column].sum())
                for band in BANDS:
                    row[band+'_per_100000_working_steps']=ratio(100000*row[band+'_chains'],row['working_steps'])
                    row[band+'_eligible_per_1000_pair_days']=ratio(1000*row[band+'_eligible_chains'],row['active_pair_days'])
                row['strong_target_turns_per_1000_observed_git_consumer_turns']=ratio(1000*row['strong_target_turns'],row['git_success_consumer_turns'])
                row['request_prevalence']=ratio(row['request_pair_days'],row['active_pair_days'])
                summaries.append(row);concentrations[key]=chain_summary(selected)
                if period=='shared_goals' and name!='all': groups[(sensitivity,name)]=g
        names=[('shared_goals','temporal'),('collaborative','temporal'),('collaborative','competitive'),
               ('collaborative','individual'),('collaborative','mixed')]
        measurements=[('strong_chains','working_steps'),('strong_eligible_chains','active_pair_days'),
                      ('strong_target_turns','working_steps'),('strong_supported_pair_days','active_pair_days'),
                      ('candidate_chains','working_steps'),('candidate_eligible_chains','active_pair_days'),
                      ('exposure_chains','working_steps'),('request_pair_days','active_pair_days'),
                      ('strong_target_turns','git_success_consumer_turns'),
                      ('layer1_behavior_alerts','active_pair_days'),('layer1_with_attribution_alerts','active_pair_days')]
        for left,right in names:
            for numerator,denominator in measurements:
                l,r=groups[(sensitivity,left)],groups[(sensitivity,right)]
                if min(l[denominator].sum(),r[denominator].sum()) <= 0:
                    comparisons.append(dict(sensitivity=sensitivity,left=left,right=right,numerator=numerator,
                                            denominator=denominator,status='unavailable_zero_observed_exposure'))
                    continue
                result=rate_comparison(l,r,numerator,denominator)
                comparisons.append(dict(sensitivity=sensitivity,left=left,right=right,**result))
    # Numerator-only concentration ablations: preserve all activity exposure.
    ablations=[]
    a,p,c,d=bundles['shared_goals'];strong=c[c.level>=3]
    top_repos=strong.repository.value_counts().index.tolist()[:2]
    top_pair=strong.groupby(['source_actor','target_actor']).size().idxmax()
    masks={'remove_top_repository':~c.repository.eq(top_repos[0]),
           'remove_top_two_repositories':~c.repository.isin(top_repos),
           'remove_top_actor_pair':~(c.source_actor.eq(top_pair[0]) & c.target_actor.eq(top_pair[1]))}
    for label,mask in masks.items():
        reduced=daily_table(a,p,c[mask],classification['day_assignments'])
        for left,right in [('shared_goals','temporal'),('collaborative','competitive'),('collaborative','mixed')]:
            l=reduced if left=='shared_goals' else reduced[reduced.category==left]
            r=bundles['temporal'][3] if right=='temporal' else reduced[reduced.category==right]
            ablations.append(dict(ablation=label,left=left,right=right,
                                  ablation_interpretation='Remove selected June provenance numerators only; all exposure denominators unchanged.',
                                  **rate_comparison(l,r,'strong_chains','working_steps')))
    destination.mkdir(parents=True,exist_ok=True)
    pd.concat(tables).to_csv(destination/'daily.csv',index=False)
    pd.DataFrame(summaries).to_csv(destination/'group-summary.csv',index=False)
    reports={'rate-comparisons.json':comparisons,'monitoring.json':monitor,
             'chain-concentration.json':concentrations,'concentration-ablations.json':ablations}
    for name,value in reports.items():
        (destination/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    manifest=dict(schema='village-cooperation-followup-v1',inputs_sha256={str(x):sha(x) for x in inputs},
                  code_sha256=sha(__file__),seed=SEED,day_bootstrap_repetitions=REPETITIONS,
                  outputs_sha256={x.name:sha(x) for x in destination.iterdir() if x.is_file() and x.name!='manifest.json'},
                  analysis_status='Post-result follow-up of frozen outputs; no new detector fitting or source reads.',
                  limitations=['Sparse chains share projects, actors and days; these are not independent cooperation events.',
                               'Goal regimes are confounded with calendar time, actor composition, work type and observability.',
                               'Day-block percentile intervals omit between-day dependence and detector/rule uncertainty.',
                               'Active-pair denominators use only chains whose two actors enter the saved same-day pair universe.',
                               'Observed successful Git consumer-turn exposure is an extraction-opportunity sensitivity, not a complete operation census.',
                               'Whole-day request-proxy scoring is not an evaluated online monitor.',
                               'Raw strong-chain counts retain the frozen automated evidence levels; audits do not relabel them.'])
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return pd.DataFrame(summaries)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('outputs/generalization-20261106'))
    parser.add_argument('--output',type=Path,default=Path('outputs/cooperation-followup-20261106/village'))
    args=parser.parse_args()
    print(build(args.study,args.output)[['sensitivity','period','group','working_steps','active_pair_days','strong_chains','candidate_chains','request_pair_days']].to_string(index=False))
