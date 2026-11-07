"""Post-review descriptive co-use diagnostics from already extracted Village traces.

No fitting, label-driven selection, source extraction, or changes to prior outputs.
The primary unweighted measure uses exact raw repository, document and page IDs.
It does not use July rarity weights or learned identity mappings. A shared step
contains a resource observed for another active agent on the same Pacific day;
this is co-occurrence, not verified communication or use of another's work.
"""
import argparse
from collections import Counter
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PERIODS = {'shared_goals': ('2026-06-01', '2026-07-05'),
           'temporal': ('2026-08-17', '2026-09-15')}
SPECIFIC_TYPES = ('repo', 'pages', 'page', 'gdoc')
LEGACY_TYPES = SPECIFIC_TYPES + ('commit', 'issue', 'pipeline', 'project', 'site')
CONTEXTS = ('cmd', 'cwd', 'typed', 'gui')
VARIANTS = ('specific', 'specific_command_context', 'all_legacy')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_days(frame, period):
    if period not in PERIODS:
        raise ValueError('Only the two previously authorized target periods are allowed')
    first, last = PERIODS[period]
    for day in frame.day.unique():
        if not isinstance(day, str) or dt.date.fromisoformat(day).isoformat() != day or not first <= day <= last:
            raise ValueError('Input contains an out-of-scope or malformed day')


def load_mentions(directory, period):
    """Read derived TSVs, count malformed records, reject valid out-of-period dates."""
    rows, diagnostics = [], Counter()
    for path in sorted(Path(directory).glob('*.tsv.gz')):
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
            reader = csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE)
            if next(reader) != ['agent', 't', 'day', 'step', 'type', 'id', 'op', 'src', 'k']:
                raise ValueError('Unexpected cached mention header')
            for row in reader:
                diagnostics['source_rows'] += 1
                if len(row) != 9:
                    diagnostics['malformed_field_count'] += 1
                    continue
                agent, _, day, step, kind, entity, operation, source, _ = row
                try:
                    parsed_day = dt.date.fromisoformat(day).isoformat()
                except ValueError:
                    diagnostics['malformed_date'] += 1
                    continue
                first, last = PERIODS[period]
                if parsed_day != day or not first <= day <= last:
                    raise ValueError('Cached mentions include a forbidden/out-of-period day')
                if kind not in LEGACY_TYPES:
                    diagnostics['nonresource_rows'] += 1
                    continue
                if not step.isdigit() or not agent or not entity:
                    diagnostics['malformed_resource_identity'] += 1
                    continue
                rows.append((agent, day, int(step), kind, entity, operation, source))
    if not rows:
        raise ValueError('No eligible derived resource mentions')
    frame = pd.DataFrame(rows, columns=['agent', 'day', 'step', 'type', 'id', 'op', 'src'])
    for column in ('agent', 'day', 'type', 'op', 'src'):
        frame[column] = frame[column].astype('category')
    diagnostics['eligible_resource_rows'] = len(frame)
    return frame, dict(diagnostics)


def co_use_counts(mentions, activity, *, variant='specific'):
    """Count unique actor/day/step exposures; repetitions never create new steps."""
    if variant not in VARIANTS:
        raise ValueError('Unknown predeclared sensitivity')
    if activity[['agent', 'day']].duplicated().any() or (activity.working < 0).any():
        raise ValueError('Activity needs unique agent-days with nonnegative working counts')
    if variant == 'all_legacy':
        selected = mentions[mentions.type.isin(LEGACY_TYPES)].copy()
    else:
        selected = mentions[mentions.type.isin(SPECIFIC_TYPES)].copy()
        if variant == 'specific_command_context':
            selected = selected[selected.src.isin(CONTEXTS) & selected.op.isin(['read', 'write', 'create'])]
    selected = selected.drop_duplicates(['agent', 'day', 'step', 'type', 'id'])
    active = activity.loc[activity.working > 0, ['agent', 'day', 'working', 'shell', 'gui']].copy()
    for frame in (active, selected):
        for key in ('agent', 'day'):
            frame[key] = frame[key].astype(str)
    universe = set(active[['agent', 'day']].itertuples(index=False, name=None))
    if set(selected[['agent', 'day']].itertuples(index=False, name=None)) - universe:
        raise ValueError('A trace belongs to an inactive or missing agent-day')
    shared = selected.groupby(['day', 'type', 'id'], observed=True).agent.transform('nunique') >= 2
    traced = selected.groupby(['agent', 'day'], observed=True).step.nunique().rename('trace_steps')
    common = selected.loc[shared].groupby(['agent', 'day'], observed=True).step.nunique().rename('shared_steps')
    agents = active.merge(traced, on=['agent', 'day'], how='left', validate='one_to_one')
    agents = agents.merge(common, on=['agent', 'day'], how='left', validate='one_to_one').fillna(0)
    agents[['trace_steps', 'shared_steps']] = agents[['trace_steps', 'shared_steps']].astype(int)
    if ((agents.shared_steps > agents.trace_steps) | (agents.trace_steps > agents.working)).any():
        raise ValueError('Step numerator exceeds its observed denominator')
    agents['shared_fraction_of_working'] = agents.shared_steps / agents.working
    agents['shared_fraction_of_traced'] = agents.shared_steps / agents.trace_steps.replace(0, np.nan)
    daily = agents.groupby('day').agg(working_steps=('working', 'sum'), trace_steps=('trace_steps', 'sum'),
                                     shared_steps=('shared_steps', 'sum'), active_agents=('agent', 'nunique'),
                                     trace_covered_agents=('trace_steps', lambda x: int((x > 0).sum())),
                                     actor_mean_shared_fraction_of_working=('shared_fraction_of_working', 'mean'),
                                     actor_mean_shared_fraction_of_traced=('shared_fraction_of_traced', 'mean')).reset_index()
    daily['shared_fraction_of_working'] = daily.shared_steps / daily.working_steps
    daily['shared_fraction_of_traced'] = daily.shared_steps / daily.trace_steps.replace(0, np.nan)
    daily['trace_coverage'] = daily.trace_steps / daily.working_steps
    return daily, agents


def attach_frozen_series(daily, predictions, regime_daily, activity, period, goals):
    """Keep classifier alerts, unsupervised co-use flags and proxy requests separate."""
    for frame in (daily, predictions, regime_daily, activity):
        validate_days(frame, period)
    if predictions[['a', 'b', 'day']].duplicated().any() or (predictions.a >= predictions.b).any():
        raise ValueError('Invalid frozen active pair-day universe')
    pred = predictions.groupby('day').agg(pair_days=('a', 'size'),
        l1_alerts=('layer1_behavior__alert', 'sum'), request_pair_days=('y', 'sum')).reset_index()
    pred['classifier_alert_share'] = pred.l1_alerts / pred.pair_days
    pred['request_prevalence'] = pred.request_pair_days / pred.pair_days
    regime = regime_daily[['day', 'flagged', 'flagged_share', 'null_calibrated']].rename(
        columns={'flagged':'co_use_flags', 'flagged_share':'co_use_threshold_share',
                 'null_calibrated':'co_use_null_calibrated'})
    merged = daily.merge(pred, on='day', validate='one_to_one').merge(regime, on='day', validate='one_to_one')
    if len(merged) != len(daily) or set(merged.day) != set(pred.day):
        raise ValueError('Frozen daily universes disagree')
    act = activity[activity.working > 0].groupby('day').agg(shell_steps=('shell','sum'), gui_steps=('gui','sum')).reset_index()
    merged = merged.merge(act, on='day', validate='one_to_one')
    merged['steps_per_active_agent'] = merged.working_steps / merged.active_agents
    merged['shell_fraction'] = merged.shell_steps / merged.working_steps
    merged['gui_fraction'] = merged.gui_steps / merged.working_steps
    merged['period'] = period
    if period == 'shared_goals':
        intervals = {x['goal_id']:x for x in goals['intervals']}
        days = {x['day']:x for x in goals['day_assignments']}
        if set(merged.day) - set(days):
            raise ValueError('An observed June day lacks a frozen goal assignment')
        merged['goal_id'] = merged.day.map(lambda day: days[day]['primary_goal_id'])
        merged['goal'] = merged.goal_id.map(lambda key: intervals[key]['description'])
        merged['category'] = merged.day.map(lambda day: days[day]['primary_category'])
        merged['transition'] = merged.day.map(lambda day: days[day]['transition_day'])
    else:
        merged['goal_id'] = 'later_period'; merged['goal'] = 'August–September reference'
        merged['category'] = 'later_period'; merged['transition'] = False
    return merged


def summarize(daily):
    records = []
    metrics = ['classifier_alert_share', 'co_use_threshold_share', 'request_prevalence',
               'shared_fraction_of_traced', 'shared_fraction_of_working', 'trace_coverage',
               'actor_mean_shared_fraction_of_traced', 'actor_mean_shared_fraction_of_working',
               'steps_per_active_agent', 'active_agents', 'shell_fraction', 'gui_fraction']
    for sensitivity in ('all_days', 'without_transition_days'):
        subset = daily if sensitivity == 'all_days' else daily[~daily.transition]
        for (variant, period, goal_id), frame in subset.groupby(['variant', 'period', 'goal_id']):
            row = dict(variant=variant, period=period, goal_id=goal_id, goal=frame.goal.iloc[0],
                       category=frame.category.iloc[0], sensitivity=sensitivity, observed_days=len(frame),
                       first_observed_day=frame.day.min(), last_observed_day=frame.day.max(),
                       working_steps=int(frame.working_steps.sum()), trace_steps=int(frame.trace_steps.sum()),
                       shared_steps=int(frame.shared_steps.sum()), active_agent_days=int(frame.active_agents.sum()),
                       pair_days=int(frame.pair_days.sum()), request_pair_days=int(frame.request_pair_days.sum()))
            for metric in metrics:
                values = frame[metric].dropna()
                row[metric+'_median'] = float(values.median()) if len(values) else None
                row[metric+'_min'] = float(values.min()) if len(values) else None
                row[metric+'_max'] = float(values.max()) if len(values) else None
            row['pooled_shared_fraction_of_traced'] = row['shared_steps']/row['trace_steps'] if row['trace_steps'] else None
            row['pooled_shared_fraction_of_working'] = row['shared_steps']/row['working_steps'] if row['working_steps'] else None
            records.append(row)
    return records


def figure(daily, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    primary = daily[(daily.period == 'shared_goals') & (daily.variant == 'specific')].sort_values('day')
    sensitivity = daily[(daily.period == 'shared_goals') & (daily.variant == 'specific_command_context')].set_index('day')
    reference = daily[(daily.period == 'temporal') & (daily.variant == 'specific')]
    x = np.arange(len(primary)); dates = pd.to_datetime(primary.day)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.labelsize':9,
                         'axes.titlesize':10, 'legend.fontsize':8, 'svg.fonttype':'none',
                         'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,1,figsize=(7.0,4.2),sharex=True,gridspec_kw={'hspace':.15})
    colors = {'collaborative':'#cce8d5','mixed':'#e5dcf2','individual':'#d5e7f5','competitive':'#f7ddbe'}
    short = {'Follow your leader!':'Follow leader','Organise an event!':'Organise event',
             'Reduce global suffering as much as you can!':'Reduce suffering',
             'Help Gemini 2.5 Pro!':'Help\nGemini','Beat the hardest game you can!':'Game',
             'Compete to be the best AI Assistant!':'Compete'}
    for goal, group in primary.groupby('goal',sort=False):
        ix = np.flatnonzero(primary.goal.to_numpy()==goal)
        left,right=ix[0]-.5,ix[-1]+.5
        for ax in axes:ax.axvspan(left,right,color=colors[group.category.iloc[0]],alpha=.65,lw=0,zorder=0)
        axes[0].text((left+right)/2,1.04,short.get(goal,goal),transform=axes[0].get_xaxis_transform(),
                     ha='center',va='bottom',fontsize=8 if len(ix)>1 else 7)
    axes[0].plot(x,primary.classifier_alert_share,color='#222222',marker='o',markersize=3,lw=1.4,label='Frozen L1 classifier alerts')
    axes[0].axhline(reference.classifier_alert_share.median(),color='#666666',ls=':',lw=1.3,label='Aug–Sep median classifier share')
    axes[0].set_ylabel('Share of active pairs')
    axes[1].plot(x,primary.shared_fraction_of_traced,color='#087f8c',marker='o',markersize=3,lw=1.4,label='Unweighted resource co-occurrence')
    axes[1].plot(x,sensitivity.loc[primary.day].shared_fraction_of_traced,color='#166347',ls='--',lw=1.3,label='Command-context sensitivity')
    axes[1].axhline(reference.shared_fraction_of_traced.median(),color='#666666',ls=':',lw=1.3,label='Aug–Sep median co-occurrence')
    axes[1].set_ylabel('Share of trace-bearing steps')
    for ax in axes:
        ax.set_ylim(0,1);ax.set_xlim(-.5,len(primary)-.5);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y',alpha=.25,lw=.5)
    axes[0].legend(loc='upper right',framealpha=.9)
    axes[1].legend(loc='lower left',framealpha=.9)
    tick = sorted(set([0,len(primary)-1]+[i for i,d in enumerate(dates) if d.weekday()==0]))
    axes[1].set_xticks(tick);axes[1].set_xticklabels([dates.iloc[i].strftime('%b %d') for i in tick])
    axes[1].set_xlabel('Observed Pacific days, June–July 2026 (non-observed dates omitted)')
    fig.subplots_adjust(left=.1,right=.985,bottom=.13,top=.87)
    fig.savefig(Path(output)/'goal-co-use-robustness.pdf',bbox_inches='tight')
    fig.savefig(Path(output)/'goal-co-use-robustness.svg',bbox_inches='tight')
    plt.close(fig)


def run(study, inputs, output):
    study, inputs, output = Path(study), Path(inputs), Path(output)
    if output.exists():raise FileExistsError('Reporting outputs must be new; prior results remain unchanged')
    output.mkdir(parents=True)
    source_paths = [study/'goal-regime-classification.json', study/'historical-freeze.json', study/'village-results-lock.json']
    goals = json.loads(source_paths[0].read_text())
    frozen_hashes = json.loads(source_paths[2].read_text())['files_sha256']
    all_daily, all_agents, read_diagnostics = [], [], {}
    for period in PERIODS:
        paths = {name:study/period/name for name in ['activity.csv.gz','predictions.csv.gz','secondary-daily.csv','feature-report.json']}
        for path in paths.values():
            if sha(path) != frozen_hashes[str(path.relative_to(study))]:
                raise ValueError('Existing frozen result hash mismatch: '+str(path))
        source_paths.extend(paths.values());source_paths.extend(sorted((inputs/period/'mentions').glob('*')))
        activity = pd.read_csv(paths['activity.csv.gz']).fillna(0);validate_days(activity,period)
        predictions = pd.read_csv(paths['predictions.csv.gz']);regime = pd.read_csv(paths['secondary-daily.csv'])
        mentions, diagnostics = load_mentions(inputs/period/'mentions',period)
        read_diagnostics[period] = diagnostics
        read_diagnostics[period]['frozen_rarity_diagnostics'] = json.loads(paths['feature-report.json'].read_text())['historical']
        for variant in VARIANTS:
            daily, agents = co_use_counts(mentions,activity,variant=variant)
            daily = attach_frozen_series(daily,predictions,regime,activity,period,goals)
            daily['variant']=variant;agents['variant']=variant;agents['period']=period
            all_daily.append(daily);all_agents.append(agents)
    daily=pd.concat(all_daily,ignore_index=True);agents=pd.concat(all_agents,ignore_index=True)
    daily.to_csv(output/'daily.csv',index=False);agents.to_csv(output/'agent-days.csv.gz',index=False)
    groups=summarize(daily);pd.DataFrame(groups).to_csv(output/'goal-summary.csv',index=False)
    metadata=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),periods=PERIODS,
        analysis='Exploratory reporting-only check proposed after observing prior goal contrasts; no retuning or confirmatory p-values.',
        scopes=dict(specific=list(SPECIFIC_TYPES),specific_command_context=dict(types=list(SPECIFIC_TYPES),sources=list(CONTEXTS),operations=['read','write','create']),all_legacy=list(LEGACY_TYPES)),
        numerator='Unique agent/day/step containing an eligible exact typed resource ID seen for >=2 active agents that day.',
        denominators=['Trace-bearing working steps under the same variant','All observed working steps'],
        analysis_units='Seven scheduled goals, with six receiving observed primary-assigned days. No independence of goals, days or pairs is asserted; goals are unreplicated and confounded with time and roster.',
        identity='Exact raw extracted type and ID. No July identity map, IDF, within-target rarity fit, or target request label affects co-use.',
        interpretation='Resource co-occurrence is not verified cooperation. Common sources, output text, roster, task type, tools, and coverage can explain it. Unknown GUI activity remains unobserved.',
        transitions='Existing description-only goal categories and scheduled-clock dominant-day assignment retained; sensitivity removes every recorded transition day.',
        read_diagnostics=read_diagnostics,source_sha256={str(p.resolve()):sha(p) for p in source_paths},
        script_sha256=sha(__file__),tests_sha256=sha(ROOT/'tests/test_review_regime_robustness.py'),
        goal_summaries=groups)
    json_write(output/'summary.json',metadata);figure(daily,output)
    for path, digest in metadata['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Input changed during reporting: '+path)
    json_write(output/'manifest.json',dict(inputs_sha256=metadata['source_sha256'],script_sha256=metadata['script_sha256'],
        outputs_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps(dict(output=str(output),daily_rows=len(daily),goal_summaries=len(groups))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'outputs/generalization-20261106')
    parser.add_argument('--inputs',type=Path,default=ROOT/'outputs/claude-review-20261107/regimes/inputs')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/claude-review-20261107/regimes/report')
    args=parser.parse_args();run(args.study,args.inputs,args.output)
