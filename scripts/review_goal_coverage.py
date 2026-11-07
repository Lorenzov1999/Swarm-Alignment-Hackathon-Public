"""Replot saved Village diagnostics: alert workload beside observable coverage.

No raw logs, labels, model fitting, thresholds, or existing results are changed.
Uses the saved reporting table, checked against its existing output manifest.
Run: python scripts/review_goal_coverage.py [--daily DAILY.csv] [--output-dir DIR]
Writes goal-co-use-robustness.{pdf,svg} (paper Figure 4) and goal-coverage-review.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/claude-review-20261107/regimes/report/daily.csv'
DEFAULT_OUTPUT = ROOT / 'outputs/figures'
PERIODS = {'shared_goals': ('2026-06-01', '2026-07-05'),
           'temporal': ('2026-08-17', '2026-09-15')}
METRICS = ['flagged_partners_per_active_agent', 'classifier_alert_share',
           'trace_coverage', 'gui_fraction', 'active_agents']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_daily(path):
    manifest = path.with_name('manifest.json')
    expected = json.loads(manifest.read_text())['outputs_sha256']['daily.csv']
    if sha(path) != expected:
        raise ValueError('Saved daily table differs from its prior manifest')
    full = pd.read_csv(path)
    daily = full.loc[full.variant.eq('specific')].copy()
    if set(daily.period) != set(PERIODS) or daily[['period', 'day']].duplicated().any():
        raise ValueError('Unexpected periods or duplicate primary daily rows')
    for period, (first, last) in PERIODS.items():
        if not daily.loc[daily.period.eq(period), 'day'].between(first, last).all():
            raise ValueError('Date outside the previously authorized windows')
    if not (daily.pair_days == daily.active_agents * (daily.active_agents - 1) / 2).all():
        raise ValueError('Pair universe does not contain every active-agent pair')
    if not daily.l1_alerts.between(0, daily.pair_days).all():
        raise ValueError('Invalid number of flagged pairs')
    if not np.allclose(daily.classifier_alert_share, daily.l1_alerts / daily.pair_days):
        raise ValueError('Saved alert share disagrees with counts')
    if not np.allclose(daily.trace_coverage, daily.trace_steps / daily.working_steps):
        raise ValueError('Saved trace coverage disagrees with counts')
    daily['flagged_partners_per_active_agent'] = 2 * daily.l1_alerts / daily.active_agents
    # Mean degree also equals pair density times the number of possible partners.
    if not np.allclose(daily.flagged_partners_per_active_agent,
                       daily.classifier_alert_share * (daily.active_agents - 1)):
        raise ValueError('Per-agent alert calculation failed its identity check')
    return daily.sort_values(['period', 'day'])


def metrics(frame):
    return {key: {'median': float(frame[key].median()),
                  'min': float(frame[key].min()), 'max': float(frame[key].max())}
            for key in METRICS}


def summarize(daily, source):
    early = daily[daily.period.eq('shared_goals')].sort_values('day')
    june = early[early.day.lt('2026-07-01')]
    groups = []
    for (period, goal), group in daily.groupby(['period', 'goal'], sort=False):
        groups.append({'period': period, 'goal': goal, 'days': len(group),
                       'first_day': group.day.min(), 'last_day': group.day.max(),
                       **metrics(group)})
    leader = early[early.goal.eq('Follow your leader!')]
    return {
        'source': str(source.resolve()), 'source_sha256': sha(source),
        'script_sha256': sha(__file__),
        'purpose': 'Post-hoc descriptive display of existing fixed detector output and coverage; no retuning.',
        'definitions': {
            'flagged_partners_per_active_agent': '2 * flagged unordered pairs / active agents; mean degree of the daily alert graph.',
            'goal_summary': 'Median of daily quantities; days have equal weight. Transition days retained.',
            'trace_coverage': 'Unique working steps bearing a repository, project-page, page, or Google-document/artifact identifier / all working steps, under the existing parser.',
            'gui_fraction': 'Saved GUI working-step count / all working steps.',
            'spearman': 'Pearson correlation of average ranks; descriptive, no independence assumption or significance test.'},
        'interpretation_limits': [
            '26 early-window observed days span June 1–July 3; only 23 fall within calendar June.',
            'Per-agent workload is a different unit from pair density, not a complete adjustment for roster or opportunities.',
            'Coverage association is not a causal decomposition or proof that GUI activity caused missed alerts.',
            'Within-goal changes at similar aggregate coverage do not establish verified cooperation or a causal goal effect.',
            'Post-hoc comparisons of serially dependent days receive no confirmatory significance claims.',
            'Coverage here uses specific resource types; the pair-visibility diagnostic uses a broader resource parser. GUI steps containing recognizable identifiers can contribute to coverage.'],
        'correlation': {
            'early_window_days': len(early),
            'early_window_rho': float(early.classifier_alert_share.rank().corr(early.trace_coverage.rank())),
            'calendar_june_days': len(june),
            'calendar_june_rho': float(june.classifier_alert_share.rank().corr(june.trace_coverage.rank()))},
        'leader_endpoints': leader.iloc[[0, -1]][['day'] + METRICS].to_dict('records'),
        'goals': groups,
        'daily': daily[['period', 'day', 'goal', 'category', 'transition', 'active_agents',
                        'pair_days', 'l1_alerts', 'working_steps', 'trace_steps'] +
                       [key for key in METRICS if key != 'active_agents']].to_dict('records')}


def plot(daily, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    primary = daily[daily.period.eq('shared_goals')].sort_values('day')
    reference = daily[daily.period.eq('temporal')]
    x = np.arange(len(primary))
    dates = pd.to_datetime(primary.day)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 9, 'legend.fontsize': 8,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'goal-coverage-review',
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.1), sharex=True,
                             gridspec_kw={'hspace': .17})
    colors = {'collaborative': '#cce8d5', 'mixed': '#e5dcf2',
              'individual': '#d5e7f5', 'competitive': '#f7ddbe'}
    names = {'Follow your leader!': 'Follow leader', 'Organise an event!': 'Organise event',
             'Reduce global suffering as much as you can!': 'Reduce\nsuffering',
             'Help Gemini 2.5 Pro!': 'Help\nGemini',
             'Beat the hardest game you can!': 'Game',
             'Compete to be the best AI Assistant!': 'Compete'}
    for goal, group in primary.groupby('goal', sort=False):
        ix = np.flatnonzero(primary.goal.to_numpy() == goal)
        left, right = ix[0] - .5, ix[-1] + .5
        for ax in axes:
            ax.axvspan(left, right, color=colors[group.category.iloc[0]], alpha=.65,
                       lw=0, zorder=0)
        axes[0].text((left + right) / 2, 1.04, names[goal],
                     transform=axes[0].get_xaxis_transform(), ha='center', va='bottom',
                     fontsize=8 if len(ix) > 1 else 7)
    axes[0].plot(x, primary.flagged_partners_per_active_agent, color='#222222',
                 marker='o', markersize=3, lw=1.4, label='Fixed L1 alerts')
    axes[0].axhline(reference.flagged_partners_per_active_agent.median(),
                    color='#666666', ls=':', lw=1.3, label='Aug–Sep daily median')
    axes[0].set_ylabel('Flagged partners\nper active agent')
    axes[0].set_ylim(0, 12)
    axes[0].set_yticks([0, 3, 6, 9, 12])
    axes[1].plot(x, primary.trace_coverage, color='#087f8c', marker='o',
                 markersize=3, lw=1.4, label='Steps with resource identifiers')
    axes[1].axhline(reference.trace_coverage.median(), color='#666666', ls=':',
                    lw=1.3, label='Aug–Sep daily median')
    axes[1].set_ylabel('Trace coverage')
    axes[1].set_ylim(0, .70)
    axes[1].set_yticks([0, .2, .4, .6])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlim(-.5, len(primary) - .5)
        ax.grid(axis='y', alpha=.25, lw=.5)
        ax.legend(loc='upper right', framealpha=.95)
    tick = sorted(set([0, len(primary)-1] +
                      [i for i, day in enumerate(dates) if day.weekday() == 0]))
    axes[1].set_xticks(tick)
    axes[1].set_xticklabels([dates.iloc[i].strftime('%b %d') for i in tick])
    axes[1].set_xlabel('Observed Pacific days, June–July 2026 (non-observed dates omitted)')
    fig.subplots_adjust(left=.14, right=.985, bottom=.17, top=.84)
    for suffix in ('pdf', 'svg'):
        metadata = {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {'Date': None}
        fig.savefig(destination / ('goal-co-use-robustness.' + suffix),
                    bbox_inches='tight', metadata=metadata)
    svg = destination / 'goal-co-use-robustness.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daily', type=Path, default=DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    daily = read_daily(args.daily)
    report = summarize(daily, args.daily)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(daily, args.output_dir)
    report['figure_sha256'] = {suffix: sha(args.output_dir / ('goal-co-use-robustness.' + suffix))
                              for suffix in ('svg', 'pdf')}
    output = args.output_dir / 'goal-coverage-review.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(output), 'correlation': report['correlation']}))


if __name__ == '__main__':
    main()
