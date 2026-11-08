"""Plot recall against average alerts per observed day in the test period.

Replots saved curves only: no raw logs, labels, fitting or thresholds. The
curves and operating point come from scripts/review_alert_budget.py; their hash
is checked against that run's summary.
Run: python scripts/plot_alert_budget.py [--source DIR] [--output-dir DIR]
Writes alert-budget.{pdf,svg} (paper Figure 3) and alert-budget.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/alert-budget'
DEFAULT_OUTPUT = ROOT / 'outputs/figures'
X_MAX = 200          # alerts per day shown; the saved curves run to every active pair
# Emphasis: the two detector curves take categorical slots 1-2 (validated on white);
# baselines recede to neutral inks. Text always uses ink colors, never series colors.
SERIES = [  # column, legend label, color, line width (pt), draw order
    ('random', 'Random ranking', '#c3c2b7', 0.9, 1),
    ('activity', 'Activity only', '#898781', 1.5, 2),
    ('shared_resources', 'Shared resources alone', '#eb6834', 1.5, 4),
    ('generic_detector', 'Generic detector', '#2a78d6', 1.5, 5),
]
INK, INK_2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(source):
    summary = json.loads((source / 'summary.json').read_text())
    curves_path = source / 'curves.csv'
    if sha(curves_path) != summary['outputs_sha256']['curves.csv']:
        raise ValueError('Saved curves differ from their run summary')
    curves = pd.read_csv(curves_path)
    if curves.alerts_per_day.iloc[0] != 0 or curves.alerts_per_day.max() < X_MAX:
        raise ValueError('Saved curves do not cover the plotted budget range')
    return curves, summary


def plot(curves, summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 9, 'legend.fontsize': 8,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'alert-budget',
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': AXIS,
                         'axes.labelcolor': INK, 'xtick.color': INK_2, 'ytick.color': INK_2,
                         'xtick.major.size': 0, 'ytick.major.size': 0,
                         'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round'})
    shown = curves[curves.alerts_per_day <= X_MAX]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for column, label, color, width, order in SERIES:
        ax.plot(shown.alerts_per_day, shown[column], color=color, lw=width, label=label,
                zorder=order)

    point = summary['operating_point']
    budget = summary['at_operating_budget']
    x0 = point['alerts_per_day']
    ax.axvline(x0, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0.5)
    ax.text(x0 + 2.5, 0.035, f'July threshold: {x0:.0f} alerts/day on average',
            color=INK_2, fontsize=8, ha='left', va='bottom')
    labels = {'generic_detector': (-3.5, 0.0, 'right'),
              'activity': (3.5, -0.035, 'left')}
    colors = {column: color for column, _, color, _, _ in SERIES}
    for column, (dx, dy, align) in labels.items():
        recall = budget[column]['recall']
        ax.plot([x0], [recall], 'o', ms=6, color=colors[column], mec='white', mew=1.5,
                zorder=6)
        ax.text(x0 + dx, recall + dy, f'{100 * recall:.1f}%', color=INK, fontsize=8,
                ha=align, va='center', zorder=7)

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(0, X_MAX + 1, 25))
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(f'Average alerts per observed day '
                  f'(about {summary["pairs_per_day"]:.0f} active pairs per day)')
    ax.set_ylabel('Recall (request\npair-days flagged)')
    handles, names = ax.get_legend_handles_labels()
    order = [names.index(name) for name in ('Generic detector', 'Shared resources alone',
                                            'Activity only', 'Random ranking')]
    # The upper-left corner is empty at every budget: no curve reaches 60% before 45 alerts/day.
    legend = ax.legend([handles[i] for i in order], [names[i] for i in order], loc='upper left',
                       bbox_to_anchor=(0.005, 1.0), frameon=True, framealpha=1,
                       edgecolor='white', borderpad=0.3, labelspacing=0.3, handlelength=1.8,
                       labelcolor=INK)
    # Above the grid and threshold line, below the curves, so its box never cuts a curve.
    legend.set_zorder(2.5)
    fig.subplots_adjust(left=.11, right=.985, bottom=.18, top=.97)
    for suffix in ('pdf', 'svg'):
        metadata = {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {'Date': None}
        fig.savefig(destination / ('alert-budget.' + suffix), bbox_inches='tight',
                    metadata=metadata)
    svg = destination / 'alert-budget.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    curves, summary = read(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(curves, summary, args.output_dir)
    report = {
        'source': str((args.source / 'summary.json').resolve()),
        'source_sha256': sha(args.source / 'summary.json'),
        'script_sha256': sha(__file__),
        'purpose': 'Post-hoc display of saved frozen test-period rankings against average alerts per observed day.',
        'shown_alerts_per_day': [0, X_MAX],
        'shown_series': [column for column, *_ in SERIES],
        'operating_point': summary['operating_point'],
        'at_operating_budget': {
            column: summary['at_operating_budget'][column]
            for column, *_ in SERIES if column in summary['at_operating_budget']},
        'figure_sha256': {suffix: sha(args.output_dir / ('alert-budget.' + suffix))
                          for suffix in ('svg', 'pdf')}}
    output = args.output_dir / 'alert-budget.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(output), 'figure_sha256': report['figure_sha256']}))


if __name__ == '__main__':
    main()
