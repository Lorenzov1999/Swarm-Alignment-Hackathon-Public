"""Draw the single-signal paper figures from saved results.

Run: python scripts/make_shared_resource_figures.py --help
Requires the existing matplotlib environment and rsvg-convert. No fitting or
raw-log reading occurs here. SVG files are the editable figure sources.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/alert-budget'
FIGURES = ROOT / 'outputs/current-paper/figures'
RESULTS = ROOT / 'outputs/current-paper/provenance/shared-resource-results.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pipeline(figures=FIGURES):
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="6.5in" height="1.95in" viewBox="0 0 936 281" role="img" aria-labelledby="title desc">
  <title id="title">From resource identifiers in work logs to ranked agent pairs</title>
  <desc id="desc">Work logs, excluding explicit chat, supply normalized resource identifiers. Daily profiles give more weight to rare shared identifiers and produce a ranking of agent pairs. July work and request records set rarity weights and alert threshold. Recorded help requests from the validation and test periods evaluate the ranking.</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#46505c"/></marker>
    <style>
      text { font-family: "DejaVu Sans", sans-serif; fill: #18232d; text-anchor: middle; }
      .heading { font-size: 18px; font-weight: 600; }
      .detail { font-size: 16px; }
      .flow { fill: none; stroke: #46505c; stroke-width: 1.8; marker-end: url(#arrow); }
      .box { stroke-width: 1.5; rx: 7; }
    </style>
  </defs>
  <rect width="936" height="281" fill="white"/>
  <rect class="box" x="293" y="3" width="430" height="53" fill="#f7f8fa" stroke="#a8b0b8"/>
  <text class="heading" x="508" y="25">July work logs + help requests</text>
  <text class="detail" x="508" y="46">Set rarity weights and alert threshold</text>
  <path class="flow" d="M 585 57 V 81"/>

  <rect class="box" x="2" y="85" width="192" height="105" fill="#f7f8fa" stroke="#707b85"/>
  <text class="heading" x="98" y="113">Work logs</text>
  <text class="detail" x="98" y="145">Actions and outputs</text>
  <text class="detail" x="98" y="169">Explicit chat excluded</text>
  <path class="flow" d="M 196 138 H 220"/>

  <rect class="box" x="226" y="85" width="210" height="105" fill="#edf3fb" stroke="#38659a"/>
  <text class="heading" x="331" y="113">Resource identifiers</text>
  <text class="detail" x="331" y="145">Match references to</text>
  <text class="detail" x="331" y="169">the same object</text>
  <path class="flow" d="M 438 138 H 462"/>

  <rect class="box" x="468" y="85" width="235" height="105" fill="#edf3fb" stroke="#38659a"/>
  <text class="heading" x="585.5" y="113">Compare daily profiles</text>
  <text class="detail" x="585.5" y="145">Rare shared identifiers</text>
  <text class="detail" x="585.5" y="169">count more</text>
  <path class="flow" d="M 705 138 H 729"/>

  <rect class="box" x="735" y="85" width="199" height="105" fill="#edf7f4" stroke="#307568"/>
  <text class="heading" x="834.5" y="113">Rank agent pairs</text>
  <text class="detail" x="834.5" y="145">One score per pair</text>
  <text class="detail" x="834.5" y="169">per day</text>
  <path class="flow" d="M 834.5 192 V 220"/>

  <rect class="box" x="414" y="226" width="283" height="44" fill="#f7f8fa" stroke="#a8b0b8"/>
  <text class="detail" x="555.5" y="243">Recorded help requests</text>
  <text class="detail" x="555.5" y="262">Validation and test periods</text>
  <path class="flow" d="M 699 248 H 729"/>
  <rect class="box" x="735" y="226" width="199" height="44" fill="#f7f8fa" stroke="#a8b0b8"/>
  <text class="heading" x="834.5" y="253">Evaluate ranking</text>
</svg>
'''
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / 'shared-resource-pipeline.svg'
    path.write_text(svg)
    subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(path.with_suffix('.pdf')), str(path)], check=True)


def workload(source=SOURCE, results_path=RESULTS, figures=FIGURES):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figures.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / 'summary.json').read_text())
    if sha(source / 'curves.csv') != summary['outputs_sha256']['curves.csv']:
        raise ValueError('Saved workload curves differ from their source summary')
    results = json.loads(results_path.read_text())
    with (source / 'curves.csv').open() as handle:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    if rows[0]['alerts_per_day'] != 0 or rows[-1]['alerts_per_day'] < 200:
        raise ValueError('Saved curves do not span the plotted range')
    rows = [row for row in rows if row['alerts_per_day'] <= 200]
    point = results['operating_point']
    budget = point['alerts_per_day']
    recalls = {'shared_resources': point['recall'],
               'activity': results['at_operating_budget']['activity']['recall']}
    if abs(point['alerts'] / results['observed_days'] - budget) > 1e-10:
        raise ValueError('Operating-point alert count and daily budget disagree')
    if not (0 < budget < 200 and all(0 <= v <= 1 for v in recalls.values())):
        raise ValueError('Invalid operating point')
    # The saved display grid advances by 0.5 alerts/day. Insert the exact
    # operating budget so line interpolation cannot suggest a different recall
    # at the marker (a few request pairs can fall between grid points).
    lower = max((r for r in rows if r['alerts_per_day'] <= budget),
                key=lambda r: r['alerts_per_day'])
    upper = min((r for r in rows if r['alerts_per_day'] >= budget),
                key=lambda r: r['alerts_per_day'])
    for key, value in recalls.items():
        if not lower[key] - 1e-6 <= value <= upper[key] + 1e-6:
            raise ValueError('Operating-point recall conflicts with the saved curve')
    rows = [r for r in rows if abs(r['alerts_per_day'] - budget) > 1e-9]
    rows.append(dict(alerts_per_day=budget, random=budget / summary['pairs_per_day'],
                     **recalls))
    rows.sort(key=lambda r: r['alerts_per_day'])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 9, 'legend.fontsize': 8,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'shared-resource-workload',
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#c3c2b7',
                         'axes.labelcolor': '#18232d', 'xtick.color': '#52514e',
                         'ytick.color': '#52514e', 'xtick.major.size': 0,
                         'ytick.major.size': 0, 'lines.solid_capstyle': 'round',
                         'lines.solid_joinstyle': 'round'})
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    series = [('shared_resources', 'Shared resources', '#2a78d6', 1.8, 4),
              ('activity', 'Activity only', '#898781', 1.5, 3),
              ('random', 'Random ranking', '#c3c2b7', 1.0, 2)]
    for key, label, color, width, order in series:
        ax.plot([row['alerts_per_day'] for row in rows], [row[key] for row in rows],
                color=color, lw=width, label=label, zorder=order)
    ax.axvline(budget, color='#898781', lw=.8, ls=(0, (3, 3)), zorder=1)
    ax.text(budget + 3, .038, f'July threshold: {budget:.0f} alerts/day on average',
            color='#52514e', fontsize=8, ha='left', va='bottom')
    for key, color, dx, dy, align in [('shared_resources', '#2a78d6', -4, .018, 'right'),
                                       ('activity', '#898781', 4, -.015, 'left')]:
        recall = recalls[key]
        ax.plot(budget, recall, 'o', ms=6, color=color, mec='white', mew=1.4, zorder=6)
        ax.text(budget + dx, recall + dy, f'{100 * recall:.1f}%',
                color='#18232d', fontsize=8, ha=align, va='center', zorder=7)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 201, 25))
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis='y', color='#e1e0d9', lw=.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Average alerts per observed day')
    ax.set_ylabel('Request pair-days flagged')
    ax.legend(loc='upper left', bbox_to_anchor=(.005, 1), frameon=False,
              borderpad=.3, labelspacing=.35, handlelength=1.8)
    fig.subplots_adjust(left=.105, right=.985, bottom=.18, top=.97)
    for suffix in ('svg', 'pdf'):
        metadata = {'Date': None} if suffix == 'svg' else {'CreationDate': None, 'ModDate': None}
        fig.savefig(figures / f'shared-resource-workload.{suffix}',
                    bbox_inches='tight', metadata=metadata)
    path = figures / 'shared-resource-workload.svg'
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    return dict(alerts=point['alerts'], alerts_per_day=budget, recalls=recalls)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE, help='Saved alert-budget directory')
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--output-dir', type=Path, default=FIGURES)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    pipeline(args.output_dir)
    marker = workload(args.source, args.results, args.output_dir)
    report = {'purpose': 'Single shared-resource detector only; diagrams and saved workload curves.',
              'script_sha256': sha(__file__),
              'input_sha256': {name: sha(p) for name, p in
                              [('workload_summary', args.source / 'summary.json'),
                               ('workload_curves', args.source / 'curves.csv'), ('results', args.results)]},
              'shown_series': ['shared_resources', 'activity', 'random'],
              'curve_sampling': 'Saved 0.5-alert/day display grid, augmented with the exact July-threshold budget and corresponding shared-resource/activity recalls.',
              'shown_alerts_per_day': [0, 200],
              'marker': marker,
              'output_sha256': {p.name: sha(p) for p in args.output_dir.glob('shared-resource-*.*')}}
    report_path = args.report or args.output_dir / 'shared-resource-figures.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(marker))


if __name__ == '__main__':
    main()
