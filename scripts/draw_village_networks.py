"""Render illustrative Village networks from saved, reviewed provenance.

No raw-log access, fitting, threshold changes, or inferred narrative occurs here.
Run from any directory with the existing matplotlib/numpy environment and neato:
    python scripts/draw_village_networks.py
Use --input and --output-dir for a preview without changing paper artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import textwrap
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'swarmcollab-mpl-networks'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK = '#263746'
MUTED = '#63717b'
EDGE = '#899ba7'
NODE = '#536f82'
ISOLATE = '#a2a8ad'
FOCUS = '#b45320'
BACKGROUND = '#ffffff'
SVG = 'http://www.w3.org/2000/svg'


def short_name(name):
    """Keep the distinguishing model family/version readable at paper scale."""
    name = name.removeprefix('Claude ')
    name = name.replace('Gemini ', 'Gem ').replace('DeepSeek-', 'DS ')
    name = name.replace('-Pro', '').replace(' Pro', '').replace(' Flash', 'F')
    name = name.replace(' Luna', 'L').replace(' Terra', 'T').replace(' Sol', 'S')
    return name


def full_name(name):
    if name.startswith('Claude '):
        return name.replace('Claude ', 'Claude\n', 1)
    if name.startswith('Gemini '):
        return name.replace('Gemini ', 'Gemini\n', 1)
    if name.startswith('DeepSeek-'):
        return name.replace('DeepSeek-', 'DeepSeek\n', 1)
    return name


def pair(a, b):
    return tuple(sorted((a, b)))


def linewidth(score):
    """One physical line-width scale for every date, bounded for roundoff."""
    return .35 + 2.0 * min(1., max(0., float(score)))


def validate(data):
    snapshots = data['snapshots']
    if len(snapshots) != 4:
        raise ValueError('The composite requires exactly four reviewed snapshots')
    if len({s['id'] for s in snapshots}) != 4:
        raise ValueError('Snapshot ids must be unique')
    for s in snapshots:
        if not '2026-06-01' <= s['day'] <= '2026-09-15':
            raise ValueError('Snapshot outside the authorized Village interval')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', s['id']):
            raise ValueError('Snapshot id must be a safe filename stem')
        nodes = set(s['nodes'])
        if len(nodes) != len(s['nodes']):
            raise ValueError('Duplicate node')
        seen = set()
        degree = {a: 0 for a in nodes}
        for e in s['edges']:
            key = pair(e['a'], e['b'])
            if e['a'] not in nodes or e['b'] not in nodes or e['a'] == e['b']:
                raise ValueError('Edge endpoints must be distinct displayed nodes')
            if key in seen or not math.isfinite(e['score']) or not 0 <= e['score'] <= 1.0000000001:
                raise ValueError('Duplicate edge or invalid cosine')
            seen.add(key)
            degree[e['a']] += 1
            degree[e['b']] += 1
        if not set(s.get('focus_agents', [])).issubset(nodes):
            raise ValueError('A focus agent is missing from the displayed nodes')
        if any(pair(*e) not in seen for e in s.get('focus_edges', [])):
            raise ValueError('Focus edges must be present in the saved graph')
        expected = dict(nodes=len(nodes), edges=len(seen), isolates=sum(v == 0 for v in degree.values()))
        if any(s['metrics'][key] != value for key, value in expected.items()):
            raise ValueError('Saved graph counts disagree with the displayed graph')
    return snapshots


def neato_positions(nodes, edges, labels, font_size, seed):
    """Use label-sized invisible boxes to reserve space; render circles ourselves."""
    ids = {a: 'n' + str(i) for i, a in enumerate(sorted(nodes))}
    lines = ['graph G {',
             f'graph [layout=neato, start={seed}, overlap=prism, sep="+10", splines=false, pack=true, packmode="graph", maxiter=2000];',
             'node [shape=box, fixedsize=true, label="", margin=0];']
    for a in sorted(nodes):
        label = labels[a]
        width = max(.36, max(map(len, label.split('\n'))) * font_size * .55 / 72 + .13)
        height = font_size * 1.18 * len(label.split('\n')) / 72 + .14
        lines.append(f'{ids[a]} [width={width:.6f}, height={height:.6f}];')
    for e in sorted(edges, key=lambda e: pair(e['a'], e['b'])):
        lines.append(f'{ids[e["a"]]} -- {ids[e["b"]]} [len=1.1];')
    lines.append('}')
    result = subprocess.run(['neato', '-Tplain'], input='\n'.join(lines), text=True,
                            capture_output=True, check=True)
    inverse = {v: k for k, v in ids.items()}
    positions = {}
    for line in result.stdout.splitlines():
        fields = shlex.split(line)
        if fields and fields[0] == 'node':
            positions[inverse[fields[1]]] = np.array([float(fields[2]), float(fields[3])])
    if set(positions) != set(nodes):
        raise ValueError('Neato did not place every connected node')
    return positions


def transform_positions(raw, rectangle, labels, font_size, rotation=0):
    names = sorted(raw)
    values = np.array([raw[a] for a in names])
    angle = np.deg2rad(rotation)
    values = values @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    x0, y0, width, height = rectangle
    # Reserve room for the widest label and for its vertical offset from the dot.
    margin_x = max(max(map(len, labels[a].split('\n'))) for a in names) * font_size * .55 / 144 + .035
    margin_y = font_size * (1.18 * max(len(labels[a].split('\n')) for a in names) + .6) / 72
    extent = np.ptp(values, axis=0)
    scale = np.array([(width - 2 * margin_x) / max(extent[0], .01),
                      (height - margin_y - .10) / max(extent[1], .01)])
    values = (values - (values.max(axis=0) + values.min(axis=0)) / 2) * scale
    values += np.array([x0 + width / 2, y0 + (height - margin_y) / 2 + .04])
    return dict(zip(names, values))


def estimated_overlap(positions, labels, font_size):
    """Select geometry only; graph topology and scores never enter this choice."""
    boxes = []
    for a, (x, y) in positions.items():
        width = max(map(len, labels[a].split('\n'))) * font_size * .56 / 72
        height = len(labels[a].split('\n')) * font_size * 1.18 / 72
        boxes.append((x - width / 2, y + .045, x + width / 2, y + .045 + height))
    count = 0
    area = 0.
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
            h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
            if w and h:
                count += 1
                area += w * h
    return count, area


def layout(snapshot, rectangle, labels, font_size):
    connected = sorted({e[k] for e in snapshot['edges'] for k in ('a', 'b')})
    if not connected:
        return {}, dict(seed=None, rotation=0, estimated_label_overlaps=0)
    candidates = []
    for seed in (17, 29, 43, 71):
        raw = neato_positions(connected, snapshot['edges'], labels, font_size, seed)
        for rotation in (0, 30, 60, 90, 120, 150):
            positions = transform_positions(raw, rectangle, labels, font_size, rotation)
            score = estimated_overlap(positions, labels, font_size)
            candidates.append((score, seed, rotation, positions))
    score, seed, rotation, positions = min(candidates, key=lambda row: row[:3])
    return positions, dict(seed=seed, rotation=rotation, estimated_label_overlaps=score[0])


def draw_snapshot(ax, snapshot, letter=None, individual=False):
    width, height = (8.0, 6.2) if individual else (3.14, 3.0)
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.set_axis_off()
    font_size = 10 if individual else 7.0
    labels = {a: full_name(a) if individual else short_name(a) for a in snapshot['nodes']}
    connected = {e[k] for e in snapshot['edges'] for k in ('a', 'b')}
    isolates = sorted(set(snapshot['nodes']) - connected)
    focus_agents = set(snapshot.get('focus_agents', []))
    focus_edges = {pair(*e) for e in snapshot.get('focus_edges', [])}
    title = snapshot['title']
    date = snapshot['day']
    heading = f'{letter}  {date}' if letter else date
    ax.text(.02, height - .06, heading, ha='left', va='top', fontsize=10 if individual else 8.3,
            color=INK, weight='bold')
    title_lines = textwrap.wrap(title, width=72 if individual else 42)
    ax.text(.02, height - (.29 if individual else .25), '\n'.join(title_lines), ha='left', va='top',
            fontsize=12 if individual else 8.3, color=INK, weight='bold', linespacing=1.1)
    title_extra = .15 * (len(title_lines) - 1)
    m = snapshot['metrics']
    metrics = f'{m["nodes"]} agents · {m["edges"]} edges · {m["isolates"]} isolates'
    ax.text(.02, height - (.62 if individual else .46) - title_extra, metrics, ha='left', va='top',
            fontsize=9 if individual else 7.0, color=MUTED)

    columns = 4 if individual else 3
    rows = math.ceil(len(isolates) / columns)
    row_height = .43 if individual else .205
    isolate_height = (rows * row_height + (.27 if individual else .19)) if isolates else 0
    graph_top = height - (.95 if individual else .77) - title_extra
    graph_bottom = isolate_height + (.16 if individual else .11)
    positions, layout_report = layout(snapshot, (.02, graph_bottom, width - .04, graph_top - graph_bottom), labels, font_size)
    for e in sorted(snapshot['edges'], key=lambda e: (pair(e['a'], e['b']) in focus_edges, e['score'])):
        a, b = positions[e['a']], positions[e['b']]
        focused = pair(e['a'], e['b']) in focus_edges
        ax.plot([a[0], b[0]], [a[1], b[1]], color=FOCUS if focused else EDGE,
                lw=linewidth(e['score']), alpha=.88 if focused else .36,
                zorder=2 if focused else 1, solid_capstyle='round')
    plotted_labels = []
    tooltips = {}
    for name, (x, y) in positions.items():
        color = FOCUS if name in focus_agents else NODE
        token = f'{snapshot["id"]}-{"full" if individual else "short"}-{snapshot["nodes"].index(name)}'
        dot = ax.plot(x, y, marker='o', markersize=4.4, markeredgewidth=.55,
                      markeredgecolor=BACKGROUND, color=color, zorder=4)[0]
        dot.set_gid('node-' + token)
        label = ax.annotate(labels[name], (x, y), xytext=(0, 4.3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=font_size, color=color,
                            zorder=5, linespacing=1.05,
                            bbox=dict(facecolor=BACKGROUND, edgecolor='none', alpha=.91, pad=.10))
        label.set_gid('label-' + token)
        plotted_labels.append((name, label))
        tooltips['node-' + token] = tooltips['label-' + token] = name
    if isolates:
        ax.axhline(isolate_height, xmin=.008, xmax=.992, lw=.5, color='#e0e4e7')
        ax.text(.02, isolate_height - .035, 'Agents with no detected links', fontsize=8.5 if individual else 7,
                color=MUTED, ha='left', va='top')
        for i, name in enumerate(isolates):
            row, col = divmod(i, columns)
            x = .07 + col * width / columns
            y = isolate_height - (.30 if individual else .23) - row * row_height
            token = f'{snapshot["id"]}-{"full" if individual else "short"}-isolate-{i}'
            dot = ax.plot(x, y, 'o', markersize=4.4, color=ISOLATE,
                          markeredgecolor=BACKGROUND, markeredgewidth=.55)[0]
            dot.set_gid('node-' + token)
            label = ax.text(x + (.13 if individual else .09), y, labels[name], va='center', ha='left',
                            fontsize=font_size, color=MUTED, linespacing=1.05)
            label.set_gid('label-' + token)
            plotted_labels.append((name, label))
            tooltips['node-' + token] = tooltips['label-' + token] = name + ' — no above-threshold edges'
    return dict(snapshot=snapshot['id'], day=date, nodes=m['nodes'], edges=m['edges'],
                isolates=m['isolates'], font_size_pt=font_size, layout=layout_report), plotted_labels, tooltips


def add_tooltips(path, titles, description):
    ET.register_namespace('', SVG)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    root = ET.parse(path).getroot()
    root.set('role', 'img')
    title = ET.Element(f'{{{SVG}}}title')
    title.text = 'Illustrative Village shared-resource networks'
    root.insert(0, title)
    desc = ET.Element(f'{{{SVG}}}desc')
    desc.text = description
    root.insert(1, desc)
    for element in root.iter():
        if element.get('id') in titles:
            tooltip = ET.Element(f'{{{SVG}}}title')
            tooltip.text = titles[element.get('id')]
            element.insert(0, tooltip)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def overlap_report(fig, labels):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlaps = []
    for group in labels:
        boxes = [(a, text.get_window_extent(renderer)) for a, text in group]
        for i, (a, box) in enumerate(boxes):
            for b, other in boxes[i + 1:]:
                intersection = Bbox.intersection(box, other)
                if intersection is not None and intersection.width > .5 and intersection.height > .5:
                    overlaps.append([a, b])
    return overlaps


def save(fig, path, tooltips, description, preview):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('svg', 'pdf'):
        metadata = {'Date': None} if suffix == 'svg' else {'CreationDate': None, 'ModDate': None}
        fig.savefig(path.with_suffix('.' + suffix), metadata=metadata, facecolor=BACKGROUND)
    add_tooltips(path.with_suffix('.svg'), tooltips, description)
    if preview:
        fig.savefig(path.with_suffix('.png'), dpi=220, facecolor=BACKGROUND)


def legend(fig, y):
    fig.text(.027, y, 'Edge width: shared-resource cosine', fontsize=7.0, color=MUTED, va='center')
    for i, value in enumerate((.1, .5, 1.0)):
        x = .40 + i * .105
        fig.add_artist(Line2D([x, x + .034], [y, y], transform=fig.transFigure,
                             color=EDGE, linewidth=linewidth(value)))
        fig.text(x + .042, y, f'{value:.1f}', fontsize=7.0, color=MUTED, va='center')
    fig.add_artist(Line2D([.748], [y], transform=fig.transFigure, marker='o', markersize=4.4,
                         color=FOCUS, linestyle='none'))
    fig.text(.763, y, 'Discussed case', fontsize=7.0, color=MUTED, va='center')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'outputs/current-paper/provenance/village-network-snapshots.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/current-paper/figures')
    parser.add_argument('--report', type=Path, default=None)
    parser.add_argument('--preview', action='store_true', help='Also save PNG review copies')
    args = parser.parse_args(argv)
    data = json.loads(args.input.read_text())
    snapshots = validate(data)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'village-network-snapshots',
                         'pdf.fonttype': 42, 'lines.solid_capstyle': 'round'})
    fig = plt.figure(figsize=(6.5, 6.55))
    reports, labels, titles = [], [], {}
    for i, snapshot in enumerate(snapshots):
        col, row = i % 2, i // 2
        ax = fig.add_axes([(.08 + col * 3.25) / 6.5, (.47 + (1 - row) * 3.07) / 6.55, 3.14 / 6.5, 3.0 / 6.55])
        report, text_items, tooltips = draw_snapshot(ax, snapshot, chr(65 + i))
        reports.append(report); labels.append(text_items); titles.update(tooltips)
    legend(fig, .037)
    fig.text(.027, .011, 'Pacific dates · separate layouts · node size is constant · isolates remain visible',
             fontsize=7, color=MUTED, va='bottom')
    collisions = overlap_report(fig, labels)
    save(fig, args.output_dir / 'village-network-snapshots', titles,
         'Four deliberately selected daily networks. Edge widths show cosine scores; orange identifies the discussed case. '
         'Gray isolated agents have no above-threshold edges. Layout is independently computed for display and has no inferential meaning.', args.preview)
    plt.close(fig)
    individual_reports = []
    for snapshot in snapshots:
        fig = plt.figure(figsize=(8.5, 8.2))
        ax = fig.add_axes([.25 / 8.5, 1.6 / 8.2, 8.0 / 8.5, 6.2 / 8.2])
        report, text_items, tooltips = draw_snapshot(ax, snapshot, individual=True)
        report['label_overlaps'] = overlap_report(fig, [text_items])
        context = snapshot.get('context', '')
        if isinstance(context, list):
            context = ' '.join(context)
        fig.text(.034, .169, textwrap.fill(context, width=115), ha='left', va='top',
                 fontsize=9, color=INK, linespacing=1.4)
        legend(fig, .068)
        fig.text(.034, .026, 'Pacific date. Display layout is independent of other dates. Edges measure shared resources;\n'
                 'they do not establish the direction, amount, or success of help.', fontsize=8, color=MUTED, linespacing=1.4)
        save(fig, args.output_dir / 'network-snapshots' / snapshot['id'], tooltips, context, args.preview)
        plt.close(fig)
        individual_reports.append(report)
    output_files = [args.output_dir / f'village-network-snapshots.{suffix}' for suffix in ('svg', 'pdf')]
    output_files += [args.output_dir / 'network-snapshots' / f'{s["id"]}.{suffix}' for s in snapshots for suffix in ('svg', 'pdf')]
    report = dict(input=args.input.name, input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  reproduction_mode=data.get('reproduction_mode', 'unspecified saved snapshots'),
                  score_version=data.get('score_version', data.get('score')),
                  timezone=data.get('timezone', 'America/Los_Angeles'),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  composite_size_inches=[6.5, 6.55], edge_width_points='0.35 + 2.0 * clip(score, 0, 1)',
                  node_diameter_points=4.4, composite=reports, composite_label_overlaps=collisions,
                  individuals=individual_reports, layout='Deterministic neato; separate geometry per date; isolates in labeled grids.',
                  svg_text_editable=True, svg_full_name_tooltips=True,
                  output_sha256={str(p.relative_to(args.output_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output_files})
    default_report = args.output_dir / 'village-network-render-report.json'
    report_path = args.report or default_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report=str(report_path), composite_label_overlaps=collisions,
                          individual_label_overlaps={r['snapshot']: r['label_overlaps'] for r in individual_reports}), indent=2))


if __name__ == '__main__':
    main()
