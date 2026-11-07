"""Validation labels: the village's request records.

Source: the props of the request chart on the public agent page
https://theaidigest.org/village/agent/gemini-2-5-pro, saved as
data/labels/ai-village-chart-props.json (see scripts/fetch_labels.py).

Each request has a sender, a Pacific-time timestamp, a quote, and the site's
`accepted` and `followedThrough` judgements. Recipients are not stored on the
request; they come from the chart's edge list, where each edge names the
requests (`instanceIds`) it aggregates. Edges are undirected: an edge holds the
requests in both directions, so the recipient is the endpoint that is not the
sender. Days are Pacific dates, the village's workdays.

These labels are held out: detectors never read them. Only the scoring code does.

Run: python -I -m swarmcollab.labels   (prints a summary of the first slice)
"""
import collections
import datetime as dt
import json
from zoneinfo import ZoneInfo

from . import config

PACIFIC = ZoneInfo('America/Los_Angeles')
UTC = dt.timezone.utc


def _to_utc(stamp):
    """'2026-07-06 09:06:50 PT' -> aware UTC datetime."""
    local = dt.datetime.strptime(stamp.replace(' PT', ''), '%Y-%m-%d %H:%M:%S').replace(tzinfo=PACIFIC)
    return local.astimezone(UTC)


def load_requests(path=config.LABELS, start=None, end=None):
    """One row per (request, recipient), optionally bounded by Pacific date.

    Bounds are inclusive. With a bound, inspect each request's timestamp before
    its other fields, and ignore recipient edges for excluded requests. Returned
    request/orphan counts then describe the selected interval. Without bounds,
    preserve the original whole-file behavior. JSON parsing still reads the
    container; excluded request payloads are not interpreted.
    """
    start = config.as_date(start) if start is not None else None
    end = config.as_date(end) if end is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError('Request start date must not follow end date')
    with open(path, encoding='utf-8') as f:
        props = json.load(f)
    panel = next(p for p in props if 'instances' in p)
    instances = panel['instances']
    selected = None
    if start is not None or end is not None:
        selected = {}
        for i, request in enumerate(instances):
            when = _to_utc(request['timestamp'])
            day = when.astimezone(PACIFIC).date()
            if (start is None or day >= start) and (end is None or day <= end):
                selected[i] = when
    everything = next(o for o in panel['options'] if o['label'].startswith('All'))
    recipients = collections.defaultdict(list)
    for e in everything['edges']:
        for i in e.get('instanceIds', []):
            if selected is not None and i not in selected:
                continue
            sender = instances[i]['from']
            if sender == e['from']:
                recipients[i].append(e['to'])
            elif sender == e['to']:
                recipients[i].append(e['from'])
    rows = []
    for i, r in enumerate(instances):
        if selected is not None and i not in selected:
            continue
        for to in recipients.get(i, []):
            when = selected[i] if selected is not None else _to_utc(r['timestamp'])
            rows.append({
                'request': i,
                'from': r['from'],
                'to': to,
                'utc': when,
                'day': when.astimezone(PACIFIC).date(),
                'accepted': r['accepted'],
                'followed_through': r['followedThrough'],
                'weight': r['weight'],
                'n_recipients': len(recipients[i]),
            })
    chosen = range(len(instances)) if selected is None else selected
    return rows, len(chosen), sum(1 for i in chosen if i not in recipients)


def pair(a, b):
    return (a, b) if a < b else (b, a)


def pair_days(rows, followed_only=False):
    """Undirected pair -> set of UTC days with at least one request between them."""
    out = collections.defaultdict(set)
    for r in rows:
        if r['from'] == r['to']:
            continue
        if followed_only and r['followed_through'] != 'yes':
            continue
        out[pair(r['from'], r['to'])].add(r['day'])
    return out


def directed_days(rows):
    """(sender, recipient) -> set of UTC days."""
    out = collections.defaultdict(set)
    for r in rows:
        if r['from'] != r['to']:
            out[(r['from'], r['to'])].add(r['day'])
    return out


def main():
    rows, n_req, n_orphan = load_requests()
    print(f'{n_req} requests, {n_orphan} with no recipient on the chart edges, {len(rows)} request-recipient rows')
    weights = collections.Counter()
    for r in rows:
        weights[(round(r['weight'], 3), r['n_recipients'])] += 1
    print('weight vs recipients found:', dict(weights.most_common(8)))
    dev = [r for r in rows if config.DEV_START <= r['day'] <= config.DEV_END]
    print(f'first slice ({config.DEV_START} .. {config.DEV_END}): {len(dev)} rows, '
          f'{len({r["request"] for r in dev})} requests')
    hours = collections.Counter(r['utc'].hour for r in dev)
    print('UTC hours of first-slice requests:', dict(sorted(hours.items())))
    pd_ = pair_days(dev)
    print(f'first slice: {sum(len(v) for v in pd_.values())} labelled pair-days over {len(pd_)} pairs')
    senders = collections.Counter(r['from'] for r in dev)
    print('top senders:', senders.most_common(8))
    print('accepted:', collections.Counter(r['accepted'] for r in dev),
          'followed through:', collections.Counter(r['followed_through'] for r in dev))


if __name__ == '__main__':
    main()
