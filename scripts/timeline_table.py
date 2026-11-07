"""Summarise outputs/timeline/day_agent_kind.csv by month and by goal period.

Usage: python -I scripts/timeline_table.py
Prints Markdown tables. An agent counts as active in a period when it has at least
one working step (shell, GUI or other, but not waiting or chat) on 3 or more days.
"""
import gzip
import json

import pandas as pd

from swarmcollab import config
from swarmcollab import labels as L


def markdown(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


T = pd.read_csv(config.OUT / 'timeline' / 'day_agent_kind.csv')
T = T[T.agent != '?']
T['month'] = T.day.str[:7]
work = T[T.kind.isin(['shell', 'gui', 'other'])]

goals = []
with gzip.open(config.VILLAGE / 'village_goals.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        g = json.loads(line)
        goals.append((g['start_time'][:10], (g['end_time'] or '9999')[:10], g['goal']))
goals.sort()

rows, _, _ = L.load_requests()
req = pd.Series([r['day'].isoformat()[:7] for r in rows if r['from'] != r['to']]).value_counts()


def active_agents(sub):
    d = sub[sub.steps > 0].groupby('agent').day.nunique()
    return int((d >= 3).sum())


out = []
for month, g in T.groupby('month'):
    k = g.groupby('kind').steps.sum()
    tot = k.sum()
    out.append({
        'month': month,
        'active agents': active_agents(work[work.month == month]),
        'steps (k)': round(tot / 1000, 1),
        'shell %': round(100 * k.get('shell', 0) / tot),
        'GUI %': round(100 * k.get('gui', 0) / tot),
        'chat-side %': round(100 * k.get('chat', 0) / tot),
        'waiting %': round(100 * k.get('waiting', 0) / tot),
        'requests': int(req.get(month, 0)),
        'goals starting': sum(1 for s, e, _ in goals if s[:7] == month),
    })
print(markdown(pd.DataFrame(out)))

print()
grows = []
for s, e, goal in goals:
    sub = T[(T.day >= s) & (T.day < e)]
    if sub.empty:
        continue
    k = sub.groupby('kind').steps.sum()
    tot = k.sum()
    grows.append({'start': s, 'goal': goal[:70], 'days': sub.day.nunique(),
                  'active agents': active_agents(sub[sub.kind.isin(['shell', 'gui', 'other'])]),
                  'steps (k)': round(tot / 1000, 1), 'shell %': round(100 * k.get('shell', 0) / tot),
                  'GUI %': round(100 * k.get('gui', 0) / tot)})
print(markdown(pd.DataFrame(grows)))
