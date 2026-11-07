"""Survey the shape of computer-use turn records.

Usage: python -I scripts/survey_turns.py TURNS.jsonl [examples_per_kind]

Reads turn records (one JSON object per line), and prints the action
vocabulary, the shapes of agent_messages and output, and a few truncated
examples per action. Used to design the extractor; prints no narration text.
"""
import collections
import json
import sys


def short(x, n=160):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = s.replace('\n', '\\n')
    return s if len(s) <= n else s[:n] + f'...(+{len(s) - n})'


def tool_names(msgs):
    """Tool names in agent_messages, whatever the provider's message shape."""
    names = []
    if isinstance(msgs, list):
        for m in msgs:
            names += tool_names(m)
        return names
    if not isinstance(msgs, dict):
        return names
    for c in msgs.get('content') or []:
        if isinstance(c, dict) and c.get('type') in ('tool_use', 'function_call', 'tool_call'):
            names.append(c.get('name'))
    for tc in msgs.get('tool_calls') or []:
        if isinstance(tc, dict):
            names.append((tc.get('function') or {}).get('name') or tc.get('name'))
    return names


def main():
    path = sys.argv[1]
    per_kind = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    action_keys = collections.Counter()
    actions = collections.Counter()
    msg_shapes = collections.Counter()
    tools = collections.Counter()
    out_types = collections.Counter()
    out_len = collections.defaultdict(list)
    other_fields = collections.Counter()
    examples = collections.defaultdict(list)
    n = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            n += 1
            act = r.get('agent_action')
            kind = act.get('action') if isinstance(act, dict) else type(act).__name__
            if isinstance(act, dict):
                action_keys[(kind, tuple(sorted(act)))] += 1
            actions[kind] += 1
            msgs = r.get('agent_messages')
            if isinstance(msgs, dict):
                msg_shapes[tuple(sorted(msgs))] += 1
            else:
                msg_shapes[type(msgs).__name__] += 1
            for t in tool_names(msgs):
                tools[(kind, t)] += 1
            out = r.get('output')
            out_types[(kind, type(out).__name__)] += 1
            out_len[kind].append(len(out) if isinstance(out, str) else len(json.dumps(out)) if out else 0)
            for k in ('error', 'system'):
                if r.get(k):
                    other_fields[(k, kind)] += 1
            if len(examples[kind]) < per_kind:
                examples[kind].append({
                    'agent_action': short(act, 300),
                    'output': short(out, 300),
                    'error': short(r.get('error'), 120) if r.get('error') else None,
                    'system': short(r.get('system'), 120) if r.get('system') else None,
                })
    print(f'records: {n}')
    print('\n== actions')
    for k, v in actions.most_common():
        lens = sorted(out_len[k])
        med = lens[len(lens) // 2] if lens else 0
        p95 = lens[int(0.95 * (len(lens) - 1))] if lens else 0
        print(f'  {str(k):<34s} {v:>6d}   output length median {med:>6d}  p95 {p95:>7d}')
    print('\n== agent_action key sets')
    for (k, keys), v in action_keys.most_common(40):
        print(f'  {str(k):<34s} {v:>6d}  {keys}')
    print('\n== agent_messages shapes')
    for k, v in msg_shapes.most_common(10):
        print(f'  {v:>6d}  {k}')
    print('\n== tool names by action')
    for (k, t), v in tools.most_common(40):
        print(f'  {str(k):<34s} {str(t):<34s} {v:>6d}')
    print('\n== output types')
    for (k, t), v in out_types.most_common(40):
        print(f'  {str(k):<34s} {t:<8s} {v:>6d}')
    print('\n== error / system present')
    for (k, kind), v in other_fields.most_common(20):
        print(f'  {k:<7s} {str(kind):<34s} {v:>6d}')
    print('\n== examples')
    for k, exs in examples.items():
        print(f'-- {k}')
        for e in exs:
            for fk, fv in e.items():
                if fv is not None:
                    print(f'   {fk}: {fv}')
            print()


if __name__ == '__main__':
    main()
