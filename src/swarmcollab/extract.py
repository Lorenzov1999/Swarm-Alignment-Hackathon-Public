"""Stage A: cut the computer-use turns down to the main period, one file per agent.

One streaming pass over computer_use_turns.jsonl.gz:
- keeps the steps dated inside the main period (UTC dates);
- maps each step to its agent through its session;
- drops leak steps whole: chat sends, history searches, room moves, and any
  action carrying a session goal (the evidence boundary);
- never writes the agents' narration (agent_messages) or the `system` field;
- cuts long commands and outputs to a head and a tail.

Output: outputs/steps/<agent>.jsonl.gz with one compact JSON object per step,
sorted by time, and outputs/steps/summary.json with kept and dropped counts.

Step fields: i turn id, t UTC timestamp, s session id, k kind (shell, restart,
empty, or the GUI action name), x command or GUI text, o output, e error.

Run: python -I -m swarmcollab.extract [turns file, default the full dataset]
"""
import collections
import gzip
import json
import re
import subprocess
import sys
import time

from . import config

LEAK_ACTIONS = {'send_message_back_to_chat', 'search_history', 'move_to_room', 'request_human_helper'}
LEAK_KEYS = {'sessionGoal', 'nextSessionGoal', 'shortDisplayedSessionGoal'}
PAYLOAD_KEYS = ('text', 'description', 'seconds', 'medium', 'recipient')

CMD_HEAD, CMD_TAIL = 40000, 10000
OUT_HEAD, OUT_TAIL = 6000, 2000
ERR_HEAD = 2000
STOP_AFTER = 50000  # consecutive lines past the period end before the scan stops


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, file=sys.stderr, flush=True)


def cap(s, head, tail=0):
    if s is None:
        return None
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    if len(s) <= head + tail:
        return s
    cut = len(s) - head - tail
    return s[:head] + f'\n[... {cut} characters cut ...]' + ('\n' + s[-tail:] if tail else '')


def is_leak(act):
    if not isinstance(act, dict):
        return False
    return act.get('action') in LEAK_ACTIONS or bool(LEAK_KEYS & act.keys())


def kind_and_payload(act):
    if act is None:
        return 'empty', None
    if not isinstance(act, dict):
        return 'other', cap(act, 2000)
    if 'command' in act:
        cmd = act.get('command')
        if cmd in (None, '') and act.get('restart'):
            return 'restart', None
        return 'shell', cap(cmd, CMD_HEAD, CMD_TAIL)
    a = act.get('action')
    if a is None:
        if act.get('restart'):
            return 'restart', None
        if set(act) <= {'coordinate', 'text'}:
            return 'gui', act.get('text') or None
        return 'other', cap(act, 2000)
    parts = [str(act[k]) for k in PAYLOAD_KEYS if act.get(k) not in (None, '')]
    return str(a), ('\n'.join(parts) if parts else None)


def slug(name):
    return re.sub(r'[^A-Za-z0-9.]+', '-', name).strip('-')


def load_maps():
    agents = {}
    with gzip.open(config.AGENTS, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            agents[r['id']] = r['name']
    sessions = {}
    with gzip.open(config.SESSIONS, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            sessions[r['id']] = r['agent_id']
    return agents, sessions


def resort(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    rows.sort(key=lambda r: r['t'])
    with gzip.open(path, 'wt', compresslevel=4, encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    t0 = time.time()
    agents, sessions = load_maps()
    log(f'{len(agents)} agents, {len(sessions)} sessions')
    config.STEPS.mkdir(parents=True, exist_ok=True)
    lo, hi = config.MAIN_START.isoformat(), config.MAIN_END.isoformat()
    lo_b, hi_b = lo.encode(), hi.encode()
    key = b'"created_at":"'

    writers, files = {}, {}
    kept = collections.defaultdict(collections.Counter)
    dropped = collections.defaultdict(collections.Counter)
    signatures = collections.Counter()
    first, last, disorder = {}, {}, collections.Counter()
    system_seen = collections.Counter()
    unknown_session = 0
    n_lines = n_period = 0
    past = 0
    stopped_at = None

    src = sys.argv[1] if len(sys.argv) > 1 else str(config.TURNS)
    cmd = ['pigz', '-dc', src] if src.endswith('.gz') else ['cat', src]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 22)
    try:
        for line in proc.stdout:
            n_lines += 1
            if n_lines % 500000 == 0:
                log(f'{n_lines} lines read, {n_period} in period')
            k = line.rfind(key)
            if k < 0:
                continue
            day = line[k + 14:k + 24]
            if day < lo_b:
                continue
            if day > hi_b:
                past += 1
                if past >= STOP_AFTER:
                    stopped_at = day.decode()
                    break
                continue
            past = 0
            r = json.loads(line)
            ts = r.get('created_at') or ''
            if not (lo <= ts[:10] <= hi):
                continue
            n_period += 1
            aid = sessions.get(r.get('session_id'))
            if aid is None:
                unknown_session += 1
                continue
            name = agents.get(aid, aid)
            act = r.get('agent_action')
            if is_leak(act):
                dropped[name][act.get('action') or 'session-goal keys'] += 1
                continue
            kind, payload = kind_and_payload(act)
            if isinstance(act, dict):
                signatures[f'{kind} {sorted(act)}'] += 1
            rec = {'i': r.get('id'), 't': ts, 's': r.get('session_id'), 'k': kind}
            if payload:
                rec['x'] = payload
            out = r.get('output')
            if out not in (None, ''):
                rec['o'] = cap(out, OUT_HEAD, OUT_TAIL)
            err = r.get('error')
            if err:
                rec['e'] = cap(err, ERR_HEAD)
            if r.get('system'):
                system_seen[name] += 1
            w = writers.get(name)
            if w is None:
                files[name] = config.STEPS / f'{slug(name)}.jsonl.gz'
                w = writers[name] = gzip.open(files[name], 'wt', compresslevel=4, encoding='utf-8')
            w.write(json.dumps(rec, ensure_ascii=False) + '\n')
            kept[name][kind] += 1
            if name in last and ts < last[name]:
                disorder[name] += 1
            first.setdefault(name, ts)
            if ts > last.get(name, ''):
                last[name] = ts
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for w in writers.values():
            w.close()

    for name, n in disorder.items():
        log(f'resorting {name}: {n} steps out of order')
        resort(files[name])

    summary = {
        'period': [lo, hi],
        'lines_read': n_lines,
        'steps_in_period': n_period,
        'unknown_session': unknown_session,
        'stopped_early_at_date': stopped_at,
        'seconds': round(time.time() - t0, 1),
        'leak_actions_dropped': sorted(LEAK_ACTIONS),
        'agents': {
            name: {
                'file': files[name].name,
                'kept': sum(kept[name].values()),
                'dropped': sum(dropped[name].values()),
                'kept_by_kind': dict(kept[name].most_common()),
                'dropped_by_kind': dict(dropped[name].most_common()),
                'first': first[name],
                'last': last[name],
                'out_of_order': disorder[name],
                'system_field_present': system_seen[name],
            }
            for name in sorted(kept)
        },
        'agents_with_only_dropped_steps': sorted(set(dropped) - set(kept)),
        'action_signatures': dict(signatures.most_common(200)),
    }
    with open(config.STEPS / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    log(f'done: {sum(sum(c.values()) for c in kept.values())} steps kept for {len(kept)} agents, '
        f'{sum(sum(c.values()) for c in dropped.values())} leak steps dropped, {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
