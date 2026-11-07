"""How the whole AI Village log is laid out over time.

One streaming pass over computer_use_turns (all dates, not just the main period)
that classifies each step from the start of its `agent_action` field without
parsing the rest of the record, and counts steps per Pacific day, agent and kind:

  shell    a shell command
  gui      a GUI action (click, type, key, scroll, screenshot, ...)
  chat     a chat-side action: chat send, history search, room move
  waiting  pause, wait, or an empty turn
  other    anything else (approval requests, clipboard, restarts, ...)

Output: outputs/timeline/day_agent_kind.csv. No message content is read.

Run: python -I -m swarmcollab.timeline
"""
import collections
import datetime as dt
import gzip
import json
import re
import subprocess
import sys
import time

from . import config

OUT = config.OUT / 'timeline'
CHAT = {b'send_message_back_to_chat', b'search_history', b'move_to_room'}
WAITING = {b'pause', b'wait'}
GUI = {b'left_click', b'right_click', b'double_click', b'triple_click', b'middle_click', b'mouse_move',
       b'left_click_drag', b'scroll', b'key', b'hold_key', b'type', b'screenshot', b'cursor_position',
       b'get_pixel_coords_of_element', b'left_mouse_down', b'left_mouse_up', b'zoom'}
ACTION_RE = re.compile(rb'"action":\s*"([A-Za-z_]+)"')
PACIFIC_OFFSET = {True: dt.timedelta(hours=-7), False: dt.timedelta(hours=-8)}


def pacific_day(stamp):
    """UTC 'YYYY-MM-DD HH:MM:SS' -> Pacific date, with US daylight-saving rules."""
    t = dt.datetime.strptime(stamp[:19], '%Y-%m-%d %H:%M:%S')
    y = t.year
    # DST: second Sunday of March 10:00 UTC to first Sunday of November 09:00 UTC
    mar = dt.datetime(y, 3, 8) + dt.timedelta(days=(6 - dt.date(y, 3, 8).weekday()) % 7)
    nov = dt.datetime(y, 11, 1) + dt.timedelta(days=(6 - dt.date(y, 11, 1).weekday()) % 7)
    dst = mar + dt.timedelta(hours=10) <= t < nov + dt.timedelta(hours=9)
    return (t + PACIFIC_OFFSET[dst]).date().isoformat()


def classify(line, k):
    head = line[k + 15:k + 27].lstrip()
    if head.startswith(b'null'):
        return 'waiting'
    if head.startswith(b'{"command"') or head.startswith(b'{"restart"'):
        return 'shell'
    m = ACTION_RE.search(line, k, k + 20000)
    if not m:
        return 'other'
    a = m.group(1)
    if a in CHAT:
        return 'chat'
    if a in WAITING:
        return 'waiting'
    if a in GUI:
        return 'gui'
    return 'other'


def main():
    t0 = time.time()
    agents = {}
    with gzip.open(config.AGENTS, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            agents[r['id']] = r['name']
    sessions = {}
    with gzip.open(config.SESSIONS, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            sessions[r['id']] = agents.get(r['agent_id'], r['agent_id'])
    counts = collections.Counter()
    day_cache = {}
    proc = subprocess.Popen(['pigz', '-dc', str(config.TURNS)], stdout=subprocess.PIPE, bufsize=1 << 22)
    n = 0
    for line in proc.stdout:
        n += 1
        if n % 500000 == 0:
            print(time.strftime('%H:%M:%S'), n, file=sys.stderr, flush=True)
        c = line.rfind(b'"created_at":"')
        s = line.find(b'"session_id":"')
        k = line.find(b'"agent_action":')
        if c < 0 or s < 0 or k < 0:
            counts[('?', '?', 'unparsed')] += 1
            continue
        stamp = line[c + 14:c + 33].decode()
        hour_key = stamp[:13]
        day = day_cache.get(hour_key)
        if day is None:
            day = day_cache[hour_key] = pacific_day(stamp)
        agent = sessions.get(line[s + 14:s + 50].decode(), '?')
        counts[(day, agent, classify(line, k))] += 1
    proc.wait()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'day_agent_kind.csv', 'w', encoding='utf-8') as f:
        f.write('day,agent,kind,steps\n')
        for (day, agent, kind), v in sorted(counts.items()):
            f.write(f'{day},"{agent}",{kind},{v}\n')
    print(f'{n} lines, {len(counts)} rows, {time.time() - t0:.0f} s', file=sys.stderr)


if __name__ == '__main__':
    main()
