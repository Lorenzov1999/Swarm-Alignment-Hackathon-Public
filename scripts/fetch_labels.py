"""Fetch the village's request records from a public AI Village agent page.

Usage: python -I scripts/fetch_labels.py OUT_DIR

Downloads https://theaidigest.org/village/agent/gemini-2-5-pro to OUT_DIR/page.html,
pulls the chart props out of the page's React Server Components payload, and
writes them to OUT_DIR/ai-village-chart-props.json. The request chart is the
object with an `instances` list; every agent page carries the village-wide list.
"""
import hashlib
import json
import pathlib
import re
import sys
import urllib.request

URL = 'https://theaidigest.org/village/agent/gemini-2-5-pro'
PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)')
START_RE = re.compile(r'\{"(?:options|focal)":')


def extract(html):
    text = ''.join(json.loads(s) for s in PUSH_RE.findall(html))
    dec = json.JSONDecoder()
    found, seen = [], set()
    for m in START_RE.finditer(text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and 'options' in obj and 'focal' in obj:
            key = json.dumps(obj, sort_keys=True)
            if key not in seen:
                seen.add(key)
                found.append(obj)
    return found


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0 (research data fetch)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode('utf-8', errors='replace')
    (out / 'page.html').write_text(html, encoding='utf-8')
    props = extract(html)
    path = out / 'ai-village-chart-props.json'
    path.write_text(json.dumps(props, ensure_ascii=False), encoding='utf-8')
    panel = next((p for p in props if 'instances' in p), None)
    n = len(panel['instances']) if panel else 0
    stamps = [x['timestamp'] for x in panel['instances']] if panel else []
    print(f'{len(props)} chart props objects; {n} requests'
          + (f' from {min(stamps)} to {max(stamps)}' if stamps else ''))
    print('sha256', hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
