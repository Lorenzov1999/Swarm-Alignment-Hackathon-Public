"""Checks for the trace parser and the label loader.

Run: python -I tests/test_parsing.py
"""
import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from swarmcollab import traces as T


def test_urls():
    cases = {
        'https://gitlab.com/ai-village-agents/village/wellbeing-compass/-/issues/3':
            [('repo', 'gitlab:ai-village-agents/village/wellbeing-compass'),
             ('issue', 'ai-village-agents/village/wellbeing-compass#3')],
        'https://gitlab.com/ai-village-agents/village/x.git': [('repo', 'gitlab:ai-village-agents/village/x')],
        'https://gitlab.com/ai-village-agents/village': [('repo', 'gitlab?:ai-village-agents/village')],
        'https://oauth2:SECRET@gitlab.com/group/proj.git': [('repo', 'gitlab:group/proj')],
        'https://gitlab.com/api/v4/projects/71234567/merge_requests/3/notes':
            [('project', '71234567'), ('issue', 'pid:71234567!3')],
        'https://gitlab.com/api/v4/projects/group%2Fproj/pipelines/998877':
            [('repo', 'gitlab:group/proj'), ('pipeline', 'p998877')],
        'https://my-proj-a1b2c3.gitlab.io/es/page.html':
            [('pages', 'my-proj-a1b2c3.gitlab.io'), ('page', 'my-proj-a1b2c3.gitlab.io/es/page.html')],
        'https://ai-village-agents.gitlab.io/hub/index.html': [('pages', 'ai-village-agents.gitlab.io/hub')],
        'https://docs.google.com/document/d/1eqspRUMIyU8l9pK7NsRDftEni-Ogor6f2-gk2Cp4ecY/edit':
            [('gdoc', '1eqspRUMIyU8l9pK7NsRDftEni-Ogor6f2-gk2Cp4ecY')],
        'https://github.com/someone/repo/commit/abc1234def': [('repo', 'github:someone/repo'), ('commit', 'abc1234')],
        'https://arxiv.org/abs/2601.01234': [('site', 'arxiv.org/abs'), ('page', 'arxiv.org/abs/2601.01234')],
        'http://localhost:8000/x': [],
    }
    for url, want in cases.items():
        got = T.parse_url(url)
        assert got == want, (url, got)


def test_ip_endpoint_identity():
    endpoint = 'http://10.108.0.39:6080'
    url = endpoint + '/vnc.html?path=WebSockify%2FOne#View!'
    assert T.parse_url(url) == [('site', endpoint), ('page', url)]

    # Different transport endpoints must not become a common site or page.
    endpoints = [endpoint, 'http://10.108.0.39:6081',
                 'https://10.108.0.39:6080', 'http://10.108.0.40:6080']
    profiles = [set(T.parse_url(e + '/vnc.html')) for e in endpoints]
    for i, profile in enumerate(profiles):
        for other in profiles[i + 1:]:
            assert profile.isdisjoint(other), (profile, other)

    # Sharing a server is allowed, while the exact page distinctions survive.
    suffixes = ['/VNC', '/vnc', '/vnc?room=A', '/vnc?room=a', '/vnc#A',
                '/vnc#B', '/vnc?', '/vnc#', '/vnc/A%2FB', '/vnc/A/B']
    parsed = [dict(T.parse_url(endpoint + suffix)) for suffix in suffixes]
    assert {p['site'] for p in parsed} == {endpoint}
    assert {p['page'] for p in parsed} == {endpoint + suffix for suffix in suffixes}


def test_ip_address_and_default_port_normalization():
    cases = {
        'http://10.108.0.39:80/vnc.html':
            [('site', 'http://10.108.0.39'), ('page', 'http://10.108.0.39/vnc.html')],
        'https://192.0.2.1:443/':
            [('site', 'https://192.0.2.1'), ('page', 'https://192.0.2.1/')],
        'https://[2001:0DB8:0:0:0:0:0:1]:443/Control?room=A#Screen':
            [('site', 'https://[2001:db8::1]'),
             ('page', 'https://[2001:db8::1]/Control?room=A#Screen')],
        'http://[fd00::1]:6080/vnc.html':
            [('site', 'http://[fd00::1]:6080'), ('page', 'http://[fd00::1]:6080/vnc.html')],
        'http://[::ffff:192.0.2.1]/':
            [('site', 'http://[::ffff:192.0.2.1]'), ('page', 'http://[::ffff:192.0.2.1]/')],
    }
    for url, want in cases.items():
        assert T.parse_url(url) == want, (url, T.parse_url(url))
    assert T.parse_url('http://10.108.0.39:80') == T.parse_url('http://10.108.0.39/')
    assert T.parse_url('https://[2001:db8::1]:443') == T.parse_url('https://[2001:db8::1]/')


def test_invalid_and_machine_local_ip_endpoints_are_excluded():
    urls = [
        'http://localhost:6080/vnc.html', 'http://machine.local:6080/vnc.html',
        'http://127.0.0.1:6080/vnc.html', 'http://127.20.30.40:6080/vnc.html',
        'http://0.0.0.0:6080/vnc.html', 'http://[::]:6080/vnc.html',
        'http://[::1]:6080/vnc.html', 'http://[::ffff:127.0.0.1]:6080/vnc.html',
        'http://[::ffff:0.0.0.0]:6080/vnc.html',
        'http://999.1.2.3/vnc.html', 'http://10.108.0/vnc.html',
        'http://010.108.0.39/vnc.html', 'http://[2001:db8::zzz]/vnc.html',
        'http://[2001:db8::1/vnc.html', 'http://2001:db8::1/vnc.html',
        'http://[2001:db8::1]junk:6080/vnc.html', 'http://[v1.a]/vnc.html',
        'http://10.108.0.39:abc/vnc.html', 'http://10.108.0.39:65536/vnc.html',
        'http://10.108.0.39:-1/vnc.html', 'http://10.108.0.39:/vnc.html',
        'http://[2001:db8::1]:abc/vnc.html', 'http://[2001:db8::1]:65536/vnc.html',
        'http://[2001:db8::1]:/vnc.html', 'ftp://10.108.0.39/vnc.html',
    ]
    for url in urls:
        assert T.parse_url(url) == [], (url, T.parse_url(url))


def test_ip_credentials_do_not_become_identifiers():
    urls = [
        'http://user:password@10.108.0.39:6080/vnc.html',
        'http://user@[2001:db8::1]:6080/vnc.html',
        'http://10.108.0.39:6080/vnc.html?token=secret',
        'http://10.108.0.39:6080/vnc.html?room=A&api_key=secret',
        'http://10.108.0.39:6080/vnc.html?%74oken=secret',
        'http://[2001:db8::1]:6080/vnc.html#password=secret',
        'http://[2001:db8::1]:6080/vnc.html#tab=console&access_key=secret',
    ]
    p = T.AgentParser('GPT-5', ['GPT-5', 'GPT-5.1'])
    for i, url in enumerate(urls):
        assert T.parse_url(url) == [], (url, T.parse_url(url))
        p.typed(float(i), 'd', i, url)
    assert p.rows == [], p.rows


def test_ip_urls_reach_shell_output_and_typed_traces():
    names = ['GPT-5', 'GPT-5.1']
    p = T.AgentParser(names[0], names)
    ipv4 = 'http://10.108.0.39:6080/vnc.html?path=WebSockify#View'
    ipv6 = 'https://[2001:db8::1]:443/Control?room=A#Screen'
    p.typed(1., 'd', 0, ipv4)
    p.shell(2., 'd', 1, 's', f"curl '{ipv6}'", f'Available at [{ipv6}]')
    p.typed(3., 'd', 2, ipv6)
    rows = {(r[3], r[4], r[5], r[6]) for r in p.rows}
    page6 = 'https://[2001:db8::1]/Control?room=A#Screen'
    assert ('page', ipv4, 'read', 'typed') in rows, rows
    assert ('site', 'http://10.108.0.39:6080', 'read', 'typed') in rows, rows
    for operation, source in [('read', 'cmd'), ('seen', 'out'), ('read', 'typed')]:
        assert ('page', page6, operation, source) in rows, rows
        assert ('site', 'https://[2001:db8::1]', operation, source) in rows, rows
    assert len(rows) == 8, rows

    # Identical loopback URLs on two machines must not create shared traces.
    for actor in names:
        local = T.AgentParser(actor, names)
        local.typed(1., 'd', 0, 'http://127.0.0.1:8888/Sharesite/Edit/0')
        local.shell(2., 'd', 1, 's', "curl 'http://[::1]:8888/'", '')
        assert local.rows == [], local.rows


def test_domain_url_normalization_and_delimiters_unchanged():
    cases = {
        'https://News.test:8443/Case/Folder/index.html?mode=X#Section':
            [('site', 'news.test/case'), ('page', 'news.test/case/folder')],
        'https://arxiv.org/abs/2601.01234.':
            [('site', 'arxiv.org/abs'), ('page', 'arxiv.org/abs/2601.01234')],
        'http://www.10.108.0.39/vnc.html': [],
        'https://example.com/path': [],
    }
    for url, want in cases.items():
        assert T.parse_url(url) == want, (url, T.parse_url(url))
    p = T.AgentParser('GPT-5', ['GPT-5', 'GPT-5.1'])
    expected = [('site', 'arxiv.org/abs'), ('page', 'arxiv.org/abs/2601.01234')]
    for text in ['[https://arxiv.org/abs/2601.01234]',
                 '(https://arxiv.org/abs/2601.01234)',
                 'https://arxiv.org/abs/2601.01234, next item']:
        assert p.urls(text) == expected, (text, p.urls(text))


def test_repo_refs():
    assert T.repo_from_ref('ai-village-agents/village/x') == 'gitlab:ai-village-agents/village/x'
    assert T.repo_from_ref('git@gitlab.com:group/proj.git') == 'gitlab:group/proj'
    assert T.repo_from_ref('https://gitlab.com/group/sub/proj') == 'gitlab:group/sub/proj'


def test_valid_uniform_hex_commits():
    for sha in ('abcdefa', '1234567', 'a' * 40, '1' * 40, 'ABCDEFAB'):
        assert T.is_commitish(sha), sha
        url = 'https://gitlab.com/group/project/-/commit/' + sha
        assert ('commit', sha[:7].lower()) in T.parse_url(url)
        assert ('commit', sha[:7].lower()) in T.parse_url('https://github.com/group/project/commit/' + sha)
    for invalid in ('123456', 'a' * 41, 'xyz1234', 'release', '123456g'):
        assert not T.is_commitish(invalid), invalid


def test_parse_agent_filters_pacific_scope_before_payload():
    with tempfile.TemporaryDirectory(prefix='swarm-parser-scope-') as directory:
        root = Path(directory)
        steps, mentions = root / 'steps', root / 'mentions'
        steps.mkdir()
        filename = 'GPT-5.jsonl.gz'
        (steps / 'summary.json').write_text(json.dumps({'agents': {
            'GPT-5': {'file': filename}, 'Claude Opus 4.8': {'file': 'other.jsonl.gz'}}}))
        # Excluded rows deliberately omit all payload fields: parsing one would
        # raise before reaching the assertions below.
        records = [
            {'t': '2026-07-06 06:59:59.999999'},
            {'t': '2026-07-06 07:00:00', 'k': 'type', 's': 's',
             'x': 'https://gitlab.com/group/project/-/commit/abcdefa'},
            {'t': '2026-08-15 06:59:59.999999', 'k': 'type', 's': 's',
             'x': 'https://gitlab.com/group/project/-/commit/1234567'},
            {'t': '2026-08-15 07:00:00'},
            {'t': '2026-08-17 16:00:00'},
            {'t': '2026-10-01 16:00:00'},
        ]
        with gzip.open(steps / filename, 'wt', encoding='utf-8') as stream:
            stream.writelines(json.dumps(record) + '\n' for record in records)
        with patch.object(T.config, 'STEPS', steps), patch.object(T, 'MENTIONS', mentions):
            T.parse_agent(filename)
        activity = json.loads((mentions / 'GPT-5.activity.json').read_text())
        assert set(activity['days']) == {'2026-07-06', '2026-08-14'}, activity
        assert sum(day['steps'] for day in activity['days'].values()) == 2
        with gzip.open(mentions / 'GPT-5.tsv.gz', 'rt', encoding='utf-8') as stream:
            rows = [line.rstrip('\n').split('\t') for line in stream.readlines()[1:]]
        assert {row[3] for row in rows} == {'1', '2'}, rows  # Original source offsets survive filtering.
        assert {row[5] for row in rows if row[4] == 'commit'} == {'abcdefa', '1234567'}


def test_names():
    rx, g = T.build_name_regex(['Claude Opus 4.8', 'GPT-5', 'GPT-5.1', 'GPT-5.6 Luna', 'DeepSeek-V3.2',
                                'Gemini 2.5 Pro', 'Kimi K3', 'Kimi K2.6', 'Grok 4.5', 'Claude Sonnet 5'])
    cases = {
        'ask Opus 4.8 and gpt-5.1': {'Claude Opus 4.8', 'GPT-5.1'},
        'GPT-5 said': {'GPT-5'},
        'GPT-5.6 Luna waits': {'GPT-5.6 Luna'},
        'claude-opus-4-8 branch': {'Claude Opus 4.8'},
        'deepseek 3.2': {'DeepSeek-V3.2'},
        'Gemini 2.5 rocks': {'Gemini 2.5 Pro'},
        'kimi k3.1 pro': set(),
        'grok said': {'Grok 4.5'},
        'Sonnet 5.5 is not Sonnet 5': {'Claude Sonnet 5'},
        'gpt-5-2025-08-07': set(),
    }
    for text, want in cases.items():
        got = T.find_names(rx, g, text)
        assert got == want, (text, got)


def test_shell_clone_mapping():
    p = T.AgentParser('Claude Opus 4.8', ['Claude Opus 4.8', 'GPT-5.1'])
    p.shell(1.0, 'd', 0, 's1', 'cd ~/work && git clone https://gitlab.com/ai-village-agents/village/hub.git', '')
    p.shell(2.0, 'd', 1, 's1', 'cd hub && git add -A && git commit -m "fix for GPT-5.1"',
            '[main 1a2b3c4] fix for GPT-5.1\n 1 file changed')
    p.shell(3.0, 'd', 2, 's1', 'git push',
            'To https://gitlab.com/ai-village-agents/village/hub.git\n   0f0f0f0..1a2b3c4  main -> main')
    rows = {(r[3], r[4], r[5], r[6]) for r in p.rows}
    assert ('repo', 'gitlab:ai-village-agents/village/hub', 'write', 'cwd') in rows, rows
    assert ('commit', '1a2b3c4', 'create', 'out') in rows, rows
    assert ('repo', 'gitlab:ai-village-agents/village/hub', 'write', 'out') in rows, rows
    assert ('name', 'GPT-5.1', 'read', 'code') in rows, rows


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
