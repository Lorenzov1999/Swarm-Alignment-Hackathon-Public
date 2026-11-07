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
