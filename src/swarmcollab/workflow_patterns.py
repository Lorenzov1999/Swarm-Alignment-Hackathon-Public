"""Dependency-backed work sequences and frozen Village workflow candidates.

Generic motifs join version-linked events, never just neighboring timestamps.
Village motifs add the observed directory/review conventions. Every pattern is
available only at its last supporting observation; none proves a causal effect.
"""
import collections
import hashlib
import io
import json
import re
import token
import tokenize
from urllib.parse import unquote, urlsplit

from . import config
from .provenance_common import PACIFIC, in_scope, timestamp

PRODUCES = {'generate', 'transform', 'incorporate', 'review'}
CHANGES = {'generate', 'transform', 'incorporate', 'write'}
FOLLOWUPS = {'test', 'execute', 'install', 'transform', 'incorporate', 'generate'}
STRICT_IDENTITIES = {'exact', 'same_content', 'recorded_build_lineage'}
BAD_IDENTITIES = {'shared_entity', 'unresolved_or_ambiguous_prefix'}
WORDS = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
RECOMMENDATION = re.compile(r'(?i)recommendation\s*\*?\s*:\s*\*?')

# Frozen from the authorized training inventory, without labels. These are
# workflow conventions, not actor pairs asserted to have collaborated.
SCRIPT_WORKFLOW_RULES = [dict(
    rule_id='village_fsa_script_v1',
    repository='gitlab:ai-village-agents/village/llm-psychoactive-prompts',
    relative_path='collaborations/fsa_analysis_script.py',
    working_directories=['~/llm-psychoactive-prompts', '/home/computeruse/llm-psychoactive-prompts'],
    read_marker='Formative Self-Audit (FSA) Analysis Script',
    execution_markers=['FORMATIVE SELF-AUDIT ANALYSIS', '009 S1 Demo'],
    available_from='2026-07-28T21:08:52.123781+00:00',
    evidence=[
        dict(actor='Kimi K2.6', turn_id='7ce3f6b3-3378-422f-93d6-2e094c7ab46b', time='2026-07-28T20:52:51.728523+00:00', role='script_creation'),
        dict(actor='Kimi K2.6', turn_id='9898a583-7c16-43a9-a422-85d957f3e640', time='2026-07-28T20:53:44.051586+00:00', role='commit_and_publication'),
        dict(actor='GLM-5.2', turn_id='e3ff7199-dcd3-4230-9c03-0bd945c5db1e', time='2026-07-28T21:03:55.973034+00:00', role='version_acquisition'),
        dict(actor='GLM-5.2', turn_id='60eda1b6-7ff7-4ae2-a462-de46a5a4d984', time='2026-07-28T21:04:00.609114+00:00', role='script_read'),
        dict(actor='GLM-5.2', turn_id='86837242-7104-472d-b0e3-038b844cbeee', time='2026-07-28T21:08:52.123781+00:00', role='script_execution'),
    ])]


def extract_workflow_snippets(actor, record, max_field_chars=12000):
    """Retain commands/results for the frozen script and review conventions.

    Retained fields are bounded and preserve their original character offsets.
    """
    if not in_scope(record['t']):
        raise ValueError('Workflow record outside development scope')
    if record.get('k') != 'shell':
        return []
    command, output = record.get('x', ''), record.get('o', '')
    script = any(r['relative_path'] in command or r['relative_path'] in output
                 for r in SCRIPT_WORKFLOW_RULES)
    # Training-observed review document and its application script. Retain the
    # Recommendation section in the document middle for this specific workflow.
    review = ('/reviews/wellbeing_compass_de_review.md' in command
              or '/tmp/apply_de_review_fixes.py' in command)
    if not script and not review:
        return []
    when = timestamp(record['t'])
    result = []
    for field, value in [('x', command), ('o', output)]:
        if not value:
            continue
        text = value[:max_field_chars]
        row = dict(actor=actor, turn_id=record['i'], session=record.get('s', ''),
                   time=when.isoformat(), day=when.astimezone(PACIFIC).date().isoformat(),
                   field=field, text=text, start=0, end=len(text), kind='workflow',
                   source_role='command_text' if field == 'x' else 'output_exposure',
                   workflow_only=True, truncated=len(value) > max_field_chars)
        row['snippet_id'] = _digest(row)
        result.append(row)
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def _when(value):
    return timestamp(value).timestamp()


def _event_time(event):
    return _when(event['time'])


def _authority(event):
    return event.get('status') == 'success' and bool(event.get('relation_authority'))


def _new_outputs(event):
    return set(event.get('outputs', ())) - set(event.get('inputs', ()))


def _scoped_event(event):
    # Date validation precedes any inspection of the event's content or identity.
    if not in_scope(event['time']):
        raise ValueError('Workflow input contains an event outside development scope')
    if timestamp(event['time']).astimezone(PACIFIC).date().isoformat() != event['day']:
        raise ValueError('Workflow event day disagrees with timestamp')
    return event


def _pattern(name, layer, actors, events=(), chains=(), confidence='candidate',
             extra_times=(), interpretation='', **extra):
    times = [e['time'] for e in events] + [c['target_time'] for c in chains]
    times += list(extra_times)
    last = max(times, key=_when)
    available = timestamp(last)
    edges = list(dict.fromkeys((a, b) for a, b in zip(actors, actors[1:]) if a != b))
    if not edges:
        return None
    value = dict(pattern=name, family=name, research_layer=layer,
                 source_actor=edges[0][0], target_actor=edges[0][1], actors=list(actors),
                 actor_edges=[dict(source_actor=a, target_actor=b) for a, b in edges],
                 time=available.isoformat(), available_time=available.isoformat(),
                 day=available.astimezone(PACIFIC).date().isoformat(), confidence=confidence,
                 evidence_level='observed_dependency' if confidence in ('strict', 'qualified') else 'candidate',
                 relation_authority=False,
                 support_events=sorted({e['event_id'] for e in events}),
                 support_turns=sorted({e['turn_id'] for e in events}),
                 support_chains=sorted({c['chain_id'] for c in chains}),
                 interpretation=interpretation)
    value.update(extra)
    value['pattern_id'] = _digest(value)
    return value


def _valid_chains(chains, lookup, counts):
    result = []
    for chain in chains:
        if not in_scope(chain['source_time']) or not in_scope(chain['target_time']):
            raise ValueError('Workflow chain outside development scope')
        if chain.get('level', 0) < 2 or chain.get('identity_confidence') in BAD_IDENTITIES:
            counts['exposure_or_ambiguous_chains_excluded'] += 1
            continue
        identifiers = [chain['source_event'], chain['target_event']] + chain.get('relation_events', [])
        if any(key not in lookup for key in identifiers):
            counts['missing_chain_support'] += 1
            continue
        source, target = lookup[chain['source_event']], lookup[chain['target_event']]
        support = [lookup[key] for key in identifiers]
        if (source['actor'] != chain['source_actor'] or target['actor'] != chain['target_actor']
                or source['actor'] == target['actor'] or source['turn_id'] == target['turn_id']
                or _event_time(source) >= _event_time(target)
                or _when(chain['source_time']) != _event_time(source)
                or _when(chain['target_time']) != _event_time(target)
                or any(e['status'] == 'failed' for e in support)
                or any(_event_time(e) > _event_time(target) for e in support)
                or source['operation'] not in PRODUCES or not _new_outputs(source)):
            counts['invalid_chain_support'] += 1
            continue
        confidence = 'candidate'
        if chain['level'] >= 3 and all(_authority(e) for e in support):
            confidence = 'strict' if chain['identity_confidence'] in STRICT_IDENTITIES else 'qualified'
        result.append((chain, support, confidence))
    return sorted(result, key=lambda row: (_when(row[0]['target_time']), row[0]['chain_id']))


def _confidence(*values):
    return 'candidate' if 'candidate' in values else 'qualified' if 'qualified' in values else 'strict'


def _explicit_bridge(receipt, production, chain):
    """A later production must explicitly use the received version or its output."""
    if receipt['actor'] != production['actor'] or production['operation'] not in PRODUCES:
        return False
    new = _new_outputs(production)
    if not new:
        return False
    if receipt['event_id'] == production['event_id']:
        return True
    if receipt['turn_id'] == production['turn_id'] or _event_time(receipt) >= _event_time(production):
        return False
    previous = set(receipt.get('outputs', ())) | {chain.get('consumed_key'), chain.get('version_key')}
    previous.discard(None)
    if previous & set(production.get('inputs', ())):
        return True
    return any(r.get('kind') in ('derived_from', 'revision_of') and r.get('source') in new
               and r.get('target') in previous for r in production.get('relations', ()))


def _generic_patterns(events, chains, lookup, counts, horizon, max_fanout):
    valid = _valid_chains(chains, lookup, counts)
    by_source = collections.defaultdict(list)
    actor_events = collections.defaultdict(list)
    for event in events:
        if event['status'] != 'failed' and event['operation'] in FOLLOWUPS:
            actor_events[event['actor']].append(event)
    for item in valid:
        by_source[item[0]['source_actor']].append(item)
    for first, first_support, first_confidence in valid:
        start = _when(first['target_time'])
        receipt = lookup[first['target_event']]
        considered = 0
        for second, second_support, second_confidence in by_source[first['target_actor']]:
            if second['chain_id'] == first['chain_id']:
                continue
            middle = lookup[second['source_event']]
            finish = lookup[second['target_event']]
            if _event_time(finish) <= start or _event_time(finish) - _when(first['source_time']) > horizon:
                continue
            if not _explicit_bridge(receipt, middle, first):
                continue
            considered += 1
            if considered > max_fanout:
                counts['dependency_motifs_capped'] += 1
                continue
            actors = [first['source_actor'], first['target_actor'], second['target_actor']]
            returned = actors[0] == actors[-1]
            name = 'artifact_return_revision' if returned and finish['operation'] in CHANGES else (
                   'artifact_return_use' if returned else 'artifact_relay')
            bridge_confidence = 'strict' if _authority(middle) else 'candidate'
            yield _pattern(name, 1, actors, events=first_support + second_support,
                           chains=[first, second], confidence=_confidence(first_confidence, second_confidence, bridge_confidence),
                           interpretation='Two observed dependency edges with an explicit middle transformation/version bridge. This is a work sequence, not a counterfactual causal effect.')
        # Acquisition followed by use/transform is distinct from one acquisition.
        previous = set(receipt.get('outputs', ())) | {first.get('consumed_key'), first.get('version_key')}
        previous.discard(None)
        considered = 0
        for later in actor_events[first['target_actor']]:
            if (later['event_id'] == receipt['event_id'] or later['turn_id'] == receipt['turn_id']
                    or not start < _event_time(later) <= start + horizon
                    or not (previous & set(later.get('inputs', ())))):
                continue
            considered += 1
            if considered > max_fanout:
                counts['followup_motifs_capped'] += 1
                continue
            name = 'acquire_then_test' if later['operation'] in ('test', 'execute', 'install') else 'acquire_then_transform'
            yield _pattern(name, 1, [first['source_actor'], first['target_actor']],
                           events=first_support + [later], chains=[first],
                           confidence=_confidence(first_confidence, 'strict' if _authority(later) else 'candidate'),
                           interpretation='A recipient later uses the explicitly linked received version. Unverified outcomes remain candidates.')


def _under(value, root):
    a, b = urlsplit(value), urlsplit(root)
    if a.hostname != b.hostname:
        return False
    path = b.path.rstrip('/')
    return a.path.rstrip('/') == path or a.path.startswith(path + '/')


def _resource(event):
    entity = event['entity']
    if entity.startswith('git:'):
        entity = entity[4:].split('@', 1)[0]
    if entity.startswith(('gitlab:', 'github:')):
        host, path = entity.split(':', 1)
        return 'https://' + host + '.com/' + path
    if entity.startswith('file:repo:'):
        return unquote(entity[len('file:repo:'):].split(':', 1)[0])
    match = re.match(r'^file:(gitlab|github):([^:]+):', entity)
    if match:
        return 'https://' + match[1] + '.com/' + match[2]
    if entity.startswith('url:'):
        return entity[4:]
    if entity.startswith('village-project:'):
        return entity[len('village-project:'):]
    if entity.startswith(('http://', 'https://')):
        return entity
    return None


def _project(event, projects, known_at=None):
    value = _resource(event)
    if value is None:
        return None
    now = _event_time(event) if known_at is None else known_at
    matches = []
    for project in projects:
        if project.get('shared_resource') or now < _when(project['available_from']):
            continue
        if _under(value, project['site']):
            matches.append(project)
        elif (project.get('repository') and project.get('repository_available_from')
              and now >= _when(project['repository_available_from'])
              and value.rstrip('/') == project['repository'].rstrip('/')):
            matches.append(project)
    if len({p['owner'] for p in matches}) != 1:
        return None
    return min(matches, key=lambda p: (len(urlsplit(p['site']).path), p['key']))


def _validate_rules(rules):
    if rules is None:
        return [], []
    if rules.get('fit_start') != str(config.DEV_START) or rules.get('fit_end') != str(config.TRAIN_END):
        raise ValueError('Village workflow knowledge must be frozen on authorized training dates')
    projects, reviews = rules.get('projects', []), rules.get('reviews', [])
    for row in projects + reviews:
        when = timestamp(row['available_from']).astimezone(PACIFIC).date()
        if not config.DEV_START <= when <= config.TRAIN_END:
            raise ValueError('Village knowledge became available outside training')
        for name in ('evidence', 'repository_evidence', 'shared_evidence'):
            for support in row.get(name, []):
                day = timestamp(support['time']).astimezone(PACIFIC).date()
                if not config.DEV_START <= day <= config.TRAIN_END:
                    raise ValueError('Village knowledge cites non-training evidence')
        for availability, evidence in [('available_from', 'evidence'),
                                       ('repository_available_from', 'repository_evidence'),
                                       ('shared_available_from', 'shared_evidence')]:
            if row.get(availability):
                day = timestamp(row[availability]).astimezone(PACIFIC).date()
                if not config.DEV_START <= day <= config.TRAIN_END:
                    raise ValueError('Village knowledge availability is outside training')
                if row.get(evidence) and min(_when(e['time']) for e in row[evidence]) > _when(row[availability]):
                    raise ValueError('Village knowledge availability precedes supporting evidence')
    return projects, reviews


def _knowledge(row, now, names=('evidence',)):
    # Cached rules retain repeated later observations of the same association.
    # Only evidence already visible at the motif's time belongs in its support.
    return [support for name in names for support in row.get(name, [])
            if _when(support['time']) <= now]


def _tokens(text):
    return [(m.group().casefold(), m.start(), m.end()) for m in WORDS.finditer(text)]


def _recommendations(snippet):
    """Observed Village review convention: quoted Recommendation lines."""
    offset = 0
    for line in snippet['text'].splitlines(keepends=True):
        marker = RECOMMENDATION.search(line)
        if marker:
            tail = line[marker.end():]
            for match in re.finditer(r'`([^`\n]+)`', tail):
                text = match[1]
                words = _tokens(text)
                if 8 <= len(words) <= 80 and len({w[0] for w in words}) >= 5:
                    start = snippet['start'] + offset + marker.end() + match.start(1)
                    yield dict(words=tuple(w[0] for w in words), text=text,
                               source_span={'start': start, 'end': start + len(text)},
                               snippet_id=snippet['snippet_id'])
        offset += len(line)


def _inserted_literals(text):
    """Offsets of literal replacement/write payloads, excluding removed text.

    Tokenization tolerates a shell heredoc around Python and a truncated final
    window. It does not execute the command or interpret a quoted code example.
    """
    tokens = []
    try:
        for value in tokenize.generate_tokens(io.StringIO(text).readline):
            if value.type not in (tokenize.NL, token.NEWLINE, token.INDENT,
                                  token.DEDENT, token.COMMENT):
                tokens.append(value)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    offsets, total = [], 0
    for line in text.splitlines(keepends=True):
        offsets.append(total)
        total += len(line)
    for i, value in enumerate(tokens):
        if (value.type != token.NAME or value.string not in ('replace', 'write', 'write_text')
                or i < 1 or tokens[i - 1].string != '.' or i + 2 >= len(tokens)
                or tokens[i + 1].string != '('):
            continue
        j = i + 2
        if value.string == 'replace':
            if j + 2 >= len(tokens) or tokens[j].type != token.STRING or tokens[j + 1].string != ',':
                continue
            j += 2
        literal = tokens[j]
        if literal.type == token.STRING:
            yield offsets[literal.start[0] - 1] + literal.start[1], offsets[literal.end[0] - 1] + literal.end[1]


def _recommendation_support(review, access, change, snippet_index, counts, max_fanout):
    sources = [s for s in snippet_index.get((access['actor'], access['turn_id']), ())
               if s['field'] == 'o' and _when(s['time']) == _event_time(access)]
    recommendations = [r for s in sources for r in _recommendations(s)]
    if not recommendations:
        return []
    actor_snippets = snippet_index.get(access['actor'], ())
    past, candidates = [], []
    for snippet in actor_snippets:
        if snippet['field'] != 'x' or snippet.get('source_role') != 'command_text':
            continue
        when = _when(snippet['time'])
        if when < _event_time(access):
            past.append(snippet)
        elif _event_time(access) < when <= _event_time(change) and list(_inserted_literals(snippet['text'])):
            candidates.append(snippet)
    if len(candidates) > max_fanout:
        counts['review_snippet_candidates_capped'] += len(candidates) - max_fanout
        candidates = candidates[:max_fanout]
    result = []
    for rec in recommendations:
        needle = ' '.join(rec['words'])
        if any(needle in ' '.join(w[0] for w in _tokens(s['text'])) for s in past):
            counts['preexisting_review_recommendations_excluded'] += 1
            continue
        for snippet in candidates:
            # The observed Village review is applied using literal Python
            # replacements. Restrict support to the inserted side of a literal
            # replacement/write; a removed value, comment or readback is not reuse.
            inserted = list(_inserted_literals(snippet['text']))
            if not inserted:
                continue
            tokens = _tokens(snippet['text'])
            values = [w[0] for w in tokens]
            length = len(rec['words'])
            position = next((i for i in range(len(values) - length + 1)
                             if tuple(values[i:i + length]) == rec['words']
                             and any(a <= tokens[i][1] and tokens[i + length - 1][2] <= b
                                     for a, b in inserted)), None)
            if position is None:
                continue
            result.append(dict(source_snippet=rec['snippet_id'], target_snippet=snippet['snippet_id'],
                               source_span=rec['source_span'], target_span={
                                   'start': snippet['start'] + tokens[position][1],
                                   'end': snippet['start'] + tokens[position + length - 1][2]},
                               matched_tokens=length, target_turn=snippet['turn_id'], target_time=snippet['time'],
                               interpretation='New-to-reader quoted recommendation appears in a subsequent edit-like command before an observed project change. This is a candidate application, not verified file contents.'))
            break
    return result


def _village_patterns(events, rules, snippets, counts, horizon, max_fanout):
    projects, reviews = _validate_rules(rules)
    if not projects:
        return
    projected = {}
    actor_changes = collections.defaultdict(list)
    cutoff = max((_event_time(e) for e in events), default=0)
    for event in events:
        if event['status'] == 'failed':
            continue
        project = _project(event, projects)
        if project:
            projected[event['event_id']] = project
        if event['operation'] in CHANGES:
            actor_changes[event['actor']].append(event)
    snippet_index = collections.defaultdict(list)
    for snippet in snippets:
        if not in_scope(snippet['time']):
            raise ValueError('Workflow snippet outside development scope')
        snippet_index[(snippet['actor'], snippet['turn_id'])].append(snippet)
        snippet_index[snippet['actor']].append(snippet)
    for values in snippet_index.values():
        values.sort(key=lambda s: (_when(s['time']), s['snippet_id']))
    by_review = {r['url']: r for r in reviews}
    for access in events:
        if access.get('family') != 'village_review_reference' or access['status'] == 'failed':
            continue
        review = by_review.get(access['entity'].removeprefix('village-review:'))
        if not review or _event_time(access) < _when(review['available_from']) or review['reviewer'] == access['actor']:
            continue
        potential = []
        for change in actor_changes[access['actor']]:
            if (change['turn_id'] == access['turn_id'] or not _event_time(access) < _event_time(change) <= _event_time(access) + min(horizon, 2 * 86400)):
                continue
            project = _project(change, projects, known_at=cutoff)
            if not project or not _under(review['target'], project['site']):
                continue
            potential.append((change, project))
        # One earliest observed change per review access, avoiding a review that
        # precedes many routine commits becoming dozens of supposed applications.
        if potential:
            change, project = min(potential, key=lambda item: (_event_time(item[0]), item[0]['event_id']))
            knowledge_times = [review['available_from'], project['available_from']]
            if project.get('repository_available_from'):
                knowledge_times.append(project['repository_available_from'])
            available = max([_event_time(change)] + [_when(t) for t in knowledge_times])
            knowledge = _knowledge(review, available) + _knowledge(project, available, ('evidence', 'repository_evidence'))
            support_turns = {access['turn_id'], change['turn_id']} | {s['turn_id'] for s in knowledge}
            yield _pattern('village_review_then_project_change', 2, [review['reviewer'], access['actor']],
                           events=[access, change], extra_times=knowledge_times,
                           verbal_channel=True, review_url=review['url'], target_project=review['target'],
                           knowledge_evidence=knowledge, support_turns=sorted(support_turns),
                           interpretation='An attributed Village review was accessed before a change in its specifically mapped target project. Temporal/project context is a candidate, not proof the review caused the edit.')
            if _authority(change):
                reused = _recommendation_support(review, access, change, snippet_index, counts, max_fanout)
                if reused:
                    yield _pattern('village_review_recommendation_in_change', 2, [review['reviewer'], access['actor']],
                                   events=[access, change], extra_times=[r['target_time'] for r in reused] + knowledge_times,
                                   verbal_channel=True, review_url=review['url'], target_project=review['target'],
                                   knowledge_evidence=knowledge, recommendation_matches=reused,
                                   support_turns=sorted(support_turns | {r['target_turn'] for r in reused}),
                                   support_snippets=sorted({r[k] for r in reused for k in ('source_snippet', 'target_snippet')}),
                                   interpretation='A new-to-reader quoted recommendation reappears in an edit-like command before a successful target-project change. The operation is observed; exact final file content/application remains unverified.')
    # Version-linked production/publication followed by another actor's site
    # access uses the Village deployment convention. The access has no immutable
    # deployed-version key, therefore the complete motif remains a candidate.
    produced = collections.defaultdict(list)
    published = collections.defaultdict(dict)
    seen_access = set()
    ordered = sorted(events, key=lambda e: (_event_time(e), 0 if e['operation'] in PRODUCES else 1, e['event_id']))
    for event in ordered:
        if event['status'] == 'failed':
            continue
        project = projected.get(event['event_id'])
        if not project:
            continue
        if _authority(event) and event['operation'] in PRODUCES:
            for key in _new_outputs(event):
                produced[(event['actor'], key)].append(event)
        if _authority(event) and event['operation'] == 'publish':
            candidates = []
            for key in event.get('inputs', ()):
                for producer in produced[(event['actor'], key)]:
                    if _event_time(producer) <= _event_time(event):
                        candidates.append(producer)
            if candidates:
                producer = max(candidates, key=lambda e: (_event_time(e), e['event_id']))
                published[project['key']][event['actor']] = (producer, event)
        if event['family'] != 'village_project_access' or event['operation'] != 'read':
            continue
        for actor, (producer, publication) in sorted(published[project['key']].items()):
            if actor == event['actor'] or not _event_time(publication) < _event_time(event) <= _event_time(publication) + min(horizon, 86400):
                continue
            identity = (publication['event_id'], event['actor'], event['day'])
            if identity in seen_access:
                continue
            seen_access.add(identity)
            yield _pattern('village_publish_then_external_access', 2, [actor, event['actor']],
                           events=[producer, publication, event], project=project['key'],
                           knowledge_evidence=_knowledge(project, _event_time(event), ('evidence', 'repository_evidence')),
                           interpretation='A version-linked project production/publication precedes another actor accessing the associated Village site. Its served version is unobserved, so this is a deployment/access candidate.')


def _script_patterns(events, rules, snippets, counts, horizon):
    if rules is None:
        return
    specifications = rules.get('script_workflows', SCRIPT_WORKFLOW_RULES)
    cutoff = max([_event_time(e) for e in events] + [_when(s['time']) for s in snippets], default=0)
    turns = collections.defaultdict(list)
    for snippet in snippets:
        turns[(snippet['actor'], snippet['turn_id'])].append(snippet)
    observations = []
    for (actor, turn), values in turns.items():
        if len({_when(s['time']) for s in values}) != 1:
            counts['inconsistent_script_turn_times'] += 1
            continue
        fields = {}
        for field in ('x', 'o'):
            choices = [s for s in values if s['field'] == field]
            if choices:
                full = [s for s in choices if s.get('workflow_only')]
                fields[field] = max(full or choices, key=lambda s: len(s['text']))
        observations.append(dict(actor=actor, turn_id=turn, time=values[0]['time'], fields=fields))
    observations.sort(key=lambda r: (_when(r['time']), r['turn_id']))
    by_turn = {(r['actor'], r['turn_id']): r for r in observations}
    for spec in specifications:
        for value in [spec['available_from']] + [e['time'] for e in spec['evidence']]:
            day = timestamp(value).astimezone(PACIFIC).date()
            if not config.DEV_START <= day <= config.TRAIN_END:
                raise ValueError('Script workflow convention must be frozen on training evidence')
        if max((_when(e['time']) for e in spec['evidence']), default=0) > _when(spec['available_from']):
            raise ValueError('Script convention availability precedes supporting evidence')
        if cutoff < _when(spec['available_from']):
            continue
        path = spec['relative_path']
        workdirs = '|'.join(re.escape(p) for p in spec['working_directories'])
        scoped = re.compile(r'(?m)^\s*cd\s+[\'"]?(?:' + workdirs + r')[\'"]?\s*(?:&&|$)')
        read = re.compile(r'(?m)^\s*(?:cat|head|tail)\s+(?:-[\w]+\s+)*' + re.escape(path) + r'(?:\s|$)')
        execute = re.compile(r'(?m)(?:^|&&)\s*python(?:3(?:\.\d+)?)?\s+' + re.escape(path) + r'(?:\s|$)')
        producers = [e for e in events if e['entity'] == spec['repository']
                     and e['operation'] == 'generate' and _authority(e) and _new_outputs(e)]
        publications = [e for e in events if e['entity'] == spec['repository']
                        and e['operation'] == 'publish' and _authority(e)]
        for producer in producers:
            source = by_turn.get((producer['actor'], producer['turn_id']))
            if not source or path not in source['fields'].get('o', {}).get('text', ''):
                continue
            for publication in publications:
                keys = _new_outputs(producer) & set(publication.get('inputs', ()))
                if (publication['actor'] != producer['actor'] or not keys
                        or not _event_time(producer) <= _event_time(publication) <= _event_time(producer) + horizon):
                    continue
                hashes = [key.rsplit('@', 1)[-1] for key in keys]
                for acquisition in observations:
                    when = _when(acquisition['time'])
                    cmd = acquisition['fields'].get('x', {}).get('text', '')
                    output = acquisition['fields'].get('o', {}).get('text', '')
                    if (acquisition['actor'] == producer['actor'] or not _event_time(publication) < when <= _event_time(publication) + horizon
                            or not scoped.search(cmd) or not re.search(r'(?m)^\s*git\s+pull\b', cmd)
                            or path not in output or 'Fast-forward' not in output
                            or not any(re.search(r'\bUpdating\s+[0-9a-f]{7,40}\.\.' + re.escape(h) + r'\b', output) for h in hashes)):
                        continue
                    reads = []
                    for later in observations:
                        if later['actor'] != acquisition['actor'] or not when < _when(later['time']) <= when + min(horizon, 86400):
                            continue
                        command = later['fields'].get('x', {}).get('text', '')
                        result = later['fields'].get('o', {}).get('text', '')
                        if not scoped.search(command):
                            continue
                        if read.search(command) and spec['read_marker'] in result:
                            reads.append(later)
                        if (not reads or not execute.search(command)
                                or not all(marker in result for marker in spec['execution_markers'])
                                or re.search(r'Traceback|SyntaxError|No such file|command not found', result)):
                            continue
                        observed = [source, acquisition, reads[0], later]
                        if len({r['turn_id'] for r in observed}) != 4:
                            continue
                        evidence = spec['evidence']
                        yield _pattern('village_script_acquire_read_execute', 2,
                                       [producer['actor'], acquisition['actor']], events=[producer, publication],
                                       extra_times=[r['time'] for r in observed] + [spec['available_from']],
                                       support_turns=sorted({r['turn_id'] for r in observed} |
                                                            {publication['turn_id']} | {r['turn_id'] for r in evidence}),
                                       support_snippets=sorted({s['snippet_id'] for r in observed for s in r['fields'].values()}),
                                       knowledge_evidence=evidence, rule_id=spec['rule_id'], repository=spec['repository'],
                                       relative_path=path, matched_versions=sorted(keys),
                                       interpretation='Observed publication, scoped pull receipt with matching abbreviated version and full path, script read, and characteristic execution output. Frozen Village working-directory/output conventions resolve an otherwise local trace; this remains a candidate, not immutable execution provenance.')
                        break


def build_workflows(events, chains, village_rules=None, snippets=(),
                    max_horizon_days=7, max_fanout=32, max_patterns=20000):
    """Return (patterns, diagnostics), without labels or runtime model calls.

    Events use the existing make_event schema. Optional snippets enable the
    training-observed Village review and script workflows. Quoted recommendation
    checks apply only within an attributed review and its target-project change.
    """
    if not 0 < max_horizon_days <= 31 or max_fanout < 1 or max_patterns < 1:
        raise ValueError('Workflow bounds must be positive; horizon must be at most 31 days')
    events = [_scoped_event(event) for event in events]
    lookup = {}
    for event in events:
        identifier = event['event_id']
        if identifier in lookup and lookup[identifier] != event:
            raise ValueError('Conflicting duplicate workflow event id')
        lookup[identifier] = event
    events = sorted(lookup.values(), key=lambda e: (_event_time(e), e['event_id']))
    scoped_snippets = []
    for snippet in snippets:
        if not in_scope(snippet['time']):
            raise ValueError('Workflow snippet outside development scope')
        scoped_snippets.append(snippet)
    counts = collections.Counter()
    horizon = max_horizon_days * 86400
    generators = (_generic_patterns(events, chains, lookup, counts, horizon, max_fanout),
                  _village_patterns(events, village_rules, scoped_snippets, counts, horizon, max_fanout),
                  _script_patterns(events, village_rules, scoped_snippets, counts, horizon))
    patterns = {}
    for generator in generators:
        for pattern in generator:
            if pattern is None:
                continue
            if len(patterns) >= max_patterns and pattern['pattern_id'] not in patterns:
                counts['patterns_capped'] += 1
                continue
            patterns[pattern['pattern_id']] = pattern
    ordered = sorted(patterns.values(), key=lambda p: (_when(p['available_time']), p['pattern_id']))
    return ordered, dict(events=len(events), patterns=len(ordered),
                         by_pattern=dict(collections.Counter(p['pattern'] for p in ordered)),
                         by_confidence=dict(collections.Counter(p['confidence'] for p in ordered)),
                         diagnostics=dict(counts), max_horizon_days=max_horizon_days,
                         max_fanout=max_fanout, max_patterns=max_patterns,
                         interpretation='Dependency-backed motifs and separately labeled content/workflow candidates. Latest support time governs availability; no counterfactual causal effect is claimed.')
