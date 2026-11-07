"""Directed, auditable dependency candidates, with explicit evidence levels."""
import collections
import datetime as dt
import hashlib
import json
import re

from .provenance_common import timestamp

PRODUCES = {'generate', 'transform', 'incorporate', 'publish', 'review'}
CONSUMES = {'acquire', 'read', 'incorporate', 'transform', 'test', 'execute', 'install', 'generate'}
DOWNSTREAM = {'incorporate', 'transform', 'test', 'execute', 'install', 'generate'}
GIT_KEY = re.compile(r'^(git:.+)@([0-9a-f]{7,40})$')
CONTENT_KEY = re.compile(r'^file:.*@((?:[a-z0-9_-]*sha256|md5):[a-f0-9]{32,64})$')


def version_keys(event):
    keys = event.get('inputs', []) + event.get('outputs', [])
    version = event.get('version')
    if isinstance(version, str) and '@' in version and version.startswith(('git:', 'file:', 'content:', 'doc:')):
        keys = keys + [version]
    return list(dict.fromkeys(keys))


def content_key(key):
    if key.startswith('content:'):
        return key
    match = CONTENT_KEY.fullmatch(key)
    return 'content:' + match[1] if match else None


def training_common_content(events, train_end, min_actors=4):
    """Freeze popularity controls before validation, independent of labels."""
    actors = collections.defaultdict(set)
    for event in events:
        if event['day'] <= train_end:
            for key in version_keys(event):
                digest = content_key(key)
                if digest:
                    actors[digest].add(event['actor'])
    return {key for key, seen in actors.items() if len(seen) >= min_actors}


def build_links(events, common_content=(), max_delay_days=7, include_exposure=True):
    """Link only earlier observed producers to later actors.

    Full hashes and immutable content IDs are preferred. Abbreviated Git hashes
    resolve only within a repository and against full hashes observed by the
    consumer's time. Conflicts downgrade the evidence rather than picking a hash.
    No relation inferred solely from API observation attributes production to
    the observer. Automation relations remain in the event graph separately.
    """
    common_content = set(common_content)
    producers = collections.defaultdict(list)
    prefix_producers = collections.defaultdict(set)
    full_hashes = collections.defaultdict(set)
    aliases = collections.defaultdict(dict)
    ancestry = collections.defaultdict(dict)
    exposures = collections.defaultdict(dict)
    exposure_seen, chain_seen = set(), {}
    chains = []
    diagnostics = collections.Counter()
    horizon = dt.timedelta(days=max_delay_days)

    def register(key):
        match = GIT_KEY.fullmatch(key)
        if match and len(match[2]) == 40:
            full_hashes[(match[1], match[2][:7])].add(match[2])

    def origins(key):
        if key.startswith('content:'):
            if key in common_content or len({p['actor'] for p in producers[key]}) != 1:
                diagnostics['ambiguous_or_popular_content'] += 1
                return []
            return [(p, 'same_content', key) for p in producers[key]]
        match = GIT_KEY.fullmatch(key)
        if match:
            repo, sha = match.groups()
            known = {h for h in full_hashes[(repo, sha[:7])] if h.startswith(sha)}
            if len(sha) == 40:
                known = {sha}
            matches = []
            for candidate in sorted(prefix_producers[(repo, sha[:7])]):
                other = GIT_KEY.fullmatch(candidate)[2]
                if sha.startswith(other) or other.startswith(sha):
                    # A short producer prefix is ambiguous if any known full SHA
                    # sharing that producer prefix differs from the target.
                    possibilities = {h for h in full_hashes[(repo, other[:7])] if h.startswith(other)}
                    exact = len(sha) == len(other) == 40 and sha == other
                    unique = len(known) == 1 and len(possibilities) == 1 and known == possibilities
                    confidence = 'exact' if exact else ('observed_prefix_resolution' if unique else 'unresolved_or_ambiguous_prefix')
                    matches.extend((p, confidence, candidate) for p in producers[candidate])
            return matches
        found = [(p, 'exact', key) for p in producers[key]]
        digest = content_key(key)
        if digest and digest not in common_content:
            origin_actors = {p['actor'] for p in producers[digest]}
            # Multiple independent producers of identical bytes can be templates.
            if len(origin_actors) == 1:
                found.extend((p, 'same_content', digest) for p in producers[digest])
            elif len(origin_actors) > 1:
                diagnostics['ambiguous_common_source'] += 1
        elif digest:
            diagnostics['popular_content_excluded'] += 1
        return found

    def emit(source, target, key, level, confidence, reason, relation_events=(), consumed_key=None, consumption_id=None):
        if source['actor'] == target['actor']:
            return
        elapsed = timestamp(target['time']) - timestamp(source['time'])
        # Exact version dependencies can persist for weeks. The temporal cutoff
        # applies only to broad shared exposure, not to recorded version lineage.
        if elapsed <= dt.timedelta(0) or (level == 1 and elapsed > horizon):
            return
        identity = (source['actor'], target['actor'], consumption_id or consumed_key or key, target['turn_id'])
        if identity in chain_seen:
            prior = chains[chain_seen[identity]]
            rank = {'exact': 5, 'observed_prefix_resolution': 4, 'same_content': 3, 'recorded_build_lineage': 2}
            if (prior['level'], rank.get(prior['identity_confidence'], 0)) >= (level, rank.get(confidence, 0)):
                return
        chain = dict(source_actor=source['actor'], target_actor=target['actor'],
                     source_event=source['event_id'], target_event=target['event_id'],
                     source_turn=source['turn_id'], target_turn=target['turn_id'],
                     source_time=source['time'], target_time=target['time'], day=target['day'],
                     family=target['family'], version_key=key, level=level,
                     identity_confidence=confidence, reason=reason,
                     delay_seconds=elapsed.total_seconds(),
                     source_status=source['status'], target_status=target['status'])
        chain['relation_events'] = list(relation_events)
        chain['consumed_key'] = consumed_key or key
        chain['verbal_channel'] = bool(source.get('verbal_channel') or target.get('verbal_channel'))
        chain['chain_id'] = hashlib.sha256(json.dumps(chain, sort_keys=True).encode()).hexdigest()[:24]
        if identity in chain_seen:
            chains[chain_seen[identity]] = chain
        else:
            chain_seen[identity] = len(chains)
            chains.append(chain)

    ordered = sorted(events, key=lambda e: (e['time'], e['actor'], e['turn_id'], e['event_id']))
    for event in ordered:
        if event.get('relation_authority') and event['status'] == 'success':
            for key in version_keys(event):
                register(key)
        for relation in event.get('relations', []):
            if relation.get('kind') == 'same_content':
                source, target = relation.get('source'), relation.get('target')
                if isinstance(source, str) and isinstance(target, str):
                    if event.get('relation_authority') and event['status'] == 'success':
                        aliases[source].setdefault(target, event['event_id'])
                        aliases[target].setdefault(source, event['event_id'])
                    else:
                        ancestry[source].setdefault(target, (event['event_id'], 'unverified_identity'))
                        ancestry[target].setdefault(source, (event['event_id'], 'unverified_identity'))
            elif relation.get('kind') in ('derived_from', 'revision_of'):
                source, target = relation.get('source'), relation.get('target')
                if isinstance(source, str) and isinstance(target, str):
                    family = event['family'] if event.get('relation_authority') and event['status'] == 'success' else 'unverified_relation'
                    previous = ancestry[source].get(target)
                    if previous is None or previous[1].startswith('unverified'):
                        ancestry[source][target] = (event['event_id'], family)
        candidates = list(event.get('inputs', []))
        if event['operation'] in ('observe', 'snapshot'):
            candidates += version_keys(event)
        for key in dict.fromkeys(candidates):
            alias_paths = {key: ()}
            for _ in range(2):
                for child, path in list(alias_paths.items()):
                    for alias, evidence_id in aliases[child].items():
                        alias_paths.setdefault(alias, path + (evidence_id,))
            equivalents = set(alias_paths)
            digests = {content_key(alias) for alias in equivalents} - {None}
            origin_actors = {p['actor'] for alias in equivalents for p, _, _ in origins(alias)}
            generic_safe = not (digests & common_content) and len(origin_actors) == 1
            original_git = GIT_KEY.fullmatch(key)
            permitted = {key}
            for alias in sorted(equivalents - {key}):
                alias_git = GIT_KEY.fullmatch(alias)
                if generic_safe or (original_git and alias_git and original_git[1] == alias_git[1]):
                    permitted.add(alias)
            equivalents = permitted
            consumption_id = key
            if generic_safe and digests:
                consumption_id = sorted(digests)[0]
            elif original_git:
                hashes = {h for h in full_hashes[(original_git[1], original_git[2][:7])] if h.startswith(original_git[2])}
                if len(hashes) == 1:
                    consumption_id = original_git[1] + '@' + next(iter(hashes))
            found = []
            for alias in sorted(equivalents):
                for producer, confidence, matched in origins(alias):
                    if alias != key and not original_git:
                        confidence = 'same_content'
                    found.append((producer, confidence, matched, alias_paths[alias], ()))
            pending = [(alias, alias_paths[alias], ()) for alias in sorted(equivalents)]
            visited = set(equivalents)
            while pending:
                child, path, families = pending.pop()
                if len(path) >= 4:
                    continue
                for parent, (evidence_id, family) in ancestry[child].items():
                    if parent in visited:
                        continue
                    visited.add(parent)
                    next_path, next_families = path + (evidence_id,), families + (family,)
                    for producer, confidence, matched in origins(parent):
                        if confidence != 'unresolved_or_ambiguous_prefix':
                            confidence = 'recorded_build_lineage' if all(f in ('ci', 'package', 'release', 'deployment') for f in next_families) else 'ancestry_candidate'
                        found.append((producer, confidence, matched, next_path, next_families))
                    pending.append((parent, next_path, next_families))
            for producer, confidence, matched_key, path, families in found:
                # Git itself resolves abbreviations during successful transport
                # or checkout. Keep this explicitly weaker than a full hash and
                # require the same independently established global repository.
                match = GIT_KEY.fullmatch(key)
                if (confidence == 'unresolved_or_ambiguous_prefix' and not path
                        and matched_key == key and match and 7 <= len(match[2]) < 40
                        and match[1].startswith(('git:gitlab:', 'git:github:'))
                        and not full_hashes[(match[1], match[2][:7])]
                        and len(prefix_producers[(match[1], match[2][:7])]) == 1
                        and producer.get('method') == 'commit'
                        and event.get('method') in ('fetch', 'pull', 'pull-rebase', 'rebase', 'merge', 'cherry-pick', 'checkout', 'switch', 'reset')):
                    confidence = 'git_scoped_abbreviation'
                level = 2
                if (event['status'] == 'success' and event.get('relation_authority')
                        and producer.get('relation_authority') and event['operation'] in CONSUMES
                        and confidence not in ('unresolved_or_ambiguous_prefix', 'ancestry_candidate')):
                    level = 4 if event['operation'] in DOWNSTREAM else 3
                emit(producer, event, matched_key, level, confidence,
                     'observed downstream operation' if level == 4 else
                     ('successful version acquisition' if level == 3 else 'subsequent identifier observation or attempt'), path, key, consumption_id)
        if include_exposure:
            for actor, earlier in exposures[event['entity']].items():
                identity = (actor, event['actor'], event['entity'], event['day'])
                if actor != event['actor'] and identity not in exposure_seen:
                    emit(earlier, event, event['entity'], 1, 'shared_entity', 'ordered shared exposure')
                    exposure_seen.add(identity)
            exposures[event['entity']][event['actor']] = event
        if event['status'] != 'success' or event['operation'] not in PRODUCES:
            continue
        if event['operation'] == 'review' and not event.get('verbal_channel'):
            continue
        for key in event.get('outputs', []):
            if key in event.get('inputs', []):
                continue  # Republishing/adopting an unchanged input is not its origin.
            if not any(p['actor'] == event['actor'] for p in producers[key]):
                producers[key].append(event)
                match = GIT_KEY.fullmatch(key)
                if match:
                    prefix_producers[(match[1], match[2][:7])].add(key)
                digest = content_key(key)
                if digest and digest != key:
                    producers[digest].append(event)
    diagnostics['events'] = len(ordered)
    diagnostics['chains'] = len(chains)
    diagnostics.update({'level_' + str(i): sum(c['level'] == i for c in chains) for i in range(1, 5)})
    diagnostics['confirmed_with_git_abbreviation'] = sum(c['level'] >= 3 and c['identity_confidence'] == 'git_scoped_abbreviation' for c in chains)
    diagnostics['confirmed_without_git_abbreviation'] = sum(c['level'] >= 3 and c['identity_confidence'] != 'git_scoped_abbreviation' for c in chains)
    diagnostics['invalid_confirmed_status'] = sum(c['level'] >= 3 and (c['source_status'] != 'success' or c['target_status'] != 'success') for c in chains)
    return chains, dict(diagnostics)


def directed_features(chains):
    """Count distinct evidence chains by directed pair/day; never symmetrize here."""
    rows = collections.defaultdict(collections.Counter)
    for chain in chains:
        key = (chain['source_actor'], chain['target_actor'], chain['day'])
        row = rows[key]
        prefix = 'prov_verbal_' if chain.get('verbal_channel') else 'prov_'
        row[prefix + 'level_' + str(chain['level'])] += 1
        if chain['level'] >= 3:
            row[prefix + 'confirmed_' + chain['family']] += 1
            if chain['identity_confidence'] != 'git_scoped_abbreviation':
                row[prefix + 'strict_identity'] += 1
            if chain['identity_confidence'] == 'same_content':
                row[prefix + 'cross_artifact'] += 1
    return [dict(source=a, target=b, day=day, **values) for (a, b, day), values in sorted(rows.items())]
