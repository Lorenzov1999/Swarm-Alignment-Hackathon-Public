"""Descriptive lifecycle-aware timing controls for observed provenance chains.

These are sensitivity diagnostics, not independent null trials or causal tests.
All activity in an actor/day block moves together, within its original Pacific
day. Thus source versions, actions, outcomes, within-block delays, and the data
split are retained. Known production cannot be moved after an observed use.
"""
import collections
import datetime as dt
import random

from . import config
from .provenance_common import PACIFIC, in_scope, timestamp
from .provenance_links import GIT_KEY, PRODUCES, build_links, content_key, version_keys

DEFAULT_OFFSETS = (-3600, -1800, 1800, 3600)


def phase(day):
    if day <= config.TRAIN_END.isoformat():
        return 'train'
    if day >= config.VALID_START.isoformat():
        return 'validation'
    return 'gap'


class Availability:
    """Index observable birth/use constraints, excluding inconsistent birth claims.

    A version seen before any recorded generation already existed; that recorded
    generation is not its birth. Such identities cannot supply availability
    bounds. Unobserved creation times are left unknown, never invented.
    """

    def __init__(self, events):
        self.events = events
        self.times = [timestamp(event['time']) for event in events]
        if any(not in_scope(event['time']) for event in events):
            raise ValueError('Controls require the authorized development slice')
        if any(when.astimezone(PACIFIC).date().isoformat() != event['day']
               for when, event in zip(self.times, events)):
            raise ValueError('Event day must match its Pacific timestamp')
        full = collections.defaultdict(set)
        for event in events:
            for key in version_keys(event):
                match = GIT_KEY.fullmatch(key)
                if match and len(match[2]) == 40:
                    full[(match[1], match[2][:7])].add(match[2])

        def canonical(key):
            match = GIT_KEY.fullmatch(key)
            if match and len(match[2]) < 40:
                candidates = {sha for sha in full[(match[1], match[2][:7])] if sha.startswith(match[2])}
                if len(candidates) == 1:
                    return match[1] + '@' + next(iter(candidates))
            return key

        self.keys = []
        self.producers, self.users = collections.defaultdict(set), collections.defaultdict(set)
        for index, event in enumerate(events):
            keys = {canonical(key) for key in version_keys(event)}
            keys.update(digest for digest in (content_key(key) for key in list(keys)) if digest)
            self.keys.append(keys)
            for key in keys:
                self.users[key].add(index)
            if event['status'] == 'success' and event['operation'] in PRODUCES:
                for key in set(event.get('outputs', ())) - set(event.get('inputs', ())):
                    normalized = canonical(key)
                    self.producers[normalized].add(index)
                    digest = content_key(normalized)
                    if digest:
                        self.producers[digest].add(index)
        self.preexisting = set()
        for key, producers in self.producers.items():
            if min(self.times[i] for i in self.users[key]) < min(self.times[i] for i in producers):
                self.preexisting.add(key)
        self.known = set(self.producers) - self.preexisting

    def violations(self, times, keys=None):
        bad = []
        for key in self.known if keys is None else self.known.intersection(keys):
            first_production = min(times[i] for i in self.producers[key])
            if any(times[i] < first_production for i in self.users[key]):
                bad.append(key)
        return bad


def shift_actor_day_blocks(events, rng, offsets_seconds=DEFAULT_OFFSETS, availability=None):
    """Return shifted copies and feasibility diagnostics; never mutate input events."""
    offsets = tuple(dict.fromkeys(float(x) for x in offsets_seconds if x != 0))
    if not offsets:
        raise ValueError('At least one nonzero shift is required')
    availability = availability or Availability(events)
    times = availability.times.copy()
    blocks = collections.defaultdict(list)
    for index, event in enumerate(events):
        blocks[(event['actor'], event['day'])].append(index)
    order = sorted(blocks)
    rng.shuffle(order)
    stats = collections.Counter(total_blocks=len(order), eligible_blocks=0, moved_blocks=0,
                                boundary_ineligible_blocks=0, availability_rejected_blocks=0,
                                availability_rejected_proposals=0, moved_events=0)
    moved_by_phase = collections.Counter()
    for actor, day in order:
        indexes = blocks[(actor, day)]
        feasible = []
        for offset in offsets:
            delta = dt.timedelta(seconds=offset)
            if all((times[i] + delta).astimezone(PACIFIC).date().isoformat() == day for i in indexes):
                feasible.append(offset)
        if not feasible:
            stats['boundary_ineligible_blocks'] += 1
            continue
        stats['eligible_blocks'] += 1
        rng.shuffle(feasible)
        affected = set().union(*(availability.keys[i] for i in indexes))
        original = {i: times[i] for i in indexes}
        moved = False
        for offset in feasible:
            delta = dt.timedelta(seconds=offset)
            for i in indexes:
                times[i] = original[i] + delta
            if availability.violations(times, affected):
                stats['availability_rejected_proposals'] += 1
                for i in indexes:
                    times[i] = original[i]
                continue
            moved = True
            stats['moved_blocks'] += 1
            stats['moved_events'] += len(indexes)
            moved_by_phase[phase(day)] += 1
            break
        if not moved:
            stats['availability_rejected_blocks'] += 1
    violations = availability.violations(times)
    if violations:
        raise AssertionError('A shifted stream violated known availability constraints')
    shifted = [dict(event, time=when.isoformat()) for event, when in zip(events, times)]
    diagnostics = dict(stats, moved_blocks_by_phase=dict(moved_by_phase),
                       known_availability_keys=len(availability.known),
                       preexisting_keys_without_birth_bound=len(availability.preexisting),
                       keys_without_observed_production=len(set(availability.users) - set(availability.producers)),
                       availability_violations=0,
                       unchanged=stats['moved_blocks'] == 0)
    return shifted, diagnostics


def confirmed_edges(chains):
    return {(chain['source_actor'], chain['target_actor'], chain['source_event'],
             chain['target_event'], chain['version_key'], chain['level'])
            for chain in chains if chain['level'] >= 3}


def time_shift_control(events, common_content=(), repetitions=5, seed=20261106,
                       offsets_seconds=DEFAULT_OFFSETS):
    """Rebuild links after constrained shifts; report degeneracy instead of a p-value."""
    if repetitions < 0:
        raise ValueError('repetitions must be nonnegative')
    offsets_seconds = tuple(offsets_seconds)
    availability = Availability(events)
    original, original_diagnostics = build_links(events, common_content, include_exposure=False)
    real_edges = confirmed_edges(original)
    rng = random.Random(seed)
    draws = []
    for rep in range(repetitions):
        shifted, diagnostics = shift_actor_day_blocks(events, rng, offsets_seconds, availability)
        chains, _ = build_links(shifted, common_content, include_exposure=False)
        edges = confirmed_edges(chains)
        draws.append(dict(repetition=rep + 1, **diagnostics,
                          confirmed_chains=sum(c['level'] >= 3 for c in chains),
                          changed_confirmed_edges=len(edges.symmetric_difference(real_edges))))
    counts = [draw['confirmed_chains'] for draw in draws]
    return dict(repetitions=repetitions, seed=seed, offsets_seconds=list(offsets_seconds),
                availability_reference='whole authorized development slice; retrospective sensitivity diagnostic only',
                original_confirmed_chains=sum(original_diagnostics.get('level_' + str(i), 0) for i in (3, 4)),
                draws=draws,
                no_blocks_moved=bool(draws) and all(draw['unchanged'] for draw in draws),
                count_degenerate=bool(counts) and len(set(counts)) == 1,
                edge_degenerate=bool(draws) and all(draw['changed_confirmed_edges'] == 0 for draw in draws),
                interpretation='Descriptive timing sensitivity, not a significance or causal test. Whole actor/day blocks stay in their original Pacific day and phase. Version identities, operations, outcomes and within-block order are unchanged. Constraints use observed production and available identity matches; unknown creation times are not validated. Preexisting versions are not assigned a false birth time.')


def direction_diagnostic(chains):
    """Check structural ordering, independently of request direction or labels."""
    problems = collections.Counter(backwards=0, simultaneous=0, same_actor=0,
                                   inconsistent_delay=0, confirmed_without_success=0)
    for chain in chains:
        elapsed = (timestamp(chain['target_time']) - timestamp(chain['source_time'])).total_seconds()
        problems['backwards'] += elapsed < 0
        problems['simultaneous'] += elapsed == 0
        problems['same_actor'] += chain['source_actor'] == chain['target_actor']
        problems['inconsistent_delay'] += abs(elapsed - chain.get('delay_seconds', elapsed)) > 1e-6
        problems['confirmed_without_success'] += chain['level'] >= 3 and (
            chain.get('source_status') != 'success' or chain.get('target_status') != 'success')
    return dict(chains_checked=len(chains), violations=dict(problems),
                valid=not any(problems.values()),
                interpretation='Implementation invariant: directed chains must point forward between distinct actors. Passing is not evidence of causal direction or successful request fulfillment.')
