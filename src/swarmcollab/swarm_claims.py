"""Exploratory claim candidates and a fixed, body-level diagnostic audit.

These are bounded lexical/structural signals, not observations of cooperation,
actors, execution, success, or time. Unmarked text has unknown meaning. Exact
whole-payload hashing is administrative duplicate accounting, never near-copy
or passage matching. The module has no source loading or network access.
"""
from collections import defaultdict
import hashlib
import re


SCHEMA_VERSION = 1
AUDIT_SEED = 20261107
AUDIT_PER_STRATUM = 24
ARTIFACT_ROLES = frozenset({'summary', 'description', 'path', 'version'})
CATEGORIES = ('role_or_task_allocation', 'publication_intent', 'result_request_intent')
# A match is a candidate for contextual review. Bounded gaps deliberately avoid
# pretending that a general keyword, POST, or common resource is a joint action.
ACTOR = r'(?:teammates?|other\s+(?:agents?|workers?)|another\s+(?:agent|worker))'
WORK = r'(?:own(?:s|ed)?|handl(?:e|es|ing)|tak(?:e|es|ing)\s+(?:care\s+of|on)|work(?:s|ing)?\s+on|responsible\s+for)'
GAP = r'[^\r\n.;{}]{1,120}?'
RESULT = r'(?:results?|findings?|reports?|summar(?:y|ies)|artifacts?|deliverables?|conclusions?)'
RULES = {
    'explicit_other_role': ('role_or_task_allocation', rf'\b{ACTOR}\s+(?:(?:is|are|will|can)\s+)?{WORK}\b'),
    'delegate_to_recipient': ('role_or_task_allocation', rf'\b(?:assign(?:ed|ing)?|delegat(?:e|ed|ing)|hand(?:ed|ing)?\s+off)\b{GAP}\bto\s+(?:you|{ACTOR})\b'),
    'parallel_task_assignment': ('role_or_task_allocation', rf'\b(?:I|we)\s+(?:will\s+|can\s+|am\s+|are\s+)?{WORK}\b{GAP}\b(?:you|they)\s+(?:will\s+|can\s+|are\s+)?{WORK}\b'),
    'coordinator_role_claim': ('role_or_task_allocation', r'\b(?:I\s+am|we\s+are|I\x27m|acting\s+as|my\s+role\s+is)\s+(?:(?:a|the)\s+)?(?:coordinator|task\s+allocator)\b'),
    'publication_of_result': ('publication_intent', rf'\b(?:publish(?:ed|ing)?|upload(?:ed|ing)?|shar(?:e|ed|ing)|post(?:ed|ing)?|submit(?:ted|ting)?)\s+(?:(?:my|our|your|the|a|these|this|final)\s+){{0,3}}{RESULT}\b'),
    'result_request': ('result_request_intent', rf'\b(?:please\s+|(?:can|could|would)\s+you\s+|I\s+need\s+you\s+to\s+)(?:send|share|provide|return|summarize|report)\b[^\r\n.;{{}}]{{0,100}}\b{RESULT}\b'),
    'send_me_results': ('result_request_intent', rf'\b(?:send|share|give)\s+(?:me|us)\s+(?:(?:your|the|a|our|final)\s+){{0,3}}{RESULT}\b'),
    'report_back_request': ('result_request_intent', r'\b(?:please\s+)?report\s+back\s+(?:with|when|once|after)\b'),
}
COMPILED = {key: (category, re.compile(pattern, re.I))
            for key, (category, pattern) in RULES.items()}


def _hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def detector_specification():
    """JSON-safe specification that can be frozen before a target census."""
    return dict(schema_version=SCHEMA_VERSION,
                rules={name: dict(category=category, regex=regex)
                       for name, (category, regex) in RULES.items()},
                regex_flags=['IGNORECASE'], categories=list(CATEGORIES),
                artifact_roles=sorted(ARTIFACT_ROLES),
                root_evidence='Payload views and uniquely exactly contained child views may enter payload candidates. '
                              'Ambiguous/noncontained children remain association-only evidence.',
                interpretation='Automatic candidates require review. Intent, quoted examples, negation, '
                               'and code/artifact text may match. No execution or cooperation label.')


def audit_plan():
    """Predeclared design; no population statistics or outcomes are inspected."""
    return dict(schema_version=SCHEMA_VERSION, seed=AUDIT_SEED,
                planned_sample_size=4 * AUDIT_PER_STRATUM,
                per_stratum=AUDIT_PER_STRATUM,
                stratification=['operation_bearing_payload', 'artifact_view_available'],
                population='All exact-unique payload bodies except every body represented among the '
                           'prior narrow development candidates; new detector flags never affect selection.',
                canonical_record='Lexicographically smallest record ID among each exact payload-body group.',
                selection='Rank exact bodies within each stratum by SHA-256(seed:stratum:bodyhash), '
                          'breaking ties by body hash; select the first 24, cap at population size, '
                          'with no redistribution of unused quota.',
                inclusion_probability='min(24, stratum_population) / stratum_population',
                inference='Diagnostic audit of unmarked text; not a prevalence estimate or a claim '
                          'that zero observed positives establishes absence. Annotations can remain unknown.',
                child_views='Artifact-view availability includes root fields and uniquely contained child fields. '
                            'Noncontained or ambiguous children cannot alter root strata. Exact-body flags union member views.')


def detect_claims(text):
    """Return literal offsets and rule names, with semantics left unresolved."""
    if not isinstance(text, str):
        raise ValueError('Claim text must be a string')
    matches = [dict(category=category, rule=rule, start=match.start(), end=match.end(),
                    literal=match.group(), status='candidate_requires_review')
               for rule, (category, regex) in COMPILED.items()
               for match in regex.finditer(text)]
    return sorted(matches, key=lambda item: (item['start'], item['end'], item['rule']))


def _boolean(value, field):
    if type(value) is not bool:
        raise ValueError(field + ' must be an explicit boolean')
    return value


def _digest(value, field):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(field + ' must be a lowercase SHA-256 digest')
    return value


def select_unmarked_audit(payload_rows, development_hashes=()):
    """Select roots only; return IDs/weights separately from blank annotations.

    Input rows must bind each payload ID to its exact body hash and have explicit
    syntax support and structural artifact-view availability flags. New claim
    decisions never affect selection. Syntax support must agree within duplicate
    bodies; artifact availability unions eligible views across members.
    """
    excluded = {_digest(value, 'development hash') for value in development_hashes}
    groups, identifiers = defaultdict(list), set()
    fields = ('operation_bearing_payload', 'artifact_view_available')
    for original in payload_rows:
        row = dict(original)
        identifier = row.get('record_id')
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('Payload record IDs must be unique nonempty strings')
        identifiers.add(identifier)
        digest = _digest(row.get('payload_sha256'), 'payload_sha256')
        for field in fields:
            _boolean(row.get(field), field)
        groups[digest].append(row)
    strata = {f'syntax_{syntax}__artifact_{artifact}': []
              for syntax in ('no', 'yes') for artifact in ('no', 'yes')}
    development_count = 0
    for digest, rows in sorted(groups.items()):
        if len({row['operation_bearing_payload'] for row in rows}) != 1:
            raise ValueError('Exact duplicate bodies have inconsistent syntax support')
        if digest in excluded:
            development_count += 1
            continue
        row = min(rows, key=lambda item: item['record_id'])
        artifact_members = sorted(item['record_id'] for item in rows if item['artifact_view_available'])
        name = ('syntax_' + ('yes' if row['operation_bearing_payload'] else 'no')
                + '__artifact_' + ('yes' if artifact_members else 'no'))
        strata[name].append(dict(record_id=row['record_id'], payload_sha256=digest,
                                 duplicate_record_count=len(rows),
                                 body_member_ids=sorted(item['record_id'] for item in rows),
                                 artifact_view_member_ids=artifact_members))
    selected, populations = [], {}
    for name, population in sorted(strata.items()):
        population = [dict(**row, selection_rank_hash=_hash(str(AUDIT_SEED) + ':' + name + ':' + row['payload_sha256']))
                      for row in population]
        population.sort(key=lambda row: (row['selection_rank_hash'], row['payload_sha256']))
        n, take = len(population), min(AUDIT_PER_STRATUM, len(population))
        populations[name] = dict(population=n, selected=take,
                                inclusion_probability=take / n if n else None)
        for row in population[:take]:
            selected.append(dict(**row, stratum=name, inclusion_probability=take / n,
                                 inverse_probability_weight=n / take))
    annotation_template = blank_annotations(selected)
    return dict(plan=audit_plan(), rubric=review_rubric(), payload_records=len(identifiers), exact_unique_payloads=len(groups),
                excluded_development_bodies=development_count,
                eligible_unmarked_bodies=sum(len(rows) for rows in strata.values()),
                strata=populations, selection=selected, annotation_template=annotation_template,
                sampling_unit='Exact whole-payload body; record multiplicities are separate metadata.')


def review_rubric():
    """Rules for independent annotations, frozen before selecting audit items."""
    return dict(
        fields={
            'role_or_task_allocation': 'Visible explicit wording assigns, accepts, or describes distinct tasks or a coordination role.',
            'publication_intent': 'Visible wording claims, proposes, or requests publishing/sharing an artifact or result; not proof of receipt or use.',
            'result_request_intent': 'Visible wording requests another result, summary, report, or report-back; recipient identity is not inferred.',
            'artifact_evidence': 'Structured or artifact-like summary, description, path, or version wording is visible; not an immutable or used artifact claim.'},
        allowed_decisions=['yes', 'no_visible_evidence', 'unknown', 'unreviewed'],
        context_status=['explicit_claim_or_intent', 'generic_or_conditional_or_quoted_artifact_wording',
                        'no_visible_explicit_claim', 'unresolved_or_truncated', 'unreviewed'],
        evidence='For yes decisions record source view/record IDs, exact character spans in the inspected text, and a short rationale. '
                 'Decoded view offsets are not raw payload offsets. Review source/context, not detector flags.',
        review_extent='Record body_fully_reviewed and inspected source/view spans; partial or truncated context may require unknown.',
        negative='No visible evidence is only a statement about the supplied text; it is not a negative cooperation label.',
        actual_cooperation='Not identifiable from these claim annotations. Response containment, common resources, POST, '
                           'document upload and same-body duplicates do not establish another actor or downstream use.',
        retuning='Do not alter the frozen detector after either audit. Disagreement remains visible or is adjudicated with rationale.')


def blank_annotations(selection):
    return [dict(record_id=row['record_id'], payload_sha256=row['payload_sha256'],
                 role_or_task_allocation='unreviewed', publication_intent='unreviewed',
                 result_request_intent='unreviewed', artifact_evidence='unreviewed',
                 context_status='unreviewed', body_fully_reviewed=None, reviewed_extent=[],
                 evidence=[], rationale='') for row in selection]


def select_candidate_audit(payload_rows, development_hashes=()):
    """Separate capped audit of new candidates, with no change to primary draw.

    Ranking is deterministic seeded SHA-256 ranking of exact body identities.
    The reported inclusion probability is the nominal fraction for this fixed
    hash-sampling design, not evidence that content bodies are independent.
    """
    excluded = {_digest(value, 'development hash') for value in development_hashes}
    groups, identifiers = defaultdict(list), set()
    for original in payload_rows:
        row = dict(original)
        identifier = row.get('record_id')
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('Payload record IDs must be unique nonempty strings')
        identifiers.add(identifier)
        digest = _digest(row.get('payload_sha256'), 'payload_sha256')
        _boolean(row.get('claim_candidate'), 'claim_candidate')
        groups[digest].append(row)
    candidates = []
    for digest, rows in groups.items():
        if digest in excluded or not any(row['claim_candidate'] for row in rows):
            continue
        canonical = min(rows, key=lambda row: row['record_id'])
        candidates.append(dict(record_id=canonical['record_id'], payload_sha256=digest,
                               duplicate_record_count=len(rows),
                               body_member_ids=sorted(row['record_id'] for row in rows),
                               candidate_member_ids=sorted(row['record_id'] for row in rows if row['claim_candidate']),
                               selection_rank_hash=_hash(str(AUDIT_SEED) + ':candidate-audit:' + digest)))
    candidates.sort(key=lambda row: (row['selection_rank_hash'], row['payload_sha256']))
    n, take = len(candidates), min(len(candidates), 4 * AUDIT_PER_STRATUM)
    selected = [dict(**row, inclusion_probability=take / n,
                     inverse_probability_weight=n / take) for row in candidates[:take]]
    return dict(population=n, selected=take, seed=AUDIT_SEED, cap=4 * AUDIT_PER_STRATUM,
                sampling='Seeded SHA-256 ranking; all when population is at most the cap.',
                selection=selected, annotation_template=blank_annotations(selected),
                interpretation='Targeted new-candidate audit, separate from the fixed prior-unmarked audit; '
                               'not a prevalence, recall, cooperation, or independent-event estimate.')


def classify_payload(group, operation_row=None):
    """Classify one ``swarm_views.payload_views`` result without executing text.

    Uniquely contained views can supply decoded artifact wording. All other
    child evidence stays in a separate association table. Multiple views never
    create multiple payload units. Evidence offsets refer to the analyzed view;
    raw offsets are supplied only for exact, untransformed substrings.
    """
    payload, views = group['payload'], group['views']
    identifier = payload['record_id']
    digest = _digest(payload['payload_sha256'], 'payload_sha256')
    source = _digest(payload['source_sha256'], 'source_sha256')
    if not isinstance(identifier, str) or not identifier or payload['root_id'] != identifier:
        raise ValueError('Payload requires a consistent nonempty root ID')
    source_views = [view for view in views if view['view_kind'] == 'root_source']
    if len(source_views) > 1:
        raise ValueError('Multiple root-source views')
    root_text = source_views[0]['text'] if source_views else None
    if root_text is not None and _hash(root_text) != digest:
        raise ValueError('Root text does not match its payload hash')
    if bool(source_views) != payload['text_available']:
        raise ValueError('Root-source availability disagrees with view metadata')
    syntax = None
    if operation_row is not None:
        if operation_row['record_id'] != identifier or operation_row['payload_sha256'] != digest:
            raise ValueError('Operation coverage does not bind this payload identity')
        syntax = _boolean(operation_row['operation_bearing_payload'], 'operation_bearing_payload')
    evidence, association_evidence, artifacts, seen = [], [], [], set()
    available_artifact = False
    for view in views:
        if view['view_id'] in seen:
            raise ValueError('Duplicate view ID')
        seen.add(view['view_id'])
        if (view['root_id'] != identifier or view['payload_sha256'] != digest
                or view['source_sha256'] != source):
            raise ValueError('View identity does not match payload/source')
        text, raw, root_span = view['text'], view['source_text'], view['root_span']
        if not isinstance(text, str) or not isinstance(raw, str) or _hash(text) != view['text_sha256']:
            raise ValueError('View text is not bound to its recorded hash')
        if _hash(raw) != view['source_text_sha256']:
            raise ValueError('View source text is not bound to its recorded hash')
        exact = _boolean(view['text_is_exact_source_substring'], 'text_is_exact_source_substring')
        if exact != (text == raw):
            raise ValueError('View transform flag disagrees with text')
        eligible = root_span is not None and view['containment'] == 'unique' and not view['association_only']
        if (root_span is not None) != eligible:
            raise ValueError('Root span requires unique exact containment')
        if eligible:
            if (root_text is None or not isinstance(root_span, list) or len(root_span) != 2
                    or any(type(value) is not int for value in root_span)
                    or not 0 <= root_span[0] <= root_span[1] <= len(root_text)
                    or root_text[root_span[0]:root_span[1]] != raw):
                raise ValueError('Contained view source span does not match payload bytes')
        provenance = dict(record_id=identifier, payload_sha256=digest, view_id=view['view_id'],
                          origin_record_id=view['origin_record_id'], origin_kind=view['origin_kind'],
                          view_kind=view['view_kind'], field_role=view['field_role'],
                          source_class=view['source_class'], source_span=view['source_span'],
                          root_container_span=root_span, text_sha256=view['text_sha256'],
                          source_text_sha256=view['source_text_sha256'],
                          text_is_exact_source_substring=exact,
                          physical_evidence_id=view['physical_evidence_id'],
                          containment=view['containment'], counts_toward_payload_candidate=eligible)
        for match in detect_claims(text):
            row = dict(**provenance, **match)
            row['offset_coordinate_system'] = 'analyzed_view_text_unicode_characters'
            row['raw_payload_match_span'] = ([root_span[0] + match['start'], root_span[0] + match['end']]
                                             if eligible and exact else None)
            row['context_excerpt'] = text[max(0, match['start'] - 80):match['end'] + 120]
            (evidence if eligible else association_evidence).append(row)
        if view['view_kind'] == 'structured_field' and view['field_role'] in ARTIFACT_ROLES:
            available_artifact |= eligible
            artifacts.append(dict(**provenance, artifact_signal=view['field_role'],
                                  value_excerpt=text[:160], value_characters=len(text),
                                  interpretation='Structured field text; no filesystem, immutable version, publication, execution, or downstream-use assertion.'))
    if available_artifact != payload['artifact_view_available']:
        raise ValueError('Structural artifact availability disagrees with supplied views')
    categories = {item['category'] for item in evidence}
    row = dict(record_id=identifier, payload_sha256=digest, source_sha256=source,
               operation_bearing_payload=syntax, artifact_view_available=available_artifact,
               claim_candidate=bool(evidence), claim_scan_status=(
                   'bounded_views_scanned' if root_text is not None else 'root_text_unavailable_due_to_limit'),
               text_available=bool(source_views),
               associated_child_claim_evidence=bool(association_evidence),
               eligible_claim_evidence_records=len(evidence),
               association_only_claim_evidence_records=len(association_evidence),
               **{category: category in categories for category in CATEGORIES})
    return dict(payload=row, evidence=evidence, association_evidence=association_evidence,
                artifact_signals=artifacts)


def summarize_claims(payload_rows, *, evidence_records=None, association_records=None):
    """Payload counts plus explicit any-member duplicate-body sensitivity."""
    groups, seen = defaultdict(list), set()
    for row in payload_rows:
        identifier = row['record_id']
        if identifier in seen:
            raise ValueError('Duplicate payload record ID')
        seen.add(identifier)
        groups[_digest(row['payload_sha256'], 'payload_sha256')].append(row)
    fields = ('claim_candidate', 'artifact_view_available') + CATEGORIES
    for row in payload_rows:
        for field in fields:
            _boolean(row[field], field)
    unique = []
    for digest, rows in groups.items():
        syntax = {row['operation_bearing_payload'] for row in rows}
        if len(syntax) != 1:
            raise ValueError('Exact duplicate bodies have inconsistent syntax support')
        unique.append(dict(payload_sha256=digest, **{
            field: any(row[field] for row in rows) for field in fields}))
    cohorts = {}
    for name, rows in [('all_payload_records', payload_rows), ('exact_unique_payload_bodies', unique)]:
        n = len(rows)
        cohorts[name] = dict(denominator=n, **{
            field: dict(count=sum(row[field] for row in rows),
                        per_1000=1000 * sum(row[field] for row in rows) / n if n else None)
            for field in fields})
    return dict(cohorts=cohorts, eligible_claim_evidence_records=evidence_records,
                association_only_claim_evidence_records=association_records,
                body_aggregation='Union across exact-body member records, retaining different child attachments. '
                                 'Raw payload flags reflect each recorded provenance separately.',
                actual_cooperation_rate=None, executed_actions=None, actors=None, chronology=None,
                interpretation='Automatic claim candidates and structured artifact signals, requiring contextual review. '
                               'Unmarked, unsupported, limited, quoted, and ambiguous text is not a negative cooperation label. '
                               'Evidence records and views are not independent payloads or events.')


def classify_views(bundle, operation_rows=None):
    """Convenience wrapper over a complete ``swarm_views.build_views`` result."""
    views = defaultdict(list)
    for view in bundle['views']:
        views[view['root_id']].append(view)
    operations = None
    if operation_rows is not None:
        operations = {}
        for row in operation_rows:
            if row['record_id'] in operations:
                raise ValueError('Duplicate operation coverage record ID')
            operations[row['record_id']] = row
        if set(operations) != {row['record_id'] for row in bundle['payloads']}:
            raise ValueError('Operation coverage and payload universes differ')
    if set(views) - {row['record_id'] for row in bundle['payloads']}:
        raise ValueError('View without supplied payload')
    payloads, evidence, associated, artifacts = [], [], [], []
    for payload in bundle['payloads']:
        result = classify_payload(dict(payload=payload, views=views[payload['record_id']]),
                                  operations[payload['record_id']] if operations is not None else None)
        payloads.append(result['payload']); evidence.extend(result['evidence'])
        associated.extend(result['association_evidence']); artifacts.extend(result['artifact_signals'])
    return dict(payloads=payloads, evidence=evidence, association_evidence=associated,
                artifact_signals=artifacts,
                summary=summarize_claims(payloads, evidence_records=len(evidence), association_records=len(associated)))
