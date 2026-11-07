"""Bounded, source-preserving artifact views of an unordered record export.

These are text observations, never execution events. A child's exact occurrence
in its declared parent can establish a text span, not authorship or chronology.
No source is removed to expose surrounding code; no code, base64, fuzzy search,
or external source is evaluated. Offsets count Python Unicode characters.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re


SOURCE_SHA256 = '7b66ab21674de52fcd3f557652f68b1801170c998e2f266862124e6edf283488'
DIRECT_CHILD_KINDS = frozenset({'response', 'recovered_text'})
FIELD_ROLES = {
    **{key: 'path' for key in ('path', 'filepath', 'file_path', 'filename', 'file_name',
                              'source_path', 'target_path', 'document_path')},
    **{key: 'version' for key in ('version', 'revision', 'commit', 'commit_sha', 'sha',
                                 'sha256', 'content_hash', 'etag', 'version_id', 'document_version')},
    'summary': 'summary', 'description': 'description',
    **{key: 'text' for key in ('text', 'content', 'body', 'message')},
}
BLOCKED_FIELDS = frozenset({'code', 'command', 'script', 'program', 'javascript',
                            'python', 'bash', 'shell', 'base64', 'encoded', 'binary'})


@dataclass(frozen=True)
class Limits:
    max_source_chars: int = 2_000_000
    max_json_chars: int = 262_144
    max_json_depth: int = 32
    max_json_nodes: int = 20_000
    max_nested_decodes: int = 2
    max_fields_per_record: int = 512
    max_child_occurrences: int = 32

    def __post_init__(self):
        for key, value in asdict(self).items():
            minimum = 0 if key == 'max_nested_decodes' else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError('Invalid bounded-view limit: ' + key)


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _identity(*values):
    return _sha(json.dumps(values, ensure_ascii=False, separators=(',', ':')))


def _check_source(source_sha256):
    if not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise ValueError('A lowercase SHA-256 source identity is required')


def _pointer(value):
    return str(value).replace('~', '~0').replace('/', '~1')


class _InvalidJSON(ValueError):
    pass


@dataclass
class _Node:
    kind: str
    start: int
    end: int
    value: object = None
    children: object = None


class _Parser:
    """Strict complete JSON with value spans and duplicate-key rejection."""
    def __init__(self, text, limits):
        self.text, self.limits, self.index, self.nodes = text, limits, 0, 0
        self.decoder = json.JSONDecoder(parse_constant=self.invalid_constant)

    @staticmethod
    def invalid_constant(value):
        raise _InvalidJSON('Nonstandard numeric constant')

    def whitespace(self):
        while self.index < len(self.text) and self.text[self.index] in ' \t\r\n':
            self.index += 1

    def value(self, depth=0):
        self.whitespace(); start = self.index; self.nodes += 1
        if depth > self.limits.max_json_depth or self.nodes > self.limits.max_json_nodes:
            raise _InvalidJSON('JSON structural limit')
        if start >= len(self.text):
            raise _InvalidJSON('Missing JSON value')
        initial = self.text[start]
        if initial in '{[':
            object_kind = initial == '{'; closer = '}' if object_kind else ']'
            self.index += 1; children = []; keys = set(); self.whitespace()
            if self.index < len(self.text) and self.text[self.index] == closer:
                self.index += 1
                return _Node('object' if object_kind else 'array', start, self.index, children=children)
            while True:
                key = len(children)
                if object_kind:
                    self.whitespace()
                    if self.index >= len(self.text) or self.text[self.index] != '"':
                        raise _InvalidJSON('Object key is not a JSON string')
                    key, self.index = self.decoder.raw_decode(self.text, self.index)
                    key.encode('utf-8')
                    if key in keys:
                        raise _InvalidJSON('Duplicate object key')
                    keys.add(key); self.whitespace()
                    if self.index >= len(self.text) or self.text[self.index] != ':':
                        raise _InvalidJSON('Missing object separator')
                    self.index += 1
                children.append((key, self.value(depth + 1)))
                self.whitespace()
                if self.index >= len(self.text):
                    raise _InvalidJSON('Incomplete JSON container')
                token = self.text[self.index]; self.index += 1
                if token == closer:
                    break
                if token != ',':
                    raise _InvalidJSON('Invalid JSON separator')
            return _Node('object' if object_kind else 'array', start, self.index, children=children)
        value, self.index = self.decoder.raw_decode(self.text, start)
        if isinstance(value, (dict, list)):
            raise _InvalidJSON('Unexpected container')
        if isinstance(value, float) and not math.isfinite(value):
            raise _InvalidJSON('Nonfinite number')
        if isinstance(value, str):
            # Reject unpaired escaped surrogates rather than inventing UTF-8.
            value.encode('utf-8')
        return _Node('string' if isinstance(value, str) else 'scalar', start, self.index, value=value)

    def complete(self):
        node = self.value(); self.whitespace()
        if self.index != len(self.text):
            raise _InvalidJSON('Trailing non-JSON source')
        return node


def _document(text, limits):
    """Do not emit a valid prefix from malformed JSON or mixed-language text."""
    if len(text) > limits.max_json_chars:
        return 'json_size_limit', []
    try:
        node = _Parser(text, limits).complete()
        return 'complete_json', [(None, node)]
    except (ValueError, UnicodeError, RecursionError):
        pass
    lines, offset, nodes = [], 0, 0
    try:
        for line_number, line in enumerate(text.splitlines(keepends=True)):
            if line.strip():
                parser = _Parser(line, limits)
                node = parser.complete(); nodes += parser.nodes
                if node.kind not in ('object', 'array') or nodes > limits.max_json_nodes:
                    raise _InvalidJSON('NDJSON requires complete structured lines')
                _shift(node, offset)
                lines.append((line_number, node))
            offset += len(line)
        if len(lines) >= 2:
            return 'complete_ndjson', lines
    except (ValueError, UnicodeError, RecursionError):
        pass
    return 'unstructured_source', []


def _shift(node, offset):
    node.start += offset; node.end += offset
    for _, child in node.children or []:
        _shift(child, offset)


def _string_mapping(text, node, mapping):
    """Map each decoded character to its exact encoded source interval."""
    intervals, index = [], node.start + 1
    while index < node.end - 1:
        start = index
        if text[index] != '\\':
            index += 1
        elif text[index + 1] != 'u':
            index += 2
        else:
            value = int(text[index + 2:index + 6], 16); index += 6
            if 0xD800 <= value <= 0xDBFF:
                # Strict parser already rejected a lone surrogate.
                index += 6
        intervals.append((start, index) if mapping is None
                         else (mapping[start][0], mapping[index - 1][1]))
    if len(intervals) != len(node.value):
        raise ValueError('JSON string source mapping disagrees with decoded text')
    return intervals


def _source_span(node, mapping):
    return [node.start, node.end] if mapping is None else [mapping[node.start][0], mapping[node.end - 1][1]]


def _value_span(node, mapping):
    if node.kind != 'string':
        return _source_span(node, mapping)
    start, end = node.start + 1, node.end - 1
    if mapping is None:
        return [start, end]
    if start == end:
        return [mapping[start][0], mapping[start][0]]
    return [mapping[start][0], mapping[end - 1][1]]


def structured_fields(text, *, limits=None):
    """Return selected field values and exact spans in this unchanged text.

    Decoded field text need not equal its encoded source slice. Nested JSON
    strings retain a pointer chain and an escape-aware source mapping; all
    returned ``source_span`` values still index the original input string.
    """
    limits = limits or Limits()
    status, documents = _document(text, limits)
    fields, diagnostics = [], Counter()

    def visit(node, current, mapping, pointer, line_index, nested_depth, chain, inherited_role=None, blocked=False):
        if len(fields) >= limits.max_fields_per_record:
            diagnostics['field_limit_reached'] = 1
            return
        if node.kind in ('object', 'array'):
            for key, child in node.children:
                is_object = node.kind == 'object'
                normalized = key.lower() if is_object else None
                child_role = FIELD_ROLES.get(normalized) if is_object else inherited_role
                child_blocked = blocked or (is_object and normalized in BLOCKED_FIELDS)
                visit(child, current, mapping, pointer + '/' + _pointer(key), line_index,
                      nested_depth, chain, child_role, child_blocked)
            return
        if blocked:
            diagnostics['blocked_program_fields'] += 1
            return
        if inherited_role and (node.kind == 'string' or (
                inherited_role == 'version' and isinstance(node.value, (int, float)) and not isinstance(node.value, bool))):
            span = _value_span(node, mapping)
            value = node.value if node.kind == 'string' else current[node.start:node.end]
            raw = text[span[0]:span[1]]
            fields.append(dict(field_role=inherited_role, text=value, source_span=span,
                               source_text=raw, text_is_exact_source_substring=(value == raw),
                               source_literal_span=_source_span(node, mapping),
                               json_pointer=pointer, json_pointer_chain=chain + [pointer],
                               ndjson_line_index=line_index, nested_decode_depth=nested_depth))
        if node.kind != 'string' or not node.value.strip().startswith(('{', '[')):
            return
        if nested_depth >= limits.max_nested_decodes:
            diagnostics['nested_decode_limit_reached'] += 1
            return
        inner_status, inner_documents = _document(node.value, limits)
        if inner_status not in ('complete_json', 'complete_ndjson'):
            return
        inner_mapping = _string_mapping(current, node, mapping)
        for inner_line, inner in inner_documents:
            if inner.kind in ('object', 'array'):
                visit(inner, node.value, inner_mapping, '', inner_line, nested_depth + 1,
                      chain + [pointer], None, False)

    for line_index, node in documents:
        visit(node, text, None, '', line_index, 0, [])
    return dict(source_class=status, fields=fields, diagnostics=dict(diagnostics))


def _occurrences(parent, child, maximum):
    if not child:
        return [], 'empty_child', False
    spans, start = [], 0
    while True:
        at = parent.find(child, start)
        if at < 0:
            break
        if len(spans) == maximum:
            return spans, 'ambiguous', True
        spans.append([at, at + len(child)])
        start = at + 1
    return spans, ('unique' if len(spans) == 1 else 'ambiguous' if spans else 'not_contained'), False


def payload_views(payload, children=(), *, source_sha256=SOURCE_SHA256, limits=None):
    """Build one payload-root group; children must be its direct records only."""
    _check_source(source_sha256); limits = limits or Limits()
    if payload.get('kind') != 'payload' or not isinstance(payload.get('id'), str) or not payload['id']:
        raise ValueError('A named payload record is required')
    root_id, root_text = payload['id'], payload['text']
    if not isinstance(root_text, str):
        raise ValueError('Payload text must be a string')
    root_hash = _sha(root_text)
    root = dict(root_id=root_id, record_id=root_id, payload_sha256=root_hash,
                source_sha256=source_sha256, source_characters=len(root_text),
                text_available=len(root_text) <= limits.max_source_chars)
    views, associations, diagnostics, seen = [], [], Counter(), {root_id}

    def add_source(record, containment, candidates, truncated=False):
        text, record_id = record['text'], record['id']
        if len(text) > limits.max_source_chars:
            diagnostics['source_size_limit_records'] += 1
            return
        parsed = structured_fields(text, limits=limits)
        diagnostics.update(parsed['diagnostics'])
        original_kind = record['kind']
        origin_hash = _sha(text)
        fields = [dict(view_kind='root_source' if record_id == root_id else 'direct_child',
                       text=text, source_text=text, source_span=[0, len(text)],
                       text_is_exact_source_substring=True, field_role=None,
                       json_pointer=None, json_pointer_chain=[], ndjson_line_index=None,
                       nested_decode_depth=0)]
        fields += [dict(field, view_kind='structured_field') for field in parsed['fields']]
        for field in fields:
            span = field['source_span']
            possible = [[candidate[0] + span[0], candidate[0] + span[1]] for candidate in candidates]
            root_span = possible[0] if containment == 'unique' else None
            raw_hash = _sha(field['source_text']); text_hash = _sha(field['text'])
            view = dict(field, root_id=root_id, origin_record_id=record_id, origin_kind=original_kind,
                        source_sha256=source_sha256, origin_text_sha256=origin_hash,
                        payload_sha256=root_hash, text_sha256=text_hash, source_text_sha256=raw_hash,
                        source_class=parsed['source_class'], root_span=root_span,
                        root_span_candidates=possible, root_span_candidates_truncated=truncated,
                        containment=containment, association_only=containment != 'unique',
                        execution_status='not_established')
            view['view_id'] = _identity(source_sha256, root_id, record_id, field['view_kind'],
                                        span, field['json_pointer_chain'], field['ndjson_line_index'], text_hash)
            # Ambiguous/uncontained child occurrences are not assigned a false
            # physical identity. Exact same-parent text duplication is separate.
            view['physical_evidence_id'] = (_identity(source_sha256, root_id, root_span, raw_hash)
                                             if root_span is not None else None)
            view['parent_text_duplicate_id'] = _identity(source_sha256, root_id, text_hash)
            views.append(view)

    add_source(payload, 'unique', [[0, len(root_text)]])
    for child in children:
        if (child.get('kind') not in DIRECT_CHILD_KINDS or child.get('parent_id') != root_id
                or not isinstance(child.get('id'), str) or not child['id']):
            raise ValueError('Child must be a named direct response/recovered record')
        if child['id'] in seen:
            raise ValueError('Duplicate source record ID')
        seen.add(child['id'])
        child_text = child['text']
        if not isinstance(child_text, str):
            raise ValueError('Child text must be a string')
        if len(root_text) > limits.max_source_chars or len(child_text) > limits.max_source_chars:
            spans, status, truncated = [], 'source_size_limit', False
        else:
            spans, status, truncated = _occurrences(root_text, child_text, limits.max_child_occurrences)
        associations.append(dict(root_id=root_id, child_record_id=child['id'], child_kind=child['kind'],
                                 source_sha256=source_sha256, payload_sha256=root_hash,
                                 child_text_sha256=_sha(child_text), containment=status,
                                 root_span=spans[0] if status == 'unique' else None,
                                 root_span_candidates=spans, occurrences_truncated=truncated))
        add_source(child, status, spans, truncated)
    root['root_structured_artifact_available'] = any(v['origin_record_id'] == root_id and
        v['view_kind'] == 'structured_field' and v['field_role'] in {'summary', 'description', 'path', 'version'}
        for v in views)
    root['artifact_view_available'] = any(v['root_span'] is not None and
        v['view_kind'] == 'structured_field' and v['field_role'] in {'summary', 'description', 'path', 'version'}
        for v in views)
    root['direct_child_available'] = bool(associations)
    return dict(payload=root, views=views, associations=associations, diagnostics=dict(diagnostics))


def build_views(records, *, source_sha256=SOURCE_SHA256, limits=None):
    """Pure convenience API; production may stream groups via payload_views.

    This function binds the supplied source identity but does not verify a file
    on disk. The caller must verify source bytes before a production scan.
    Unknown kinds and non-direct child text are not inspected.
    """
    _check_source(source_sha256); limits = limits or Limits()
    metadata, selected, counts = {}, {}, Counter()
    for record in records:
        record_id, kind, parent = record['id'], record['kind'], record['parent_id']
        if (not isinstance(record_id, str) or not record_id or not isinstance(kind, str) or not kind
                or (parent is not None and (not isinstance(parent, str) or not parent))):
            raise ValueError('Invalid record metadata')
        if record_id in metadata:
            raise ValueError('Duplicate source record ID')
        metadata[record_id] = (kind, parent); counts[kind] += 1
        if kind == 'payload' or kind in DIRECT_CHILD_KINDS:
            selected[record_id] = record
    from .swarm_static import _validate_graph
    _validate_graph(metadata)
    children, excluded = defaultdict(list), Counter()
    for record_id, (kind, parent) in metadata.items():
        if kind not in DIRECT_CHILD_KINDS:
            continue
        if parent is None or metadata[parent][0] != 'payload':
            excluded['no_direct_payload_parent'] += 1
        else:
            children[parent].append(selected[record_id])
    payloads, views, associations, diagnostics = [], [], [], Counter()
    for root_id in sorted(key for key, (kind, _) in metadata.items() if kind == 'payload'):
        group = payload_views(selected[root_id], sorted(children[root_id], key=lambda row: row['id']),
                              source_sha256=source_sha256, limits=limits)
        payloads.append(group['payload']); views.extend(group['views'])
        associations.extend(group['associations']); diagnostics.update(group['diagnostics'])
    summary = summarize_views(payloads, views, associations)
    summary.update(source_record_kind_counts=dict(counts), excluded_children=dict(excluded),
                   diagnostics=dict(diagnostics), limits=asdict(limits), source_sha256=source_sha256,
                   source_file_bytes_verified=False)
    return dict(payloads=payloads, views=views, associations=associations, summary=summary)


def summarize_views(payloads, views, associations):
    """Root denominators and view counts remain different observation units."""
    return dict(payload_records=len(payloads), exact_unique_payload_bodies=len({p['payload_sha256'] for p in payloads}),
                payload_roots_with_artifact_fields=sum(p['root_structured_artifact_available'] for p in payloads),
                payload_roots_with_contained_artifact_fields=sum(p['artifact_view_available'] for p in payloads),
                payload_roots_with_direct_children=sum(p['direct_child_available'] for p in payloads),
                source_view_records=sum(v['view_kind'] != 'structured_field' for v in views),
                structured_field_views=sum(v['view_kind'] == 'structured_field' for v in views),
                view_counts_by_origin=dict(Counter(v['origin_kind'] for v in views)),
                view_counts_by_source_class=dict(Counter(v['source_class'] for v in views)),
                direct_child_records=len(associations),
                child_containment_counts=dict(Counter(a['containment'] for a in associations)),
                unique_physical_evidence_spans=len({v['physical_evidence_id'] for v in views if v['physical_evidence_id']}),
                exact_parent_text_groups=len({v['parent_text_duplicate_id'] for v in views}),
                execution_events=None, actor_interactions=None, chronological_order=None,
                interpretation='Payload roots are denominators; views are source-linked text observations. '
                    'Exact containment and duplicates do not establish execution, authorship, chronology, or cooperation.')
