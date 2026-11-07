"""Conservative semantic disposition of local time candidates, not a timeline."""
import argparse
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

SIGNING = re.compile(r'(?i)(?P<field>X-Amz-Date|Timestamp)[\x22\x27]?\s*[:=]\s*[\x22\x27]?(?P<value>\d{8}T\d{6}Z|\d{4}-\d{2}-\d{2}T\d{2}(?::|%3A)\d{2}(?::|%3A)\d{2}Z)(?!\w)')


def sha(value):return hashlib.sha256(value).hexdigest()


def immediate_field(candidate):
    before=candidate['context_excerpt'].split('[[CANDIDATE:',1)[0]
    before=re.sub(r'\\+(?=[\x22\x27])','',before)
    match=re.search(r'([A-Za-z_][A-Za-z0-9_-]*)[\x22\x27]?(?:\])?\s*[:=]\s*[\x22\x27]?$',before)
    return match[1] if match else None


def disposition(candidate):
    family=candidate['family'];context=candidate['context_excerpt'];lower=context.lower()
    field=candidate.get('field_name') or immediate_field(candidate)
    key=(field or '').lower().replace('_','').replace('-','')
    if candidate.get('parse_status')=='invalid_calendar_or_clock':
        category='invalid_syntax';reason='Calendar or clock string is invalid.'
    elif candidate.get('header_context')=='date' and family in ('iso_datetime','rfc_datetime'):
        category='http_generation_clock_unattributed';reason='Date-header value can date a returned HTTP representation; original agent-request provenance is unverified, and caching/retrieval may intervene.'
    elif family=='date_only':
        category='calendar_reference_without_clock';reason='No within-day clock or action-observation semantics.'
    elif candidate.get('header_context')=='expires' or key in ('exp','expiry','expires','expiration','expiresat') or ('expires=' in lower and family=='rfc_datetime'):
        category='expiry_or_duration_claim';reason='Expiry or duration does not date an observed action; units and semantics must not be guessed.'
    elif family=='typed_numeric_time' and key=='mtime':
        category='archive_metadata_or_literal';reason='mtime is artifact metadata and is often deliberately assigned in code.'
    elif family=='typed_numeric_time' and key in ('iat','nbf'):
        category='credential_time_claim';reason='Credential claim may be authored, truncated, or redacted; no signature or observation provenance is established.'
    elif family=='typed_numeric_time' and key=='timestamp' and '"amr"' in re.sub(r'\\+(?=[\x22\x27])','',lower):
        category='authentication_event_claim';reason='An embedded authentication-method claim does not timestamp later payload execution.'
    elif family=='typed_numeric_time':
        category='unresolved_numeric_field';reason='May be a partial date, projection flag, literal, duration, or clock; neither units nor action binding is established.'
    elif key in ('startcapturedat','endcapturedat') or '"literal"' in lower:
        category='query_filter_or_literal';reason='Query bounds and literals do not timestamp observations.'
    elif key in ('created','createdat','updatedat','lastupdated','lastmodified','last','date','generated','datetime'):
        category='artifact_metadata_or_authored_datetime';reason='Creation, modification, display, or generated-document metadata is not a current execution clock; authored values can be backdated.'
    elif key in ('createtime','start','end','finishedat','tos'):
        category='embedded_service_or_document_event_claim';reason='Time describes an embedded event or declaration, whose relation to this exported record is not established.'
    else:
        category='unresolved_datetime_context';reason='Literal time is parseable but its event and observation provenance remain unresolved.'
    return dict(semantic_category=category,immediate_field=field,reason=reason,
                accepted_original_action_clock=False,acceptance_status='not_established',
                normalized_literal_utc=candidate.get('utc_value'),
                timezone_status=candidate.get('timezone_status'),
                normalized_literal_is_execution_time=False)


def run(source,audit_root):
    source,audit_root=Path(source),Path(audit_root)
    summary_path=audit_root/'acceptance-summary.json'
    if summary_path.exists():raise FileExistsError('Refusing to overwrite an acceptance audit')
    source_hash=sha(source.read_bytes())
    if source_hash!='7b66ab21674de52fcd3f557652f68b1801170c998e2f266862124e6edf283488':raise ValueError('Unexpected source')
    categories=Counter();fields=Counter();total=0
    with gzip.open(audit_root/'profile/candidates.jsonl.gz','rt') as f,gzip.open(audit_root/'acceptance-table.jsonl.gz','wt') as out:
        for candidate in map(json.loads,f):
            result=disposition(candidate);total+=1
            categories[candidate['kind'],result['semantic_category']]+=1
            fields[result['immediate_field'] or '<none>']+=1
            row={k:candidate[k] for k in ['record_id','kind','parent_id','family','offset_start','offset_end','value','text_sha256','cached_record_sha256','context_sha256']}
            out.write(json.dumps(dict(row,**result))+'\n')
    signing=[]
    with gzip.open(source,'rt') as f:
        for line in f:
            record=json.loads(line);text=record['text']
            for match in SIGNING.finditer(text):
                value=unquote(match['value'])
                try:
                    when=dt.datetime.strptime(value,'%Y%m%dT%H%M%SZ' if '-' not in value else '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc)
                except ValueError:continue
                signing.append(dict(record_id=record['id'],kind=record['kind'],parent_id=record['parent_id'],field=match['field'],literal=match['value'],normalized_literal_utc=when.isoformat(),offset_start=match.start('value'),text_sha256=sha(text.encode()),semantic_category='request_signing_time_claim',accepted_original_action_clock=False,reason='Typed request-signing literal, not an observed execution receipt; copied or replayed URLs retain it.'))
    (audit_root/'signing-time-claims.json').write_text(json.dumps(signing,indent=2)+'\n')
    result=dict(schema='conservative-time-acceptance-v1',created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256=source_hash,code_sha256=sha(Path(__file__).read_bytes()),
        input_candidates=total,category_counts=[dict(kind=k,category=c,count=n) for (k,c),n in sorted(categories.items())],
        immediate_fields=dict(fields),additional_signing_candidates=len(signing),signing_candidate_kinds=dict(Counter(r['kind'] for r in signing)),
        accepted_original_action_anchors=0,
        interpretation='No original-action anchors were established by this audit. This is an evidential acceptance decision, not proof that the export contains no temporal information.',
        preserved_information=['Unattributed HTTP generation dates','Artifact and service-event time claims','Request-signing and credential claims','Original precision, timezone ambiguity, and record/parent/hash provenance'],
        timeline_status='An original-agent cooperation timeline remains unidentified. No global ordering, interpolation, day bins, or cooperation join was produced.',
        partial_order_status='Extraction parents remain structural links. Embedded telemetry may describe its own service events, but does not establish succession of exported payload executions.',
        acceptance_requirements=['Explicit time with known zone and units','Evidence the value is an observation receipt, not authored artifact/expiry/template metadata','Attribution to the particular original execution/request, distinguishing later retrieval and cached responses','Duplicate-view handling and uncertainty; no order filled in from row IDs'],
        heuristic_limit='Semantic buckets describe why additional provenance is needed; they are not independently validated temporal labels. Unresolved candidates remain auditable rather than discarded.',
        no_cooperation_candidate_selection=True,no_cooperation_join=True,no_time_curves=True,no_external_sources=True,tags_not_used=True,
        post_exposure_status='Exploratory after prior project exposure; no external reports or prior external comparisons opened in this phase.',
        output_sha256={name:sha((audit_root/name).read_bytes()) for name in ['acceptance-table.jsonl.gz','signing-time-claims.json']})
    summary_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['input_candidates','additional_signing_candidates','signing_candidate_kinds','accepted_original_action_anchors','category_counts']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--audit-root',required=True,type=Path)
    args=parser.parse_args();run(args.source,args.audit_root)
