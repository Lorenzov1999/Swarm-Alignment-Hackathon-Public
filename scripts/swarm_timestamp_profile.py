"""Mechanical timestamp-candidate inventory; no chronology or clock inference.

Only supplied local JSONL records are read. Text is never executed. Sampling
does not depend on calendar dates, incident knowledge, or cooperation signals.
"""
import argparse
from collections import Counter, defaultdict
import datetime as dt
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import json
from pathlib import Path
import re


ISO = re.compile(r'(?<!\d)\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(?:\.\d{1,12})?(?:[Zz]|[+-]\d{2}:?\d{2})?(?!\d)')
RFC = re.compile(r'(?<!\w)(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+)?\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+(?:[A-Za-z]{2,5}|[+-]\d{2}:?\d{2})(?!\w)', re.I)
DATE = re.compile(r'(?<!\d)\d{4}-\d{2}-\d{2}(?![\dTt])')
TIME_FIELD = r'(?:timestamp|time|datetime|date|epoch|unix_time|server_time|request_time|response_time|received_at|created_at|updated_at|modified_at|started_at|finished_at|start_time|end_time|mtime|ctime|atime|iat|exp|nbf|expires|expiry)(?:_s|_ms|_us|_ns|Millis|Milliseconds|Seconds)?'
EPOCH = re.compile(r'(?<![\w])(?P<key>'+TIME_FIELD+r')[\x22\x27]?\s*[:=]\s*[\x22\x27]?(?P<number>-?\d{1,19}(?:\.\d{1,9})?)(?![\w.])',re.I)
HINT = re.compile(r'(?i)\b(expir\w*|creat\w*|modif\w*|updat\w*|timestamp|date|server|clock|template|example|fixture|test|build|copyright|issued|iat|exp|nbf)\b')
HEADER = re.compile(r'(?i)(?:^|[\r\n]|\\r\\n|\\n)\s*(Date|Last-Modified|Expires)\s*:\s*$')
EXPECTED_SOURCE = '7b66ab21674de52fcd3f557652f68b1801170c998e2f266862124e6edf283488'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def redact(value):
    value=re.sub(r'https?://[^\s\x22\x27<>]+','<URL>',value)
    value=re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9_]{15,}|github_pat_[A-Za-z0-9_]+|glpat-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{15,}|olp_[A-Za-z0-9_]+)\b','[REDACTED TOKEN]',value)
    value=re.sub(r'(?i)((?:password|secret|api[_-]?key|authorization)\s*[:=]\s*[\x22\x27]?)[^\s\x22\x27,;]+',r'\1[REDACTED]',value)
    return value


def parsed_literal(value,family):
    """Parse syntax; never assign missing zone, year, or epoch units."""
    result={'parse_status':'valid','timezone_status':'not_applicable','utc_value':None}
    try:
        if family=='date_only':
            result['calendar_date']=dt.date.fromisoformat(value).isoformat()
        elif family=='iso_datetime':
            normalized=re.sub(r'\.(\d+)',lambda m:'.'+m[1][:6].ljust(6,'0'),value)
            normalized=re.sub(r'([+-]\d{2})(\d{2})$',r'\1:\2',normalized)
            when=dt.datetime.fromisoformat(normalized.replace('t','T').replace('z','+00:00').replace('Z','+00:00'))
            result['timezone_status']='explicit_offset' if when.tzinfo else 'missing'
            result['utc_value']=when.astimezone(dt.timezone.utc).isoformat() if when.tzinfo else None
            result['fraction_digits']=len(value.split('.',1)[1].split('+',1)[0].split('-',1)[0].rstrip('Zz')) if '.' in value else 0
        elif family=='rfc_datetime':
            zone=value.split()[-1]
            if zone.upper() not in ('GMT','UTC') and not re.fullmatch(r'[+-]\d{2}:?\d{2}',zone):
                result['timezone_status']='ambiguous_named_zone'
                return result
            when=parsedate_to_datetime(value)
            result['timezone_status']='explicit_offset' if when.tzinfo else 'missing'
            result['utc_value']=when.astimezone(dt.timezone.utc).isoformat() if when.tzinfo else None
    except (ValueError,TypeError,OverflowError):
        result.update(parse_status='invalid_calendar_or_clock',utc_value=None)
    return result


def candidates(text):
    """Emit candidate strings and syntax only; none are accepted observations."""
    occupied=[]
    for family,pattern in [('iso_datetime',ISO),('rfc_datetime',RFC),('date_only',DATE),('typed_numeric_time',EPOCH)]:
        for match in pattern.finditer(text):
            start,end=match.span('number') if family=='typed_numeric_time' else match.span()
            if any(start<b and a<end for a,b in occupied):
                continue
            occupied.append((start,end))
            value=text[start:end]
            before=text[max(0,start-180):start]
            after=text[end:min(len(text),end+180)]
            header=HEADER.search(before)
            row=dict(family=family,offset_start=start,offset_end=end,value=value,
                line_number=text.count('\n',0,start)+1,
                context_excerpt=redact(before+'[[CANDIDATE:'+value+']]'+after),
                context_sha256=digest((before+value+after).encode()),
                lexical_context_hints=sorted(set(x.lower() for x in HINT.findall(before+after))),
                header_context=header[1].lower() if header else None)
            if family=='typed_numeric_time':
                key=match['key'];unit=None
                if key.lower().endswith(('_ms','millis','milliseconds')): unit='milliseconds'
                elif key.lower().endswith(('_s','seconds')): unit='seconds'
                elif key.lower().endswith('_us'): unit='microseconds'
                elif key.lower().endswith('_ns'): unit='nanoseconds'
                row.update(field_name=key,explicit_unit=unit,parse_status='numeric_syntax',timezone_status='numeric_semantics_unverified',utc_value=None)
            else:
                row.update(parsed_literal(value,family))
            yield row


def profile(source,output,sample_size=8):
    source,output=Path(source),Path(output)
    if output.exists(): raise FileExistsError('Refusing to overwrite an existing profile directory')
    source_hash=digest(source.read_bytes())
    if source_hash!=EXPECTED_SOURCE: raise ValueError('Unexpected logical source SHA256')
    output.mkdir(parents=True)
    records=Counter();counts=Counter();record_counts=Counter();parse_counts=Counter();zone_counts=Counter();header_counts=Counter();field_counts=Counter()
    bodies=defaultdict(set);sample={};ids=set();candidates_total=0
    with gzip.open(source,'rb') as stream,gzip.open(output/'candidates.jsonl.gz','wt') as target:
        for line in stream:
            record=json.loads(line)
            rid,kind,parent=record['id'],record['kind'],record['parent_id']
            if rid in ids: raise ValueError('Duplicate record ID')
            ids.add(rid);records[kind]+=1
            text=record['text']
            if not isinstance(text,str): raise ValueError('Text must be a string')
            body_hash=digest(text.encode())
            seen_families=set()
            for candidate in candidates(text):
                candidates_total+=1
                candidate.update(record_id=rid,kind=kind,parent_id=parent,text_sha256=body_hash,
                                 cached_record_sha256=digest(line),accepted_as_observation=False)
                family=candidate['family'];bucket=(kind,family)
                counts[bucket]+=1;seen_families.add(family);bodies[bucket].add(body_hash)
                parse_counts[family,candidate['parse_status']]+=1
                zone_counts[family,candidate['timezone_status']]+=1
                if candidate['header_context']:header_counts[kind,candidate['header_context']]+=1
                if 'field_name' in candidate:field_counts[kind,candidate['field_name']]+=1
                target.write(json.dumps(candidate,ensure_ascii=False)+'\n')
                # Each exact body/format contributes at most one deterministic
                # candidate. Selection excludes date values and incident dates.
                sample_key=(kind,family,body_hash)
                rank=digest((body_hash+'|'+str(candidate['offset_start'])+'|20261106').encode())
                old=sample.get(sample_key)
                if old is None or (rank,rid)<(old[0],old[1]['record_id']):sample[sample_key]=(rank,candidate)
            for family in seen_families:record_counts[kind,family]+=1
    groups=defaultdict(list)
    for (kind,family,_),(rank,candidate) in sample.items():groups[kind,family].append((rank,candidate))
    selected=[]
    for key,values in sorted(groups.items()):
        selected.extend(row for _,row in sorted(values,key=lambda item:(item[0],item[1]['record_id']))[:sample_size])
    def items(counter,keys):return [dict(zip(keys,key),count=value) for key,value in sorted(counter.items())]
    summary=dict(schema='swarm-embedded-time-profile-v1',created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256=source_hash,script_sha256=digest(Path(__file__).read_bytes()),source_records=sum(records.values()),
        kinds=dict(records),candidate_count=candidates_total,candidate_occurrences=items(counts,['kind','family']),
        records_with_candidate=items(record_counts,['kind','family']),
        exact_body_counts=[dict(kind=k,family=f,bodies=len(v)) for (k,f),v in sorted(bodies.items())],
        parse_status_counts=items(parse_counts,['family','status']),timezone_status_counts=items(zone_counts,['family','status']),
        header_context_counts=items(header_counts,['kind','header']),typed_numeric_field_counts=items(field_counts,['kind','field']),
        sample_size_per_kind_family=sample_size,selected_contexts=len(selected),
        sampling='Deterministic hash sample per kind/format after exact-text deduplication; no date range, incident anchor, behavior score, or outcome selection.',
        semantics='All matches are candidates, including quoted code and artifact metadata; none establishes an observation time.',
        ignored_fields=['tags','cite'],raw_code_executed=False,external_sources_accessed=False,
        post_exposure_status='Exploratory follow-up after prior project exposure; not prospectively outcome-blind.',
        outputs_sha256={'candidates.jsonl.gz':digest((output/'candidates.jsonl.gz').read_bytes())})
    (output/'sample.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n')
    summary['outputs_sha256']['sample.json']=digest((output/'sample.json').read_bytes())
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['source_records','candidate_count','candidate_occurrences','header_context_counts','selected_contexts']}))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    profile(args.source,args.output)
