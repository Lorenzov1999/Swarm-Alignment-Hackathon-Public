"""Frozen standalone census of static operation-bearing payload records.

No cooperation-marker join, execution, actor inference, or time inference.
"""
import argparse
import collections
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('swarm_operation_sites',ROOT/'scripts/swarm_operation_sites.py')
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):result.update(block)
    return result.hexdigest()


def code_hashes():
    paths=['scripts/swarm_operation_census.py','scripts/swarm_operation_sites.py',
           'scripts/javascript_operation_sites.cjs','src/swarmcollab/provenance_common.py',
           'src/swarmcollab/config.py','src/swarmcollab/__init__.py','tests/test_swarm_operation_sites.py']
    return {relative:digest(ROOT/relative) for relative in paths}


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def summarize(frame):
    total=len(frame)
    bearing=int(frame.operation_bearing_payload.sum()) if total else 0
    sites=int(frame.operation_site_count.sum()) if total else 0
    known=int(frame.method_known_site_count.sum()) if total else 0
    return dict(payload_records=total,operation_bearing_payload_records=bearing,
                operation_bearing_fraction=bearing/total if total else None,
                operation_bearing_per_1000_payload_records=1000*bearing/total if total else None,
                static_operation_expression_sites=sites,method_known_sites=known,
                method_known_fraction_of_sites=known/sites if sites else None,
                operation_bearing_payloads_by_selected_syntax={str(key):int(value) for key,value in
                    frame.loc[frame.operation_bearing_payload,'selected_syntax_family'].value_counts().items()} if total else {})


def run_census(source,output,expected_sha256,*,javascript=True):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError('Refusing to overwrite a census or its pre-census lock')
    observed_hash=digest(source)
    if observed_hash!=expected_sha256:raise ValueError('Source SHA-256 differs from the requested identity')
    source_stat=source.stat();code=code_hashes()
    parser=_module.JavascriptParser() if javascript else None
    runtime=parser.parse('',record_id='synthetic_runtime_probe') if parser else {}
    output.mkdir(parents=True)
    lock=dict(locked_before_census_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=observed_hash,
        code_sha256=code,python=sys.version,javascript_enabled=javascript,
        javascript_runtime={key:runtime.get(key) for key in ('node_version','acorn_version')},
        observation_unit='Payload record containing static operation-expression sites; never an executed call.',
        operation_families=['simple shell curl/wget','import-supported Python requests','complete-AST JavaScript fetch/XHR'],
        primary_coverage_denominator='All payload records; report operation-bearing subset separately.',
        future_marker_proxy='A separately frozen marker may label an operation-bearing payload, never every site within it.',
        duplicate_sensitivity='Exact whole-payload UTF-8 hash; lexicographically first record ID per group.',
        unsupported='Mixed-language/malformed source, nested quoted or encoded programs, unknown wrappers and ambiguous bindings are not recovered as operation sites.',
        no_marker_join=True,no_execution=True,no_actor_or_time_inference=True)
    save(output/'pre-census-lock.json',lock)
    started=time.perf_counter();payload_rows=[];site_rows=[];kind_counts=collections.Counter();seen=set()
    try:
        with gzip.open(source,'rt',encoding='utf-8') as stream:
            for line in stream:
                record=json.loads(line);kind_counts[record['kind']]+=1
                if record['kind']!='payload':continue
                record_id=record['id']
                if record_id in seen:raise ValueError('Duplicate payload record ID')
                seen.add(record_id)
                value=_module.extract_sites(record['text'],parser,record_id)
                row=dict(record_id=record_id,payload_sha256=value['payload_sha256'],
                    operation_bearing_payload=value['operation_bearing_payload'],
                    operation_site_count=len(value['operation_expression_sites']),
                    method_known_site_count=value['method_known_sites'],
                    selected_syntax_family=value['selected_syntax_family'],
                    **{key+'_parse_status':status for key,status in value['parser_status'].items()})
                payload_rows.append(row)
                site_rows.extend(dict(record_id=record_id,**site) for site in value['operation_expression_sites'])
        if code_hashes()!=code:raise ValueError('Parser implementation changed after the pre-census lock')
        if (source.stat().st_size,source.stat().st_mtime_ns)!=(source_stat.st_size,source_stat.st_mtime_ns):
            raise ValueError('Source file changed during census')
        frame=pd.DataFrame(payload_rows,columns=['record_id','payload_sha256','operation_bearing_payload',
            'operation_site_count','method_known_site_count','selected_syntax_family',
            'shell_parse_status','python_parse_status','javascript_parse_status']).sort_values('record_id')
        if frame.empty:frame['operation_bearing_payload']=frame.operation_bearing_payload.astype(bool)
        groups=frame.groupby('payload_sha256').record_id.agg(['min','size'])
        frame['duplicate_representative_id']=frame.payload_sha256.map(groups['min'])
        frame['duplicate_group_size']=frame.payload_sha256.map(groups['size'])
        frame['included_in_duplicate_sensitivity']=frame.record_id.eq(frame.duplicate_representative_id)
        unique=frame[frame.included_in_duplicate_sensitivity]
        sites=pd.DataFrame(site_rows)
        if sites.empty:sites=pd.DataFrame(columns=['record_id','site_index','syntax_family','api_shape','method','method_known','target_support'])
        else:sites=sites.sort_values(['record_id','site_index'])
        summary=dict(source_sha256=observed_hash,pre_census_lock_sha256=digest(output/'pre-census-lock.json'),
            completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
            source_record_kind_counts=dict(kind_counts),primary=summarize(frame),
            exact_whole_payload_duplicate_sensitivity=summarize(unique),
            parser_status_counts={name:{str(key):int(value) for key,value in frame[name+'_parse_status'].value_counts().items()}
                                  for name in ['shell','python','javascript']},
            site_api_shape_counts={str(key):int(value) for key,value in sites.api_shape.value_counts().items()},
            site_method_counts={str(key):int(value) for key,value in sites.method.fillna('UNKNOWN').value_counts().items()},
            target_support_counts={str(key):int(value) for key,value in sites.target_support.value_counts().items()},
            observed_cooperation_rate=None,executed_operation_rate=None,
            interpretation='Exploratory parser coverage. Static site counts are not invocation/iteration counts; operation-bearing payloads are not actors or independent episodes. No marker, time, success, or outcome join.',
            limitations=['Standard names can be rebound through unobserved/dynamic code; explicit shadows and mutations are conservatively excluded, not all possible runtime changes.',
                         'Only supported standard syntax shapes are counted. Unsupported mixed-language text and literal-string programs are not automatically decoded or reclassified.',
                         'Method defaults for JavaScript fetch are known only for literal string targets; dynamic Request inputs may inherit a method.',
                         'No cooperation marker labels are propagated to individual operation sites.'])
        frame.to_csv(output/'payload-operation-coverage.csv.gz',index=False)
        sites.to_csv(output/'operation-sites.csv.gz',index=False)
        save(output/'summary.json',summary)
        save(output/'manifest.json',dict(source_sha256=observed_hash,code_sha256=code,
             outputs_sha256={path.name:digest(path) for path in sorted(output.iterdir()) if path.is_file()},
             no_marker_join=True,no_execution=True))
        return summary
    except BaseException:
        save(output/'failed.json',dict(status='failed_closed',pre_census_lock_sha256=digest(output/'pre-census-lock.json')))
        raise
    finally:
        if parser:parser.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--without-javascript',action='store_true')
    args=parser.parse_args()
    summary=run_census(args.source,args.output,args.source_sha256,javascript=not args.without_javascript)
    print(json.dumps({'output':str(args.output),'primary':summary['primary'],'elapsed_seconds':summary['elapsed_seconds']}))


if __name__=='__main__':main()
