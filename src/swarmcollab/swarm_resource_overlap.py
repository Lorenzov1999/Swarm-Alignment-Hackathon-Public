"""Rank exact shared references in unordered payload records without an N² matrix.

Only payload text is parsed. These are resource contributions to record-pair
similarity, not identified agent interactions or threshold-calibrated alerts.
"""
import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

from scipy.stats import spearmanr

from . import swarm_static as static


FAMILIES = ('url', 'opaque_alias')
PROTOCOL = Path(__file__).resolve().parents[2] / 'docs/swarm-resource-ranking-protocol.md'
KNOWN_SOURCE = '7b66ab21674de52fcd3f557652f68b1801170c998e2f266862124e6edf283488'


def _roots(metadata):
    static._validate_graph(metadata)
    roots = {}
    for start in metadata:
        path, node = [], start
        while node not in roots and metadata[node][1] is not None:
            path.append(node)
            node = metadata[node][1]
        root = roots.get(node, node)
        roots[node] = root
        for node in path:
            roots[node] = root
    return roots


def _correlation(left, right):
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return float(spearmanr(left, right).statistic)


def aggregate(payloads, labels):
    """Exact unordered distinct-record cosine contributions, O(incidences).

    All records, including empty-resource records, enter N. Binary occurrence
    uses log(N/df); each nonzero row is L2-normalized. Summing column products
    excludes self-pairs algebraically. An unweighted normalized-binary score
    and a same-extraction-root exclusion are calculated as sensitivities.
    """
    n = len(payloads)
    support = Counter(resource for row in payloads for resource in row['resources'])
    idf = {resource: math.log(n / count) for resource, count in support.items()}
    sums, squares, uniform_sums, uniform_squares = [defaultdict(float) for _ in range(4)]
    root_sums = defaultdict(lambda: defaultdict(float))
    supporting_roots = defaultdict(set)
    incidence, nonzero = [], 0
    for row in payloads:
        resources = sorted(row['resources'])
        norm = math.sqrt(math.fsum(idf[r] ** 2 for r in resources))
        uniform = 1 / math.sqrt(len(resources)) if resources else 0.0
        nonzero += norm > 0
        weighted = []
        for resource in resources:
            value = idf[resource] / norm if norm else 0.0
            sums[resource] += value
            squares[resource] += value * value
            uniform_sums[resource] += uniform
            uniform_squares[resource] += uniform * uniform
            root_sums[resource][row['root_id']] += value
            supporting_roots[resource].add(row['root_id'])
            family, literal = labels[resource]
            weighted.append(dict(resource_id=resource, family=family, value=literal,
                                 normalized_weight=value, uniform_normalized_weight=uniform))
        incidence.append(dict(record_id=row['record_id'], payload_sha256=row['payload_sha256'],
                              root_id=row['root_id'], resources=weighted))
    rows = []
    for resource, count in support.items():
        pairs = count * (count - 1) // 2
        mass = max(0.0, (sums[resource] ** 2 - squares[resource]) / 2) if pairs else 0.0
        uniform = max(0.0, (uniform_sums[resource] ** 2 - uniform_squares[resource]) / 2) if pairs else 0.0
        cross_root = max(0.0, (sums[resource] ** 2 - math.fsum(v*v for v in root_sums[resource].values())) / 2)
        family, literal = labels[resource]
        rows.append(dict(resource_id=resource, family=family, value=literal,
                         supporting_records=count, supporting_roots=len(supporting_roots[resource]),
                         possible_record_pairs=pairs, idf=idf[resource],
                         normalized_weight_sum=sums[resource], normalized_weight_square_sum=squares[resource],
                         overlap_mass=mass, uniform_overlap_mass=uniform,
                         cross_root_overlap_mass=cross_root,
                         mean_cosine_contribution=mass/pairs if pairs else 0.0))
    total = math.fsum(row['overlap_mass'] for row in rows)
    rows.sort(key=lambda row: (-row['overlap_mass'], row['resource_id']))
    for rank, row in enumerate(rows, 1):
        row['rank'] = rank
        row['fraction_total_overlap_mass'] = row['overlap_mass']/total if total else 0.0
    for field, rank_name in [('supporting_records', 'frequency_rank'), ('uniform_overlap_mass', 'uniform_rank')]:
        for rank, row in enumerate(sorted(rows, key=lambda row: (-row[field], row['resource_id'])), 1):
            row[rank_name] = rank
    recurrent = [r for r in rows if r['supporting_records'] >= 2]
    selected = [r['resource_id'] for r in rows if r['overlap_mass'] > 0][:20]
    frequency = [r['resource_id'] for r in sorted(rows, key=lambda r: r['frequency_rank'])[:20]]
    uniform = [r['resource_id'] for r in sorted(rows, key=lambda r: r['uniform_rank']) if r['uniform_overlap_mass'] > 0][:20]
    summary = dict(payload_records=n, records_with_selected_reference=sum(bool(r['resources']) for r in payloads),
                   records_with_nonzero_weighted_profile=nonzero,
                   selected_reference_incidences=sum(support.values()), distinct_selected_references=len(support),
                   recurrent_references=len(recurrent), unordered_distinct_record_pairs=n*(n-1)//2,
                   total_cosine_overlap_mass=total,
                   mean_cosine_over_all_record_pairs=total/(n*(n-1)/2) if n > 1 else None,
                   total_uniform_overlap_mass=math.fsum(r['uniform_overlap_mass'] for r in rows),
                   total_cross_root_overlap_mass=math.fsum(r['cross_root_overlap_mass'] for r in rows),
                   top20_overlap_resources=selected, top20_frequency_resources=frequency,
                   top20_uniform_resources=uniform,
                   overlap_frequency_top20_intersection=len(set(selected)&set(frequency)),
                   overlap_uniform_top20_intersection=len(set(selected)&set(uniform)),
                   recurrent_mass_frequency_spearman=_correlation([r['overlap_mass'] for r in recurrent],
                                                                  [r['supporting_records'] for r in recurrent]),
                   recurrent_mass_uniform_spearman=_correlation([r['overlap_mass'] for r in recurrent],
                                                                [r['uniform_overlap_mass'] for r in recurrent]))
    return dict(resources=rows, incidence=incidence, summary=summary)


def analyze_records(records, config):
    fields = static._check_config(config)
    metadata, payloads, labels = {}, [], {}
    kinds, excluded, local_paths = Counter(), Counter(), Counter()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError('Each source row must be a mapping')
        record_id, kind, parent = [record[fields[k]] for k in ('id', 'kind', 'parent_id')]
        if (not isinstance(record_id, str) or not record_id or not isinstance(kind, str) or not kind
                or parent is not None and (not isinstance(parent, str) or not parent)):
            raise ValueError('Invalid extraction metadata')
        if record_id in metadata:
            raise ValueError('Duplicate extraction record ID')
        metadata[record_id] = (kind, parent)
        kinds[kind] += 1
        if kind != config['payload_kind']:
            continue
        body = record[fields['text']]
        refs, dropped = static.extract_references(body)
        excluded.update(dropped)
        local_paths['records_with_local_path'] += bool(refs['local_path'])
        local_paths['local_path_incidences'] += len(refs['local_path'])
        selected = set()
        for family in FAMILIES:
            for value in refs[family]:
                resource = static._identity(config['source_sha256'], family, value)
                labels[resource] = (family, value)
                selected.add(resource)
        payloads.append(dict(record_id=record_id, payload_sha256=static._sha_bytes(body.encode('utf-8')),
                             resources=selected))
    roots = _roots(metadata)
    payloads.sort(key=lambda r: r['record_id'])
    groups = defaultdict(list)
    for row in payloads:
        row['root_id'] = roots[row['record_id']]
        groups[row['payload_sha256']].append(row)
    unique = sorted((group[0] for group in groups.values()), key=lambda r: r['record_id'])
    primary, raw = aggregate(unique, labels), aggregate(payloads, labels)
    p_by_id = {r['resource_id']: r for r in primary['resources']}
    r_by_id = {r['resource_id']: r for r in raw['resources']}
    common = sorted(p_by_id)
    recurrent = [r for r in common if p_by_id[r]['supporting_records'] >= 2]
    review = []
    top = primary['summary']['top20_overlap_resources']
    support = defaultdict(list)
    for row in primary['incidence']:
        for item in row['resources']:
            if item['resource_id'] in top:
                support[item['resource_id']].append(dict(record_id=row['record_id'],
                    payload_sha256=row['payload_sha256'], normalized_weight=item['normalized_weight']))
    for resource in top:
        row = p_by_id[resource]
        review.append(dict(resource_id=resource, rank=row['rank'], family=row['family'], value=row['value'],
                           supporting_records=row['supporting_records'], overlap_mass=row['overlap_mass'],
                           records=sorted(support[resource], key=lambda x: (-x['normalized_weight'], x['record_id']))[:3]))
    summary = dict(schema_version='swarm-resource-overlap-v1', source_sha256=config['source_sha256'],
        total_records=len(metadata), record_kind_counts=dict(sorted(kinds.items())),
        payload_records=len(payloads), exact_unique_payloads=len(unique),
        duplicate_records=len(payloads)-len(unique), duplicate_groups=sum(len(g)>1 for g in groups.values()),
        graph=dict(unique_ids=len(metadata), dangling_parents=0, cycles=0,
                   payloads_with_parent=sum(metadata[r['record_id']][1] is not None for r in payloads),
                   distinct_payload_roots=len({r['root_id'] for r in payloads}),
                   interpretation='Extraction-parent roots, not actors, runs, or time.'),
        excluded_content_records=len(metadata)-len(payloads),
        ignored_fields=['tags', 'cite'], selected_families=list(FAMILIES),
        excluded_local_path_coverage=dict(local_paths), reference_exclusions=dict(sorted(excluded.items())),
        definitions=dict(unit='Exported payload record; exact whole-body duplicates collapsed in primary.',
            weight='Binary occurrence times ln(N/df), L2-normalized within each payload. Empty payloads enter N.',
            overlap_mass='Sum z_ir*z_jr over unordered distinct payload pairs i<j; equals ((sum z_ir)^2-sum z_ir^2)/2.',
            mean_cosine_contribution='Resource contribution divided by choose(df,2); not full pair similarity.',
            uniform_overlap_mass='Same exact calculation after L2-normalizing binary occurrence with no IDF.',
            cross_root_overlap_mass='Same contribution sum excluding pairs with a common extraction root.',
            review='Top20 positive-mass resources, then up to3 distinct exact-body representatives by decreasing resource normalized weight and lexicographic record ID.',
            frequency_comparison='Rank by supporting record count; Spearman uses recurrent resources and average ranks for ties.',
            transfer='Exploratory change of observation unit and unsupervised corpus rarity; not the validated Village agent-day threshold.'),
        primary=primary['summary'], raw_sensitivity=raw['summary'],
        sensitivity=dict(primary_raw_top20_intersection=len(set(top)&set(raw['summary']['top20_overlap_resources'])),
            primary_raw_mass_spearman_recurrent=_correlation([p_by_id[r]['overlap_mass'] for r in recurrent],
                                                            [r_by_id[r]['overlap_mass'] for r in recurrent])),
        no_external_investigation_used=True, no_execution_or_temporal_inference=True,
        limitations=['Resource overlap does not identify actors or establish executed behavior.',
                     'Total overlap contribution can track resource frequency; frequency and uniform baselines are reported.',
                     'Opaque alias equality does not guarantee unique original endpoint identity.',
                     'No pair threshold, maliciousness labels, inferential p-values, or incident timeline is fitted.'])
    return dict(summary=summary, primary=primary, raw=raw,
                review_selection=dict(method=summary['definitions']['review'], selected_resources=review))


def _code_hashes():
    return dict(static._code_hashes(), **{Path(__file__).name: static._sha_file(Path(__file__))})


def run_file(source, config_path, output, protocol_path=PROTOCOL):
    source, config_path, output, protocol_path = map(Path, (source, config_path, output, protocol_path))
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing analysis')
    protocol_hash = static._sha_file(protocol_path)
    config_hash = static._sha_file(config_path)
    config = json.loads(config_path.read_text(), object_pairs_hook=static._unique_object)
    static._check_config(config)
    source_stat, code_hashes = source.stat(), _code_hashes()
    if static._sha_file(source) != config['source_sha256']:
        raise ValueError('Source SHA-256 differs from configured source')
    opener = gzip.open if config['compression'] == 'gzip' else open
    with opener(source, 'rt', encoding='utf-8') as handle:
        result = analyze_records((json.loads(line, object_pairs_hook=static._unique_object) for line in handle), config)
    if config['source_sha256'] == KNOWN_SOURCE:
        if (result['summary']['payload_records'], result['summary']['exact_unique_payloads']) != (91037, 88672):
            raise ValueError('Payload totals differ from the source inventory')
        if result['summary']['graph']['payloads_with_parent']:
            raise ValueError('Source payload root structure differs from the inventory')
        if (result['summary']['primary']['selected_reference_incidences'],
                result['summary']['raw_sensitivity']['selected_reference_incidences']) != (95959, 98395):
            raise ValueError('Selected reference incidences differ from the independent inventory')
    if (source.stat().st_size, source.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError('Source changed during analysis')
    if (_code_hashes() != code_hashes or static._sha_file(config_path) != config_hash
            or static._sha_file(protocol_path) != protocol_hash):
        raise ValueError('Code, config, or protocol changed during analysis')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'.tmp-', dir=output.parent))
    try:
        for key in ('primary', 'raw'):
            rows = result[key]['resources']
            with (staging/(key+'-resource-ranking.csv')).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['resource_id', 'rank'])
                writer.writeheader(); writer.writerows(rows)
            with gzip.open(staging/(key+'-incidence.jsonl.gz'), 'wt', encoding='utf-8') as handle:
                for row in result[key]['incidence']:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        for filename, value in [('summary.json', result['summary']), ('review-selection.json', result['review_selection'])]:
            (staging/filename).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        manifest = dict(schema_version='swarm-resource-overlap-manifest-v1',
                        source_sha256=config['source_sha256'], source_bytes=source_stat.st_size,
                        config_sha256=config_hash, protocol_file=str(protocol_path), protocol_sha256=protocol_hash,
                        code_sha256=code_hashes, python=sys.version,
                        outputs_sha256={p.name:static._sha_file(p) for p in sorted(staging.iterdir())},
                        no_external_investigation_used=True, no_pair_matrix_materialized=True)
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL,
                        help='Frozen ranking protocol recorded in the output manifest')
    args = parser.parse_args()
    manifest = run_file(args.source, args.config, args.output, args.protocol)
    print(json.dumps(dict(output=str(args.output), source_sha256=manifest['source_sha256'],
                          outputs=sorted(manifest['outputs_sha256']))))


if __name__ == '__main__':
    main()
