#!/usr/bin/env python3
"""Freeze the development results before the transfer study.

Copies the detector code and the 6 July-14 August development outputs into
outputs/generalization-20261106/frozen/ and writes freeze-manifest.json next to
it. The transfer code reads the frozen copies and refuses to run if a frozen
source file changes afterwards. This recreates the freeze made in the original
study, with the same file list.

It also reports whether the code being frozen is byte-identical to the code
frozen in the original study (hashes below). A difference does not stop the
freeze, but results may then differ from the reported ones.

Usage: python scripts/freeze_development.py [--require-original-code]
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs' / 'generalization-20261106'

# SHA-256 of every code file frozen in the original study (6 November 2026).
ORIGINAL_CODE_SHA256 = {
    'data/manifest.json':
        '1256cc1851f7c2b05e395aa996626c1c70293759be57103d6d5ede35474282ea',
    'pyproject.toml':
        '572ee4481dd01611375fcddf84c39d19527e49ac27cd1b33428081564f40956a',
    'src/swarmcollab/__init__.py':
        'ad5aa30010cce9d696b09f2728bb9bb33dbc6f1c77c9a9c8747b6afb1e2fce79',
    'src/swarmcollab/agent_traces.py':
        'd1e0da27b2e382e30940662baad2e69493ed559126dad3af3e904bd4e8747ec9',
    'src/swarmcollab/config.py':
        '4d6343c5777fde2b63a6722db4b76e24d8a766a98ac77fdf6a987827170b0d56',
    'src/swarmcollab/extract.py':
        'df5f3e8a9216179dce8c30c22219131ab54b4f6457a7e6f99d08eb4e27146698',
    'src/swarmcollab/hawkes.py':
        '08e4dceeb8a3855ce37fbbbe59bd0d98d4b490f614808ce900a2a236abe7125a',
    'src/swarmcollab/labels.py':
        '0e7208950b84ea451edf23c81e6a7bc8d28fc6135d9a52b7c88c49df8dc0037b',
    'src/swarmcollab/layer_scoring.py':
        '3230e0bdb614cced7cd314d5ed7db284d2ee92e6d0464b9419cb5475c7e5b76d',
    'src/swarmcollab/metrics.py':
        '53d01d6fd19948a6a123b4318fab61c56711298f592d36c9d3658617a0957156',
    'src/swarmcollab/monitor_evaluation.py':
        'a0cd739c53dc6a4aad910fa602460c7308a098ae343a906553fbb680948f0829',
    'src/swarmcollab/pairs.py':
        '43a7298e1fa4a20f1b8bfe6b018419b7baaa4e611471668521ed952c0fc8d488',
    'src/swarmcollab/provenance.py':
        'bcfe8e773345bafa56d32059b6770f714a48af3dcb65850c6c8ff2dfe603f553',
    'src/swarmcollab/provenance_common.py':
        'dd07bce752043012a2d13f6e101d888bbfda9aad72c280d47c5e959c6cdafb5c',
    'src/swarmcollab/provenance_controls.py':
        'd3f5e47b5b86f3743b78e31a810774e8d6c7001cf181e7a4ac0f78377d53ce0f',
    'src/swarmcollab/provenance_git.py':
        '8b20509952ee34f775c0e5803cd3463a98eec241cd7933a26bb5cc3c86015761',
    'src/swarmcollab/provenance_links.py':
        '8c5faece387da78c71467a3a15ebb28d40dc2e7df1ab5a86840bfd5b7b6444cc',
    'src/swarmcollab/provenance_structured.py':
        '0668ca39c5a1d139bf2e6e834decc5b86b01d8c0943a38047e86f8815a553a21',
    'src/swarmcollab/regimes.py':
        'b764c27445d17633bd607ed46bf3503959c1c514ce28035ccffc0eccf672b728',
    'src/swarmcollab/research_layers.py':
        '165632e4fea15639216f5320290b7b644b14937ca0f20bdc9413f60c1492734c',
    'src/swarmcollab/spread.py':
        'c6d0e6946d9de7ba08590ba37c692c520c99e2a3ca85708b5b9f07d9bf21b1e0',
    'src/swarmcollab/timeline.py':
        'a2c4fdd151b043d931fad421ecd1ab9186f975bb2bf9f63e68fafe8ab950d7d2',
    'src/swarmcollab/timescale.py':
        'ef568054cf54b7cbf938f908141bfbd3bd6a9cd946d096deff121e907ea1ecb1',
    'src/swarmcollab/traces.py':
        '87c9b7e1d0e45b0df07a28f2b061426f18afcd2c2b1b3f6b9e9cd8248283a9bf',
    'src/swarmcollab/village_traces.py':
        'a36a3f72e626c4aabda034fba3d79e47fca69630bdf1f4e12998f3c09c764290',
    'src/swarmcollab/visibility.py':
        'dfab1627e5229061433a36f3aa9668ba8f8d914a0f117bded68cf4084aefe3f3',
    'src/swarmcollab/workflow_patterns.py':
        '0154236aacc6138be4b2a2ba3b8f640731b0eff7f503f17a9b665ac92b8c8854',
    'tests/test_agent_traces.py':
        '3575d966635769b1587d76c553dbcd4321d73606e685a4ece3a0c93e3464984b',
    'tests/test_labels.py':
        'aaf54ceab68d324c783301be3e926ba0a20bc78d60249903617c1644eb16fdee',
    'tests/test_layer_scoring.py':
        '0ee01d472242b5a424975f179e487780667a85a25fea042f3586299e1ecf6cd9',
    'tests/test_monitor_evaluation.py':
        'b081e73caf306ef15b9618c03308e827620e0a54b5b7e0d123afa2e5180853c5',
    'tests/test_parsing.py':
        '9e183d26ead951bf91d4469459cd92cdb11e85e6c132b8de165e72dfbbbd2b28',
    'tests/test_provenance_controls.py':
        '68384192fb2ef664cd893118e797003d44797899354af01aff30428a5306c8cd',
    'tests/test_provenance_git.py':
        '75cb455a009f929b08a8b757a5a44f71f66fde36375c21c0c8ea3e752162a7a9',
    'tests/test_provenance_links.py':
        'a6c280781cb0b0625ecfb9db6d93556be8ca25ed37af5626a93c33d439c53f93',
    'tests/test_provenance_structured.py':
        '6079e3b4d0ab05c7f08bbe57bcea4f845dfa9c6737a18b2f5fba721195c9aa85',
    'tests/test_research_layers.py':
        '95cb1e73836a67638939ccfa379848ad4325d3566db5db90ea5258a78cb92569',
    'tests/test_validation.py':
        'adc75113cedd8f39b4b69d17fab9b9a6c4a629752974352c319f60d5cbf4a619',
    'tests/test_village_traces.py':
        '2335eac449af2725348e48b33de8d891dad3bb27d1f8bf4483af2bde0752709b',
    'tests/test_workflow_patterns.py':
        'ae0924e7beedd20d1af8ebef68253f430093817ecb14a6897a98de845e5aa6a8',
}

# Development outputs read by the transfer and secondary analyses.
OUTPUT_FILES = [
    'outputs/research-layers/scores.json',
    'outputs/research-layers/feature_manifest.json',
    'outputs/research-layers/village_rules.json',
    'outputs/research-layers/extraction.json',
    'outputs/research-layers/summary.json',
    'outputs/research-layers/features_dev.csv.gz',
    'outputs/research-layers/predictions_dev.csv.gz',
    'outputs/pairs/features_dev.csv.gz',
    'outputs/pairs/scores_dev.json',
    'outputs/provenance/summary.json',
    'outputs/provenance/extraction.json',
    'outputs/provenance/features_dev.csv.gz',
    'outputs/provenance/chains.jsonl.gz',
    'outputs/spread/alpha_dev.csv',
    'outputs/spread/alpha_excess_dev.csv',
    'outputs/spread/summary_dev.json',
    'outputs/hawkes/alpha_dev.csv',
    'outputs/hawkes/alpha_excess_dev.csv',
    'outputs/hawkes/summary_dev.json',
    'outputs/regimes/summary_dev.json',
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions():
    from importlib import metadata
    versions = {}
    for name in ('numpy', 'pandas', 'scipy', 'matplotlib'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--require-original-code', action='store_true',
                        help='fail if any frozen code file differs from the original study')
    args = parser.parse_args()

    manifest_path = STUDY / 'freeze-manifest.json'
    if manifest_path.exists():
        raise SystemExit(f'{manifest_path} already exists; refusing to overwrite a freeze.')
    missing = [rel for rel in list(ORIGINAL_CODE_SHA256) + OUTPUT_FILES if not (ROOT / rel).is_file()]
    if missing:
        raise SystemExit('Missing files; run the development stage first:\n  ' + '\n  '.join(missing))

    differing = [rel for rel, expected in ORIGINAL_CODE_SHA256.items()
                 if sha256(ROOT / rel) != expected]
    if differing and args.require_original_code:
        raise SystemExit('Code differs from the original freeze:\n  ' + '\n  '.join(differing))

    frozen = STUDY / 'frozen'
    files = {}
    for rel in list(ORIGINAL_CODE_SHA256) + OUTPUT_FILES:
        target = frozen / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
        files[rel] = {'sha256': sha256(target), 'bytes': target.stat().st_size}

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        'created_utc': now,
        'git_head': git_head(),
        'files': files,
        'code_matches_original_study': not differing,
        'code_files_differing_from_original_study': differing,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'packages': package_versions(), 'recorded_utc': now}
    (STUDY / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')

    print(f'Froze {len(files)} files into {frozen.relative_to(ROOT)}')
    if differing:
        print('WARNING: code differs from the original study in:', *differing, sep='\n  ')
    else:
        print('Frozen code is byte-identical to the code frozen in the original study.')


if __name__ == '__main__':
    main()
