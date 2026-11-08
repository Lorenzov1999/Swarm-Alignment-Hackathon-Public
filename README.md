# Monitoring LLM Agents Through Shared Resources

Code for the report *Monitoring LLM Agents Through Shared Resources*
(Apart Research, Swarm Alignment sprint, November 2026).

The current paper uses a generic parser and a rarity-weighted shared-resource
score to identify likely collaboration in AI Village work logs, with an
activity-only control. July weights are held fixed for August validation and
later testing through 15 September. Four selected daily networks illustrate
shared work and GUI assistance. The Swarm Traces case study applies resource
overlap to distinct payload records and ranks resources for source review.
Extraction and scoring run on CPUs without LLM calls.

This release includes the IPv4/IPv6 URL correction, Swarm resource ranking,
the current paper's numerical checks, and SVG/PDF network rendering. Older
Hawkes, provenance and combined-detector experiments remain available as
historical research code; they are not methods in the current paper.

## Prior work and authorship

All code in this repository was written during the sprint, 6 to 8 November
2026. No code predates it. Pre-sprint preparation (downloading data, reading
the literature, exploratory scans) used temporary code that was deleted and is
not included. As stated in the report, LLM coding assistants wrote most of the
implementation under the author's direction.

## Requirements

- Python 3.9 or newer (tested with 3.9). Pinned dependencies, installed from
  `pyproject.toml`: numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, matplotlib 3.9.4.
- Optional: Node.js 20. The Swarm request-expression census parses JavaScript
  with the Acorn parser bundled in Node. Without Node, JavaScript is skipped
  and that census covers far fewer payloads than reported.
- Figures: Graphviz (`neato`) and librsvg (`rsvg-convert`). These are
  system tools; numerical analysis does not need them.
- Optional: git, used by one test (skipped without it).
- About 4 GB of downloads and a few GB of outputs. The full run was developed
  on 4 CPUs with 8 GB of RAM.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -I tests/test_parsing.py
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

## Get the data

The data are not included. `data/manifest.json` records the size and SHA-256
of every analysed file; `scripts/check_data.py` compares your copies with it.

**1. AI Village logs.** The dataset at
<https://huggingface.co/datasets/aidigestorg/ai-village> is gated: sign in and
accept AI Digest's research terms first (access is reviewed manually). Then
download three files into `data/ai-village/main-village/`:

```bash
pip install huggingface_hub     # provides the `hf` command (older versions: `huggingface-cli`)
hf auth login
hf download aidigestorg/ai-village --repo-type dataset --local-dir data/ai-village \
  main-village/computer_use_turns.jsonl.gz \
  main-village/computer_use_sessions.jsonl.gz \
  main-village/agents.jsonl.gz
```

AI Digest refreshes the dataset regularly, so a new download will not match the
recorded SHA-256. The registered analysis windows lie between 1 June and 15 September 2026.
Changed records or request annotations within those dates can change results.

**2. Request labels.** The evaluation labels are the agent-to-agent requests in
AI Digest's public request chart:

```bash
.venv/bin/python scripts/fetch_labels.py data/labels
```

This writes `data/labels/ai-village-chart-props.json`. The analysed snapshot was
taken on 5 November 2026. A new snapshot also contains later requests, so its
SHA-256 differs. The pipeline only uses requests up to 15 September 2026; the
label diagnostic checks the snapshot strictly and needs
`--allow-label-snapshot-mismatch` (see below).

**3. Swarm Traces export.** Only needed for the Swarm stage:

```bash
mkdir -p data/swarm-traces
curl -L -o data/swarm-traces/redacted.jsonl.gz https://swarmtraces.org/data/final/redacted.jsonl.gz
```

The analysed file has SHA-256
`7b66ab21674de52fcd3f557652f68b1801170c998e2f266862124e6edf283488`; the Swarm
scripts refuse any other file.

Check everything with:

```bash
.venv/bin/python scripts/check_data.py
```

## Run the current code

After obtaining the data and installing the package:

```bash
.venv/bin/python scripts/reproduce.py --dry-run
.venv/bin/python scripts/reproduce.py --allow-label-snapshot-mismatch
```

The runner retains the historical preparation stages because the saved feature
and transfer tables feed the current paper checks. It then runs the current
paper reports and Swarm resource ranking. Use `--stages` or `--from` to select
stages; `--dry-run` prints commands. Logs are written to `outputs/logs/`.
Optional numeric-correction replay is described below.

The current additions are `current-paper`, `swarm-ranking`, and the explicitly
selected `numeric-correction` stage. For saved inputs:

```bash
.venv/bin/python scripts/reproduce.py --stages current-paper
.venv/bin/python scripts/reproduce.py --stages current-paper --paper-mode frozen-paper
.venv/bin/python scripts/reproduce.py --stages numeric-correction --paper-mode frozen-paper
.venv/bin/python scripts/reproduce.py --stages swarm-ranking
```

Use `current-paper`'s default `current-run` mode for newly computed features;
use `frozen-paper` only for the original retained baseline. The selected stages
consume existing preparation outputs rather than automatically running their
prerequisites.

Preparation stages resume by skipping existing outputs; current-paper reports
and figures are refreshed when selected. Use a clean checkout/output
workspace when changing source data or extraction code: the existence of a
file does not certify that it was produced by the current parser. Raw-data
preparation can take one to two hours on four CPUs; reporting from saved tables
is much faster.

A fresh run uses the corrected parser. It is a new computation, not a
byte-for-byte recreation of the earlier evaluation. New dataset or request
snapshots can also change the numbers. Input and source hashes are recorded.

### Original evaluation and the later parser correction

The paper preserves the original evaluation and separately reports the numeric
URL correction, which is included in its network illustrations. The correction
leaves the original training, validation and test alert sets unchanged; test
AUROC still rounds to 0.817. On June 22 it adds 11 links among users of one
remote desktop, bringing the illustrated graph from 16 to 27 links.

Public commit `0b97d55` contains the exact pre-correction code used by the
original freeze. To recreate that baseline, run its preparation and diagnostic
stages in a separate checkout with the recorded inputs, preserve the resulting
`outputs/` and frozen source, then use the current reporting and correction
scripts on those artifacts. For example:

```bash
git worktree add ../swarm-original 0b97d55
```

Obtain/install the data and environment in that checkout using its README.
Do not rebuild its old freeze with the corrected parser: the comparison needs
both versions. The correction runner deliberately rejects a baseline that
already admits numeric IP endpoints. See
[the numeric endpoint audit](docs/numeric-endpoint-fix.md) for the required
caches and commands. No original data, score cache or review annotations are
bundled in this code release.

## Current paper scripts

Every script offers `--help` and accepts explicit input/output paths; no sibling
private repository or Overleaf checkout is required. Default paper artifacts
are written under `outputs/current-paper/`.

| Script/module | Purpose |
|---|---|
| `scripts/check_shared_resources.py` | Shared-resource and activity comparisons, within-pair/day checks, follow-through, operating point and descriptive intervals. |
| `scripts/make_shared_resource_figures.py` | Pipeline schema and shared-resource alert-workload SVG/PDF figures. |
| `scripts/build_village_network_snapshots.py` | Build the four selected daily networks, retaining agents without detected links. |
| `scripts/build_corrected_village_network_snapshots.py` | Build the same views from the separate numeric-IP correction scores. |
| `scripts/draw_village_networks.py` | Render network SVGs/PDFs using one edge-width scale, with independently chosen layouts. |
| `scripts/review_numeric_endpoint_fix.py` | Compare newly admitted IP resources against the original frozen baseline, preserving July weights and threshold. |
| `swarmcollab.swarm_resource_overlap` | Rank shared resources across distinct Swarm payload records; compare frequency and deduplication controls. |
| `scripts/review_swarm_resource_contexts.py` | Export the fixed, rank-selected source contexts without executing payload code. |
| `scripts/summarize_swarm_resource_review.py` | Summarize separately supplied reviewer annotations; does not create annotations automatically. |

The network builder distinguishes the original reviewed snapshots from a fresh
run. A new run must not inherit historical topology or coverage claims merely
because it uses the same dates. The graph is undirected resource overlap; source
review supplies the contextual description of selected partnerships.

## Swarm resource ranking

This stage can run without the Village data:

```bash
.venv/bin/python -m swarmcollab.swarm_resource_overlap \
  --source data/swarm-traces/redacted.jsonl.gz \
  --config config/swarm-schema-adapter.json \
  --protocol docs/swarm-resource-ranking-protocol.md \
  --output outputs/swarm-resource-ranking-20261108
.venv/bin/python scripts/review_swarm_resource_contexts.py \
  --source data/swarm-traces/redacted.jsonl.gz \
  --ranking outputs/swarm-resource-ranking-20261108
```

The input hash is fixed. Ranking and review-selection artifacts are saved before
source-context inspection. The ranker refuses to overwrite an existing output
folder. Source annotations and the paper's contextual assessments require
separate review; the automatic ranking does not classify maliciousness or
establish that code executed. No external incident-investigation outcomes are
used by these scripts. The [protocol](docs/swarm-resource-ranking-protocol.md)
defines the record population, resource score and review selection.

## Historical stages and data boundaries

The retained preparation runner includes `development`, `freeze`, `transfer`,
`report`, `followup`, `diagnostics`, and `swarm`. These retain the earlier
combined-detector and static-census analyses for reproducibility. Historical
output directories retain their original dates because scripts refer to them
by name. They should not be interpreted as the current paper's table numbers.

The registered Village windows are training (6–31 July), validation
(3–14 August), test (17 August–15 September), and earlier shared goals
(1 June–5 July), all in 2026 and using Pacific dates. The registered runners
exclude 16 September onward. The legacy exploratory `timeline` module and
`pairs --final` option are not part of this workflow.

The parser filters explicit chat/history operations, room moves, session goals
and waiting turns as documented by each stage. Text retained inside tool outputs
can still contain messages. GUI activity remains in scope, but screenshots and
other non-text content do not supply resource identifiers.

## Repository layout

| Path | Contents |
|---|---|
| `src/swarmcollab/` | Current parser, resource methods and retained historical analyses. |
| `scripts/` | Reproduction, reporting and rendering commands. |
| `tests/` and `scripts/test_*.py` | Unit tests and focused numerical/graph checks. |
| `config/` | Frozen goal categories and Swarm export/schema protocols. |
| `docs/` | Numeric-endpoint audit guide and resource-ranking protocol. |
| `data/manifest.json` | File sizes and SHA-256 hashes for the analyzed snapshots. |

Raw datasets, output caches, credentials and local environments are excluded
from Git. The published code does not include gated-log excerpts or the
researcher's private source-review files.

## Data terms and citation

The AI Village data are subject to AI Digest's research terms: research use
only, no training or fine-tuning of AI systems without written permission, and
citation of AI Digest, "AI Village dataset", 2026,
<https://theaidigest.org/village>. The Swarm Traces export is described in
Forman et al., "Revealing the details of how OpenAI agents hacked Hugging
Face", Swarm Traces, 2026, <https://swarmtraces.org/>.

## License

The code is released under the MIT License (see `LICENSE`). The license does
not cover the datasets.
