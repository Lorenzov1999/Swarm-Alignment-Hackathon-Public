# What Work Traces Reveal About AI Agent Interactions

Code for the report *What Work Traces Reveal About AI Agent Interactions*
(Apart Research, Swarm Alignment sprint, November 2026).

The analysis asks how much of the interaction between AI agents can be recovered
from the work they leave behind, without reading their messages. It uses the
[AI Village](https://theaidigest.org/village) computer-use logs with explicit
chat and history operations removed, and:

- ranks agent pairs that may be interacting on each day from shared resources,
  timing, handovers and provenance (a portable layer, plus a layer of
  Village-specific rules);
- links artifact versions from the agent that produced them to the agent that
  acquired or incorporated them (provenance, with evidence levels);
- freezes models fitted on July 2026 and tests them, without retuning, on later
  work (17 August to 15 September) and on earlier goal episodes (1 June to
  5 July), against the request records published by AI Digest;
- applies the portable reference matching to the redacted
  [Swarm Traces](https://swarmtraces.org/) export of the Hugging Face incident.

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
- Optional: git, used by one test (skipped without it).
- About 4 GB of downloads and a few GB of outputs. The full run was developed
  on 4 CPUs with 8 GB of RAM.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
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
recorded SHA-256. The analysis only reads 1 June to 15 September 2026, so the
results should not change unless records from that period were edited.

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

## Reproduce the results

```bash
.venv/bin/python scripts/reproduce.py --allow-label-snapshot-mismatch
```

`--allow-label-snapshot-mismatch` is only needed with a newly fetched label
snapshot. The run takes one to two hours on 4 CPUs. Each step is skipped when
its output already exists, so an interrupted run resumes when you run the same
command again. Logs are written to `outputs/logs/`. Use `--dry-run` to print
the commands, `--stages report,diagnostics` to run selected stages, or
`--from transfer` to run one stage and everything after it.

| Stage | What it does |
|---|---|
| `data` | Checks the input files against `data/manifest.json`. |
| `development` | 6 July to 14 August pipeline: working steps (chat, history, room and session-goal operations dropped), resource traces, pair-day features and scores, daily regimes, Hawkes models, versioned provenance, and the portable and Village-specific layers. Training is 6 to 31 July; 3 to 14 August is the development validation. |
| `freeze` | Copies the code and development outputs into `outputs/generalization-20261106/frozen/` and records their hashes; fits the frozen July state and the secondary models. Reports whether the code is byte-identical to the code frozen in the original study. |
| `transfer` | Applies the frozen models, unchanged, to 17 August to 15 September and to 1 June to 5 July. Predictions are saved before the request labels are joined. |
| `report` | Comparison tables, provenance by goal episode, and a hash lock of the transfer results that later stages check. |
| `followup` | Alert workload and goal-episode comparisons. |
| `diagnostics` | Baselines (activity, July pair history, within-pair), visibility and follow-through, goal-episode coverage, and Figure 4. |
| `swarm` | Swarm Traces: static reference analysis, wording census, request-expression census, embedded date/time inventory, artifact-text census. |

Output folders keep the dates of the original study (for example
`generalization-20261106`), because later stages look for them by name.

## Where the results are

| Report item | File |
|---|---|
| Tables 1 and 2 (AUROC and average precision) | `outputs/generalization-20261106/report/primary-comparison.csv` |
| Activity, July pair-history and within-pair baselines | `outputs/claude-review-20261107/baselines/summary.json` |
| Within-pair and within-day comparisons with activity | `outputs/claude-review-20261107/activity-concordance-review.json` |
| Table 3 (recall and false-positive rate at July thresholds) | `outputs/generalization-20261106/report/primary-comparison.csv` |
| Alert workload and precision | `outputs/cooperation-followup-20261106/village/monitoring.json` |
| Follow-through and visibility | `outputs/claude-review-20261107/labels-visibility/` (`summary.json`, `single-request-sensitivity.json`) |
| Figure 4 and the coverage correlations | `outputs/figures/goal-co-use-robustness.pdf`, `goal-coverage-review.json` |
| Table 4 and the provenance counts by period and goal | `outputs/cooperation-followup-20261106/village/group-summary.csv`, `rate-comparisons.json` |
| Provenance chains (including the inspected handoff) | `outputs/generalization-20261106/{temporal,shared_goals}/chains.jsonl.gz` |
| Spread control (development) | `outputs/spread/summary_dev.json` |
| Swarm Traces results | `outputs/generalization-20261106/swarm/static/`, `outputs/cooperation-followup-20261106/swarm/`, `outputs/swarm-improvement-20261107/` |

## What the scripts do not reproduce

Some results rest on annotations made by reviewers, not by code. The
annotation files are not distributed, so these steps are not run:

- **Provenance chain audits.** `swarmcollab.provenance` writes an audit sample
  (`outputs/provenance/audit_sample.jsonl`). Annotate a copy and score it with
  `python -m swarmcollab.provenance --skip-extract --annotations FILE`.
- **Swarm contextual reviews.** The classification of wording candidates, the
  review of date/time contexts and the artifact-text audits are read by
  `scripts/summarize_swarm_cooperation.py`, `scripts/swarm_time_acceptance.py`
  and `scripts/summarize_swarm_content_audits.py`. The census stage selects the
  audit samples and writes blank annotation forms.

## Repository layout

| Path | Contents |
|---|---|
| `src/swarmcollab/` | The analysis package |
| `scripts/` | Stage runners, reporting and diagnostics; `reproduce.py` runs everything |
| `tests/` | Unit tests |
| `config/goal-regime-classification.json` | Goal categories for 1 June to 5 July, assigned from the goal descriptions before any result was seen; later stages check this exact file by hash |
| `config/swarm-schema-adapter.json` | Field mapping for the Swarm Traces export |
| `config/swarm-content-protocol.txt` | The artifact-text census design, recorded by that census |
| `data/manifest.json` | Size and SHA-256 of the analysed data files |

Steps run with `PYTHONHASHSEED=0` and single-threaded linear algebra. Small
floating-point differences across platforms are still possible. Data from
16 September 2026 onward were held in reserve and never analysed; the code
refuses date windows other than the registered ones.

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
