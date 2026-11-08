# Numeric IP endpoints and the original evaluation

The current parser retains valid IPv4/IPv6 HTTP URLs, including endpoint ports
and exact page paths. Default ports and address spelling are normalized;
different ports, protocols, path case, queries and fragments remain distinct.
Loopback and unspecified addresses are excluded from cross-agent matching.
Matching a private address assumes the agents share the relevant network
context. The implementation contains no Village-specific VNC addresses or rules.

The original paper evaluation used a parser that excluded numeric hosts. Its
separate correction replay held July weights and the threshold fixed and added
330 mentions across 162 steps. Training, validation and test alert sets were
unchanged; test AUROC changed from 0.817314 to 0.817301. The June 22 network
gained 11 links among users of a shared remote desktop, from 16 to 27 edges.
Gemini 2.5's four neighbors were unchanged because its own eligible work steps
never named that endpoint. The other three illustrated networks kept their
edge sets. These are results on the retained study inputs, not promises about
new upstream snapshots.

## Reproduce the comparison

Use a baseline generated with pre-correction public commit `0b97d55`, then
upgrade the code while retaining that checkout's ignored data and outputs.
All 40 code/data-manifest hashes listed by the original freeze match that
commit. A fresh freeze built from the current parser is a different experiment
and cannot substitute for the old reference.

The audit requires:

- The pre-fix parser under `outputs/generalization-20261106/frozen/src/` and
  its `historical-freeze.json` with July identities and rarity weights.
- Development mentions in `outputs/mentions/` and pair features in
  `outputs/research-layers/features_dev.csv.gz`.
- Original June and test features under `outputs/generalization-20261106/`.
- June/test mention caches under
  `outputs/claude-review-20261107/regimes/inputs/{shared_goals,temporal}/mentions`.
  The historical diagnostics stage generates these inputs.
- Complete per-agent compact step caches for all three periods. Test dates
  use Pacific time: an older cache cut off at September 15 UTC omits 37
  eligible final-day steps and must not be used.

Inspect prerequisites before executing:

```bash
.venv/bin/python scripts/review_numeric_endpoint_fix.py --plan \
  --shared-steps outputs/generalization-20261106/shared_goals/steps \
  --temporal-steps outputs/generalization-20261106/temporal/steps
.venv/bin/python scripts/review_numeric_endpoint_fix.py \
  --shared-steps outputs/generalization-20261106/shared_goals/steps \
  --temporal-steps outputs/generalization-20261106/temporal/steps \
  --output outputs/numeric-endpoint-fix-20261108
```

`--study` and `--input-cache` select other archived locations;
`--published-results` optionally cross-checks the threshold against a supplied
paper-result JSON. There is no dependency on a private manuscript checkout.
An output directory must be new. Original inputs are never overwritten.

The runner reconstructs the original scores, reconciles cached activity for
every agent/day, and checks that additions come only from newly admitted IP
resources. It preserves untouched scores exactly and compares updated scores
with a full sparse-matrix recomputation. A separate July-only recalibration is
reported alongside the primary fixed-threshold comparison. No test labels
are used for fitting, and no later Village dates are read.

The optional `numeric-correction` reproduction stage builds and renders the
corrected network views from these results. It is excluded from a fresh-run
default because it requires the original pre-correction baseline.
