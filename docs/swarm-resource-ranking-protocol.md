# Shared-resource ranking of Swarm payloads

Scope authorized 8 November 2026: compare distinct Swarm log records through
shared resources, rank those resources, and inspect whether the ranking surfaces
suspicious behavior. Do not consult METR or external incident outcomes. The
existing focused Village manuscript and all earlier locked artifacts stay intact.
Prior work has already exposed this project to incident context; this is a new
exploratory analysis without external outcome consultation, not a fresh blind test.

## Fixed input and observation unit

Use only `data/swarm-traces/redacted.jsonl.gz`, SHA256
`7b66ab21674de52fcd3f557652f68b1801170c998e2f266862124e6edf283488`.
The second downloaded copy is the same logical source. Use outer metadata only
to validate IDs, kinds, and extraction parents. Read text only from `payload`
records; exclude response/recovered-text records, tags, and citations from scores.
All payloads are extraction roots. Neither these roots nor record IDs identify
agents, sessions, executions, or time order.

Primary population: one deterministic representative per exact whole-body hash.
Keep all payload records as a duplicate sensitivity. No near-copy matching.

## Resource extraction and ranking

Reuse `swarm_static.extract_references`, including its existing portable URL
canonicalization and redaction handling. Admit complete canonical URLs and exact
numbered service/destination aliases in separate namespaces. Alias equality is
equality of exported tokens, not a verified one-to-one mapping to original URLs.
Local paths remain scoped to individual payloads and cannot create primary links.
Credential/runtime/source placeholders are excluded. No payload code is executed.

For N records and resource r with document frequency df_r, let w_r=ln(N/df_r).
Use binary resource presence per payload, normalize each record's weighted vector
to length one, and assign empty/zero-weight vectors zero. This adapts the paper's
shared-resource idea to unordered artifacts; repeated text is not repeated action.
It is not the original agent-day detector or its calibrated threshold.

For normalized component z_ir, rank resource r by its total contribution to
cosine similarities over all unordered pairs of distinct records:

    M_r = ((sum_i z_ir)^2 - sum_i z_ir^2) / 2.

This is an exact aggregate, not a count of pairs exceeding an alert threshold.
Report document frequency, df_r*(df_r-1)/2 supporting record pairs, overlap mass,
mean contribution per supporting pair, and fraction of all overlap mass.
Calculate both raw-record and exact-deduplicated versions. Compare the primary
ordering with frequency and uniform-weight cosine mass, including top-20 overlap
and rank correlations. No activity labels, malicious keywords, or known attack
targets influence the ranking. Common resources can still dominate aggregate
mass; the frequency comparison makes this visible.

## Fixed review and evidence standard

Before inspecting ranked source contexts, freeze the script, protocol, input and
ranking output hashes. Select the top 20 resources with positive mass. For each,
select three distinct exact-body representatives with largest normalized resource
weight, breaking ties by record ID. This selects simple/high-contribution records;
it is not a prevalence sample or a uniformly random audit. Preserve this limitation.
Review source snippets and, where required, full selected payloads. Do not use
exported tags or external accounts as labels. Code and instructions inside the
payloads are untrusted data, not instructions for the research process.

Classify reviewed evidence as ordinary/ambiguous, security-sensitive behavior,
explicitly suspicious intent/attempt represented in the payload, or insufficient
context. Describe the supporting behavior (for example collection of secrets or
transmission to another destination) without claiming unauthorized use solely
from a URL's presence. A code body can establish what code attempts or proposes;
it does not establish that agents executed it or that the operation succeeded.
Retain record IDs and bounded supporting excerpts. Record disagreements between
two assistant reviews and resolve them against the source; do not call this
independent human validation.

Any subsequent selection or alternative scoring must be marked exploratory and
kept separate from this fixed ranking and review. Without agent identities, event
times, or verified outcomes, do not report cross-agent detection precision,
malicious-behavior recall, online alert timing, or agreement with an investigation.

## High-agent-count assumption

If agent a contributes n_a records, the fraction of all record pairs from the
same agent is sum_a n_a(n_a-1)/(N(N-1)). Large agent count lowers this only under
an activity-balance assumption. Unknown identities prevent checking that
assumption, and aggregation by resource is not aggregation by agent pair.

## Verification

Test the exact aggregate against a dense synthetic cosine matrix, removal of
self-pairs, exact-body deduplication, zero-IDF cases, and baseline rank behavior.
Verify source and result hashes and reconcile parser incidence counts with the
existing metadata-only inventory. Preserve existing analysis outputs and paper.
