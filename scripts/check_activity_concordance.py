"""Reproduce descriptive activity comparisons from frozen later-window scores.

Run: python3 scripts/check_activity_concordance.py
Only saved predictions within August 17–September 15 are read.
"""
from pathlib import Path
from collections import defaultdict
import csv,gzip,hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'outputs/generalization-20261106/temporal/predictions.csv.gz'
manifest=ROOT/'outputs/claude-review-20261107/baselines/manifest.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected=json.loads(manifest.read_text())['inputs_sha256'][str(source.relative_to(ROOT))]
assert sha(source)==expected
with gzip.open(source,'rt') as f:rows=list(csv.DictReader(f))
seen=set()
for r in rows:
 assert '2026-08-17'<=r['day']<='2026-09-15'
 key=(tuple(sorted((r['a'],r['b']))),r['day']); assert key not in seen;seen.add(key)
 assert r['y'] in ['True','False'];r['positive']=r['y']=='True'
 for m in ['activity','layer1_behavior']:r[m]=float(r[m]);assert math.isfinite(r[m])
def compare(keys):
 groups=defaultdict(list)
 for r in rows:groups[keys(r)].append(r)
 out={}
 for m in ['activity','layer1_behavior']:
  detail=[]
  for key,grp in sorted(groups.items()):
   pos=[r[m] for r in grp if r['positive']];neg=[r[m] for r in grp if not r['positive']]
   if not pos or not neg:continue
   n=len(pos)*len(neg)
   wins=sum(1. if p>q else .5 if p==q else 0. for p in pos for q in neg)
   ties=sum(p==q for p in pos for q in neg)
   detail.append({'group':key,'rows':len(grp),'positive_rows':len(pos),'negative_rows':len(neg),'comparisons':n,'wins_with_half_ties':wins,'tied_comparisons':ties,'concordance':wins/n})
  comparisons=sum(d['comparisons'] for d in detail)
  out[m]={'eligible_groups':len(detail),'eligible_rows':sum(d['rows'] for d in detail),'positive_rows':sum(d['positive_rows'] for d in detail),'positive_negative_comparisons':comparisons,'tied_comparisons':sum(d['tied_comparisons'] for d in detail),'comparison_weighted_auc':sum(d['wins_with_half_ties'] for d in detail)/comparisons,'macro_group_auc':sum(d['concordance'] for d in detail)/len(detail)}
 return out
report={'scope':'Post-hoc verification of saved later-window prediction scores; no raw data, model refitting, new thresholds, or reserved data.',
'inputs':{str(source):sha(source),str(manifest):sha(manifest)},'script_sha256':sha(__file__),
'rows':len(rows),'positive_rows':sum(r['positive'] for r in rows),'observed_days':len({r['day'] for r in rows}),
'definitions':{'within_pair':'Compare every positive day with every negative day of the same undirected actor pair. Only pairs with both labels contribute.','within_day':'Compare every positive pair with every negative pair observed on the same Pacific day. Only days with both labels contribute.','weighting':'Headline AUC pools within-group positive-negative comparisons; each comparison has equal weight. Macro statistic instead weights eligible groups equally.','ties':'Equal scores count one half.','activity':'Existing frozen July-fitted score from act_min and act_max (pair minimum and maximum working-step counts).'},
'within_pair':compare(lambda r:tuple(sorted((r['a'],r['b'])))),
'within_day':compare(lambda r:r['day']),
'interpretation':['Within-pair comparison holds pair identity fixed, not activity counts.','Within-day comparison holds day fixed, removing explanations based solely on common day-level conditions; different pairs can still have different activity.','Comparing with activity-only scores tests a particular activity baseline; it is not conditional activity matching, causal adjustment, or proof of activity-independent interaction detection.','These checks are additional post-hoc descriptive diagnostics. No confidence interval or significance claim is added.']}
old=json.loads((ROOT/'outputs/claude-review-20261107/baselines/summary.json').read_text())['periods']['temporal']['within_pair']
assert math.isclose(report['within_pair']['layer1_behavior']['comparison_weighted_auc'],old['comparison_weighted_auc'],abs_tol=1e-12)
assert report['within_pair']['layer1_behavior']['positive_negative_comparisons']==old['positive_negative_comparisons']
(ROOT/'outputs/claude-review-20261107/activity-concordance-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['within_pair','within_day']},indent=2))
