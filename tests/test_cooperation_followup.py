import json
import unittest

import numpy as np
import pandas as pd

from swarmcollab import cooperation_followup as C


class RateTests(unittest.TestCase):
    def frame(self, counts, exposure=None):
        return pd.DataFrame({'events':counts,'exposure':exposure or [10]*len(counts)})

    def test_zero_baseline_draws_remain_unbounded(self):
        x=C.rate_comparison(self.frame([2,2,2]),self.frame([1,0,0]),'events','exposure',1000)
        self.assertEqual(x['rate_ratio'],6.)
        self.assertGreater(x['positive_over_zero_draws'],0)
        self.assertEqual(x['ratio_bound_status'][1],'unbounded')
        self.assertIsNone(x['ratio_interval_95'][1])
        json.dumps(x,allow_nan=False)

    def test_both_zero_is_undefined_not_evidence_of_equivalence(self):
        x=C.rate_comparison(self.frame([0,0]),self.frame([0,0]),'events','exposure',100)
        self.assertEqual(x['rate_ratio_status'],'undefined_both_zero')
        self.assertEqual(x['undefined_zero_over_zero_draws'],100)
        self.assertTrue(x['zero_observed_group'])
        self.assertEqual(x['ratio_bound_status'],['undefined','undefined'])

    def test_count_and_exposure_resample_together(self):
        a=self.frame([1,100],[10,1000])
        x=C.rate_comparison(a,a,'events','exposure',500)
        self.assertEqual(x['ratio_interval_95'],[1.,1.])
        np.testing.assert_allclose(x['difference_interval_95'],[0.,0.],atol=1e-15)

    def test_no_confidence_interval_invented_for_zero_observed_baseline(self):
        x=C.rate_comparison(self.frame([1,1]),self.frame([0,0]),'events','exposure',100)
        self.assertEqual(x['rate_ratio_status'],'unbounded_zero_baseline')
        self.assertEqual(x['ratio_interval_95'],[None,None])
        self.assertEqual(x['ratio_bound_status'],['undefined','undefined'])
        self.assertEqual(x['positive_over_zero_draws'],100)

    def test_invalid_exposure_or_values_fail(self):
        for a in [self.frame([1],[0]),self.frame([-1]),self.frame([np.nan])]:
            with self.assertRaises(ValueError):
                C.rate_comparison(a,self.frame([1]),'events','exposure',10)


class ExposureAndClusterTests(unittest.TestCase):
    def fixtures(self):
        p=pd.DataFrame([dict(a='A',b='B',day=d,y=False,agent_stratum='both_known',
                            **{m:0. for m in C.MODELS},**{m+'__alert':False for m in C.MODELS})
                        for d in ['2026-06-01','2026-06-02']])
        a=pd.DataFrame([dict(agent=x,day=d,working=10,shell=8,gui=2)
                        for d in p.day for x in ['A','B']])
        c=pd.DataFrame([dict(chain_id='c',level=4,source_actor='C',target_actor='A',
                            source_turn='s',target_turn='t',day='2026-06-01',version_key='git:r@abc')])
        return a,p,C.prepare_chains(c,p)

    def test_nonactive_source_excluded_only_from_pair_supported_numerator(self):
        a,p,c=self.fixtures();d=C.daily_table(a,p,c)
        self.assertEqual(d.working_steps.sum(),40)
        self.assertEqual(d.strong_chains.sum(),1)
        self.assertEqual(d.strong_eligible_chains.sum(),0)
        self.assertEqual(d.strong_supported_pair_days.sum(),0)
        self.assertEqual(len(d),2)

    def test_shared_version_clusters_without_merging_other_versions_in_repository(self):
        _,p,c=self.fixtures()
        other=c.iloc[0].to_dict();other.update(chain_id='d',source_turn='s2',target_turn='t2')
        third=other.copy();third.update(chain_id='e',source_turn='s3',target_turn='t3',version_key='git:r@def')
        result=C.chain_summary(pd.concat([c,pd.DataFrame([other,third])],ignore_index=True))
        self.assertEqual(result['strong_chains'],3)
        self.assertEqual(result['version_or_turn_linked_components'],2)
        self.assertEqual(result['unique_target_turns'],3)
        self.assertEqual(result['repository_actor_pair_groups'],1)

    def test_git_opportunities_deduplicate_successful_consumer_turns(self):
        a,p,c=self.fixtures()
        e=pd.DataFrame([dict(actor='A',turn_id='one',day='2026-06-01',operation=o,status='success')
                        for o in ['acquire','acquire','incorporate','generate']]+
                       [dict(actor='B',turn_id='two',day='2026-06-01',operation='acquire',status='unknown')])
        d=C.daily_table(a,p,c,git_events=e)
        self.assertEqual(d.git_observed_turns.sum(),2)
        self.assertEqual(d.git_success_consumer_turns.sum(),1)
        self.assertEqual(d.git_success_producer_turns.sum(),1)

    def test_frozen_alert_workload_and_proxy_precision(self):
        _,p,_=self.fixtures()
        p['layer1_behavior__alert']=True
        p['layer1_behavior']=[1.,2.]
        p['y']=[True,False]
        m=C.monitoring(p,'layer1_behavior',100)
        self.assertEqual(m['alerts_per_day'],1.)
        self.assertEqual(m['precision'],.5)
        self.assertEqual(m['request_negative_alerts_per_day'],.5)
        self.assertEqual(m['unique_alerted_pairs'],1)
        self.assertEqual(m['max_alert_days_per_pair'],2)


if __name__=='__main__':
    unittest.main()
