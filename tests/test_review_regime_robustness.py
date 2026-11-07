import importlib.util
from pathlib import Path
import unittest

import pandas as pd


PATH=Path(__file__).resolve().parents[1]/'scripts/review_regime_robustness.py'
SPEC=importlib.util.spec_from_file_location('review_regimes',PATH)
R=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(R)


def activity():
    return pd.DataFrame([dict(agent=a,day=d,working=4,shell=3,gui=1)
                         for a in ['A','B'] for d in ['2026-06-01','2026-06-02']])


def mentions(rows):
    return pd.DataFrame(rows,columns=['agent','day','step','type','id','op','src'])


class CoUseTests(unittest.TestCase):
    def test_repetition_and_multiple_resources_do_not_inflate_step_denominator(self):
        m=mentions([['A','2026-06-01',1,'repo','r','read','cmd'],
                    ['A','2026-06-01',1,'repo','r','read','cmd'],
                    ['A','2026-06-01',1,'page','p','read','cmd'],
                    ['B','2026-06-01',3,'repo','r','seen','out']])
        d,a=R.co_use_counts(m,activity());x=d.set_index('day').loc['2026-06-01']
        self.assertEqual(x.shared_steps,2);self.assertEqual(x.trace_steps,2)
        self.assertEqual(x.shared_fraction_of_working,.25);self.assertEqual(x.shared_fraction_of_traced,1.)

    def test_other_actor_same_day_required_and_type_identity_is_scoped(self):
        m=mentions([['A','2026-06-01',1,'repo','x','read','cmd'],
                    ['A','2026-06-01',2,'repo','x','read','cmd'],
                    ['B','2026-06-02',1,'repo','x','read','cmd'],
                    ['B','2026-06-01',2,'page','x','read','cmd']])
        d,_=R.co_use_counts(m,activity());self.assertEqual(d.shared_steps.sum(),0)

    def test_broad_site_and_opaque_commit_are_only_sensitivity(self):
        m=mentions([[a,'2026-06-01',1,'site','common','read','cmd'] for a in ['A','B']]+
                   [[a,'2026-06-01',2,'commit','abcdef0','seen','out'] for a in ['A','B']])
        p,_=R.co_use_counts(m,activity());b,_=R.co_use_counts(m,activity(),variant='all_legacy')
        self.assertEqual(p.trace_steps.sum(),0);self.assertEqual(b.shared_steps.sum(),4)
        self.assertTrue(p.shared_fraction_of_traced.isna().all())

    def test_command_context_sensitivity_excludes_output_only_exposure(self):
        m=mentions([['A','2026-06-01',1,'repo','r','read','cmd'],
                    ['B','2026-06-01',1,'repo','r','seen','out']])
        p,_=R.co_use_counts(m,activity());c,_=R.co_use_counts(m,activity(),variant='specific_command_context')
        self.assertEqual(p.shared_steps.sum(),2);self.assertEqual(c.shared_steps.sum(),0)
        self.assertEqual(c.trace_steps.sum(),1)

    def test_activity_universe_and_denominators_fail_closed(self):
        m=mentions([['X','2026-06-01',1,'repo','r','read','cmd']])
        with self.assertRaisesRegex(ValueError,'inactive or missing'):R.co_use_counts(m,activity())
        m=mentions([['A','2026-06-01',i,'repo','r','read','cmd'] for i in range(5)])
        with self.assertRaisesRegex(ValueError,'denominator'):R.co_use_counts(m,activity())
        a=activity();a=pd.concat([a,a.iloc[:1]])
        with self.assertRaisesRegex(ValueError,'unique'):R.co_use_counts(m,a)

    def test_scope_rejects_reserved_period_and_other_windows(self):
        with self.assertRaises(ValueError):R.validate_days(pd.DataFrame({'day':['2026-09-16']}),'temporal')
        with self.assertRaises(ValueError):R.validate_days(pd.DataFrame({'day':['2026-06-01']}),'training')

    def test_equal_actor_rates_do_not_equal_activity_pooled_rates(self):
        a=activity().iloc[:1].copy();a.loc[0,'working']=10
        a=pd.concat([a,pd.DataFrame([dict(agent='B',day='2026-06-01',working=2,shell=2,gui=0)])])
        m=mentions([['A','2026-06-01',i,'repo','shared' if i==0 else 'own','read','cmd'] for i in range(8)]+
                   [['B','2026-06-01',0,'repo','shared','read','cmd']])
        d,_=R.co_use_counts(m,a);row=d.iloc[0]
        self.assertAlmostEqual(row.actor_mean_shared_fraction_of_traced,(1/8+1)/2)
        self.assertAlmostEqual(row.shared_fraction_of_traced,2/9)

    def test_classifier_and_co_use_regime_are_distinct_and_transitions_remain_visible(self):
        m=mentions([['A','2026-06-01',1,'repo','r','read','cmd']])
        d,_=R.co_use_counts(m,activity())
        predictions=pd.DataFrame([dict(a='A',b='B',day=day,layer1_behavior__alert=alert,y=0)
                                  for day,alert in [('2026-06-01',True),('2026-06-02',False)]])
        regime=pd.DataFrame([dict(day=day,flagged=flag,flagged_share=flag,null_calibrated=False)
                             for day,flag in [('2026-06-01',0),('2026-06-02',1)]])
        goals={'intervals':[{'goal_id':'g','description':'Goal'}],
               'day_assignments':[dict(day=day,primary_goal_id='g',primary_category='mixed',transition_day=transition)
                                  for day,transition in [('2026-06-01',True),('2026-06-02',False)]]}
        joined=R.attach_frozen_series(d,predictions,regime,activity(),'shared_goals',goals)
        self.assertEqual(list(joined.classifier_alert_share),[1,0])
        self.assertEqual(list(joined.co_use_threshold_share),[0,1])
        self.assertEqual(list(joined.transition),[True,False])
        joined['variant']='specific'
        summaries=R.summarize(joined)
        excluded=next(row for row in summaries if row['sensitivity']=='without_transition_days')
        self.assertEqual(excluded['observed_days'],1)
        self.assertEqual(excluded['classifier_alert_share_median'],0)
        self.assertEqual(excluded['co_use_threshold_share_median'],1)

    def test_frozen_series_requires_the_same_observed_day_universe(self):
        m=mentions([['A','2026-06-01',1,'repo','r','read','cmd']])
        d,_=R.co_use_counts(m,activity())
        predictions=pd.DataFrame([dict(a='A',b='B',day='2026-06-01',layer1_behavior__alert=True,y=0)])
        regime=pd.DataFrame([dict(day='2026-06-01',flagged=0,flagged_share=0,null_calibrated=False)])
        with self.assertRaisesRegex(ValueError,'universes disagree'):
            R.attach_frozen_series(d,predictions,regime,activity(),'shared_goals',{})


if __name__=='__main__':unittest.main()
