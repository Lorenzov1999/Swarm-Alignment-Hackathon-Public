import datetime as dt
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PATH=Path(__file__).resolve().parents[1]/'scripts/review_label_visibility.py'
SPEC=importlib.util.spec_from_file_location('review_label_visibility',PATH)
report=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(report)


def predictions():
    f=pd.DataFrame(dict(a=['A','A','A','A'],b=['B','B','B','B'],
        day=['2026-08-17','2026-08-18','2026-08-19','2026-08-20'],
        y=[True,True,True,False]))
    for model in report.MODELS:
        f[model]=[1.,-1.,0.,2.];f[model+'__alert']=f[model]>0
    return f


def request(index,day,followed,accepted='yes',sender='A',recipient='B'):
    return dict(request=index,day=dt.date.fromisoformat(day),from_=sender,to=recipient,
                accepted=accepted,followed_through=followed,**{'from':sender})


class SavedPredictionValidationTests(unittest.TestCase):
    def test_frozen_threshold_is_verified_without_recalibration(self):
        f=predictions();spec={model:dict(threshold=0.) for model in report.MODELS}
        report.bounded_predictions(f,spec)
        changed=f.copy();changed.loc[2,report.MODELS[0]+'__alert']=True
        with self.assertRaisesRegex(ValueError,'frozen training threshold'):
            report.bounded_predictions(changed,spec)

    def test_future_dates_duplicates_and_noncanonical_pairs_rejected(self):
        spec={model:dict(threshold=0.) for model in report.MODELS}
        for change in ('date','duplicate','pair'):
            f=predictions()
            if change=='date':f.loc[0,'day']='2026-09-16'
            elif change=='duplicate':f=pd.concat([f,f.iloc[:1]])
            else:f.loc[0,'a']='Z'
            with self.assertRaises(ValueError):report.bounded_predictions(f,spec)


class VisibilityTests(unittest.TestCase):
    def test_less_visible_actor_controls_pair_count_and_share(self):
        f=predictions().iloc[:1]
        activity=pd.DataFrame([dict(agent='A',day='2026-08-17',working=100,trace_steps=40),
                               dict(agent='B',day='2026-08-17',working=20,trace_steps=15)])
        r=report.add_visibility(f,activity)
        self.assertEqual(r.visibility_count.iloc[0],15)
        self.assertEqual(r.visibility_share.iloc[0],.4)

    def test_missing_or_impossible_activity_is_not_zero_filled(self):
        f=predictions().iloc[:1]
        activity=pd.DataFrame([dict(agent='A',day='2026-08-17',working=5,trace_steps=2)])
        with self.assertRaisesRegex(ValueError,'Missing activity'):report.add_visibility(f,activity)
        activity.loc[0,'trace_steps']=6
        with self.assertRaisesRegex(ValueError,'invalid'):report.add_visibility(f,activity)

    def test_group_cuts_are_label_independent_and_ties_stay_together(self):
        f=predictions();f['visibility_count']=[1.,1.,3.,4.]
        rows,cuts=report.visibility_table(f,'visibility_count')
        g=f.copy();g.y=~g.y
        changed,changed_cuts=report.visibility_table(g,'visibility_count')
        self.assertEqual(cuts,changed_cuts)
        self.assertEqual([r['pair_days'] for r in rows],[r['pair_days'] for r in changed])
        self.assertEqual(rows[0]['pair_days'],2)


class FollowThroughTests(unittest.TestCase):
    def test_repeated_and_opposite_direction_requests_collapse_to_one_pair_day(self):
        f=predictions()
        a=request(1,'2026-08-17','yes');b=request(2,'2026-08-17','yes',sender='B',recipient='A')
        rows=[a,a,b,request(3,'2026-08-18','no'),request(4,'2026-08-19','ambiguous')]
        joined,coverage=report.join_request_annotations(f,rows)
        self.assertEqual(len(joined),3)
        self.assertEqual(coverage['duplicate_request_recipient_rows'],1)
        self.assertEqual(coverage['unique_requests'],4)
        self.assertEqual(joined.loc[joined.day=='2026-08-17','requests'].iloc[0],2)

    def test_mixed_and_ambiguous_pair_days_are_excluded_from_clean_contrast(self):
        f=predictions()
        rows=[request(1,'2026-08-17','yes'),request(2,'2026-08-18','no'),
              request(3,'2026-08-19','yes'),request(4,'2026-08-19','no')]
        joined,coverage=report.join_request_annotations(f,rows)
        self.assertEqual(set(joined.followed_group),{'yes','no','yes_and_no'})
        s=report.followthrough_summary(joined,draws=50)
        self.assertEqual(s['clean_contrast']['only_yes_pair_days'],1)
        self.assertEqual(s['clean_contrast']['only_no_pair_days'],1)
        self.assertEqual(s['clean_contrast']['excluded_mixed_or_ambiguous_pair_days'],1)
        self.assertEqual(s['clean_contrast']['models'][report.MODELS[0]]['auroc'],1)
        self.assertEqual(s,report.followthrough_summary(joined,draws=50))

    def test_label_mismatch_inconsistent_status_and_future_request_fail(self):
        f=predictions()
        rows=[request(1,'2026-08-17','yes'),request(2,'2026-08-18','no'),
              request(3,'2026-08-19','ambiguous')]
        with self.assertRaisesRegex(ValueError,'reproduce frozen'):report.join_request_annotations(f,rows[:1])
        with self.assertRaisesRegex(ValueError,'inconsistent'):report.join_request_annotations(f,rows+[request(1,'2026-08-17','no')])
        with self.assertRaisesRegex(ValueError,'outside'):report.join_request_annotations(f,rows+[request(5,'2026-09-16','yes')])

    def test_outside_eligible_pairs_are_reported_not_silently_labeled(self):
        f=predictions()
        rows=[request(1,'2026-08-17','yes'),request(2,'2026-08-18','no'),
              request(3,'2026-08-19','ambiguous'),request(4,'2026-08-19','yes',recipient='C')]
        joined,coverage=report.join_request_annotations(f,rows)
        self.assertEqual(coverage['recipient_rows_outside_eligible_pairs'],1)
        self.assertEqual(coverage['outside_pair_days'],[('A','C','2026-08-19')])
        self.assertEqual(len(joined),3)


if __name__=='__main__':unittest.main()
