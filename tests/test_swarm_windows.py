"""Synthetic-only checks for the predeclared, blinded call-window statistics."""
import copy
import json
import unittest

import numpy as np

from swarmcollab.swarm_windows import analyze_windows, summarize_cut


def run(values, run_id='run', counts=None):
    family = {'binary': values}
    if counts is not None:
        family['counts'] = counts
    return {'run_id': run_id, 'families': {'resource': family}, 'n_calls': len(values)}


def anchor(cut, run_id='run', anchor_id='message', kind='external_message'):
    return dict(run_id=run_id, anchor_id=anchor_id, kind=kind, cut=cut)


class BoundaryTests(unittest.TestCase):
    def test_cut_is_first_post_call_and_rates_keep_raw_denominators(self):
        sample = run([1]*100+[0]*100, counts=[2]*100+[0]*100)
        value = summarize_cut(sample, 100)
        self.assertEqual(value['before_index_start'], 0)
        self.assertEqual(value['before_index_end_exclusive'], 100)
        self.assertEqual(value['after_index_start'], 100)
        self.assertEqual(value['after_index_end_exclusive'], 200)
        self.assertEqual(value['before_denominator'], 100)
        self.assertEqual(value['after_denominator'], 100)
        binary = value['families']['resource']['binary']
        counts = value['families']['resource']['counts']
        self.assertEqual(binary['before_sum'], 100)
        self.assertEqual(binary['after_sum'], 0)
        self.assertEqual(binary['before_fraction'], 1.)
        self.assertEqual(binary['change_fraction'], -1.)
        self.assertEqual(binary['change_per_1000_calls'], -1000.)
        self.assertEqual(counts['change_per_1000_calls'], -2000.)
        self.assertNotIn('change_fraction', counts)

    def test_full_symmetric_windows_exclude_one_call_short_and_boundaries(self):
        sample = run([0]*200)
        for cut in (0, 99, 101, 200):
            value = summarize_cut(sample, cut)
            self.assertEqual(value['status'], 'excluded_insufficient_calls')
            self.assertIsNone(value['before_denominator'])
            self.assertIsNone(value['after_denominator'])
            self.assertEqual(value['families'], {})
        self.assertEqual(summarize_cut(sample, 100)['status'], 'available')

    def test_windows_never_cross_run_boundaries(self):
        report = analyze_windows([run([0]*150, 'a'), run([1]*150, 'b')])
        self.assertTrue(report['fallback']['triggered'])
        self.assertEqual(report['fallback']['status'], 'insufficient_data')
        self.assertEqual(report['fallback']['candidate_cuts_per_run'], {'a': [], 'b': []})
        self.assertEqual(report['fallback']['families']['resource']['binary'], [])

    def test_existing_but_boundary_only_messages_do_not_trigger_fallback(self):
        report = analyze_windows([run([0]*300)], [anchor(0)])
        self.assertFalse(report['fallback']['triggered'])
        ref = report['anchor_references']['external_message']['100']
        self.assertEqual(ref['status'], 'insufficient_data')
        self.assertEqual(ref['excluded_anchors'], 1)
        self.assertIsNone(ref['families']['resource']['binary']['reference_tail_probability'])

    def test_sensitivity_eligibility_is_separate_from_primary(self):
        report = analyze_windows([run([0]*250)], [anchor(75)])
        windows = report['anchors'][0]['windows']
        self.assertEqual(windows['100']['status'], 'excluded_insufficient_calls')
        self.assertEqual(windows['50']['status'], 'available')
        self.assertEqual(windows['200']['missing_before_calls'], 125)
        self.assertEqual(report['anchor_references']['external_message']['50']['eligible_anchors'], 1)


class PlaceboTests(unittest.TestCase):
    def test_whole_anchor_maximum_matches_independent_seeded_reference(self):
        a = (np.arange(510) % 7 == 0).astype(int)
        b = (np.arange(360) % 11 < 3).astype(int)
        runs = [run(a, 'a', a*3), run(b, 'b', b*2)]
        anchors = [anchor(140, 'a', 'a1'), anchor(380, 'a', 'a2'), anchor(160, 'b', 'b1')]
        report = analyze_windows(runs, anchors)
        ref = report['anchor_references']['external_message']['100']
        self.assertEqual(ref['eligible_anchors_per_run'], {'a': 2, 'b': 1})
        rng = np.random.default_rng(20261106)
        sampled_a = rng.integers(100, 510-100+1, size=(500, 2))
        sampled_b = rng.integers(100, 360-100+1, size=(500, 1))
        for measure, arrays in [('binary', {'a': a, 'b': b}), ('counts', {'a': a*3, 'b': b*2})]:
            def change(values, cut):
                return float(values[cut:cut+100].sum()-values[cut-100:cut].sum())*10
            observed = max(abs(change(arrays[x['run_id']], x['cut'])) for x in anchors)
            null = [max([abs(change(arrays['a'], int(cut))) for cut in sampled_a[i]] +
                        [abs(change(arrays['b'], int(cut))) for cut in sampled_b[i]]) for i in range(500)]
            actual = ref['families']['resource'][measure]
            self.assertEqual(actual['observed_max_absolute_change_per_1000_calls'], observed)
            self.assertEqual(actual['placebo_max_absolute_changes_per_1000_calls'], null)
            exceed = sum(value >= observed for value in null)
            self.assertEqual(actual['reference_tail_probability'], (1+exceed)/501)
            self.assertEqual(actual['null_at_least_observed'], exceed)

    def test_reproducible_and_invariant_to_run_anchor_order_without_mutating_inputs(self):
        runs = [run([0]*100+[1]*200, 'a'), run([1]*150+[0]*150, 'b')]
        anchors = [anchor(100, 'a', 'a1'), anchor(150, 'b', 'b1')]
        original = copy.deepcopy((runs, anchors))
        one = analyze_windows(runs, anchors)
        two = analyze_windows(list(reversed(runs)), list(reversed(anchors)))
        self.assertEqual(one, two)
        self.assertEqual((runs, anchors), original)
        json.dumps(one, allow_nan=False)

    def test_message_and_start_references_are_separate(self):
        sample = run([0]*100+[1]*200+[0]*100)
        only = analyze_windows([sample], [anchor(100)])
        both = analyze_windows([sample], [anchor(100), anchor(300, anchor_id='task', kind='task_start')])
        self.assertEqual(only['anchor_references']['external_message'], both['anchor_references']['external_message'])
        self.assertEqual(both['anchor_counts']['task_start'], 1)
        self.assertEqual(both['anchor_references']['task_start']['100']['eligible_anchors'], 1)
        self.assertEqual(both['anchor_references']['session_start']['100']['status'], 'no_metadata_anchors')

    def test_zero_change_ties_have_tail_probability_one_and_single_cut_is_valid(self):
        report = analyze_windows([run([0]*200)], [anchor(100)])
        reference = report['anchor_references']['external_message']['100']['families']['resource']['binary']
        self.assertEqual(reference['reference_tail_probability'], 1.)
        self.assertEqual(reference['null_at_least_observed'], 500)
        self.assertEqual(reference['placebo_max_absolute_changes_per_1000_calls'], [0.]*500)


class FallbackTests(unittest.TestCase):
    def test_tied_ranks_are_earliest_and_exclude_distance_equal_to_200(self):
        report = analyze_windows([run([0]*2200)])
        selected = report['fallback']['families']['resource']['binary']
        self.assertEqual([x['cut'] for x in selected], [100, 400, 700, 1000, 1300])
        self.assertEqual(selected[0]['windows']['200']['status'], 'excluded_insufficient_calls')
        self.assertEqual(selected[1]['windows']['200']['status'], 'available')
        self.assertNotIn('reference_tail_probability', json.dumps(report['fallback']))

    def test_suppression_is_within_run_and_same_cut_ties_use_run_id(self):
        report = analyze_windows([run([0]*500, 'b'), run([0]*500, 'a')])
        selected = report['fallback']['families']['resource']['binary']
        self.assertEqual([(x['run_id'],x['cut']) for x in selected], [('a',100),('b',100),('a',400),('b',400)])

    def test_counts_and_binary_find_their_own_ranked_maximum(self):
        binary = [0]*300+[1]*100+[0]*600
        counts = [0]*700+[20]*100+[0]*200
        report = analyze_windows([run(binary, counts=counts)])
        family = report['fallback']['families']['resource']
        self.assertEqual(family['binary'][0]['cut'], 300)
        self.assertEqual(family['counts'][0]['cut'], 700)
        for measure, selections in family.items():
            self.assertTrue(all(row['selected_measure'] == measure for row in selections))
            self.assertTrue(all(set(row['windows']) == {'50','100','200'} for row in selections))

    def test_only_start_metadata_still_allows_external_anchor_fallback(self):
        report = analyze_windows([run([0]*300)], [anchor(0, kind='session_start')])
        self.assertTrue(report['fallback']['triggered'])
        self.assertEqual(report['anchor_references']['session_start']['100']['excluded_anchors'], 1)


class ValidationTests(unittest.TestCase):
    def test_array_and_metadata_validation(self):
        invalid = [run([0, np.nan]), run([0, .5]), run([0, 2]), run([[0, 1]]),
                   run([0, 1], counts=[-1, 0]), run([0, 1], counts=[0]),
                   {'run_id':'x','n_calls':3,'families':{'resource':{'binary':[0,1]}}}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                analyze_windows([value])
        for value in [anchor(-1), anchor(301), anchor(True), anchor(100, run_id='unknown'),
                      anchor(100, kind='tool_response')]:
            with self.subTest(anchor=value), self.assertRaises(ValueError):
                analyze_windows([run([0]*300)], [value])
        with self.assertRaises(ValueError):
            analyze_windows([run([0]*300)], [anchor(100), anchor(120)])

    def test_families_and_count_availability_must_be_explicitly_identical(self):
        other = {'run_id':'other','families':{'other_family':{'binary':[0]*300}}}
        for other_run in [other, run([0]*300,'other',counts=[0]*300), run([0]*300)]:
            with self.assertRaises(ValueError):
                analyze_windows([run([0]*300), other_run])

    def test_empty_recorded_run_is_insufficient_data_not_zero_change(self):
        report = analyze_windows([run([])])
        self.assertEqual(report['fallback']['status'], 'insufficient_data')
        self.assertEqual(report['runs'], [{'run_id':'run','n_calls':0}])
        self.assertEqual(report['fallback']['families']['resource']['binary'], [])


if __name__ == '__main__':
    unittest.main()
