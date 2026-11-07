"""Work sequences require linked evidence and become visible at their last support."""
import unittest

from swarmcollab.provenance_common import git_key, make_event
from swarmcollab.provenance_links import build_links
from swarmcollab.workflow_patterns import build_workflows, extract_workflow_snippets


REPO = 'gitlab:team/demo'
KEY1 = git_key(REPO, 'a1' * 20)
KEY2 = git_key(REPO, 'b2' * 20)
KEY3 = git_key(REPO, 'c3' * 20)
SITE = 'https://demo.gitlab.io/'
REVIEW = 'https://gitlab.com/team/reviewer/-/raw/main/review.md'


def event(actor, minute, operation, inputs=(), outputs=(), family='git', entity=REPO,
          status='success', **extra):
    record = dict(i=actor + str(minute) + operation + family,
                  t='2026-07-06T16:%02d:00+00:00' % minute, s='test')
    extra.setdefault('relation_authority', True)
    return make_event(actor, record, family, operation, status, entity,
                      inputs=list(inputs), outputs=list(outputs), **extra)


def workflows(events, **kwargs):
    chains = build_links(events)[0]
    return build_workflows(events, chains, **kwargs)[0]


def rules():
    evidence = [dict(actor='Observer', turn_id='directory', time='2026-07-06T16:00:00+00:00')]
    return dict(fit_start='2026-07-06', fit_end='2026-07-31', projects=[dict(
        key=SITE, site=SITE, owner='Directory contact', repository='https://gitlab.com/team/demo',
        available_from=evidence[0]['time'], repository_available_from=evidence[0]['time'],
        evidence=evidence, repository_evidence=evidence, shared_resource=False)], reviews=[dict(
        url=REVIEW, target=SITE + 'de/', reviewer='Reviewer',
        available_from=evidence[0]['time'], evidence=evidence)])


def review_access(minute=1):
    return event('Editor', minute, 'read', family='village_review_reference',
                 entity='village-review:' + REVIEW, status='attempted', relation_authority=False)


def site_access(actor='Visitor', minute=5):
    return event(actor, minute, 'read', family='village_project_access',
                 entity='village-project:' + SITE, status='attempted', relation_authority=False)


def snippet(actor, minute, turn_id, text, field='x'):
    return dict(snippet_id=actor + str(minute) + field, actor=actor, turn_id=turn_id,
                time='2026-07-06T16:%02d:00+00:00' % minute, day='2026-07-06', field=field,
                start=0, end=len(text), text=text,
                source_role='command_text' if field == 'x' else 'output_exposure')


class DependencyWorkflowTests(unittest.TestCase):
    def sequence(self, last_actor='A'):
        return [event('A', 0, 'generate', outputs=[KEY1]),
                event('B', 1, 'acquire', inputs=[KEY1]),
                event('B', 2, 'transform', inputs=[KEY1], outputs=[KEY2]),
                event(last_actor, 3, 'incorporate', inputs=[KEY2], outputs=[KEY3])]

    def test_return_uses_explicit_middle_bridge_and_latest_time(self):
        events = self.sequence()
        found = [p for p in workflows(events) if p['pattern'] == 'artifact_return_revision']
        self.assertTrue(found)
        for pattern in found:
            self.assertEqual(pattern['actors'], ['A', 'B', 'A'])
            self.assertEqual(pattern['available_time'], events[-1]['time'])
            self.assertEqual(pattern['confidence'], 'strict')
            self.assertFalse(pattern['relation_authority'])
            self.assertIn(events[-1]['event_id'], pattern['support_events'])

    def test_three_actor_relay_reports_only_observed_edges(self):
        found = [p for p in workflows(self.sequence('C')) if p['pattern'] == 'artifact_relay']
        self.assertTrue(found)
        self.assertEqual(found[0]['actor_edges'], [dict(source_actor='A', target_actor='B'),
                                                  dict(source_actor='B', target_actor='C')])
        self.assertEqual((found[0]['source_actor'], found[0]['target_actor']), ('A', 'B'))

    def test_time_and_same_repo_are_not_a_bridge(self):
        events = self.sequence()
        events[2]['inputs'] = []
        self.assertFalse(any(p['pattern'].startswith('artifact_') for p in workflows(events)))

    def test_one_transform_can_be_middle_of_two_real_edges(self):
        events = self.sequence()
        del events[1]
        self.assertTrue(any(p['pattern'] == 'artifact_return_revision' for p in workflows(events)))

    def test_observation_does_not_become_producer(self):
        events = self.sequence()
        chains = build_links(events)[0]
        events[0]['operation'] = 'observe'
        found, _ = build_workflows(events, chains)
        self.assertFalse(any(p['source_actor'] == 'A' for p in found))

    def test_failed_support_invalidates_precomputed_chains(self):
        events = self.sequence()
        chains = build_links(events)[0]
        events[2]['status'] = 'failed'
        found, _ = build_workflows(events, chains)
        self.assertFalse(any(p['pattern'].startswith('artifact_') for p in found))

    def test_future_relation_cannot_backdate_a_chain(self):
        events = self.sequence()
        chains = build_links(events)[0]
        future = event('Observer', 4, 'observe', outputs=[KEY1])
        for chain in chains:
            chain['relation_events'] = [future['event_id']]
        self.assertEqual(build_workflows(events + [future], chains)[0], [])

    def test_acquire_then_test_needs_the_received_version(self):
        events = self.sequence()[:2] + [event('B', 3, 'test', inputs=[KEY1])]
        found = [p for p in workflows(events) if p['pattern'] == 'acquire_then_test']
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['time'], events[-1]['time'])
        events[-1]['inputs'] = [KEY2]
        self.assertFalse(any(p['pattern'] == 'acquire_then_test' for p in workflows(events)))

    def test_unknown_outcomes_stay_candidates(self):
        events = self.sequence()[:2] + [event('B', 3, 'test', inputs=[KEY1], status='unknown')]
        found = [p for p in workflows(events) if p['pattern'] == 'acquire_then_test']
        self.assertEqual(found[0]['confidence'], 'candidate')

    def test_wide_coincidences_and_same_turn_followups_are_excluded(self):
        events = self.sequence()[:2] + [event('B', 3, 'test', inputs=[KEY1])]
        events[-1]['turn_id'] = events[1]['turn_id']
        self.assertEqual(workflows(events), [])
        events[-1]['turn_id'] = 'later'
        events[-1]['time'], events[-1]['day'] = '2026-07-20T16:03:00+00:00', '2026-07-20'
        self.assertEqual(workflows(events), [])

    def test_scope_and_cached_support_validation(self):
        events = self.sequence()
        events[-1]['time'], events[-1]['day'] = '2026-08-17T16:03:00+00:00', '2026-08-17'
        with self.assertRaisesRegex(ValueError, 'outside development'):
            build_workflows(events, [])
        events = self.sequence()
        chains = build_links(events)[0]
        self.assertEqual(build_workflows(events[1:], chains)[0], [])

    def test_deterministic_and_qualified_grade_not_upgraded(self):
        events = self.sequence()
        chains = build_links(events)[0]
        for chain in chains:
            chain['identity_confidence'] = 'abbreviated_observed'
        first = build_workflows(events, chains)[0]
        second = build_workflows(events[::-1], chains[::-1])[0]
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertTrue(all(p['confidence'] == 'qualified' for p in first))


class WorkflowBoundaryTests(unittest.TestCase):
    def test_unrelated_matching_snippets_do_not_create_a_workflow(self):
        body = 'The same long passage can appear in several observations without any recorded dependency.'
        snippets = [snippet('A', 1, 'first', body), snippet('B', 2, 'second', body),
                    snippet('A', 3, 'third', body)]
        patterns, diagnostics = build_workflows([], [], snippets=snippets)
        self.assertEqual(patterns, [])
        self.assertEqual(diagnostics['patterns'], 0)

    def test_removed_global_matching_arguments_are_not_accepted(self):
        for argument in ('text_matches', 'text_model'):
            with self.assertRaises(TypeError):
                build_workflows([], [], **{argument: []})


class VillageWorkflowTests(unittest.TestCase):
    def review_sequence(self):
        return [review_access(), event('Editor', 3, 'generate', outputs=[KEY1])]

    def test_review_is_attributed_to_reviewer_not_observer(self):
        found = workflows(self.review_sequence(), village_rules=rules())
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['pattern'], 'village_review_then_project_change')
        self.assertEqual(found[0]['source_actor'], 'Reviewer')
        self.assertEqual(found[0]['target_actor'], 'Editor')
        self.assertEqual(found[0]['confidence'], 'candidate')
        self.assertTrue(found[0]['verbal_channel'])

    def test_review_requires_actual_target_project_scope(self):
        for entity in ('gitlab:team/other', 'gitlab:team/demo-unrelated', 'https://demo.gitlab.io.evil.test/'):
            events = self.review_sequence()
            events[-1]['entity'] = entity
            self.assertEqual(workflows(events, village_rules=rules()), [])

    def test_review_is_not_applied_to_every_later_commit(self):
        events = self.review_sequence() + [event('Editor', 4, 'generate', outputs=[KEY2])]
        found = workflows(events, village_rules=rules())
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['time'], events[1]['time'])

    def test_cached_rules_cannot_enable_earlier_attribution(self):
        known = rules()
        known['reviews'][0]['available_from'] = '2026-07-06T16:02:00+00:00'
        self.assertEqual(workflows(self.review_sequence(), village_rules=known), [])

    def test_later_mapping_releases_earlier_change_only_at_knowledge_time(self):
        known = rules()
        known['projects'][0]['repository_available_from'] = '2026-07-06T16:04:00+00:00'
        events = self.review_sequence()
        self.assertEqual(workflows(events, village_rules=known), [])
        events += [event('Observer', 4, 'observe')]
        found = workflows(events, village_rules=known)
        self.assertEqual(found[0]['time'], '2026-07-06T16:04:00+00:00')
        self.assertIn(events[1]['event_id'], found[0]['support_events'])
        known = rules()
        known['projects'][0]['repository_available_from'] = '2026-07-06T16:04:00+00:00'
        self.assertEqual(workflows(self.review_sequence(), village_rules=known), [])

    def test_failed_access_and_shared_board_do_not_attribute_owner(self):
        events = self.review_sequence()
        events[0]['status'] = 'failed'
        self.assertEqual(workflows(events, village_rules=rules()), [])
        known = rules()
        known['projects'][0]['shared_resource'] = True
        self.assertEqual(workflows(self.review_sequence(), village_rules=known), [])
        events = [event('Editor', 2, 'generate', outputs=[KEY1]),
                  event('Editor', 3, 'publish', inputs=[KEY1]), site_access()]
        self.assertEqual(workflows(events, village_rules=known), [])

    def test_recommendation_in_edit_is_candidate_with_all_support_turns(self):
        events = self.review_sequence()
        phrase = 'Please retain this distinctive recommendation phrase for the revised German banner.'
        snippets = [snippet('Editor', 1, events[0]['turn_id'], '* Recommendation: `'+phrase+'`', 'o'),
                    snippet('Editor', 2, 'edit-turn', 'path.write_text("'+phrase+'")')]
        found = workflows(events, village_rules=rules(), snippets=snippets)
        specific = [p for p in found if p['pattern'] == 'village_review_recommendation_in_change']
        self.assertEqual(len(specific), 1)
        self.assertEqual(specific[0]['confidence'], 'candidate')
        self.assertIn('edit-turn', specific[0]['support_turns'])
        self.assertEqual(len(specific[0]['support_snippets']), 2)
        self.assertEqual(specific[0]['time'], events[-1]['time'])
        snippets.append(snippet('Editor', 0, 'old-turn', 'path.write_text("'+phrase+'")'))
        found = workflows(events, village_rules=rules(), snippets=snippets)
        self.assertFalse(any(p['pattern'] == 'village_review_recommendation_in_change' for p in found))

    def test_current_text_or_readback_or_future_snippet_is_not_recommendation_reuse(self):
        events = self.review_sequence()
        phrase = 'Please retain this distinctive recommendation phrase for the revised German banner.'
        sources = [snippet('Editor', 1, events[0]['turn_id'], 'Current Text: `'+phrase+'`', 'o'),
                   snippet('Editor', 4, events[0]['turn_id'], 'Recommendation: `'+phrase+'`', 'o')]
        for source in sources:
            snippets = [source, snippet('Editor', 2, 'edit-turn', 'path.write_text("'+phrase+'")')]
            found = workflows(events, village_rules=rules(), snippets=snippets)
            self.assertFalse(any(p['pattern'] == 'village_review_recommendation_in_change' for p in found))
        snippets = [snippet('Editor', 1, events[0]['turn_id'], 'Recommendation: `'+phrase+'`', 'o'),
                    snippet('Editor', 2, 'read-turn', 'print("'+phrase+'")')]
        self.assertFalse(any(p['pattern'] == 'village_review_recommendation_in_change'
                             for p in workflows(events, village_rules=rules(), snippets=snippets)))

    def test_removed_phrase_or_comment_is_not_inserted_recommendation(self):
        events = self.review_sequence()
        phrase = 'Please retain this distinctive recommendation phrase for the revised German banner.'
        source = snippet('Editor', 1, events[0]['turn_id'], 'Recommendation: `'+phrase+'`', 'o')
        for command in ['content.replace("'+phrase+'", "something else")',
                        '# '+phrase+'\npath.write_text("unrelated")',
                        'example = \'content.write_text("'+phrase+'")\'']:
            snippets = [source, snippet('Editor', 2, 'edit-turn', command)]
            self.assertFalse(any(p['pattern'] == 'village_review_recommendation_in_change'
                                 for p in workflows(events, village_rules=rules(), snippets=snippets)))

    def test_publication_uses_observed_producer_not_directory_contact(self):
        events = [event('Writer', 2, 'generate', outputs=[KEY1]),
                  event('Writer', 3, 'publish', inputs=[KEY1]), site_access()]
        found = workflows(events, village_rules=rules())
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['pattern'], 'village_publish_then_external_access')
        self.assertEqual(found[0]['source_actor'], 'Writer')
        self.assertEqual(found[0]['target_actor'], 'Visitor')
        self.assertEqual(found[0]['confidence'], 'candidate')
        self.assertEqual(found[0]['time'], events[-1]['time'])

    def test_publication_needs_observed_version_dependency(self):
        events = [event('Writer', 2, 'generate', outputs=[KEY1]),
                  event('Writer', 3, 'publish', inputs=[KEY2]), site_access()]
        self.assertEqual(workflows(events, village_rules=rules()), [])
        events[1]['inputs'], events[1]['status'] = [KEY1], 'failed'
        self.assertEqual(workflows(events, village_rules=rules()), [])

    def test_late_repeated_directory_evidence_is_not_backdated(self):
        known = rules()
        known['projects'][0]['evidence'] += [dict(actor='Observer', turn_id='future-directory',
                                                 time='2026-07-07T16:00:00+00:00')]
        events = [event('Writer', 2, 'generate', outputs=[KEY1]),
                  event('Writer', 3, 'publish', inputs=[KEY1]), site_access()]
        found = workflows(events, village_rules=known)
        self.assertTrue(found)
        self.assertNotIn('future-directory', {s['turn_id'] for s in found[0]['knowledge_evidence']})

    def test_nontraining_rules_fail_before_use(self):
        known = rules()
        known['reviews'][0]['evidence'][0]['time'] = '2026-08-01T16:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'non-training'):
            workflows(self.review_sequence(), village_rules=known)

    def test_cached_knowledge_cannot_claim_availability_before_its_evidence(self):
        known = rules()
        known['reviews'][0]['evidence'] = [dict(actor='Observer', turn_id='future-review',
                                              time='2026-07-06T16:02:00+00:00')]
        with self.assertRaisesRegex(ValueError, 'precedes supporting evidence'):
            workflows(self.review_sequence(), village_rules=known)

class ScriptWorkflowTests(unittest.TestCase):
    path = 'collaborations/fsa_analysis_script.py'

    def sequence(self):
        events = [event('Writer', 2, 'generate', outputs=[KEY1]), event('Writer', 3, 'publish', inputs=[KEY1])]
        snips = [snippet('Writer', 2, events[0]['turn_id'], 'create mode 100644 '+self.path, 'o'),
                 snippet('Reader', 4, 'acquire', 'cd ~/demo\ngit pull'),
                 snippet('Reader', 4, 'acquire', 'Updating aaaaaaa..'+'a1'*20+'\nFast-forward\n '+self.path+' | 310 ++++', 'o'),
                 snippet('Reader', 5, 'read', 'cd ~/demo\nhead -60 '+self.path),
                 snippet('Reader', 5, 'read', 'Specific Analysis Script Header', 'o'),
                 snippet('Reader', 6, 'run', 'cd ~/demo\npython3 '+self.path),
                 snippet('Reader', 6, 'run', 'SPECIFIC SCRIPT ANALYSIS\nKnown Demo', 'o')]
        known = rules()
        known['script_workflows'] = [dict(rule_id='test_script', repository=REPO, relative_path=self.path,
             working_directories=['~/demo'], read_marker='Specific Analysis Script Header',
             execution_markers=['SPECIFIC SCRIPT ANALYSIS', 'Known Demo'],
             available_from='2026-07-06T16:06:00+00:00', evidence=[dict(actor='Reader', turn_id='run', time='2026-07-06T16:06:00+00:00')])]
        return events, snips, known

    def patterns(self, events, snips, known):
        return [p for p in workflows(events, snippets=snips, village_rules=known)
                if p['pattern'] == 'village_script_acquire_read_execute']

    def test_linked_script_sequence_is_candidate_with_actual_producer(self):
        events, snips, known = self.sequence()
        found = self.patterns(events, snips, known)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['source_actor'], 'Writer')
        self.assertEqual(found[0]['target_actor'], 'Reader')
        self.assertEqual(found[0]['confidence'], 'candidate')
        self.assertEqual(found[0]['available_time'], '2026-07-06T16:06:00+00:00')
        self.assertIn('acquire', found[0]['support_turns'])
        self.assertIn('run', found[0]['support_turns'])

    def test_global_basename_wrong_scope_or_missing_receipt_cannot_link(self):
        for which, text in [(1, 'cd ~/other\ngit pull'), (2, 'log a1a1a1a\n'+self.path),
                            (3, 'cd ~/demo\nhead -60 fsa_analysis_script.py'),
                            (5, 'cd ~/other\npython3 '+self.path),
                            (6, 'Traceback\nSPECIFIC SCRIPT ANALYSIS\nKnown Demo')]:
            events, snips, known = self.sequence()
            snips[which]['text'] = text
            self.assertEqual(self.patterns(events, snips, known), [])

    def test_source_metadata_observation_is_not_script_creation(self):
        events, snips, known = self.sequence()
        events[0]['operation'] = 'observe'
        self.assertEqual(self.patterns(events, snips, known), [])

    def test_future_convention_waits_and_invalid_training_support_rejected(self):
        events, snips, known = self.sequence()
        known['script_workflows'][0]['available_from'] = '2026-07-06T16:07:00+00:00'
        self.assertEqual(self.patterns(events, snips, known), [])
        events += [event('Observer', 7, 'observe')]
        self.assertEqual(self.patterns(events, snips, known)[0]['time'], '2026-07-06T16:07:00+00:00')
        known['script_workflows'][0]['evidence'][0]['time'] = '2026-08-03T16:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'training'):
            self.patterns(events, snips, known)

    def test_short_command_capture_is_workflow_only_and_scope_guarded(self):
        record = dict(i='run', t='2026-07-28T21:08:52+00:00', k='shell',
                      x='cd ~/llm-psychoactive-prompts\npython3 '+self.path, o='result')
        found = extract_workflow_snippets('Reader', record)
        self.assertEqual(len(found), 2)
        self.assertTrue(all(s['workflow_only'] for s in found))
        record['x'] = 'python3 fsa_analysis_script.py'
        self.assertEqual(extract_workflow_snippets('Reader', record), [])
        record['t'] = '2026-08-17T16:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'outside development'):
            extract_workflow_snippets('Reader', record)

    def test_known_review_capture_keeps_middle_recommendation_section(self):
        record = dict(i='review', t='2026-07-07T20:26:37+00:00', k='shell',
                      x='curl https://gitlab.com/team/project/-/raw/main/reviews/wellbeing_compass_de_review.md',
                      o='intro '*300+'\nRecommendation: `a sufficiently distinctive quoted recommendation with eight separate words`\n'+'tail '*300)
        found = extract_workflow_snippets('Reader', record)
        self.assertIn('Recommendation:', found[1]['text'])
        self.assertTrue(found[1]['workflow_only'])


if __name__ == '__main__':
    unittest.main()
