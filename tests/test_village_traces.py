"""Village knowledge isolation, attribution and candidate-evidence checks."""
import copy
import unittest

from swarmcollab.village_traces import VillageTraceExtractor, fit_village_rules


SITE = 'https://wellbeing-compass-409cf0.gitlab.io/'
REPO = 'https://gitlab.com/ai-village-agents/village/wellbeing-compass'
REVIEW = 'https://gitlab.com/ai-village-agents/village/merch-store/-/raw/main/reviews/wellbeing_de_review.md'


def record(i='r1', t='2026-07-06T18:00:00Z', x='', o='', k='shell', e=''):
    return dict(i=i, t=t, s='session', x=x, o=o, k=k, e=e)


def directory(owner='Claude Sonnet 5', site=SITE, i='directory', t='2026-07-06T18:00:00Z'):
    return record(i=i, t=t, x='curl -s https://village-hub-1ddcad.gitlab.io/directory.html',
                  o='{a:"' + owner + '", goal:"wellbeing", proj:{t:"Wellbeing Compass", u:"' + site + '"}, collab:"cross-links"}')


def fitted(extra=()):
    return fit_village_rules([('Reader', directory()),
                              ('Reader', record(i='repo', x='git ls-remote ' + REPO))] + list(extra),
                             actor_names=['Reader', 'Claude Sonnet 5', 'Gemini 3.5 Flash'])


class VillageKnowledgeTests(unittest.TestCase):
    def test_custom_directory_schema_maps_project_and_retains_support(self):
        rules = fitted()
        self.assertEqual(len(rules['projects']), 1)
        p = rules['projects'][0]
        self.assertEqual(p['owner'], 'Claude Sonnet 5')
        self.assertEqual(p['site'], SITE)
        self.assertEqual(p['repository'], REPO)
        self.assertEqual(p['evidence'][0]['turn_id'], 'directory')
        self.assertEqual(p['repository_evidence'][0]['turn_id'], 'repo')

    def test_index_schema_and_name_punctuation(self):
        row = record(x='curl https://village-hub-1ddcad.gitlab.io/',
                     o='{a:"Claude-Sonnet-5",g:"Wellbeing Compass",href:"' + SITE + '"}')
        rules = fit_village_rules([('Reader', row)], ['Claude Sonnet 5'])
        self.assertEqual(rules['projects'][0]['owner'], 'Claude Sonnet 5')

    def test_title_preserves_escaped_quotes_and_decodes_display_escapes(self):
        row = directory()
        row['o'] = r'{a:"Claude Sonnet 5",proj:{t:"Comma Fox merch \u2014 \"a pause, not a stop\" tees",u:"' + SITE + '"}}'
        rules = fit_village_rules([('Reader', row)], ['Claude Sonnet 5'])
        self.assertEqual(rules['projects'][0]['title'], 'Comma Fox merch — "a pause, not a stop" tees')

    def test_unobserved_repo_is_not_guessed_from_site(self):
        rules = fit_village_rules([('Reader', directory())], ['Claude Sonnet 5'])
        self.assertIsNone(rules['projects'][0]['repository'])

    def test_multiple_entries_keep_url_next_to_its_actor(self):
        row = directory()
        row['o'] += '\n{a:"Gemini 3.5 Flash",proj:{t:"Merch",u:"https://merch-123abc.gitlab.io/"}}'
        rules = fit_village_rules([('Reader', row)], ['Claude Sonnet 5', 'Gemini 3.5 Flash'])
        self.assertEqual({p['owner']: p['site'] for p in rules['projects']},
                         {'Claude Sonnet 5': SITE, 'Gemini 3.5 Flash': 'https://merch-123abc.gitlab.io/'})

    def test_conflicting_ownership_is_excluded(self):
        rules = fitted([('Reader', directory(owner='Gemini 3.5 Flash', i='conflict'))])
        self.assertEqual(rules['projects'], [])
        self.assertEqual(len(rules['conflicts']), 1)

    def test_shared_public_platform_root_is_never_owned(self):
        rules = fitted([('Reader', directory(site='https://manifold.markets/', i='platform'))])
        self.assertEqual(len(rules['projects']), 1)
        self.assertEqual(rules['excluded'][0]['url'], 'https://manifold.markets/')
        self.assertEqual(VillageTraceExtractor('Reader', rules).extract(record(x='curl https://manifold.markets/someone-else')), [])

    def test_arbitrary_name_url_proximity_is_not_directory_schema(self):
        row = directory()
        row['o'] = 'Claude Sonnet 5 says see ' + SITE
        rules = fit_village_rules([('Reader', row)], ['Claude Sonnet 5'])
        self.assertEqual(rules['projects'], [])

    def test_schema_outside_known_hub_does_not_assign_projects(self):
        row = directory()
        row['x'] = 'curl https://unrelated.example/directory.html'
        self.assertEqual(fit_village_rules([('Reader', row)], ['Claude Sonnet 5'])['projects'], [])

    def test_validation_record_rejected_before_payload_access(self):
        class DateOnly(dict):
            def get(self, *args):
                raise AssertionError('Payload accessed before date guard')
        row = DateOnly(t='2026-08-03T18:00:00Z')
        with self.assertRaises(ValueError):
            fit_village_rules([('Reader', row)])

    def test_fitting_rejects_before_training_boundary(self):
        with self.assertRaises(ValueError):
            fit_village_rules([('Reader', record(t='2026-07-06T06:59:59Z'))])

    def test_review_attribution_is_frozen_with_evidence(self):
        row = record(i='review', x='curl -s ' + REVIEW,
                     o='**Reviewer:** Gemini 3.5 Flash  \n**Target Project:** Wellbeing Compass (`' + SITE + '`)\n')
        rules = fitted([('Reader', row)])
        self.assertEqual(len(rules['reviews']), 1)
        self.assertEqual(rules['reviews'][0]['reviewer'], 'Gemini 3.5 Flash')
        self.assertEqual(rules['reviews'][0]['evidence'][0]['turn_id'], 'review')


class VillageExtractionTests(unittest.TestCase):
    def setUp(self):
        self.rules = fitted()
        self.extractor = VillageTraceExtractor('Reader', self.rules)

    def test_direct_project_access_is_only_a_candidate(self):
        events = self.extractor.extract(record(t='2026-08-03T18:00:00Z', x='curl -s ' + SITE + 'de/index.html'))
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e['related_actor'], 'Claude Sonnet 5')
        self.assertEqual(e['research_layer'], 2)
        self.assertEqual(e['status'], 'attempted')
        self.assertFalse(e['verbal_channel'])
        self.assertEqual(e['inputs'], [])
        self.assertEqual(e['outputs'], [])

    def test_apostrophe_in_leading_comment_does_not_hide_access(self):
        row = record(x="# Fetch another agent's project\ncurl -s " + SITE)
        self.assertEqual(len(self.extractor.extract(row)), 1)

    def test_explicit_failure_stays_failed(self):
        e = self.extractor.extract(record(x='curl ' + SITE, e='network error'))[0]
        self.assertEqual(e['status'], 'failed')
        self.assertFalse(e['relation_authority'])

    def test_exit_status_and_stdout_failures_remain_failed(self):
        row = record(x='curl ' + SITE)
        row['returncode'] = 22
        self.assertEqual(self.extractor.extract(row)[0]['status'], 'failed')
        row = record(x='curl ' + SITE, o='HTTP/2 404 Not Found')
        self.assertEqual(self.extractor.extract(row)[0]['status'], 'failed')

    def test_output_directory_and_printed_commands_are_not_project_accesses(self):
        self.assertEqual(self.extractor.extract(directory()), [])
        self.assertEqual(self.extractor.extract(record(x='echo "curl ' + SITE + '"')), [])
        self.assertEqual(self.extractor.extract(record(x='echo ok', o='curl ' + SITE)), [])

    def test_header_body_urls_are_not_resources(self):
        row = record(x='curl -H "Referer: ' + SITE + '" -d ' + SITE + ' https://unrelated.example/endpoint')
        self.assertEqual(self.extractor.extract(row), [])

    def test_typed_url_navigation_supported_but_prose_not(self):
        self.assertEqual(len(self.extractor.extract(record(k='type', x=SITE))), 1)
        self.assertEqual(self.extractor.extract(record(k='type', x='Please visit ' + SITE)), [])

    def test_repository_access_has_independent_training_evidence(self):
        self.assertEqual(len(self.extractor.extract(record(x='git clone ' + REPO + '.git'))), 1)
        self.assertEqual(self.extractor.extract(record(x='git clone ' + REPO + '-unrelated.git')), [])

    def test_overlapping_project_aliases_emit_once_without_future_backdating(self):
        rules = fitted([('Reader', directory(site=SITE + 'gallery.html', i='gallery'))])
        extractor = VillageTraceExtractor('Reader', rules)
        for url in (SITE + 'gallery.html', REPO):
            events = extractor.extract(record(x='curl ' + url))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['entity'], 'village-project:' + SITE)
        rows = [('Reader', directory(i='root', t='2026-07-20T18:00:00Z')),
                ('Reader', directory(site=SITE + 'gallery.html', i='gallery'))]
        early = VillageTraceExtractor('Reader', fit_village_rules(rows, ['Claude Sonnet 5']))
        event = early.extract(record(x='curl ' + SITE + 'gallery.html'))[0]
        self.assertEqual(event['entity'], 'village-project:' + SITE + 'gallery.html')

    def test_knowledge_not_applied_before_observed(self):
        self.assertEqual(self.extractor.extract(record(t='2026-07-06T17:59:59Z', x='curl ' + SITE)), [])

    def test_shared_village_board_does_not_create_listed_agent_edge(self):
        board = 'https://docs.google.com/document/d/board-id/edit'
        row = directory(owner='Claude Haiku 4.5', site=board)
        row['o'] = row['o'].replace('Wellbeing Compass', 'Village Handoff Board (live doc)')
        rules = fit_village_rules([('Reader', row)], ['Claude Haiku 4.5'])
        project = rules['projects'][0]
        self.assertTrue(project['shared_resource'])
        self.assertEqual(project['owner'], 'Claude Haiku 4.5')
        self.assertEqual(project['shared_evidence'][0]['turn_id'], 'directory')
        event = VillageTraceExtractor('Reader', rules).extract(record(x='curl ' + board))[0]
        self.assertEqual(event['family'], 'village_shared_board')
        self.assertEqual(event['related_actor'], '')
        self.assertEqual(event['listed_agent'], 'Claude Haiku 4.5')
        self.assertFalse(event['relation_authority'])
        self.assertEqual(event['status'], 'attempted')

    def test_known_history_dump_is_excluded(self):
        row = record(x='bash /home/computeruse/memory/scripts/pre_send_chat.sh; curl ' + SITE)
        self.assertEqual(self.extractor.extract(row), [])

    def test_incidental_history_filename_is_not_a_history_execution(self):
        row = record(x='curl ' + SITE + 'public_comms.md')
        self.assertEqual(len(self.extractor.extract(row)), 1)

    def test_http_mutation_is_not_mislabeled_read(self):
        for flag in ('-X POST', '--request=POST', '-XPOST', '--data=x'):
            self.assertEqual(self.extractor.extract(record(x='curl ' + flag + ' ' + SITE)), [])

    def test_rules_are_frozen_against_caller_mutation(self):
        self.rules['projects'][0]['owner'] = 'Someone else'
        e = self.extractor.extract(record(x='curl ' + SITE))[0]
        self.assertEqual(e['related_actor'], 'Claude Sonnet 5')

    def test_sealed_record_is_filtered_before_payload_access(self):
        class DateOnly(dict):
            def get(self, *args):
                raise AssertionError('Payload accessed before date guard')
        self.assertEqual(self.extractor.extract(DateOnly(t='2026-08-17T18:00:00Z')), [])

    def test_review_access_separate_verbal_channel_not_incorporation(self):
        row = record(i='review', x="# Fetch Gemini 3.5 Flash's review\ncurl " + REVIEW,
                     o='**Reviewer:** Gemini 3.5 Flash\n**Target Project:** ' + SITE)
        rules = fitted([('Reader', row)])
        e = VillageTraceExtractor('Reader', rules).extract(record(t='2026-08-03T18:00:00Z', x='curl ' + REVIEW))[0]
        self.assertTrue(e['verbal_channel'])
        self.assertEqual(e['related_actor'], 'Gemini 3.5 Flash')
        self.assertEqual(e['status'], 'attempted')
        self.assertEqual(e['family'], 'village_review_reference')

    def test_deterministic_extraction_and_bad_rules_guard(self):
        r = record(x='curl ' + SITE)
        self.assertEqual(self.extractor.extract(r), self.extractor.extract(r))
        invalid = copy.deepcopy(self.rules)
        invalid['fit_end'] = '2026-08-14'
        with self.assertRaises(ValueError):
            VillageTraceExtractor('Reader', invalid)


if __name__ == '__main__':
    unittest.main()
