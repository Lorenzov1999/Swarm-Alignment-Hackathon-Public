"""Adversarial and successful examples for structured provenance extraction."""
import base64
import hashlib
import json
import unittest

from swarmcollab.provenance_structured import StructuredExtractor
from swarmcollab.provenance_common import git_key

SHA = 'a' * 40
PARENT = '1' * 40
MERGED = 'c' * 40
REPO = 'gitlab:team/project'
URL = 'https://gitlab.com/api/v4/projects/team%2Fproject'


def record(command, output, turn='1', error='', stamp='2026-07-10 16:00:00'):
    return {'i': turn, 't': stamp, 's': 'session', 'k': 'shell', 'x': command,
            'o': output if isinstance(output, str) else json.dumps(output), 'e': error}


def mr(**changes):
    obj = {'id': 10, 'iid': 2, 'project_id': 123, 'state': 'opened',
           'source_branch': 'feature', 'merge_status': 'can_be_merged', 'sha': SHA,
           'web_url': 'https://gitlab.com/team/project/-/merge_requests/2'}
    obj.update(changes)
    return obj


class StructuredTests(unittest.TestCase):
    def test_job_lineage_is_observation_not_agent_production(self):
        extractor = StructuredExtractor('B')
        obj = {'id': 9, 'stage': 'deploy', 'status': 'success',
               'web_url': 'https://gitlab.com/team/project/-/jobs/9',
               'commit': {'id': SHA, 'parent_ids': [PARENT]},
               'pipeline': {'id': 7, 'sha': SHA, 'status': 'success'},
               'artifacts_file': {'filename': 'artifacts.zip', 'size': 47065}}
        events = extractor.extract(record('curl ' + URL + '/jobs/9', obj))
        self.assertTrue(events)
        self.assertTrue(all(e['operation'] == 'observe' for e in events))
        self.assertTrue(all(e['relation_authority'] for e in events))
        edges = [r for e in events for r in e['relations']]
        self.assertIn({'kind': 'derived_from', 'source': git_key(REPO, SHA), 'target': git_key(REPO, PARENT)}, edges)
        self.assertIn('artifact:' + REPO + ':job:9', [key for e in events for key in e['outputs']])
        self.assertEqual([e for e in events if e['entity'].startswith('job:')][0]['workflow_status'], 'success')

    def test_untrusted_metadata_does_not_authorize_relations_or_poison_context(self):
        obj = {'id': 9, 'project_id': 999, 'stage': 'deploy', 'status': 'success',
               'web_url': 'https://gitlab.com/team/project/-/jobs/9',
               'commit': {'id': SHA, 'parent_ids': [PARENT]},
               'artifacts_file': {'filename': 'artifacts.zip', 'size': 10}}
        extractor = StructuredExtractor('B')
        printed = extractor.extract(record('cat cached.json', obj))
        self.assertTrue(printed)
        self.assertTrue(all(not e['relation_authority'] for e in printed))
        self.assertFalse(extractor.project_repos)
        failed = extractor.extract(record('curl https://gitlab.com/api/v4/projects/999/jobs/9', obj, error='curl: (22) HTTP 403'))
        self.assertTrue(all(not e['relation_authority'] for e in failed))
        self.assertFalse(extractor.project_repos)
        extractor.extract(record('cat state.json', mr(), turn='2'))
        self.assertFalse(extractor.last_state)

    def test_response_formatters_cannot_authorize_fabricated_metadata(self):
        obj = {'id': 9, 'stage': 'deploy', 'status': 'success',
               'web_url': 'https://gitlab.com/team/project/-/jobs/9',
               'commit': {'id': SHA, 'parent_ids': [PARENT]}}
        for formatter in ("--jq '{id:9}'", "--jq='{id:9}'", "-q '{id:9}'", "-q'{id:9}'", "--template '{{.}}'", "-t '{{.}}'"):
            events = StructuredExtractor('B').extract(record('glab api projects/team%2Fproject/jobs/9 ' + formatter, obj))
            self.assertTrue(events)
            self.assertTrue(all(e['operation'] == 'observe' and not e['relation_authority'] for e in events), formatter)
        for identity in ('--jq .', '--jq=.', '-q .', '-q.'):
            events = StructuredExtractor('B').extract(record('glab api projects/team%2Fproject/jobs/9 ' + identity, obj))
            self.assertTrue(all(e['relation_authority'] for e in events), identity)
        command = 'curl ' + URL + '/jobs/9 --write-out ' + repr(json.dumps(obj))
        events = StructuredExtractor('B').extract(record(command, obj))
        self.assertTrue(events)
        self.assertTrue(all(not e['relation_authority'] for e in events))

    def test_file_metadata_only_is_not_content_consumption(self):
        obj = {'file_path': 'data.csv', 'content_sha256': 'b' * 64, 'blob_id': SHA, 'commit_id': PARENT}
        events = StructuredExtractor('B').extract(record('curl ' + URL + '/repository/files/data.csv', obj))
        self.assertEqual(events[0]['operation'], 'observe')
        self.assertIn('gitlab_content_sha256:', events[0]['outputs'][0])
        self.assertFalse(any(key.startswith('content:sha256:') for key in events[0]['outputs']))

    def test_file_content_access_and_raw_representation(self):
        data = b'column\n42\n'
        obj = {'file_path': 'data.csv', 'content_sha256': 'b' * 64, 'blob_id': SHA, 'commit_id': PARENT,
               'content': base64.b64encode(data).decode(), 'encoding': 'base64'}
        events = StructuredExtractor('B').extract(record('curl ' + URL + '/repository/files/data.csv', obj))
        event = events[0]
        self.assertEqual(event['operation'], 'acquire')
        self.assertIn('content:sha256:' + hashlib.sha256(data).hexdigest(), event['inputs'])
        self.assertFalse(event['outputs'])

    def test_api_file_creation_requires_matching_success_response(self):
        data = {'content': 'column\n42\n', 'branch': 'main'}
        cmd = 'curl -X POST ' + URL + '/repository/files/data.csv --data ' + repr(json.dumps(data))
        extractor = StructuredExtractor('A')
        events = extractor.extract(record(cmd, {'file_path': 'data.csv', 'branch': 'main'}))
        self.assertEqual(events[0]['operation'], 'generate')
        self.assertTrue(events[0]['outputs'])
        wrong = extractor.extract(record(cmd, {'file_path': 'unrelated.csv', 'branch': 'main'}, turn='2'))
        self.assertFalse(wrong)
        failed = extractor.extract(record(cmd, {'message': '403 Forbidden'}, turn='3'))
        self.assertTrue(all(e['operation'] != 'generate' for e in failed))

    def test_failed_get_does_not_claim_acquisition(self):
        obj = {'file_path': 'data.csv', 'content_sha256': 'b' * 64, 'encoding': 'base64',
               'content': base64.b64encode(b'old cache').decode()}
        events = StructuredExtractor('B').extract(record('curl ' + URL + '/repository/files/data.csv', obj, error='curl: (22) HTTP 403'))
        self.assertTrue(all(e['operation'] != 'acquire' for e in events))

    def test_lists_partial_json_and_echoes_never_confirm_mutation(self):
        command = 'curl -X POST ' + URL + '/merge_requests'
        listed = StructuredExtractor('B').extract(record(command, [mr()]))
        self.assertTrue(all(e['operation'] == 'observe' for e in listed))
        partial = '{"id":10,"nested": {"id": 8, "parent_ids": [], "sha": "' + SHA + '"}\n'
        self.assertFalse(StructuredExtractor('B').extract(record(command, partial)))
        echoed = StructuredExtractor('B').extract(record('echo ' + repr(command), mr()))
        self.assertTrue(all(e['operation'] == 'observe' for e in echoed))

    def test_multiple_requests_cannot_attribute_one_response(self):
        command = 'curl -X POST ' + URL + '/merge_requests && curl ' + URL + '/merge_requests/2'
        events = StructuredExtractor('A').extract(record(command, mr()))
        self.assertTrue(all(e['operation'] == 'observe' for e in events))

    def test_repeated_mr_states_are_snapshots_not_transitions(self):
        extractor = StructuredExtractor('B')
        command = 'curl ' + URL + '/merge_requests/2'
        first = extractor.extract(record(command, mr()))[-1]
        second = extractor.extract(record(command, mr(), turn='2'))[-1]
        changed = extractor.extract(record(command, mr(state='closed'), turn='3'))[-1]
        self.assertFalse(first['state_changed'])
        self.assertFalse(second['state_changed'])
        self.assertTrue(changed['state_changed'])
        self.assertEqual(changed['operation'], 'observe')
        self.assertEqual(changed['previous_event_id'], second['event_id'])

    def test_successful_mr_merge_carries_input_and_result(self):
        command = 'curl -X PUT ' + URL + '/merge_requests/2/merge'
        events = StructuredExtractor('B').extract(record(command, mr(state='merged', merge_commit_sha=MERGED)))
        event = next(e for e in events if e['operation'] == 'incorporate')
        self.assertIn(git_key(REPO, SHA), event['inputs'])
        self.assertIn(git_key(REPO, MERGED), event['outputs'])
        wrong_project = StructuredExtractor('B').extract(record(command.replace('team%2Fproject', 'different%2Frepo'), mr(state='merged', merge_commit_sha=MERGED)))
        self.assertTrue(all(e['operation'] == 'observe' for e in wrong_project))

    def test_review_keeps_verbal_channel_separate(self):
        payload = {'body': 'Please adjust the schema.'}
        command = 'curl -X POST ' + URL + '/merge_requests/2/notes --data ' + repr(json.dumps(payload))
        note = {'id': 11, 'body': payload['body'], 'position': {'head_sha': SHA}, 'resolved': False}
        event = StructuredExtractor('A').extract(record(command, note))[0]
        self.assertEqual(event['operation'], 'review')
        self.assertTrue(event['verbal_channel'])
        self.assertNotIn(payload['body'], json.dumps(event))

    def test_artifact_download_needs_explicit_outcome(self):
        command = "curl -w '%{http_code}' -o output.zip " + URL + '/jobs/9/artifacts'
        extractor = StructuredExtractor('B')
        attempted = extractor.extract(record(command, ''))[0]
        acquired = extractor.extract(record(command, '200', turn='2'))[0]
        failed = extractor.extract(record(command, '404 Not Found', turn='3'))[0]
        self.assertEqual(attempted['status'], 'attempted')
        self.assertEqual(acquired['status'], 'success')
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(acquired['inputs'], ['artifact:' + REPO + ':job:9'])
        echoed_status = extractor.extract(record(command + '; echo 200', '200', turn='4'))[0]
        self.assertEqual(echoed_status['status'], 'attempted')
        unmarked = extractor.extract(record(command.replace("-w '%{http_code}' ", ''), '200', turn='5'))[0]
        self.assertEqual(unmarked['status'], 'attempted')

    def test_external_packages_have_no_invented_agent_producer(self):
        package = {'id': 6, 'name': 'public-lib', 'version': '1.0', 'package_files': [
            {'file_name': 'lib.tgz', 'file_sha256': 'd' * 64}], 'pipelines': [{'id': 7, 'sha': SHA}]}
        events = StructuredExtractor('B').extract(record('curl ' + URL + '/packages/6', package))
        self.assertEqual(events[0]['operation'], 'observe')
        self.assertTrue(events[0]['producer_unknown'])
        self.assertIn(git_key(REPO, SHA), events[0]['inputs'])

    def test_review_suggestion_application_preserves_note_link(self):
        extractor = StructuredExtractor('B')
        note = {'id': 11, 'body': 'Suggested replacement', 'suggestions': [{'id': 12, 'applied': False}]}
        seen = extractor.extract(record('curl ' + URL + '/merge_requests/2/notes/11', note))[0]
        events = extractor.extract(record('curl -X PUT https://gitlab.com/api/v4/suggestions/12/apply', {'id': 12, 'applied': True}, turn='2'))
        self.assertEqual(events[0]['operation'], 'incorporate')
        self.assertEqual(events[0]['inputs'], seen['outputs'])
        self.assertTrue(events[0]['verbal_channel'])

    def test_lockfile_declaration_and_same_step_install_are_distinct(self):
        obj = {'lockfileVersion': 3, 'packages': {'node_modules/lib': {'integrity': 'sha512-' + base64.b64encode(b'x' * 64).decode()}}}
        extractor = StructuredExtractor('B')
        extractor.set_context(repo=REPO)
        observed = extractor.extract(record('cat package-lock.json', obj))[0]
        installed = extractor.extract(record('cat package-lock.json && npm ci', json.dumps(obj) + '\nadded 1 package in 2s', turn='2'))[0]
        self.assertEqual(observed['operation'], 'observe')
        self.assertEqual(installed['operation'], 'install')
        self.assertEqual(installed['inputs'], observed['inputs'])
        failed = extractor.extract(record('cat package-lock.json && npm ci', json.dumps(obj) + '\nnpm ERR! failure', turn='3'))[0]
        self.assertEqual(failed['operation'], 'observe')

    def test_digest_pinned_install_and_container_pull(self):
        extractor = StructuredExtractor('B')
        command = 'pip install https://example.org/library.whl#sha256=' + 'a' * 64
        installed = extractor.extract(record(command, 'Successfully installed library-1.0'))[0]
        self.assertEqual(installed['operation'], 'install')
        self.assertEqual(installed['status'], 'success')
        image = 'registry.example.org/lib@sha256:' + 'b' * 64
        pulled = extractor.extract(record('docker pull ' + image, 'Digest: sha256:' + 'b' * 64 + '\nStatus: Downloaded newer image', turn='2'))[0]
        self.assertEqual(pulled['inputs'], ['container:' + image])
        self.assertEqual(pulled['status'], 'success')

    def test_document_revisions_are_user_specific_and_content_is_hashed(self):
        content = 'A substantive document paragraph with several specific research findings.\n'
        doc = {'documentId': 'document-123', 'revisionId': 'USER-SPECIFIC-SECRET-REVISION',
               'body': {'content': [{'paragraph': {'elements': [{'textRun': {'content': content}}]}}]}}
        command = 'curl https://docs.googleapis.com/v1/documents/document-123'
        a = StructuredExtractor('A').extract(record(command, doc))[0]
        b = StructuredExtractor('B').extract(record(command, {**doc, 'revisionId': 'DIFFERENT-USER-REVISION'}))[0]
        self.assertEqual(a['inputs'], b['inputs'])
        self.assertEqual(a['provider_revision_scope'], 'A')
        self.assertEqual(b['provider_revision_scope'], 'B')
        serialized = json.dumps(a)
        self.assertNotIn(content, serialized)
        self.assertNotIn('USER-SPECIFIC-SECRET-REVISION', serialized)
        self.assertFalse(StructuredExtractor('A').extract(record(command, {'documentId': 'document-123', 'revisionId': 'opaque'})))

    def test_doc_edit_without_snapshot_does_not_invent_version(self):
        payload = {'requests': [{'insertText': {'text': 'A new paragraph', 'location': {'index': 1}}}]}
        command = 'curl -X POST https://docs.googleapis.com/v1/documents/document-123:batchUpdate --data ' + repr(json.dumps(payload))
        events = StructuredExtractor('A').extract(record(command, {'documentId': 'document-123', 'replies': [{}], 'writeControl': {'requiredRevisionId': 'private'}}))
        self.assertEqual(events[0]['operation'], 'transform')
        self.assertEqual(events[0]['outputs'], [])
        self.assertTrue(events[0]['version_unobserved'])
        extractor = StructuredExtractor('A')
        extractor.extract(record('curl https://docs.googleapis.com/v1/documents/document-123', {
            'documentId': 'document-123', 'revisionId': 'stale',
            'body': {'content': [{'paragraph': {'elements': [{'textRun': {'content': 'old content'}}]}}]}}))
        later = extractor.extract(record(command, {'documentId': 'document-123', 'replies': [{}]}, turn='2'))[0]
        self.assertEqual(later['inputs'], [])

    def test_sheet_write_requires_returned_values_and_update_count(self):
        payload = {'values': [[1, 2], [3, 4]]}
        command = 'curl -X PUT https://sheets.googleapis.com/v4/spreadsheets/sheet-123/values/A1:B2 --data ' + repr(json.dumps(payload))
        response = {'spreadsheetId': 'sheet-123', 'updatedCells': 4,
                    'updatedData': {'range': 'Sheet1!A1:B2', 'values': payload['values']}}
        event = StructuredExtractor('A').extract(record(command, response))[0]
        self.assertEqual(event['operation'], 'generate')
        self.assertTrue(event['outputs'])
        response['updatedCells'] = 0
        event = StructuredExtractor('A').extract(record(command, response))[0]
        self.assertEqual(event['operation'], 'observe')

    def test_sealed_and_earlier_dates_are_not_processed(self):
        extractor = StructuredExtractor('B')
        for stamp in ('2026-07-06 06:59:59', '2026-08-15 07:00:00', '2026-08-17 16:00:00'):
            self.assertEqual(extractor.extract(record('curl ' + URL + '/merge_requests/2', mr(), stamp=stamp)), [])

    def test_malformed_schema_fields_do_not_crash_or_invent_producers(self):
        extractor = StructuredExtractor('B')
        extractor.set_context(repo=REPO)
        cases = [
            ('curl ' + URL + '/jobs/9', {'id': 9, 'stage': 'deploy', 'status': 'failed', 'commit': {'id': SHA, 'parent_ids': 1}}),
            ('curl ' + URL + '/merge_requests/2/notes', {'id': 11, 'body': 'note', 'suggestions': 1}),
            ('curl ' + URL + '/packages/6', {'file_name': {'unexpected': 'object'}, 'file_sha256': 'a' * 64}),
            ('curl https://[malformed', {}),
        ]
        for command, obj in cases:
            events = extractor.extract(record(command, obj))
            self.assertTrue(all(e['operation'] == 'observe' for e in events))

    def test_credentials_are_not_copied_into_events(self):
        command = 'curl -H "Authorization: Bearer DO-NOT-SAVE" https://user:DO-NOT-SAVE@gitlab.com/api/v4/projects/team%2Fproject/jobs/9'
        obj = {'id': 9, 'stage': 'deploy', 'status': 'success',
               'web_url': 'https://user:DO-NOT-SAVE@gitlab.com/team/project/-/jobs/9',
               'pipeline': {'id': 7, 'sha': SHA, 'status': 'success'}}
        events = StructuredExtractor('B').extract(record(command, obj))
        self.assertTrue(events)
        self.assertNotIn('DO-NOT-SAVE', json.dumps(events))


if __name__ == '__main__':
    unittest.main()
