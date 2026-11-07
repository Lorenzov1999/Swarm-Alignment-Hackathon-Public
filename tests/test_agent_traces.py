"""General agent trace identity, scope and operation regressions."""
import unittest

from swarmcollab.agent_traces import AgentTraceExtractor, canonical_url, repository_url
from swarmcollab.provenance_common import command_segments


def step(command, output='', error='', turn='1', **extra):
    record = dict(i=turn, t='2026-07-06 16:00:00', s='session', k='shell',
                  x=command, o=output, e=error)
    record.update(extra)
    return record


def family(events, name):
    return [event for event in events if event['family'] == name]


class GeneralTraceTests(unittest.TestCase):
    def test_leading_shell_comments_ignore_apostrophes_and_control_text(self):
        command = "\n  # Fetch Gemini's page; $(this is comment text)\n\t# another comment\ncurl https://example.org/Page"
        self.assertEqual(command_segments(command), [['curl', 'https://example.org/Page']])
        event, = family(AgentTraceExtractor('A').extract(step(command)), 'url')
        self.assertEqual(event['operation'], 'read')

    def test_comment_prefix_removal_does_not_enter_quoted_programs(self):
        command = 'python -c "\n# comment inside program\nprint(1)"'
        self.assertEqual(command_segments(command), [['python', '-c', '\n# comment inside program\nprint(1)']])
        quoted = 'printf "literal\n# literal text\ncurl https://example.org/Page"'
        self.assertEqual(command_segments(quoted), [['printf', 'literal\n# literal text\ncurl https://example.org/Page']])
        self.assertEqual(command_segments("# only Gemini's comment"), [])

    def test_url_preserves_path_query_and_fragment(self):
        upper = canonical_url('HTTPS://EXAMPLE.ORG:443/Report.csv?id=1#Results')
        self.assertEqual(upper, 'https://example.org/Report.csv?id=1#Results')
        for alternative in ('https://example.org/report.csv?id=1#Results',
                            'https://example.org/Report.csv?id=2#Results',
                            'https://example.org/Report.csv?id=1#results'):
            self.assertNotEqual(upper, canonical_url(alternative))
        self.assertEqual(canonical_url('https://example.org/a;)'), 'https://example.org/a;)')

    def test_credentials_are_not_emitted_or_merged(self):
        for url in ('https://user:secret@example.org/a', 'https://example.org/a?access_token=secret',
                    'https://example.org/a?api_key=secret', 'https://example.org/a#token=secret'):
            self.assertIsNone(canonical_url(url))
            self.assertFalse(family(AgentTraceExtractor('A').extract(step('curl ' + repr(url))), 'url'))

    def test_invalid_url_is_safe(self):
        for url in ('https://[bad', 'https://example.org:bad/path', 'file:///tmp/x', 'https://example.org/a b'):
            self.assertIsNone(canonical_url(url))

    def test_remote_syntax_generic_across_providers(self):
        self.assertEqual(repository_url('git@git.example.org:Team/Project.git'),
                         'https://git.example.org/Team/Project')
        self.assertEqual(repository_url('ssh://git@git.example.org/Team/Project.git'),
                         'https://git.example.org/Team/Project')

    def test_direct_url_attempt_vs_printed_observation(self):
        direct, = family(AgentTraceExtractor('A').extract(step('curl https://example.org/a')), 'url')
        printed, = family(AgentTraceExtractor('A').extract(step("echo 'curl https://example.org/a'")), 'url')
        self.assertEqual((direct['operation'], direct['status']), ('read', 'attempted'))
        self.assertEqual((printed['operation'], printed['status']), ('observe', 'unknown'))
        self.assertFalse(direct['relation_authority'])
        self.assertEqual(direct['inputs'], [])

    def test_output_url_never_becomes_operation(self):
        event, = family(AgentTraceExtractor('A').extract(step('cat notes.txt', 'curl https://example.org/a')), 'url')
        self.assertEqual(event['source'], 'o')
        self.assertEqual(event['operation'], 'observe')

    def test_unmatched_prose_parenthesis_is_not_url_identity(self):
        event, = family(AgentTraceExtractor('A').extract(step('cat notes.txt', '(https://example.org/Results)')), 'url')
        self.assertEqual(event['entity'], 'url:https://example.org/Results')

    def test_curl_data_is_not_a_read_target_and_post_is_write(self):
        events = family(AgentTraceExtractor('A').extract(step("curl -d 'https://example.org/payload' https://example.org/target")), 'url')
        by_entity = {e['entity']: e for e in events}
        self.assertEqual(by_entity['url:https://example.org/payload']['operation'], 'observe')
        self.assertEqual(by_entity['url:https://example.org/target']['operation'], 'write')
        self.assertEqual(by_entity['url:https://example.org/target']['status'], 'attempted')

    def test_same_filename_on_different_machines_stays_scoped(self):
        one = AgentTraceExtractor('A').extract(step('cat /work/research.csv'))
        two = AgentTraceExtractor('B').extract(step('cat /work/research.csv'))
        self.assertNotEqual(family(one, 'file')[0]['entity'], family(two, 'file')[0]['entity'])
        self.assertEqual(family(one, 'file_candidate')[0]['entity'], family(two, 'file_candidate')[0]['entity'])
        self.assertEqual(family(one, 'file_candidate')[0]['identity_strength'], 'candidate')

    def test_known_shared_repo_resolves_different_checkout_paths(self):
        one, two = AgentTraceExtractor('A'), AgentTraceExtractor('B')
        one.extract(step('cd /work/alpha && git remote -v', 'origin https://git.example.org/Team/Repo.git (fetch)\n'))
        two.extract(step('cd /tmp/beta && git remote -v', 'origin git@git.example.org:Team/Repo.git (fetch)\n'))
        a, = family(one.extract(step('cat data/results.csv', turn='2')), 'file')
        b, = family(two.extract(step('cat data/results.csv', turn='2')), 'file')
        self.assertEqual(a['entity'], b['entity'])
        self.assertEqual(a['identity_strength'], 'repository_scoped')

    def test_successful_clone_receipt_learns_explicit_absolute_root(self):
        parser = AgentTraceExtractor('A')
        parser.extract(step('git clone https://git.example.org/Team/Repo.git /work/repo',
                            "Cloning into '/work/repo'...\n", exit_code=0))
        event, = family(parser.extract(step('cat /work/repo/results.csv', turn='2')), 'file')
        self.assertEqual(event['identity_strength'], 'repository_scoped')
        failed = AgentTraceExtractor('B')
        failed.extract(step('git clone https://git.example.org/Team/Repo.git /work/repo',
                            "Cloning into '/work/repo'...\n"))
        event, = family(failed.extract(step('cat /work/repo/results.csv', turn='2')), 'file')
        self.assertEqual(event['identity_strength'], 'actor_session_scoped')

    def test_files_outside_checkout_do_not_inherit_repo(self):
        parser = AgentTraceExtractor('A')
        parser.set_context('session', '/repo', repo='gitlab:team/project', root='/repo')
        event, = family(parser.extract(step('cat ../outside.csv')), 'file')
        self.assertEqual(event['identity_strength'], 'actor_session_scoped')

    def test_failed_cd_invalidates_prior_context(self):
        parser = AgentTraceExtractor('A')
        parser.set_context('session', '/repo', repo='gitlab:team/project', root='/repo')
        parser.extract(step('cd /missing && cat data.csv', error='bash: cd: No such file or directory', exit_code=1))
        event, = family(parser.extract(step('cat data.csv', turn='2')), 'file')
        self.assertEqual(event['identity_strength'], 'actor_session_scoped')

    def test_unknown_change_does_not_reuse_relative_identity(self):
        parser = AgentTraceExtractor('A')
        before, = family(parser.extract(step('cat results.csv')), 'file')
        parser.extract(step('cd "$ROOT"', turn='2'))
        after, = family(parser.extract(step('cat results.csv', turn='3')), 'file')
        self.assertNotEqual(before['entity'], after['entity'])

    def test_opaque_shell_location_changes_invalidate_context(self):
        for command in ('pushd /other', 'source setup.sh', 'if true; then cd /other; fi'):
            parser = AgentTraceExtractor('A')
            parser.set_context('session', '/repo', repo='gitlab:team/project', root='/repo')
            parser.extract(step(command))
            event, = family(parser.extract(step('cat results.csv', turn='2')), 'file')
            self.assertEqual(event['identity_strength'], 'actor_session_scoped')

    def test_out_of_scope_is_filtered_before_state_and_payload(self):
        parser = AgentTraceExtractor('A')
        parser.set_context('session', '/known')
        record = step('cd /forbidden', t='2026-08-17 16:00:00', exit_code=0)
        record['x'] = object()
        self.assertEqual(parser.extract(record), [])
        self.assertEqual(parser.cwd['session'], '/known')

    def test_plain_read_success_needs_explicit_outcome(self):
        parser = AgentTraceExtractor('A')
        attempted, = family(parser.extract(step('cat /tmp/data.csv', 'contents')), 'file')
        succeeded, = family(parser.extract(step('cat /tmp/data.csv', 'contents', turn='2', exit_code=0)), 'file')
        failed, = family(parser.extract(step('cat /tmp/data.csv', error='cat: Permission denied', turn='3')), 'file')
        self.assertEqual([attempted['status'], succeeded['status'], failed['status']], ['attempted', 'success', 'failed'])
        self.assertTrue(all(not e['relation_authority'] for e in (attempted, succeeded, failed)))

    def test_script_arguments_are_references_not_inferred_consumption(self):
        events = family(AgentTraceExtractor('A').extract(step('python analysis.py --input dataset.csv --output results.csv')), 'file')
        self.assertEqual([e['operation'] for e in events], ['execute', 'observe', 'observe'])
        self.assertEqual([e['status'] for e in events], ['attempted', 'unknown', 'unknown'])

    def test_printed_or_embedded_commands_do_not_create_file_operations(self):
        for command in ("echo 'cat data.csv'", "python -c 'print(\"cat data.csv\")'", 'sh -c "cat data.csv"'):
            self.assertEqual(family(AgentTraceExtractor('A').extract(step(command)), 'file'), [])

    def test_safe_redirections_and_quoted_operators(self):
        parser = AgentTraceExtractor('A')
        events = family(parser.extract(step('cat < input.csv > output.csv')), 'file')
        self.assertEqual([e['operation'] for e in events], ['read', 'write'])
        self.assertFalse(family(parser.extract(step("echo 'hello > output.csv'", turn='2')), 'file'))
        self.assertFalse(family(parser.extract(step('echo ">" "output.csv"', turn='3')), 'file'))

    def test_later_cd_does_not_relocate_an_earlier_redirection(self):
        parser = AgentTraceExtractor('A')
        parser.set_context('session', '/old', repo='gitlab:team/project', root='/old')
        events = family(parser.extract(step('echo result > output.csv && cd /new')), 'file')
        self.assertFalse(events)

    def test_head_and_cat_flags_do_not_become_filenames(self):
        for command in ('head -n 5 results.csv', 'cat -n results.csv'):
            events = family(AgentTraceExtractor('A').extract(step(command)), 'file')
            self.assertEqual(len(events), 1)
            self.assertTrue(events[0]['entity'].endswith('results.csv'))

    def test_checksum_receipt_is_observation_of_bytes_not_production(self):
        digest = 'a' * 64
        parser = AgentTraceExtractor('A')
        event, = family(parser.extract(step('sha256sum results.csv', digest + '  results.csv\n')), 'content')
        self.assertEqual(event['entity'], 'content:sha256:' + digest)
        self.assertEqual(event['operation'], 'observe')
        self.assertFalse(event['outputs'])
        self.assertFalse(family(parser.extract(step('cat checksums.txt', digest + '  results.csv\n', turn='2')), 'content'))
        self.assertFalse(family(parser.extract(step('sha256sum other.csv', digest + '  results.csv\n', turn='3')), 'content'))

    def test_checksum_failure_and_compound_spoof_are_not_receipts(self):
        digest = 'a' * 64
        for record in (step('sha256sum results.csv', digest + '  results.csv', exit_code=1),
                       step('echo fake && sha256sum results.csv', digest + '  results.csv')):
            self.assertFalse(family(AgentTraceExtractor('A').extract(record), 'content'))

    def test_immutable_container_and_python_pins(self):
        digest = 'a' * 64
        image, = family(AgentTraceExtractor('A').extract(step('docker pull registry.example.org/model@sha256:' + digest)), 'package')
        self.assertEqual(image['status'], 'attempted')
        self.assertFalse(family(AgentTraceExtractor('A').extract(step('docker pull registry.example.org/model:latest')), 'package'))
        pin, = family(AgentTraceExtractor('A').extract(step('python -m pip install Some_Package==1.2.3')), 'package')
        self.assertEqual(pin['entity'], 'python-package-pin:some-package==1.2.3')
        self.assertEqual(pin['identity_strength'], 'candidate')

    def test_custom_package_index_is_not_assumed_public(self):
        events = AgentTraceExtractor('A').extract(step('pip install --index-url https://packages.example.org Some_Package==1.2.3'))
        self.assertFalse(family(events, 'package'))

    def test_container_command_argument_is_not_acquired_image(self):
        events = AgentTraceExtractor('A').extract(step('docker run alpine echo example.org/other@sha256:' + 'a' * 64))
        self.assertFalse(family(events, 'package'))

    def test_common_filenames_not_weak_matching_features(self):
        events = AgentTraceExtractor('A').extract(step('cat README.md requirements.txt'))
        self.assertEqual(len(family(events, 'file')), 2)
        self.assertFalse(family(events, 'file_candidate'))

    def test_event_ids_are_deterministic_and_deduplicated(self):
        record = step('cat /work/results.csv /work/results.csv')
        one = AgentTraceExtractor('A').extract(record)
        two = AgentTraceExtractor('A').extract(record)
        self.assertEqual(one, two)
        self.assertEqual(len({e['event_id'] for e in one}), len(one))
        self.assertTrue(all(e['research_layer'] == 1 for e in one))


if __name__ == '__main__':
    unittest.main()
