"""Outcome and identity checks for Git/file provenance; stdlib unittest only."""
import hashlib
import itertools
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from swarmcollab.provenance_common import git_key
from swarmcollab.provenance_git import GitExtractor


A = 'a' * 40
B = 'b' * 40
C = '1234567' + 'c' * 33
REPO = 'gitlab:group/project'
IDS = itertools.count()


def step(command, output='', error='', **extra):
    return dict(i='turn-' + str(next(IDS)), t='2026-07-06 16:00:00',
                s='session', k='shell', x=command, o=output, e=error, **extra)


def known(actor='Agent A'):
    parser = GitExtractor(actor)
    parser.extract(step('cd /work/project && git remote -v',
                        'origin https://gitlab.com/group/project.git (fetch)\n'
                        'origin https://gitlab.com/group/project.git (push)\n'))
    return parser


class GitEvidenceTests(unittest.TestCase):
    def test_confirmed_commit_preserves_full_parent_and_short_result(self):
        parser = known()
        parser.extract(step('git rev-parse HEAD', A + '\n'))
        event, = parser.extract(step('git add result.csv && git commit -m result', '[main bbbbbbb] result\n 1 file changed'))
        self.assertEqual(event['status'], 'success')
        self.assertEqual(event['operation'], 'generate')
        self.assertEqual(event['outputs'], [git_key(REPO, 'bbbbbbb')])
        self.assertEqual(event['inputs'], [git_key(REPO, A)])
        self.assertTrue(event['abbreviated'])
        self.assertTrue(event['relation_authority'])
        self.assertEqual(event['relations'][0]['target'], git_key(REPO, A))

    def test_full_resolution_keeps_explicit_alias(self):
        parser = known()
        parser.extract(step('git commit -m result', '[main bbbbbbb] result\n'))
        event, = parser.extract(step('git rev-parse HEAD', B + '\n'))
        self.assertEqual(event['relations'], [{'kind': 'same_content', 'source': git_key(REPO, 'bbbbbbb'), 'target': git_key(REPO, B)}])
        self.assertFalse(event['abbreviated'])

    def test_amend_is_revision_not_parent_claim(self):
        parser = known()
        parser.extract(step('git rev-parse HEAD', A + '\n'))
        event, = parser.extract(step('git commit --amend -m revised', '[main bbbbbbb] revised\n'))
        self.assertEqual(event['relations'][0]['kind'], 'revision_of')

    def test_failed_commit_and_printed_command_do_not_create(self):
        parser = known()
        failure, = parser.extract(step('git commit -m fix', '', 'fatal: unable to read index'))
        self.assertEqual(failure['status'], 'failed')
        self.assertEqual(failure['outputs'], [])
        self.assertFalse(failure['relation_authority'])
        self.assertEqual(parser.extract(step('echo "git commit -m fix"', '[main bbbbbbb] fix\n')), [])
        spoof, = parser.extract(step('echo "[main bbbbbbb] fix" && git commit -m fix', '[main bbbbbbb] fix\n'))
        self.assertNotEqual(spoof['status'], 'success')
        self.assertEqual(spoof['outputs'], [])
        self.assertFalse(spoof['relation_authority'])

    def test_ambiguous_two_commit_outputs_are_not_attributed(self):
        parser = known()
        events = parser.extract(step('git commit -m one && git commit -m two', '[main aaaaaaa] one\n[main bbbbbbb] two\n'))
        self.assertTrue(events)
        self.assertTrue(all(event['status'] == 'unknown' and not event['outputs'] for event in events))

    def test_passive_log_and_merge_base_are_not_consumption(self):
        parser = known()
        events = parser.extract(step('git log --oneline', A + ' first\n' + B + ' second\n'))
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event['operation'] == 'observe' and not event['inputs'] and not event['outputs'] for event in events))
        self.assertEqual(parser.extract(step('git merge-base ' + A + ' ' + B, A + '\n')), [])

    def test_successful_cherry_pick_has_input_and_new_descendant(self):
        parser = known()
        event, = parser.extract(step('git cherry-pick ' + A, '[main bbbbbbb] upstream change\n 1 file changed\n'))
        self.assertEqual(event['operation'], 'incorporate')
        self.assertEqual(event['status'], 'success')
        self.assertEqual(event['inputs'], [git_key(REPO, A)])
        self.assertEqual(event['outputs'], [git_key(REPO, 'bbbbbbb')])

    def test_direct_rebase_requires_identified_base_and_success_marker(self):
        parser = known()
        event, = parser.extract(step('git rebase ' + A, '', '\x1b[KSuccessfully rebased and updated refs/heads/main.\n'))
        self.assertEqual(event['operation'], 'incorporate')
        self.assertEqual(event['status'], 'success')
        self.assertEqual(event['inputs'], [git_key(REPO, A)])
        self.assertEqual(event['outputs'], [])
        self.assertTrue(event['relation_authority'])

    def test_rebase_verifier_can_identify_new_descendant(self):
        parser = known()
        events = parser.extract(step('git rebase ' + A + ' && git rev-parse HEAD',
                                     B + '\n', 'Successfully rebased and updated refs/heads/main.\n'))
        rebased, = [event for event in events if event.get('method') == 'rebase']
        self.assertEqual(rebased['status'], 'success')
        self.assertEqual(rebased['outputs'], [git_key(REPO, B)])
        self.assertEqual(rebased['relations'][0]['target'], git_key(REPO, A))

    def test_rebase_failure_noop_and_missing_marker_do_not_incorporate(self):
        parser = known()
        failed, = parser.extract(step('git rebase ' + A, 'CONFLICT (content): Merge conflict\n', 'error: could not apply patch'))
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['inputs'], [])
        noop, = parser.extract(step('git rebase ' + A, 'Current branch main is up to date.\n'))
        self.assertEqual(noop['operation'], 'observe')
        self.assertTrue(noop['no_change'])
        self.assertEqual(noop['inputs'], [])
        quiet, = parser.extract(step('git rebase ' + A, '', exit_code=0))
        self.assertEqual(quiet['status'], 'unknown')

    def test_rebase_resume_unknown_ref_and_ambiguous_context_remain_unknown(self):
        output = 'Successfully rebased and updated refs/heads/main.\n'
        for command in ('git rebase --continue', 'git rebase --abort', 'git rebase unknown-branch',
                        'git rebase --onto ' + B + ' ' + A, 'git rebase -i ' + A,
                        'git rebase ' + A + ' && git log --format=%B',
                        'git -C /first rebase ' + A + ' && git -C /second status'):
            events = known().extract(step(command, output))
            rebase, = [event for event in events if event.get('method') == 'rebase']
            self.assertEqual(rebase['status'], 'unknown', command)
            self.assertEqual(rebase['inputs'], [], command)

    def test_pull_rebase_is_not_promoted_without_verified_input(self):
        parser = known()
        event, = parser.extract(step('git pull --rebase origin main', 'Successfully rebased and updated refs/heads/main.\n'))
        self.assertEqual(event['status'], 'unknown')
        self.assertEqual(event['inputs'], [])

    def test_pull_rebase_uses_incoming_tracking_ref_not_outgoing_push(self):
        output = ('From https://gitlab.com/group/project\n   aaaaaaa..bbbbbbb main -> origin/main\n'
                  '\x1b[KSuccessfully rebased and updated refs/heads/main.\n'
                  'To https://gitlab.com/group/project.git\n   bbbbbbb..ccccccc main -> main\n')
        events = GitExtractor('A').extract(step('git pull --rebase origin main && git push origin main', output))
        event, = [event for event in events if event.get('method') == 'pull-rebase']
        self.assertEqual(event['status'], 'success')
        self.assertEqual(event['entity'], REPO)
        self.assertEqual(event['inputs'], [git_key(REPO, 'bbbbbbb')])
        self.assertEqual(event['outputs'], [])
        self.assertTrue(event['relation_authority'])

    def test_pull_rebase_rejects_unverified_or_conflicting_tracking_updates(self):
        success = 'Successfully rebased and updated refs/heads/main.\n'
        header = 'From https://gitlab.com/group/project\n'
        for changes in ('   aaaaaaa..bbbbbbb other -> origin/other\n',
                        '   aaaaaaa..bbbbbbb main -> origin/main\n   ccccccc..ddddddd other -> origin/other\n',
                        '   aaaaaaa..bbbbbbb main -> main\n',
                        B + ' nearby git log message\n', ''):
            event, = known().extract(step('git pull --rebase origin main', header + changes + success))
            self.assertEqual(event['status'], 'unknown', changes)
            self.assertEqual(event['inputs'], [])
        missing_receipt, = known().extract(step('git pull --rebase origin main', '   aaaaaaa..bbbbbbb main -> origin/main\n' + success))
        self.assertEqual(missing_receipt['status'], 'unknown')
        failed, = known().extract(step('git pull --rebase origin main', header + '   aaaaaaa..bbbbbbb main -> origin/main\n', 'error: could not apply patch'))
        self.assertEqual(failed['status'], 'failed')

    def test_rebase_push_receipt_supplies_same_turn_repository(self):
        output = ('Successfully rebased and updated refs/heads/main.\nTo https://gitlab.com/group/project.git\n'
                  '   aaaaaaa..bbbbbbb main -> main\n')
        events = GitExtractor('A').extract(step('git rebase ' + A + ' && git push origin main', output))
        rebase, = [event for event in events if event.get('method') == 'rebase']
        self.assertEqual(rebase['entity'], REPO)
        self.assertEqual(rebase['status'], 'success')

    def test_merge_message_that_looks_like_hash_is_not_input(self):
        parser = known()
        event, = parser.extract(step('git merge -m ' + A + ' ' + B, "Merge made by the 'ort' strategy.\n"))
        self.assertEqual(event['inputs'], [git_key(REPO, B)])

    def test_merge_failure_and_noop_are_not_successful_handoffs(self):
        parser = known()
        failure, = parser.extract(step('git merge ' + A, 'CONFLICT (content): Merge conflict in x\nAutomatic merge failed; fix conflicts'))
        self.assertEqual(failure['status'], 'failed')
        self.assertEqual(failure['inputs'], [])
        noop, = parser.extract(step('git merge ' + A, 'Already up to date.\n'))
        self.assertEqual(noop['operation'], 'observe')
        self.assertEqual(noop['status'], 'success')
        self.assertEqual(noop['inputs'], [])
        self.assertTrue(noop['no_change'])

    def test_fast_forward_adopts_but_does_not_create_existing_commit(self):
        parser = known()
        event, = parser.extract(step('git merge ' + B, 'Updating aaaaaaa..bbbbbbb\nFast-forward\n x | 1 +\n'))
        self.assertEqual(event['status'], 'success')
        self.assertEqual(event['inputs'], [git_key(REPO, B)])
        self.assertEqual(event['outputs'], [])

    def test_checkout_acquisition_is_distinct_from_incorporation(self):
        parser = known()
        event, = parser.extract(step('git checkout ' + C, '', 'HEAD is now at 1234567 update\n'))
        self.assertEqual(event['operation'], 'acquire')
        self.assertEqual(event['inputs'], [git_key(REPO, C)])
        self.assertEqual(event['outputs'], [])
        self.assertEqual(event['version'], git_key(REPO, C))

    def test_checkout_path_is_not_head_adoption(self):
        parser = known()
        event, = parser.extract(step('git checkout ' + A + ' -- data.csv', '', exit_code=0))
        self.assertEqual(event['inputs'], [])
        self.assertIsNone(event['version'])

    def test_versioned_file_read_and_unknown_branch(self):
        parser = known()
        event, = parser.extract(step('git show ' + A + ':data.csv', 'a,b\n1,2\n'))
        self.assertEqual(event['family'], 'file')
        self.assertEqual(event['status'], 'success')
        self.assertEqual(event['inputs'], [git_key(REPO, A)])
        self.assertTrue(event['version'].endswith('@' + A))
        unknown, = parser.extract(step('git show main:data.csv', 'a,b\n1,2\n'))
        self.assertIsNone(unknown['version'])
        self.assertEqual(unknown['inputs'], [])

    def test_failed_show_and_formatted_listing_are_not_file_reads(self):
        parser = known()
        failed, = parser.extract(step('git show ' + A + ':missing.csv', '', 'fatal: path does not exist'))
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['inputs'], [])
        listing, = parser.extract(step('git show --format=%H:%s ' + A, A + ':title\n'))
        self.assertEqual(listing['operation'], 'observe')
        compound = parser.extract(step('git show ' + A + ':empty.csv; git log --oneline', B + ' unrelated output\n'))
        read, = [event for event in compound if event['family'] == 'file']
        self.assertNotEqual(read['status'], 'success')
        self.assertEqual(read['inputs'], [])

    def test_fetch_only_acquires_objects(self):
        parser = known()
        event, = parser.extract(step('git fetch origin', '', 'From https://gitlab.com/group/project\n aaaaaaa..bbbbbbb main -> origin/main\n'))
        self.assertEqual(event['operation'], 'acquire')
        self.assertFalse(event['incorporation'])
        self.assertEqual(event['outputs'], [])

    def test_test_pass_needs_clean_worktree_for_exact_head(self):
        parser = known()
        parser.extract(step('git rev-parse HEAD', A + '\n'))
        unknown, = parser.extract(step('pytest -q', '3 passed in 0.1s\n'))
        self.assertEqual(unknown['status'], 'success')
        self.assertEqual(unknown['inputs'], [])
        parser.extract(step('git status', 'On branch main\nnothing to commit, working tree clean\n'))
        exact, = parser.extract(step('pytest -q', '3 passed in 0.1s\n'))
        self.assertEqual(exact['inputs'], [git_key(REPO, A)])
        self.assertTrue(exact['working_tree_confirmed_clean'])
        failure, = parser.extract(step('pytest -q', '1 failed, 2 passed in 0.1s\n'))
        self.assertEqual(failure['status'], 'failed')
        self.assertEqual(failure['outputs'], [])

    def test_test_spoofing_or_worktree_write_cannot_use_clean_head(self):
        parser = known()
        parser.extract(step('git rev-parse HEAD', A + '\n'))
        parser.extract(step('git status', 'nothing to commit, working tree clean\n'))
        events = parser.extract(step('printf changed > data.csv && pytest -q', '3 passed in 0.1s\n'))
        event, = [event for event in events if event['family'] == 'test']
        self.assertEqual(event['inputs'], [])
        self.assertNotEqual(event['status'], 'success')
        spoof, = parser.extract(step('echo "3 passed in 0.1s" && pytest -q', '3 passed in 0.1s\n'))
        self.assertNotEqual(spoof['status'], 'success')

    def test_unknown_head_change_and_unmodelled_write_invalidate_context(self):
        parser = known()
        parser.extract(step('git rev-parse HEAD', A + '\n'))
        parser.extract(step('git status', 'nothing to commit, working tree clean\n'))
        parser.extract(step('git checkout ' + B))
        parser.extract(step('git status', 'nothing to commit, working tree clean\n'))
        event, = parser.extract(step('pytest -q', '3 passed in 0.1s\n'))
        self.assertEqual(event['inputs'], [])
        parser.extract(step('git rev-parse HEAD', B + '\n'))
        parser.extract(step('git status', 'nothing to commit, working tree clean\n'))
        parser.extract(step('npm install'))
        event, = parser.extract(step('pytest -q', '3 passed in 0.1s\n'))
        self.assertEqual(event['inputs'], [])

    def test_remote_reconfiguration_invalidates_old_global_identity(self):
        parser = known()
        parser.extract(step('git remote set-url origin https://gitlab.com/other/repo.git'))
        self.assertIsNone(parser.current_repo('session'))

    def test_scope_and_shell_kind_before_state_updates(self):
        parser = GitExtractor('A')
        out = step('cd /work && git remote -v', 'origin https://gitlab.com/group/project.git (fetch)\n')
        out['t'] = '2026-08-17 16:00:00'
        self.assertEqual(parser.extract(out), [])
        self.assertEqual(parser.roots, {})
        out['t'] = '2026-07-06 16:00:00'
        out['k'] = 'type'
        self.assertEqual(parser.extract(out), [])
        self.assertEqual(parser.roots, {})

    def test_local_repositories_never_alias_across_actors_or_sessions(self):
        a = GitExtractor('A').extract(step('git commit -m one', '[main aaaaaaa] one\n'))[0]
        b = GitExtractor('B').extract(step('git commit -m one', '[main aaaaaaa] one\n'))[0]
        self.assertNotEqual(a['version'], b['version'])
        parser = known()
        other = step('git commit -m one', '[main aaaaaaa] one\n')
        other['s'] = 'other-session'
        self.assertNotEqual(parser.extract(other)[0]['entity'], REPO)

    def test_empty_clone_output_does_not_confirm_clone(self):
        parser = GitExtractor('A')
        event, = parser.extract(step('git clone https://gitlab.com/group/project.git', "Cloning into 'project'...\n"))
        self.assertEqual(event['status'], 'unknown')
        self.assertEqual(parser.roots, {})

    def test_printed_remote_output_cannot_create_repository_mapping(self):
        parser = GitExtractor('A')
        parser.extract(step('echo fake && git remote -v', 'origin https://gitlab.com/group/project.git (fetch)\n'))
        self.assertEqual(parser.roots, {})

    def test_unconfirmed_cd_invalidates_context_but_failed_cd_preserves_it(self):
        parser = known()
        parser.extract(step('cd /different'))
        self.assertIsNone(parser.current_repo('session'))
        parser = known()
        parser.extract(step('cd /different && git status', '', 'bash: cd: /different: No such file or directory'))
        self.assertEqual(parser.current_repo('session'), REPO)

    def test_commit_push_batch_uses_authoritative_same_directory_receipt(self):
        parser = GitExtractor('A')
        output = '[main bbbbbbb] update\nTo https://gitlab.com/group/project.git\n   aaaaaaa..bbbbbbb  main -> main\n'
        events = parser.extract(step('cd /work/project && git add data.csv && git commit -m update && git push origin main', output))
        created, = [event for event in events if event['operation'] == 'generate']
        self.assertEqual(created['entity'], REPO)
        self.assertEqual(created['status'], 'success')
        self.assertEqual(created['outputs'], [git_key(REPO, 'bbbbbbb')])

    def test_uncertain_cd_batch_has_turn_local_identity_without_guessed_mapping(self):
        parser = known()
        output = '[main bbbbbbb] update\nTo https://gitlab.com/new/project.git\n   aaaaaaa..bbbbbbb  main -> main\n'
        events = parser.extract(step('cd /different\ngit add data.csv\ngit commit -m update\ngit push origin main', output))
        created, = [event for event in events if event['operation'] == 'generate']
        self.assertEqual(created['entity'], 'gitlab:new/project')
        self.assertEqual(created['status'], 'success')
        self.assertEqual(parser.current_repo('session'), 'gitlab:new/project')
        self.assertIsNone(parser.repo_for_path('/different', 'session'))
        observed, = parser.extract(step('git log --oneline', 'bbbbbbb update\n'))
        self.assertTrue(observed['entity'].startswith('local:'))
        self.assertNotEqual(observed['entity'], REPO)

    def test_multiple_uncertain_directories_never_share_transport_identity(self):
        parser = known()
        output = '[main bbbbbbb] update\nTo https://gitlab.com/new/project.git\n   aaaaaaa..bbbbbbb  main -> main\n'
        events = parser.extract(step('cd /first; git commit -m update; cd /second; git push', output))
        created, = [event for event in events if event['operation'] == 'generate']
        self.assertTrue(created['entity'].startswith('local:'))
        self.assertNotEqual(created['entity'], REPO)
        other = GitExtractor('A').extract(step('git -C /first commit -m update && git -C /second push', output))
        self.assertTrue(next(event for event in other if event['operation'] == 'generate')['entity'].startswith('local:'))

    def test_fetch_receipt_scopes_both_acquisition_and_same_turn_log(self):
        parser = GitExtractor('A')
        output = 'From https://gitlab.com/group/project\n aaaaaaa..bbbbbbb main -> origin/main\nbbbbbbb update\n'
        events = parser.extract(step('cd /work/project\ngit fetch origin\ngit log --oneline', output))
        self.assertTrue(events)
        self.assertTrue(all(event['entity'] == REPO for event in events))
        self.assertIn('acquire', [event['operation'] for event in events])

    def test_later_push_explicitly_aliases_only_verified_current_head(self):
        parser = GitExtractor('A')
        created, = parser.extract(step('git commit -m update', '[main bbbbbbb] update\n'))
        pushed, = parser.extract(step('git push https://gitlab.com/group/project.git main',
                                     'To https://gitlab.com/group/project.git\n   aaaaaaa..bbbbbbb main -> main\n'))
        self.assertEqual(pushed['relations'], [{'kind': 'same_content', 'source': created['version'],
                                               'target': git_key(REPO, 'bbbbbbb')}])
        self.assertTrue(pushed['relation_authority'])
        implicit = GitExtractor('B')
        created, = implicit.extract(step('git commit -m update', '[main bbbbbbb] update\n'))
        pushed, = implicit.extract(step('git push', 'To https://gitlab.com/group/project.git\n   aaaaaaa..bbbbbbb main -> main\n'))
        self.assertEqual(pushed['relations'][0]['source'], created['version'])
        self.assertEqual(pushed['relations'][0]['target'], git_key(REPO, 'bbbbbbb'))

    def test_printable_git_payload_and_conflicting_receipts_do_not_scope_batch(self):
        output = '[main bbbbbbb] update\nTo https://gitlab.com/group/project.git\n   aaaaaaa..bbbbbbb main -> main\n'
        parser = GitExtractor('A')
        events = parser.extract(step('git commit -m update && git log --format=%B && git push', output))
        self.assertTrue(next(event for event in events if event['operation'] == 'generate')['entity'].startswith('local:'))
        other = GitExtractor('B').extract(step('git commit -m update && git push', output + 'To https://gitlab.com/other/project.git\n'))
        self.assertTrue(next(event for event in other if event['operation'] == 'generate')['entity'].startswith('local:'))

    def test_heredoc_and_embedded_script_not_interpreted_as_git(self):
        parser = known()
        self.assertEqual(parser.extract(step("python - <<'PY'\nprint('git commit -m fake')\nPY", '[main aaaaaaa] fake\n')), [])
        self.assertEqual(parser.extract(step("python -c 'print(\"git commit -m fake\")'", '[main aaaaaaa] fake\n')), [])


class FileEvidenceTests(unittest.TestCase):
    def test_checksum_observation_is_acquisition_not_creation(self):
        parser = known()
        digest = 'd' * 64
        event, = parser.extract(step('sha256sum data.csv', digest + '  data.csv\n'))
        self.assertEqual(event['operation'], 'acquire')
        self.assertEqual(event['outputs'], [])
        self.assertEqual(event['content_key'], 'content:sha256:' + digest)
        self.assertTrue(event['relation_authority'])
        self.assertTrue(event['version'].startswith('file:' + REPO + ':data.csv@sha256:'))

    def test_generation_requires_write_before_acknowledged_hash(self):
        parser = known()
        digest = hashlib.sha256(b'hello\n').hexdigest()
        event, = parser.extract(step("printf 'hello\\n' > data.csv && sha256sum data.csv", digest + '  data.csv\n'))
        self.assertEqual(event['operation'], 'generate')
        self.assertIn('content:sha256:' + digest, event['outputs'])
        earlier, = parser.extract(step("sha256sum data.csv && printf 'new\\n' > data.csv", digest + '  data.csv\n'))
        self.assertEqual(earlier['operation'], 'acquire')
        self.assertEqual(earlier['outputs'], [])

    def test_quoted_redirection_and_fake_checksum_are_not_generation(self):
        parser = known()
        digest = 'd' * 64
        self.assertEqual(parser.extract(step("echo '>data.csv' && sha256sum data.csv", digest + '  data.csv\n')), [])
        self.assertEqual(parser.extract(step("printf '" + digest + "  data.csv\\n' && sha256sum data.csv", digest + '  data.csv\n')), [])

    def test_copy_lineage_uses_both_same_step_checksums(self):
        parser = known()
        digest = 'd' * 64
        events = parser.extract(step('cp input.csv result.csv && sha256sum input.csv result.csv', digest + '  input.csv\n' + digest + '  result.csv\n'))
        source, result = sorted(events, key=lambda event: event['entity'])
        self.assertEqual(source['operation'], 'acquire')
        self.assertEqual(result['operation'], 'transform')
        self.assertIn(source['version'], result['inputs'])
        self.assertIn(result['version'], result['outputs'])

    def test_failed_or_unordered_write_is_not_generation(self):
        parser = known()
        digest = 'd' * 64
        self.assertEqual(parser.extract(step('cp missing result.csv && sha256sum result.csv', '', 'cp: cannot stat missing')), [])
        event, = parser.extract(step('cp missing result.csv; sha256sum result.csv', digest + '  result.csv\n', 'cp: cannot stat missing'))
        self.assertEqual(event['operation'], 'acquire')
        self.assertEqual(event['outputs'], [])

    def test_checksum_requires_requested_path_and_correct_algorithm(self):
        parser = known()
        self.assertEqual(parser.extract(step('sha256sum data.csv', 'd' * 64 + '  unrelated.csv\n')), [])
        self.assertEqual(parser.extract(step('md5sum data.csv', 'd' * 64 + '  data.csv\n')), [])
        self.assertEqual(parser.extract(step('sha256sum data.csv data.csv', 'd' * 64 + '  data.csv\n' + 'e' * 64 + '  data.csv\n')), [])

    def test_escaping_repository_path_remains_local(self):
        parser = known()
        event, = parser.extract(step('sha256sum /tmp/data.csv', 'd' * 64 + '  /tmp/data.csv\n'))
        self.assertTrue(event['entity'].startswith('file:local:'))
        self.assertNotIn(REPO, event['entity'])


@unittest.skipUnless(shutil.which('git'), 'git executable unavailable')
class RealGitTests(unittest.TestCase):
    def test_real_commit_checkout_and_versioned_file_output(self):
        with tempfile.TemporaryDirectory(prefix='swarm-provenance-test-') as directory:
            def run(*args):
                return subprocess.run(['git', '-C', directory, *args], capture_output=True, text=True, check=True)
            run('init', '-q')
            run('config', 'user.name', 'Fixture')
            run('config', 'user.email', 'fixture@example.invalid')
            run('remote', 'add', 'origin', 'https://gitlab.com/fixtures/provenance.git')
            parser = GitExtractor('Fixture agent')
            remote = run('remote', '-v')
            parser.extract(step('git -C ' + directory + ' remote -v', remote.stdout, remote.stderr))
            Path(directory, 'data.csv').write_text('a,b\n1,2\n')
            run('add', 'data.csv')
            committed = run('commit', '-m', 'Add data fixture')
            generated, = parser.extract(step('git -C ' + directory + ' commit -m fixture', committed.stdout, committed.stderr))
            self.assertEqual(generated['status'], 'success')
            self.assertEqual(generated['operation'], 'generate')
            sha = run('rev-parse', 'HEAD').stdout.strip()
            shown = run('show', sha + ':data.csv')
            acquired, = parser.extract(step('git -C ' + directory + ' show ' + sha + ':data.csv', shown.stdout, shown.stderr))
            self.assertEqual(acquired['status'], 'success')
            self.assertEqual(acquired['inputs'], [git_key('gitlab:fixtures/provenance', sha)])
            checked = run('checkout', '--detach', sha)
            checkout, = parser.extract(step('git -C ' + directory + ' checkout --detach ' + sha, checked.stdout, checked.stderr))
            self.assertEqual(checkout['status'], 'success')
            self.assertEqual(checkout['operation'], 'acquire')


if __name__ == '__main__':
    unittest.main()
