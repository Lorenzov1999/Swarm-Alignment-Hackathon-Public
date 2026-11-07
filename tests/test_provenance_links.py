"""Regression tests for evidence grades and leakage/identity boundaries."""
import copy
import unittest

from swarmcollab.provenance_common import command_segments, in_scope, json_objects, make_event, git_key, timestamp
from swarmcollab.provenance_links import build_links, directed_features, training_common_content

SHA = 'a1' * 20
KEY = git_key('gitlab:team/repo', SHA)


def event(actor, minute, operation='observe', status='success', inputs=(), outputs=(), entity='repo', family='git', **extra):
    record = {'i': actor + str(minute) + operation, 't': '2026-07-06 16:%02d:00' % minute, 's': 'session'}
    extra.setdefault('relation_authority', True)
    return make_event(actor, record, family, operation, status, entity, inputs=list(inputs), outputs=list(outputs), **extra)


class CommonTests(unittest.TestCase):
    def test_pacific_scope_and_fractional_precision(self):
        self.assertFalse(in_scope('2026-07-06 06:59:59.99999'))
        self.assertTrue(in_scope('2026-07-06 07:00:00.00001'))
        self.assertTrue(in_scope('2026-08-15 06:59:59'))
        self.assertFalse(in_scope('2026-08-15 07:00:00'))
        self.assertEqual(timestamp('2026-07-06 16:00:00.12345').microsecond, 123450)

    def test_commands_are_not_embedded_code(self):
        self.assertEqual(command_segments("echo 'git merge abcdef0'"), [['echo', 'git merge abcdef0']])
        self.assertEqual(command_segments("cat <<'EOF'\ngit merge abcdef0\nEOF"), [])
        self.assertEqual(command_segments('if true; then git merge abcdef0; fi'), [])
        self.assertEqual(command_segments('git status\ngit log'), [['git', 'status'], ['git', 'log']])
        self.assertIn('#sha256=123', command_segments('pip install https://x.org/a.whl#sha256=123')[0][2])

    def test_complete_json_only(self):
        self.assertEqual([x[0] for x in json_objects('log\n{"id": 1}\n{"id": 2}')], [{'id': 1}, {'id': 2}])
        self.assertEqual(list(json_objects('{"outer":\n{"id": 2}\n')), [])


class LinkTests(unittest.TestCase):
    def producer(self, key=KEY):
        return event('A', 0, 'generate', outputs=[key])

    def strong(self, values, **kwargs):
        return [c for c in build_links(values, **kwargs)[0] if c['level'] >= 3]

    def test_passive_listing_is_not_consumption(self):
        links, _ = build_links([self.producer(), event('B', 1, outputs=[KEY])])
        self.assertTrue(any(c['level'] == 2 for c in links))
        self.assertFalse(any(c['level'] >= 3 for c in links))

    def test_acquisition_and_incorporation_are_separate(self):
        links = self.strong([self.producer(), event('B', 1, 'acquire', inputs=[KEY]), event('C', 2, 'incorporate', inputs=[KEY])])
        self.assertEqual([c['level'] for c in links], [3, 4])

    def test_failed_or_unknown_actions_never_confirm(self):
        for status in ('failed', 'unknown', 'attempted'):
            self.assertEqual(self.strong([self.producer(), event('B', 1, 'incorporate', status, inputs=[KEY])]), [])
        self.assertEqual(self.strong([event('A', 0, 'observe', outputs=[KEY]), event('B', 1, 'acquire', inputs=[KEY])]), [])

    def test_future_and_self_use_are_not_handoffs(self):
        self.assertEqual(self.strong([event('B', 0, 'acquire', inputs=[KEY]), event('A', 1, 'generate', outputs=[KEY])]), [])
        self.assertEqual(self.strong([self.producer(), event('A', 1, 'acquire', inputs=[KEY])]), [])

    def test_persistent_version_use_is_not_limited_to_one_week(self):
        target = event('B', 1, 'acquire', inputs=[KEY])
        target['time'] = '2026-07-20T16:01:00+00:00'
        target['day'] = '2026-07-20'
        self.assertEqual(len(self.strong([self.producer(), target])), 1)

    def test_repository_namespace_and_prefix_ambiguity(self):
        other = git_key('gitlab:other/repo', SHA)
        self.assertEqual(self.strong([self.producer(), event('B', 1, 'acquire', inputs=[other])]), [])
        short = git_key('gitlab:team/repo', SHA[:7])
        self.assertEqual(len(self.strong([self.producer(short), event('B', 1, 'acquire', inputs=[KEY])])), 1)
        collision = git_key('gitlab:team/repo', SHA[:7] + 'c' * 33)
        self.assertEqual(self.strong([self.producer(short), event('C', 1, outputs=[collision]), event('B', 2, 'acquire', inputs=[KEY])]), [])

    def test_future_hash_resolution_does_not_rewrite_earlier_edge(self):
        short = git_key('gitlab:team/repo', SHA[:7])
        links = self.strong([self.producer(short), event('B', 1, 'acquire', inputs=[short]), event('C', 2, outputs=[KEY]), event('D', 3, 'acquire', inputs=[KEY])])
        self.assertEqual([c['target_actor'] for c in links], ['D'])

    def test_cross_artifact_content_and_common_source_control(self):
        a = 'file:local:A:report.csv@sha256:' + 'b' * 64
        b = 'file:local:B:input.csv@sha256:' + 'b' * 64
        values = [self.producer(a), event('B', 2, 'transform', inputs=[b], family='file')]
        links = self.strong(values)
        self.assertEqual(links[0]['identity_confidence'], 'same_content')
        values.insert(1, event('C', 1, 'generate', outputs=[a.replace('local:A', 'local:C')]))
        self.assertEqual(self.strong(values), [])
        self.assertEqual(self.strong(values[:1] + values[2:], common_content={'content:sha256:' + 'b' * 64}), [])

    def test_build_lineage_but_not_generic_ancestry_confirms(self):
        job, artifact = 'job:repo:1', 'artifact:repo:1:zip'
        relations = [{'kind': 'derived_from', 'source': job, 'target': KEY}, {'kind': 'derived_from', 'source': artifact, 'target': job}]
        metadata = event('C', 1, outputs=[artifact], family='ci', relations=relations)
        consume = event('B', 2, 'acquire', inputs=[artifact], family='ci')
        links = self.strong([self.producer(), metadata, consume])
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['identity_confidence'], 'recorded_build_lineage')
        self.assertEqual(links[0]['source_actor'], 'A')
        metadata['family'] = 'git'
        self.assertEqual(self.strong([self.producer(), metadata, consume]), [])

    def test_aliases_cannot_bypass_common_content_filter(self):
        digest = 'content:sha256:' + 'b' * 64
        a, b = 'file:A:x@sha256:' + 'b' * 64, 'file:B:x@sha256:' + 'b' * 64
        relations = [{'kind': 'same_content', 'source': a, 'target': digest}, {'kind': 'same_content', 'source': b, 'target': digest}]
        values = [self.producer(a), event('C', 1, relations=relations), event('B', 2, 'acquire', inputs=[b])]
        self.assertTrue(self.strong(values))
        self.assertEqual(self.strong(values, common_content={digest}), [])

    def test_observed_version_field_resolves_hash_without_claiming_origin(self):
        short = git_key('gitlab:team/repo', SHA[:7])
        values = [self.producer(short), event('A', 1, version=KEY), event('B', 2, 'acquire', inputs=[short])]
        self.assertEqual(len(self.strong(values)), 1)

    def test_relations_from_future_do_not_leak(self):
        artifact = 'artifact:repo:1:zip'
        metadata = event('C', 3, family='ci', relations=[{'kind': 'derived_from', 'source': artifact, 'target': KEY}])
        self.assertEqual(self.strong([self.producer(), event('B', 2, 'acquire', inputs=[artifact]), metadata]), [])

    def test_untrusted_metadata_cannot_promote_build_lineage(self):
        artifact = 'artifact:repo:1'
        metadata = event('C', 1, family='ci', relation_authority=False,
                         relations=[{'kind': 'derived_from', 'source': artifact, 'target': KEY}])
        self.assertEqual(self.strong([self.producer(), metadata, event('B', 2, 'acquire', inputs=[artifact])]), [])

    def test_untrusted_hash_cannot_resolve_a_short_identifier(self):
        short = git_key('gitlab:team/repo', SHA[:7])
        values = [self.producer(short), event('C', 1, version=KEY, relation_authority=False),
                  event('B', 2, 'acquire', inputs=[short])]
        self.assertEqual(self.strong(values), [])

    def test_git_resolved_abbreviation_requires_operation_and_global_namespace(self):
        short = git_key('gitlab:team/repo', SHA[:7])
        source = event('A', 0, 'generate', outputs=[short], method='commit')
        target = event('B', 1, 'incorporate', inputs=[short], method='pull')
        chain = self.strong([source, target])[0]
        self.assertEqual(chain['identity_confidence'], 'git_scoped_abbreviation')
        self.assertEqual(chain['level'], 4)
        for method in ('rebase', 'pull-rebase'):
            target['method'] = method
            self.assertEqual(self.strong([source, target])[0]['level'], 4)
        target['operation'] = 'observe'
        self.assertEqual(self.strong([source, target]), [])
        target['operation'] = 'incorporate'
        source['outputs'] = [short.replace('gitlab:team/repo', 'local:A:path')]
        target['inputs'] = source['outputs']
        self.assertEqual(self.strong([source, target]), [])

    def test_verbal_review_channel_is_separate(self):
        source = event('A', 0, 'review', outputs=['note:1'], family='review', verbal_channel=True)
        target = event('B', 1, 'incorporate', inputs=['note:1'], family='workflow', verbal_channel=True)
        features = directed_features(self.strong([source, target]))[0]
        self.assertEqual(features['prov_verbal_level_4'], 1)
        self.assertNotIn('prov_level_4', features)

    def test_validation_popularity_does_not_affect_training_filter(self):
        key = 'file:repo:x@sha256:' + 'a' * 64
        values = [event('A', 0, outputs=[key])]
        before = training_common_content(values, '2026-07-31')
        for actor in ('B', 'C', 'D', 'E'):
            extra = event(actor, 1, outputs=[key])
            extra['day'] = '2026-08-04'
            values.append(extra)
        self.assertEqual(before, training_common_content(values, '2026-07-31'))


if __name__ == '__main__':
    unittest.main()
