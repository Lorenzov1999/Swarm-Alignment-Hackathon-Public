"""Synthetic-only operation coverage checks; no source census or markers."""
import unittest
import importlib.util
from pathlib import Path
import gzip
import hashlib
import json
import tempfile

_spec=importlib.util.spec_from_file_location('swarm_operation_sites',Path(__file__).resolve().parents[1]/'scripts/swarm_operation_sites.py')
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
JavascriptParser,extract_sites,python_sites,shell_sites=(_module.JavascriptParser,_module.extract_sites,_module.python_sites,_module.shell_sites)


class JavascriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.parser=JavascriptParser()
    @classmethod
    def tearDownClass(cls):cls.parser.close()

    def parse(self,text):return self.parser.parse(text)

    def test_complete_fetch_sites_and_literal_methods(self):
        report=self.parse("fetch('[SERVICE 1 URL 2]'); window.fetch(url,{method:'POST',body: data}); globalThis.fetch(url, options);")
        self.assertEqual([site['method'] for site in report['sites']],['GET','POST',None])
        self.assertEqual([site['method_known'] for site in report['sites']],[True,True,False])
        self.assertEqual([site['target_support'] for site in report['sites']],['string_literal','dynamic_or_nonstring','dynamic_or_nonstring'])

    def test_comments_strings_templates_and_regex_are_not_code_sites(self):
        report=self.parse("// fetch('not a call')\n const s=\"fetch('literal')\"; const t=`fetch('template')`; const r=/fetch\\('regex'\\)/; fetch(real);")
        self.assertEqual(len(report['sites']),1)

    def test_function_and_loop_bodies_are_counted_once_without_execution(self):
        report=self.parse("throw new Error('must never run'); function neverCalled(){ for(let i=0;i<10000;i++) fetch(url); }")
        self.assertEqual(len(report['sites']),1)
        self.assertTrue(report['sites'][0]['enclosing_function_scope'].startswith('function:'))

    def test_dynamic_duplicate_or_spread_method_stays_unknown(self):
        report=self.parse("fetch(url,{method:m}); fetch(url,{...options}); fetch(url,{method:'GET',method:'POST'});")
        self.assertTrue(all(not site['method_known'] for site in report['sites']))

    def test_fetch_request_input_can_inherit_method_unless_explicitly_overridden(self):
        report=self.parse("fetch(requestObj);fetch(input,{headers:h});fetch(input,null);fetch(input,{method:'PUT'});fetch('literal',{headers:h});")
        self.assertEqual([site['method'] for site in report['sites']],[None,None,None,'PUT','GET'])

    def test_prototype_method_is_not_assumed_to_be_default_get(self):
        report=self.parse("fetch('url',{__proto__:{method:'POST'}});fetch('url',{__proto__:options,method:'PUT'});")
        self.assertEqual([site['method'] for site in report['sites']],[None,'PUT'])

    def test_shadowed_standard_names_are_excluded_conservatively(self):
        for text in ["const fetch=x=>x; fetch(url);", "function f(fetch){fetch(url);}",
                     "fetch=custom;fetch(url);", "const window={fetch:custom};window.fetch(url);"]:
            with self.subTest(text=text):self.assertEqual(self.parse(text)['sites'],[])

    def test_xhr_unique_same_function_configuration_and_send(self):
        report=self.parse("var x=new XMLHttpRequest();x.open('PUT',url);x.send(body);")
        self.assertEqual(len(report['sites']),1)
        self.assertEqual(report['sites'][0]['api_shape'],'javascript.XMLHttpRequest.send')
        self.assertEqual(report['sites'][0]['method'],'PUT')
        self.assertIn('unverified',report['sites'][0]['pairing_interpretation'])

    def test_xhr_ambiguous_reassignment_cross_function_and_missing_send_excluded(self):
        samples=["var x=new XMLHttpRequest();x.open('GET',url);x.open('POST',url);x.send();",
                 "var x=new XMLHttpRequest();x=custom;x.open('GET',url);x.send();",
                 "var x=new XMLHttpRequest();function f(){x.open('GET',url);x.send();}",
                 "var x=new XMLHttpRequest();x.open('GET',url);"]
        for text in samples:
            with self.subTest(text=text):self.assertEqual(self.parse(text)['sites'],[])

    def test_mutated_standard_object_members_are_not_supported_bindings(self):
        samples=["globalThis.fetch=custom;globalThis.fetch(url);",
                 "window.fetch=custom;window.fetch(url);", "self.fetch=custom;self.fetch(url);",
                 "var x=new XMLHttpRequest();x.open=custom;x.open('GET',url);x.send();",
                 "XMLHttpRequest.prototype.open=custom;var x=new XMLHttpRequest();x.open('GET',url);x.send();"]
        for text in samples:
            with self.subTest(text=text):self.assertEqual(self.parse(text)['sites'],[])

    def test_cross_global_aliases_deletes_with_and_named_classes_fail_closed(self):
        samples=["globalThis.fetch=custom;fetch('url');","fetch=custom;globalThis.fetch('url');",
                 "delete globalThis.fetch;globalThis.fetch('url');","with(custom){fetch('url');}",
                 "const X=class fetch {f(){fetch('url');}};",
                 "globalThis.XMLHttpRequest=custom;var x=new XMLHttpRequest();x.open('GET','url');x.send();"]
        for text in samples:
            with self.subTest(text=text):self.assertEqual(self.parse(text)['sites'],[])

    def test_xhr_catch_shadow_and_destructuring_reassignment_fail_closed(self):
        for text in ["const x=new XMLHttpRequest();try{}catch(x){x.open('POST','url');x.send();}",
                     "let x=new XMLHttpRequest();({x}=other);x.open('POST','url');x.send();"]:
            with self.subTest(text=text):self.assertEqual(self.parse(text)['sites'],[])

    def test_invalid_or_mixed_language_has_no_regex_recovery(self):
        report=self.parse("fetch(url);\nimport requests\nrequests.get(url)")
        self.assertEqual(report['parse_status'],'unsupported_or_invalid_javascript')
        self.assertEqual(report['sites'],[])

    def test_quoted_program_is_not_promoted_to_nested_operations(self):
        report=extract_sites("eval(\"fetch('https://example.invalid')\")",self.parser)
        self.assertFalse(report['operation_bearing_payload'])
        self.assertEqual(report['operation_expression_sites'],[])


class PythonAndShellTests(unittest.TestCase):
    def test_python_import_supported_requests_and_unknown_wrapper(self):
        report=python_sites("import requests as r\nfrom requests import post as send\nr.get(url)\nsend('literal',data=data)\nr.request(method,url)\nwrapper(url)")
        self.assertEqual([site['method'] for site in report['sites']],['GET','POST',None])
        self.assertEqual(len(report['sites']),3)

    def test_python_shadowed_alias_and_embedded_source_do_not_count(self):
        for text in ["import requests\nrequests=custom\nrequests.get(url)",
                     "import requests\ndef f(requests):\n return requests.get(url)",
                     "source=\"import requests; requests.get(url)\""]:
            with self.subTest(text=text):self.assertEqual(python_sites(text)['sites'],[])

    def test_other_import_bindings_and_wildcards_shadow_requests_aliases(self):
        for text in ["import requests as r\nimport other as r\nr.get(url)",
                     "from requests import get as g\nfrom other import get as g\ng(url)",
                     "import requests\nfrom other import *\nrequests.get(url)",
                     "import requests as r\nfrom requests import post as r\nr.get(url)"]:
            with self.subTest(text=text):self.assertEqual(python_sites(text)['sites'],[])

    def test_python_member_replacement_exception_binding_and_conflicting_aliases(self):
        samples=["import requests\nrequests.get=custom\nrequests.get(url)",
                 "import requests\ntry: pass\nexcept Exception as requests:\n requests.get(url)",
                 "from requests import get as f\nf(url)\nfrom requests import post as f\nf(url)"]
        for text in samples:
            with self.subTest(text=text):self.assertEqual(python_sites(text)['sites'],[])

    def test_python_function_loop_is_syntax_not_iteration_or_execution(self):
        report=python_sites("import requests\ndef f():\n for x in range(100000):\n  requests.get(url)\nraise RuntimeError('never execute')")
        self.assertEqual(len(report['sites']),1)

    def test_curl_direct_defaults_explicit_and_compact_flags(self):
        report=shell_sites("curl -fsSL -dBODY https://example.invalid; curl -XPUT --upload-file data.bin '$URL'; curl -sI https://example.invalid")
        self.assertEqual([site['method'] for site in report['sites']],['POST','PUT','HEAD'])
        self.assertEqual(report['sites'][1]['target_support'],'dynamic_or_nonstring')

    def test_curl_option_values_cannot_be_mistaken_for_method_switches(self):
        report=shell_sites("curl -d '-X' https://example.invalid")
        self.assertEqual(report['sites'][0]['method'],'POST')

    def test_shell_unknown_method_flags_remain_unknown(self):
        report=shell_sites("curl --future-option https://example.invalid; wget --method=POST --body-data=x https://example.invalid")
        self.assertEqual(len(report['sites']),1)
        self.assertEqual(report['sites'][0]['method'],'POST')

    def test_shell_option_values_and_redirections_do_not_invent_url_operands(self):
        samples=["curl -w 'status only'","curl > output.log","curl </dev/null", "curl -o output.bin 2>errors.log"]
        for text in samples:
            with self.subTest(text=text):self.assertEqual(shell_sites(text)['sites'],[])
        self.assertEqual(shell_sites("curl -d '--help' https://example.invalid")['sites'][0]['method'],'POST')

    def test_shell_embedded_commands_control_flow_and_help_are_not_operations(self):
        samples=["python -c \"print('curl https://example.invalid')\"", "bash -c 'curl https://example.invalid'",
                 'if true; then curl https://example.invalid; fi','curl --help','wget --version']
        for text in samples:
            with self.subTest(text=text):self.assertEqual(shell_sites(text)['sites'],[])


class CensusTests(unittest.TestCase):
    def test_census_locks_before_reading_and_reports_payload_duplicate_denominators(self):
        spec=importlib.util.spec_from_file_location('swarm_operation_census',Path(__file__).resolve().parents[1]/'scripts/swarm_operation_census.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'synthetic.jsonl.gz';output=root/'census'
            payload='import requests\nrequests.get("https://example.invalid")'
            rows=[dict(id='a',kind='payload',text=payload),dict(id='b',kind='payload',text=payload),
                  dict(id='c',kind='payload',text='"no operation"'),dict(id='r',kind='recovered_text')]
            with gzip.open(source,'wt') as stream:
                for row in rows:stream.write(json.dumps(row)+'\n')
            expected=hashlib.sha256(source.read_bytes()).hexdigest()
            result=module.run_census(source,output,expected,javascript=False)
            self.assertEqual(result['primary']['payload_records'],3)
            self.assertEqual(result['primary']['operation_bearing_payload_records'],2)
            self.assertEqual(result['primary']['static_operation_expression_sites'],2)
            self.assertEqual(result['exact_whole_payload_duplicate_sensitivity']['payload_records'],2)
            self.assertEqual(result['exact_whole_payload_duplicate_sensitivity']['operation_bearing_payload_records'],1)
            self.assertIsNone(result['observed_cooperation_rate'])
            lock=json.loads((output/'pre-census-lock.json').read_text())
            self.assertLess(lock['locked_before_census_utc'],result['completed_utc'])
            self.assertTrue(lock['no_marker_join'])
            manifest=json.loads((output/'manifest.json').read_text())
            for name,digest in manifest['outputs_sha256'].items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(),digest)
            with self.assertRaises(FileExistsError):module.run_census(source,output,expected,javascript=False)


if __name__=='__main__':unittest.main()
