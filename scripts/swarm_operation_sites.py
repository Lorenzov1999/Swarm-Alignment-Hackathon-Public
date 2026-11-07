"""Standalone static operation-site coverage tooling; no cooperation labels.

Only complete supported syntax is parsed. Node/Acorn is an optional standalone
existing-runtime stage; importing core swarmcollab does not depend on it.
Source snippets are never executed or recursively decoded as programs.
"""
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess

from swarmcollab.provenance_common import command_segments

HERE=Path(__file__).resolve().parent
METHODS={'GET','HEAD','POST','PUT','PATCH','DELETE','OPTIONS','CONNECT','TRACE'}
REQUEST_METHODS={'get','head','post','put','patch','delete','options'}


def _literal(node):
    return node.value if isinstance(node,ast.Constant) and isinstance(node.value,str) else None


def _method(value,origin):
    return dict(method=(value.upper() if value.upper() in METHODS else 'OTHER_LITERAL') if value is not None else None,
                method_known=value is not None,method_source=origin)


def python_sites(text):
    """Complete Python AST only, with import-supported requests name bindings."""
    try:
        tree=ast.parse(text)
    except (SyntaxError,ValueError,RecursionError):
        return dict(parse_status='unsupported_or_invalid_python',sites=[])
    module_aliases=set();function_aliases={};shadowed=set();wildcard_import=False
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for alias in node.names:
                if alias.name=='requests':module_aliases.add(alias.asname or 'requests')
                else:shadowed.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node,ast.ImportFrom) and node.module=='requests' and not node.level:
            for alias in node.names:
                if alias.name in REQUEST_METHODS|{'request'}:
                    bound=alias.asname or alias.name
                    if bound in function_aliases and function_aliases[bound]!=alias.name:shadowed.add(bound)
                    function_aliases[bound]=alias.name
                elif alias.name=='*':wildcard_import=True
                else:shadowed.add(alias.asname or alias.name)
        elif isinstance(node,ast.ImportFrom):
            for alias in node.names:
                if alias.name=='*':wildcard_import=True
                else:shadowed.add(alias.asname or alias.name)
        elif isinstance(node,ast.Name) and isinstance(node.ctx,(ast.Store,ast.Del)):
            shadowed.add(node.id)
        elif isinstance(node,ast.arg):shadowed.add(node.arg)
        elif isinstance(node,ast.ExceptHandler) and node.name:shadowed.add(node.name)
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            shadowed.add(node.name)
        targets=(node.targets if isinstance(node,(ast.Assign,ast.Delete)) else [node.target]
                 if isinstance(node,(ast.AnnAssign,ast.AugAssign)) else [])
        for target in targets:
            if isinstance(target,ast.Attribute):
                while isinstance(target,ast.Attribute):target=target.value
                if isinstance(target,ast.Name):shadowed.add(target.id)
    shadowed.update(module_aliases & set(function_aliases))
    module_aliases-=shadowed
    function_aliases={key:value for key,value in function_aliases.items() if key not in shadowed}
    if wildcard_import:module_aliases.clear();function_aliases.clear()
    sites=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        name=None
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id in module_aliases and node.func.attr in REQUEST_METHODS|{'request'}):
            name=node.func.attr
        elif isinstance(node.func,ast.Name):name=function_aliases.get(node.func.id)
        if name is None:continue
        keywords={item.arg:item.value for item in node.keywords if item.arg is not None}
        if name=='request':
            method_node=node.args[0] if node.args else keywords.get('method')
            target=node.args[1] if len(node.args)>1 else keywords.get('url')
            method=_method(_literal(method_node),'requests.request_argument')
        else:
            target=node.args[0] if node.args else keywords.get('url')
            method=_method(name,'requests_method_name')
        if target is None or isinstance(target,ast.Starred):continue
        sites.append(dict(api_shape='python.requests.'+name,start_line=node.lineno,start_column=node.col_offset,
                          end_line=node.end_lineno,end_column=node.end_col_offset,
                          target_support='string_literal' if _literal(target) is not None else 'dynamic_or_nonstring',**method))
    sites.sort(key=lambda row:(row['start_line'],row['start_column']))
    return dict(parse_status='complete_python_parse',sites=sites,
                import_supported_module_aliases=sorted(module_aliases),
                import_supported_function_alias_count=len(function_aliases))


def shell_sites(text):
    """Simple curl/wget argv expressions, never interpreter/eval string bodies."""
    quote=None;escaped=False
    for character in text:
        if escaped:escaped=False;continue
        if character=='\\' and quote!="'":escaped=True;continue
        if quote:
            if character==quote:quote=None
        elif character in "\"'":quote=character
        elif character in '<>':
            return dict(parse_status='unsupported_shell_redirection',sites=[])
    segments=command_segments(text)
    explicit_shell=bool(re.match(r'^#![^\n]*(?:/|\s)(?:ba|da|z|k)?sh(?:\s|$)',text))
    if not segments or (not explicit_shell and Path(segments[0][0]).name not in {'curl','wget'}):
        return dict(parse_status='unsupported_or_not_direct_shell',sites=[])
    curl_valued={'-H','--header','-d','--data','--data-raw','--data-binary','--data-urlencode',
        '-F','--form','--form-string','-X','--request','-o','--output','-T','--upload-file',
        '-u','--user','-A','--user-agent','-e','--referer','-b','--cookie','-c','--cookie-jar',
        '--proxy','-x','--connect-timeout','--max-time','--retry','--request-target','--url','-w','--write-out'}
    wget_valued={'--method','--post-data','--post-file','--body-data','--body-file','--header',
                '-O','--output-document','-o','--output-file','-U','--user-agent','-T','--timeout'}
    curl_switches={'-s','-S','-L','-k','-f','-i','-v','-I','-G','-O','-N','-g','-4','-6',
        '--silent','--show-error','--location','--insecure','--fail','--include','--verbose',
        '--head','--get','--remote-name','--no-buffer','--globoff','--ipv4','--ipv6','--help','--version','-V'}
    wget_switches={'-q','-S','-v','-c','--quiet','--server-response','--verbose','--continue',
                   '--no-check-certificate','--content-disposition','--help','--version','-h'}
    sites=[]
    for index,argv in enumerate(segments):
        head,args=Path(argv[0]).name,argv[1:]
        if head not in {'curl','wget'}:continue
        valued=curl_valued if head=='curl' else wget_valued
        switches=curl_switches if head=='curl' else wget_switches
        targets,options,unknown=[],[],[]
        position=0
        while position<len(args):
            value=args[position];position+=1
            if value=='--':targets.extend(args[position:]);break
            if value in valued:
                if position>=len(args):unknown.append(value);continue
                options.append((value,args[position]));position+=1
            elif value.startswith('--') and '=' in value and value.split('=',1)[0] in valued:
                options.append(tuple(value.split('=',1)))
            elif value.startswith('-') and not value.startswith('--') and len(value)>2 and value[:2] in valued:
                options.append((value[:2],value[2:]))
            elif value in switches:options.append((value,None))
            elif value.startswith('-') and not value.startswith('--') and len(value)>2 and all('-'+c in switches for c in value[1:]):
                options.extend(('-'+c,None) for c in value[1:])
            elif value.startswith('-'):unknown.append(value)
            else:targets.append(value)
        targets += [value for flag,value in options if flag=='--url']
        flags={flag for flag,_ in options}
        if not targets or unknown or flags&{'--help','--version','-V','-h'}:continue
        explicit=[value for flag,value in options if flag in ({'-X','--request'} if head=='curl' else {'--method'})]
        if len(explicit)==1 and not any(char in explicit[0] for char in '$`'):
            method=_method(explicit[0],'explicit_cli_method')
        elif explicit:method=_method(None,'ambiguous_cli_method')
        elif head=='curl':
            post=bool(flags&{'-d','--data','--data-raw','--data-binary','--data-urlencode','-F','--form','--form-string'})
            upload=bool(flags&{'-T','--upload-file'})
            method=_method('HEAD' if flags&{'-I','--head'} else 'GET' if flags&{'-G','--get'}
                           else 'PUT' if upload else 'POST' if post else 'GET','standard_cli_default')
        else:
            post=bool(flags&{'--post-data','--post-file'})
            method=_method('POST' if post else 'GET','standard_cli_default')
        # One CLI invocation site, even if it contains multiple URL operands.
        sites.append(dict(api_shape='shell.'+head,segment_index=index,target_operand_count=len(targets),
            target_support='dynamic_or_nonstring' if any('$' in value or '`' in value for value in targets) else 'string_literal',**method))
    return dict(parse_status='simple_direct_shell_parse',sites=sites)


class JavascriptParser:
    """Persistent parse-only bridge to the already installed Node runtime."""
    def __init__(self,node='node'):
        self.process=subprocess.Popen([node,'--expose-internals',str(HERE/'javascript_operation_sites.cjs')],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def parse(self,text,record_id='synthetic'):
        self.process.stdin.write(json.dumps(dict(record_id=record_id,text=text))+'\n')
        self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Installed Node/Acorn parser unavailable; no regex fallback is allowed')
        value=json.loads(line)
        if value['parse_status']=='input_or_parser_error':raise ValueError('JavaScript parser failed closed')
        return value
    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close();self.process.stderr.close()
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()


def extract_sites(text,javascript_parser=None,record_id='synthetic'):
    """Choose one syntax family without using any cooperation-marker content.

    Direct shell first, then import-supported Python requests, then complete
    JavaScript. Literal-string embedded programs never enter another parser.
    Zero supported sites and syntax parse failures remain explicitly distinct.
    """
    if not isinstance(text,str):raise ValueError('Payload text must be a string')
    shell=shell_sites(text)
    python=python_sites(text)
    javascript=javascript_parser.parse(text,record_id) if javascript_parser else dict(parse_status='javascript_parser_unavailable',sites=[])
    choices=[('shell',shell),('python',python),('javascript',javascript)]
    chosen=next(((language,result) for language,result in choices if result['sites']),None)
    if chosen:
        language,result=chosen
        sites=[dict(site_index=index,syntax_family=language,**site) for index,site in enumerate(result['sites'])]
    else:language=None;sites=[]
    return dict(record_id=record_id,payload_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                selected_syntax_family=language,operation_expression_sites=sites,
                operation_bearing_payload=bool(sites),method_known_sites=sum(site['method_known'] for site in sites),
                parser_status={name:result['parse_status'] for name,result in choices},
                javascript_runtime={key:javascript.get(key) for key in ('node_version','acorn_version')},
                interpretation='Static expression sites only. No source evaluation, execution count, actor, cooperation label, or temporal order inferred.')
