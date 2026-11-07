// Standalone installed-Node tooling. Parse snippets with bundled Acorn only;
// never evaluate, import, compile for execution, or follow any source URL.
"use strict";
const acorn = require("internal/deps/acorn/acorn/dist/acorn");
const readline = require("readline");
const METHODS = new Set(["GET","HEAD","POST","PUT","PATCH","DELETE","OPTIONS","CONNECT","TRACE"]);

function nodes(root) {
  const out = [], stack = [[root, "program"]];
  while (stack.length) {
    const [node, inherited] = stack.pop();
    if (!node || typeof node.type !== "string") continue;
    const scope = /^(?:FunctionDeclaration|FunctionExpression|ArrowFunctionExpression)$/.test(node.type)
      ? "function:" + node.start : inherited;
    out.push([node, scope]);
    for (const value of Object.values(node)) {
      if (Array.isArray(value)) for (const child of value) stack.push([child, scope]);
      else if (value && typeof value === "object") stack.push([value, scope]);
    }
  }
  return out;
}

function bindingNames(pattern, result) {
  if (!pattern) return;
  if (pattern.type === "Identifier") result.add(pattern.name);
  else if (pattern.type === "RestElement") bindingNames(pattern.argument,result);
  else if (pattern.type === "AssignmentPattern") bindingNames(pattern.left,result);
  else if (pattern.type === "ArrayPattern") for (const item of pattern.elements) bindingNames(item,result);
  else if (pattern.type === "ObjectPattern") for (const item of pattern.properties)
    bindingNames(item.type === "RestElement" ? item.argument : item.value,result);
}

function methodValue(value, origin) {
  if (!value || value.type !== "Literal" || typeof value.value !== "string")
    return {method:null, method_known:false, method_source:origin};
  const literal = value.value.toUpperCase();
  return {method:METHODS.has(literal) ? literal : "OTHER_LITERAL", method_known:true, method_source:origin};
}

function fetchMethod(args) {
  const defaultMethod = () => args[0] && args[0].type === "Literal" && typeof args[0].value === "string"
    ? {method:"GET",method_known:true,method_source:"standard_default_for_literal_url"}
    : {method:null,method_known:false,method_source:"dynamic_input_may_inherit_request_method"};
  if (args.length === 1 || (args[1] && args[1].type === "Literal" && args[1].value === null))
    return defaultMethod();
  const options = args[1];
  if (!options || options.type !== "ObjectExpression") return methodValue(null,"dynamic_options");
  if (options.properties.some(p => p.type !== "Property" || p.computed || p.kind !== "init"))
    return methodValue(null,"computed_or_spread_options");
  const values = options.properties.filter(p => (p.key.name || p.key.value) === "method");
  if (values.length > 1) return methodValue(null,"duplicate_method_property");
  if (!values.length && options.properties.some(p => (p.key.name || p.key.value) === "__proto__"))
    return methodValue(null,"inherited_options_method");
  return values.length ? methodValue(values[0].value,"literal_options")
    : defaultMethod();
}

function targetSupport(node) {
  return node && node.type === "Literal" && typeof node.value === "string"
    ? "string_literal" : "dynamic_or_nonstring";
}

function parseSource(text) {
  if (typeof text !== "string") throw new TypeError("text must be a string");
  let tree, sourceType;
  for (const candidate of ["script","module"]) {
    try {
      tree = acorn.parse(text,{ecmaVersion:"latest",sourceType:candidate,allowAwaitOutsideFunction:true});
      sourceType = candidate; break;
    } catch (_) { /* Fail closed; no regex recovery of embedded programs. */ }
  }
  const report = {parse_status:tree ? "complete_javascript_parse" : "unsupported_or_invalid_javascript",
    source_type:sourceType || null,node_version:process.version,acorn_version:acorn.version,
    sites:[], exclusions:{}};
  if (!tree) return report;
  const entries = nodes(tree), bound = new Set(), constructors = new Map(), assignments = new Map(), mutatedReceivers = new Set();
  if (entries.some(([node]) => node.type === "WithStatement")) {
    report.parse_status="unsupported_javascript_with_scope"; return report;
  }
  for (const [node,scope] of entries) {
    if (/Function/.test(node.type)) {
      bindingNames(node.id,bound);
      for (const param of node.params || []) bindingNames(param,bound);
    }
    if (node.type === "VariableDeclarator") bindingNames(node.id,bound);
    if (node.type === "ClassDeclaration" || node.type === "ClassExpression") bindingNames(node.id,bound);
    if (node.type === "CatchClause") {
      bindingNames(node.param,bound);
      const names=new Set();bindingNames(node.param,names);
      for (const name of names) mutatedReceivers.add(scope+":"+name);
    }
    if (node.type === "ImportDeclaration") for (const spec of node.specifiers) bindingNames(spec.local,bound);
    if (node.type === "AssignmentExpression") bindingNames(node.left,bound);
    if (node.type === "UpdateExpression") bindingNames(node.argument,bound);
    if (node.type === "AssignmentExpression" || node.type === "UpdateExpression" || (node.type === "UnaryExpression" && node.operator === "delete")) {
      let target = node.left || node.argument;
      if (target && target.type === "MemberExpression") {
        const properties=[];
        while (target.type === "MemberExpression") {
          properties.unshift(target.computed ? (target.property.type === "Literal" ? target.property.value : null) : target.property.name);
          target=target.object;
        }
        if (target.type === "Identifier") {
          mutatedReceivers.add(scope+":"+target.name);
          if (["globalThis","window","self","XMLHttpRequest"].includes(target.name)) bound.add(target.name);
          if (["globalThis","window","self"].includes(target.name)) {
            if (properties[0] === "fetch" || properties[0] === null) bound.add("fetch");
            if (properties[0] === "XMLHttpRequest" || properties[0] === null) bound.add("XMLHttpRequest");
          }
        }
      }
      if (target && (target.type === "ObjectPattern" || target.type === "ArrayPattern")) {
        const names=new Set();bindingNames(target,names);
        for (const name of names) mutatedReceivers.add(scope+":"+name);
      }
    }
    let name, init;
    if (node.type === "VariableDeclarator" && node.id.type === "Identifier") {name=node.id.name;init=node.init;}
    else if (node.type === "AssignmentExpression" && node.left.type === "Identifier") {name=node.left.name;init=node.right;}
    else if (node.type === "UpdateExpression" && node.argument.type === "Identifier") name=node.argument.name;
    if (name) {
      const key = scope + ":" + name;
      assignments.set(key,(assignments.get(key)||0)+1);
      if (init && init.type === "NewExpression" && init.callee.type === "Identifier" && init.callee.name === "XMLHttpRequest")
        constructors.set(key,node.start);
    }
  }
  const opens = new Map(), sends = [];
  for (const [node,scope] of entries) {
    if (node.type !== "CallExpression" || node.optional) continue;
    const callee = node.callee;
    let standardFetch = false;
    if (callee.type === "Identifier" && callee.name === "fetch" && !bound.has("fetch")) standardFetch=true;
    if (callee.type === "MemberExpression" && !callee.computed && callee.object.type === "Identifier"
        && ["window","globalThis","self"].includes(callee.object.name) && !bound.has(callee.object.name)
        && !bound.has("fetch") && callee.property.name === "fetch") standardFetch=true;
    if (standardFetch && node.arguments.length >= 1 && node.arguments[0].type !== "SpreadElement") {
      report.sites.push({api_shape:"javascript.fetch",start:node.start,end:node.end,
        enclosing_function_scope:scope,target_support:targetSupport(node.arguments[0]),...fetchMethod(node.arguments)});
    }
    if (callee.type === "MemberExpression" && !callee.computed && callee.object.type === "Identifier") {
      const key = scope + ":" + callee.object.name;
      if (bound.has("XMLHttpRequest") || mutatedReceivers.has(key) || !constructors.has(key) || assignments.get(key) !== 1) continue;
      if (callee.property.name === "open" && node.arguments.length >= 2) {
        const existing = opens.get(key)||[];existing.push(node);opens.set(key,existing);
      }
      if (callee.property.name === "send") sends.push([node,key,scope]);
    }
  }
  for (const [send,key,scope] of sends) {
    const configurations = opens.get(key)||[];
    if (configurations.length !== 1 || constructors.get(key) > configurations[0].start || configurations[0].start > send.start) {
      report.exclusions.ambiguous_xhr_configuration=(report.exclusions.ambiguous_xhr_configuration||0)+1;
      continue;
    }
    const open = configurations[0];
    report.sites.push({api_shape:"javascript.XMLHttpRequest.send",start:send.start,end:send.end,
      configuration_start:open.start,configuration_end:open.end,enclosing_function_scope:scope,
      target_support:targetSupport(open.arguments[1]),...methodValue(open.arguments[0],"literal_open_method"),
      pairing_interpretation:"Same-function construction/configuration/send syntax; control-flow reachability and execution are unverified."});
  }
  report.sites.sort((a,b)=>a.start-b.start || a.end-b.end);
  report.conservative_builtin_shadowing_names=[...bound].filter(name=>["fetch","window","globalThis","self","XMLHttpRequest"].includes(name)).sort();
  return report;
}

module.exports = {parseSource};
if (require.main === module) {
  const input = readline.createInterface({input:process.stdin,crlfDelay:Infinity});
  input.on("line",line => {
    try {
      const record=JSON.parse(line);
      process.stdout.write(JSON.stringify({record_id:record.record_id,...parseSource(record.text)})+"\n");
    } catch (error) {
      process.stdout.write(JSON.stringify({parse_status:"input_or_parser_error",error_type:error.name})+"\n");
    }
  });
}
