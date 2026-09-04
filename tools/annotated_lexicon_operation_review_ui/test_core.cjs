"use strict";
const assert = require("assert").strict;
const Ops = require("./core.js");
const item = {item_id: "one", term: "词", flags: [], open_questions: [], proposed_entry: {lexicon_id: "lex-one", term: "词", variants: [], senses: [{sense_id: "sense-one", definition: "依当前语境解释", categories: ["others"]}], match_policy: {require_any: [], exclude_any: []}}};
const clone = value => JSON.parse(JSON.stringify(value));
const draft = Ops.defaultDecision(item);
assert.equal(draft.resolution, null);
assert.ok(Ops.validateDecision(item, draft, {confirm: true}).resolution);
draft.resolution = "approve";
assert.deepEqual(Ops.validateDecision(item, draft, {confirm: true}), {});
draft.entry.senses[0].definition = "修改释义";
assert.ok(Ops.validateDecision(item, draft, {confirm: true}).resolution);
assert.equal(item.proposed_entry.senses[0].definition, "依当前语境解释");
draft.resolution = "revise";
assert.deepEqual(Ops.validateDecision(item, draft, {confirm: true}), {});
draft.entry.senses[0].definition = "";
assert.deepEqual(Ops.validateDecision(item, draft), {});
assert.ok(Ops.validateDecision(item, draft, {confirm: true}).senses);
for (const resolution of ["defer", "reject"]) {
  draft.resolution = resolution;
  assert.ok(Ops.validateDecision(item, draft, {confirm: true}).notes);
  draft.notes = "证据不足";
  assert.deepEqual(Ops.validateDecision(item, draft, {confirm: true}), {});
  draft.notes = "";
}
const unresolved = clone(item); unresolved.flags = ["policy_unresolved"]; unresolved.open_questions = ["请核实规则"];
const unresolvedDraft = {...Ops.defaultDecision(unresolved), resolution: "approve"};
assert.ok(Ops.validateDecision(unresolved, unresolvedDraft, {confirm: true}).policy);
assert.ok(Ops.validateDecision(unresolved, unresolvedDraft, {confirm: true}).notes);
const event = key => ({key, target: {tagName: "DIV"}});
assert.equal(Ops.reviewShortcut(event("1")), "approve");
assert.equal(Ops.reviewShortcut(event("2")), "revise");
assert.equal(Ops.reviewShortcut(event("3")), "defer");
assert.equal(Ops.reviewShortcut(event("4")), "reject");
assert.equal(Ops.reviewShortcut(event("[")), "previous-item");
assert.equal(Ops.reviewShortcut(event("]")), "next-item");
for (const key of ["1", "2", "3", "4", "[", "]", "?"]) {
  for (const tagName of ["INPUT", "TEXTAREA", "SELECT"]) assert.equal(Ops.reviewShortcut({...event(key), target: {tagName}}), null);
  for (const guard of [{dialogOpen: true}, {busy: true}, {loaded: false}]) assert.equal(Ops.reviewShortcut(event(key), guard), null);
  for (const guard of [{isComposing: true}, {keyCode: 229}, {repeat: true}, {altKey: true}, {defaultPrevented: true}]) assert.equal(Ops.reviewShortcut({...event(key), ...guard}), null);
}
for (const modifier of ["ctrlKey", "metaKey"]) {
  assert.equal(Ops.reviewShortcut({...event("s"), [modifier]: true, target: {tagName: "TEXTAREA"}}), "save");
  assert.equal(Ops.reviewShortcut({...event("Enter"), [modifier]: true}), "confirm");
  assert.equal(Ops.reviewShortcut({...event("Enter"), [modifier]: true}, {locked: true}), null);
}
assert.equal(Ops.reviewShortcut(event("1"), {locked: true}), null);
assert.equal(Ops.reviewShortcut(event("?"), {locked: true}), "help");
const summaries = [{item_id: "a", term: "乐色", operation_kind: "existing", query_preview: "乐色堆", status: "confirmed"}, {item_id: "b", term: "呆比", operation_kind: "new_entry", query_preview: "呆比呆比", status: "deferred"}, {item_id: "c", term: "它们", operation_kind: "duplicate_merge", query_preview: "它们乱吠", status: "draft"}];
assert.deepEqual(Ops.visibleItemQueue(summaries, "", "literal", "open"), ["c"]);
assert.deepEqual(Ops.visibleItemQueue(summaries, "", "literal", "deferred"), ["b"]);
assert.deepEqual(Ops.visibleItemQueue(summaries, "乐堆", "fuzzy"), ["a"]);
assert.equal(Ops.nextUnfinishedItemId(["a", "b", "c"], summaries, "a"), "c");
assert.equal(Ops.nextUnfinishedItemId(["a", "b"], summaries), null);
const text = "😀它和它<script>";
const segments = Ops.evidenceSegments(text, [{surface: "它", span: [1, 2]}, {surface: "它", span: [3, 4]}]);
assert.deepEqual(segments.filter(row => row.highlighted).map(row => row.span), [[1, 2], [3, 4]]);
assert.equal(segments.map(row => row.text).join(""), text);
assert.equal(Ops.safeEvidenceUrl("javascript:alert(1)"), null);
assert.equal(Ops.safeEvidenceUrl("data:text/html,test"), null);
assert.equal(Ops.safeEvidenceUrl("https://example.com/a"), "https://example.com/a");
console.log("PASS operation review core validation, Unicode, search and shortcut guardrails");
