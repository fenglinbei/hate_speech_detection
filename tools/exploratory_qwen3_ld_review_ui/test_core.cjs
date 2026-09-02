"use strict";

const assert = require("assert");
global.ReviewCore = require("../wp3_candidate_review_ui/core.js");
const Audit = require("./core.js");

function test(name, callback) {
  try {
    callback();
    process.stdout.write(`ok - ${name}\n`);
  } catch (error) {
    process.stderr.write(`not ok - ${name}\n${error.stack}\n`);
    process.exitCode = 1;
  }
}

const lexItem = {audit_kind: "lex_hit"};
const noHitItem = {audit_kind: "no_hit"};

test("1-3 map to accept, reject and defer", () => {
  assert.equal(Audit.dispositionForKey("1"), "accept");
  assert.equal(Audit.dispositionForKey("2"), "reject");
  assert.equal(Audit.dispositionForKey("3"), "defer");
  assert.equal(Audit.dispositionForKey("4"), null);
});

test("lex-hit dimension shortcuts map directly to the visible choices", () => {
  const expected = {
    KeyQ: {field: "relevance", value: "pass"},
    KeyA: {field: "relevance", value: "fail"},
    KeyW: {field: "boundary", value: "pass"},
    KeyS: {field: "boundary", value: "fail"},
    KeyE: {field: "definition_quality", value: "good"},
    KeyD: {field: "definition_quality", value: "usable"},
    KeyC: {field: "definition_quality", value: "poor"},
    KeyR: {field: "sense_fit", value: "pass"},
    KeyF: {field: "sense_fit", value: "fail"},
    KeyT: {field: "swap_incompatibility", value: "pass"},
    KeyG: {field: "swap_incompatibility", value: "fail"},
  };
  for (const [code, mapping] of Object.entries(expected)) {
    assert.deepEqual(Audit.dimensionShortcutForCode("lex_hit", code), mapping);
  }
});

test("no-hit safely reuses Q/A because lex dimensions are hidden", () => {
  assert.deepEqual(Audit.dimensionShortcutForCode("no_hit", "KeyQ"), {field: "no_hit_verified", value: "pass"});
  assert.deepEqual(Audit.dimensionShortcutForCode("no_hit", "KeyA"), {field: "no_hit_verified", value: "fail"});
  assert.equal(Audit.dimensionShortcutForCode("no_hit", "KeyW"), null);
});

test("pragmatic tags require Shift plus digits 1-6", () => {
  assert.equal(Audit.tagShortcutForCode("Digit1", true), "quote");
  assert.equal(Audit.tagShortcutForCode("Digit6", true), "discussion");
  assert.equal(Audit.tagShortcutForCode("Digit1", false), null);
  assert.equal(Audit.tagShortcutForCode("Digit7", true), null);
});

test("lex-hit accept requires all five applicable dimensions", () => {
  const decision = {
    ...Audit.defaultDecision(),
    disposition: "accept",
    relevance: "pass",
    boundary: "pass",
    definition_quality: "usable",
    sense_fit: "pass",
    swap_incompatibility: "pass",
  };
  assert.deepEqual(Audit.validateDecision(lexItem, decision, {confirm: true}), {});
  decision.sense_fit = null;
  assert.ok(Audit.validateDecision(lexItem, decision, {confirm: true}).sense_fit);
});

test("failed lex dimension blocks accept and enables documented reject", () => {
  const decision = {
    ...Audit.defaultDecision(),
    disposition: "reject",
    relevance: "pass",
    boundary: "fail",
    definition_quality: "good",
    sense_fit: "pass",
    swap_incompatibility: "pass",
    notes: "substring 边界只是嵌套片段",
  };
  assert.deepEqual(Audit.validateDecision(lexItem, decision, {confirm: true}), {});
  decision.disposition = "accept";
  assert.ok(Audit.validateDecision(lexItem, decision, {confirm: true}).disposition);
});

test("no-hit uses only its dedicated verification field", () => {
  const decision = {...Audit.defaultDecision(), disposition: "accept", no_hit_verified: "pass"};
  assert.deepEqual(Audit.validateDecision(noHitItem, decision, {confirm: true}), {});
  decision.relevance = "pass";
  assert.ok(Audit.validateDecision(noHitItem, decision, {confirm: true}).relevance);
});

test("queue filters and search retain shared workbench semantics", () => {
  const summaries = [
    {item_id: "a", query_preview: "女拳 示例", terms: ["女拳"], audit_kind: "lex_hit", status: "draft", disposition: "defer"},
    {item_id: "b", query_preview: "普通文本", terms: [], audit_kind: "no_hit", status: "confirmed", disposition: "accept"},
    {item_id: "c", query_preview: "边界问题", terms: ["黑"], audit_kind: "lex_hit", status: "confirmed", disposition: "reject"},
  ];
  assert.deepEqual(Audit.searchItemIds(summaries, "女拳"), ["a"]);
  assert.deepEqual(Audit.visibleItemQueue(summaries, null, "reject"), ["c"]);
  assert.equal(Audit.nextUnfinishedItemId(["a", "b", "c"], summaries, "c"), "a");
});
