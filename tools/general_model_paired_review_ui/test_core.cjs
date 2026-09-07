"use strict";
const assert = require("node:assert/strict");
const Pair = require("./core.js");

const rows = [
  {item_id: "first", query_preview: "中文 词义 示例", focus_task: "hate", initial_batch: true, status: "confirmed", disposition: "input_control"},
  {item_id: "second", query_preview: "待核验的词义", focus_task: "group", initial_batch: true, status: "draft", disposition: ""},
  {item_id: "later", query_preview: "后续定位材料", focus_task: "hate", initial_batch: false, status: "confirmed", disposition: "defer"},
];
assert.deepEqual(Pair.visibleQueue(rows), ["first", "second"]);
assert.deepEqual(Pair.visibleQueue(rows, {scope: "all", filter: "defer"}), ["later"]);
assert.deepEqual(Pair.visibleQueue(rows, {query: "中文 示例", mode: "all_terms"}), ["first"]);
assert.deepEqual(Pair.visibleQueue(rows, {query: "中词示", mode: "fuzzy"}), ["first"]);
assert.equal(Pair.nextUnfinished(["first", "second"], rows, "first"), "second");
assert.equal(Pair.nextUnfinished(["first"], rows, "first"), null);
assert.equal(Pair.labels([]), "无目标群体 ∅");
assert.notEqual(Pair.labels([]), Pair.labels(["others"]));

const notes = Pair.emptyNotes();
assert.equal(Object.keys(Pair.validate(notes, "reveal")).length, 4);
for (const key of Object.keys(Pair.RESOURCE_LABELS)) notes.resources[key] = "测试记录";
assert.deepEqual(Pair.validate(notes, "reveal"), {});
assert.ok(Pair.validate(notes, "reveal_ai").gold_verdict);
Object.assign(notes.assessment, {
  gold_verdict: "agree", stage2_candidate_explanation: "候选解释",
  alternative_explanation: "替代解释", disposition: "input_control",
});
assert.ok(Pair.validate(notes, "confirm").falsifiable_followup);
notes.assessment.falsifiable_followup = "保持长度位置的输入对照";
assert.deepEqual(Pair.validate(notes, "confirm"), {});
notes.assessment.disposition = "verify_first";
assert.ok(Pair.validate(notes, "confirm").patching_defer_reason);
notes.assessment.patching_defer_reason = "先核验计分敏感性";
assert.deepEqual(Pair.validate(notes, "confirm"), {});
console.log("PASS queue scope/search/order, group empty-set display, and phase-specific validation");
