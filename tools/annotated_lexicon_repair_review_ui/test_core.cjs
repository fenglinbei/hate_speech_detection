"use strict";

const assert = require("assert").strict;
global.ReviewCore = require("../wp3_candidate_review_ui/core.js");
const Gold = require("./core.js");

const item = {
  query_content: "厕所里畒勾叫的，还没路边蝲蝲蛄叫的好听",
  candidates: [
    {candidate_id: "old", surface: "畒", span: [3, 4]},
    {candidate_id: "full", surface: "畒勾", span: [3, 5]},
    {candidate_id: "other", surface: "蝲蝲蛄", span: [13, 16]},
  ],
};

const decision = Gold.defaultDecision(item);
assert.deepEqual(decision.candidate_actions, {old: null, full: null, other: null});
assert.ok(Gold.validateDecision(item, decision, {confirm: true}).candidate_actions);
decision.candidate_actions = {old: "drop", full: "keep", other: "keep"};
assert.deepEqual(Gold.validateDecision(item, decision, {confirm: true}), {});

decision.candidate_actions.old = "keep";
assert.ok(Gold.validateDecision(item, decision, {confirm: true}).expected_spans);

const parsed = Gold.parseAdditionalSpans("0 | 2 | 厕所 | 补充\n");
assert.deepEqual(parsed.errors, []);
assert.deepEqual(parsed.rows, [{start: 0, end: 2, surface: "厕所", reason: "补充"}]);
assert.equal(Gold.serializeAdditionalSpans(parsed.rows), "0 | 2 | 厕所 | 补充");

const summaries = [
  {item_id: "a", source_item_id: "blind-a", query_preview: "畒勾", cohort: "repair_target", surfaces: ["畒勾"], status: "draft"},
  {item_id: "b", source_item_id: "blind-b", query_preview: "对照", cohort: "accepted_control", surfaces: ["国女"], status: "confirmed"},
];
assert.deepEqual(Gold.visibleItemQueue(summaries, "畒", "literal", "all"), ["a"]);
assert.deepEqual(Gold.visibleItemQueue(summaries, "", "literal", "open"), ["a"]);
assert.equal(Gold.nextUnfinishedItemId(["a", "b"], summaries, null), "a");

console.log("annotated lexicon span-gold core tests passed");
