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

const unicodeQuery = "😀它们和它们 e\u0301 👩‍💻";
assert.deepEqual(Gold.spanFromUtf16Offsets(unicodeQuery, 5, 7), {start: 4, end: 6, surface: "它们"});
assert.deepEqual(Gold.spanFromUtf16Offsets(unicodeQuery, 0, 2), {start: 0, end: 1, surface: "😀"});
assert.deepEqual(Gold.spanFromUtf16Offsets(unicodeQuery, 8, 10), {start: 7, end: 9, surface: "e\u0301"});
for (const [start, end] of [[0, 1], [1, 2], [2, 2], [-1, 2], [7, 999]]) {
  assert.equal(Gold.spanFromUtf16Offsets(unicodeQuery, start, end), null);
}
const extra = {start: 0, end: 2, surface: "厕所", reason: "需要 | 完整解释"};
const withAdditional = Gold.defaultDecision(item);
withAdditional.additional_spans = [extra];
assert.deepEqual(Gold.validateDecision(item, withAdditional), {});
assert.ok(Gold.additionalSpanError(item, withAdditional, extra));
assert.equal(Gold.additionalSpanError(item, withAdditional, extra, 0), "");
assert.ok(Gold.additionalSpanError(item, withAdditional, {start: 3, end: 5}));
assert.ok(Gold.additionalSpanError(item, withAdditional, {start: 1, end: 3}));
withAdditional.candidate_actions.full = "keep";
assert.ok(Gold.additionalSpanError(item, withAdditional, {start: 4, end: 6}));
withAdditional.additional_spans[0].reason = " ";
assert.ok(Gold.validateDecision(item, withAdditional).additional_spans);
withAdditional.additional_spans[0].reason = "字".repeat(501);
assert.ok(Gold.validateDecision(item, withAdditional).additional_spans);
withAdditional.additional_spans = null;
assert.ok(Gold.validateDecision(item, withAdditional).additional_spans);
withAdditional.additional_spans = Array(32).fill(extra);
assert.ok(Gold.additionalSpanError(item, withAdditional, {start: 20, end: 21}));

const shortcuts = {"1": "keep-all", "2": "drop-all", "3": "clear-all", ArrowUp: "previous-candidate", ArrowDown: "next-candidate", q: "keep", Q: "keep", a: "drop", "[": "previous-item", "]": "next-item", "?": "help"};
for (const [key, action] of Object.entries(shortcuts)) {
  assert.equal(Gold.reviewShortcut({key}), action);
  for (const target of [{tagName: "INPUT"}, {tagName: "TEXTAREA"}, {tagName: "SELECT"}, {tagName: "DIV", isContentEditable: true}]) {
    assert.equal(Gold.reviewShortcut({key, target}), null);
  }
  for (const state of [{dialogOpen: true}, {busy: true}, {loaded: false}]) assert.equal(Gold.reviewShortcut({key}, state), null);
  for (const flags of [{altKey: true}, {isComposing: true}, {keyCode: 229}, {defaultPrevented: true}]) assert.equal(Gold.reviewShortcut({key, ...flags}), null);
}
assert.equal(Gold.reviewShortcut({key: "q", repeat: true}), null);
assert.equal(Gold.reviewShortcut({key: "ArrowDown", repeat: true}), "next-candidate");
assert.equal(Gold.reviewShortcut({key: "ArrowDown", shiftKey: true}), null);
assert.equal(Gold.reviewShortcut({key: "?", shiftKey: true}), "help");
assert.equal(Gold.reviewShortcut({key: "s", ctrlKey: true, target: {tagName: "TEXTAREA"}}), "save");
assert.equal(Gold.reviewShortcut({key: "Enter", metaKey: true}), "confirm");
assert.equal(Gold.reviewShortcut({key: "Enter", ctrlKey: true, isComposing: true}), null);
assert.equal(Gold.reviewShortcut({key: "Enter", ctrlKey: true, repeat: true}), null);
for (const key of ["1", "2", "3", "q", "a"]) assert.equal(Gold.reviewShortcut({key}, {locked: true}), null);
assert.equal(Gold.reviewShortcut({key: "s", ctrlKey: true}, {locked: true}), null);
assert.equal(Gold.reviewShortcut({key: "ArrowDown"}, {locked: true}), "next-candidate");

const summaries = [
  {item_id: "a", source_item_id: "blind-a", query_preview: "畒勾", cohort: "repair_target", surfaces: ["畒勾"], status: "draft"},
  {item_id: "b", source_item_id: "blind-b", query_preview: "对照", cohort: "accepted_control", surfaces: ["国女"], status: "confirmed"},
];
assert.deepEqual(Gold.visibleItemQueue(summaries, "畒", "literal", "all"), ["a"]);
assert.deepEqual(Gold.visibleItemQueue(summaries, "", "literal", "open"), ["a"]);
assert.equal(Gold.nextUnfinishedItemId(["a", "b"], summaries, null), "a");

console.log("annotated lexicon span-gold core tests passed");
