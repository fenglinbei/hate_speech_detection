"use strict";

const assert = require("assert").strict;
const C = require("./core.js");
const tests = [];
function test(name, body) { tests.push({name, body}); }

test("Chinese labels cover persisted actions, routes and reasons", () => {
  for (const action of ["accept", "trim", "expand", "split", "reject", "defer"]) {
    assert.ok(C.ACTION_LABELS[action]);
  }
  for (const route of ["A_candidate", "B_candidate", "C_candidate"]) {
    assert.ok(C.ROUTE_LABELS[route]);
    assert.ok(C.ROUTE_DESCRIPTIONS[route]);
  }
  const grouped = C.REASON_GROUPS.flatMap(group => group.values);
  assert.equal(new Set(grouped).size, grouped.length);
  for (const reason of grouped) assert.ok(C.REASON_LABELS[reason]);
});

test("quick route keys map only 1-3 to A/B/C defaults", () => {
  assert.equal(C.quickRouteForKey("1"), "A_candidate");
  assert.equal(C.quickRouteForKey("2"), "B_candidate");
  assert.equal(C.quickRouteForKey("3"), "C_candidate");
  for (const key of ["4", "5", "6"]) assert.equal(C.quickRouteForKey(key), null);
  assert.deepEqual(C.routeDefaults(C.quickRouteForKey("1")), ["stable_core_candidate"]);
  assert.deepEqual(C.routeDefaults(C.quickRouteForKey("2")), ["context_required"]);
  assert.deepEqual(C.routeDefaults(C.quickRouteForKey("3")), ["evidence_required"]);
});

test("search modes cycle through literal, all terms and fuzzy", () => {
  assert.deepEqual(C.SEARCH_MODE_ORDER, ["literal", "all_terms", "fuzzy"]);
  assert.equal(C.nextSearchMode("literal"), "all_terms");
  assert.equal(C.nextSearchMode("all_terms"), "fuzzy");
  assert.equal(C.nextSearchMode("fuzzy"), "literal");
  assert.equal(C.nextSearchMode("unknown"), "literal");
  assert.equal(C.SEARCH_MODE_LABELS.literal, "连续");
  assert.equal(C.SEARCH_MODE_LABELS.all_terms, "多词");
  assert.equal(C.SEARCH_MODE_LABELS.fuzzy, "模糊");
});

test("visible case queue preserves result order and applies status filters", () => {
  const summaries = [
    {case_id: "S21-001", complete: false},
    {case_id: "S21-002", complete: true},
    {case_id: "S21-003", complete: false},
  ];
  const searchOrder = ["S21-003", "missing", "S21-002", "S21-003", "S21-001"];
  assert.deepEqual(
    C.visibleCaseQueue(summaries, searchOrder, "all"),
    ["S21-003", "S21-002", "S21-001"],
  );
  assert.deepEqual(
    C.visibleCaseQueue(summaries, searchOrder, "open"),
    ["S21-003", "S21-001"],
  );
  assert.deepEqual(
    C.visibleCaseQueue(summaries, searchOrder, "complete"),
    ["S21-002"],
  );
  assert.deepEqual(
    C.visibleCaseQueue(summaries),
    ["S21-001", "S21-002", "S21-003"],
  );
});

test("next unfinished case follows queue order, wraps, and handles a removed current case", () => {
  const summaries = [
    {case_id: "S21-001", complete: false},
    {case_id: "S21-002", complete: true},
    {case_id: "S21-003", complete: false},
  ];
  const queue = ["S21-003", "S21-002", "S21-001"];
  assert.equal(C.nextUnfinishedCaseId(queue, summaries, "S21-003"), "S21-001");
  assert.equal(C.nextUnfinishedCaseId(queue, summaries, "S21-001"), "S21-003");
  assert.equal(C.nextUnfinishedCaseId(queue, summaries, "removed"), "S21-003");
  assert.equal(C.nextUnfinishedCaseId([], summaries, "S21-001"), null);
  assert.equal(
    C.nextUnfinishedCaseId(queue, summaries.map(row => ({...row, complete: true})), "S21-003"),
    null,
  );
});

test("occurrence helpers count overlapping matches and Unicode text", () => {
  assert.equal(C.occurrenceStart("ababa", "aba", 1), 0);
  assert.equal(C.occurrenceStart("ababa", "aba", 2), 2);
  assert.equal(C.occurrenceOrdinal("ababa", "aba", 2), 2);
  assert.equal(C.occurrenceOrdinal("甲😎甲😎", "😎", 4), 2);
  assert.equal(C.codePointOffset("甲😎乙", 3), 2);
});

test("diagnostic action transitions clear incompatible values", () => {
  const proposal = {surface: "女拳", occurrence_ordinal: 1};
  assert.deepEqual(C.defaultDiagnosticDecision(proposal), {
    action: "accept",
    result_mentions: [{surface: "女拳", occurrence_ordinal: 1}],
    reason_codes: ["stable_core_candidate"],
    notes: "",
  });
  assert.deepEqual(C.decisionForAction("reject", proposal).result_mentions, []);
  assert.deepEqual(C.decisionForAction("reject", proposal).reason_codes, []);
  assert.equal(C.decisionForAction("split", proposal).result_mentions.length, 2);
  assert.deepEqual(C.decisionForAction("defer", proposal).reason_codes, ["evidence_required"]);
});

test("client validation catches illegal action/result combinations", () => {
  const content = "女拳不行";
  const proposal = {surface: "女拳不", occurrence_ordinal: 1};
  const reject = C.decisionForAction("reject", proposal);
  assert.ok(C.validateDiagnostic(content, proposal, reject).reason_codes);
  reject.reason_codes = ["fragment"];
  assert.equal(C.hasErrors(C.validateDiagnostic(content, proposal, reject)), false);

  const trim = C.decisionForAction("trim", proposal);
  trim.result_mentions = [{surface: "女拳", occurrence_ordinal: 1}];
  assert.equal(C.hasErrors(C.validateDiagnostic(content, proposal, trim)), false);
  trim.result_mentions = [{surface: "女拳不", occurrence_ordinal: 1}];
  assert.ok(C.validateDiagnostic(content, proposal, trim).result_mentions);
});

test("mutation queue snapshots IDs and replaces a pending draft with confirm", async () => {
  const calls = [];
  const queue = new C.MutationQueue(async task => {
    calls.push(task);
    return task;
  });
  const draft = {case_id: "S21-001", expected_revision: "revision-1", confirm: false, payload: {notes: "a"}};
  queue.schedule(draft, 10_000);
  draft.case_id = "S21-002";
  draft.expected_revision = "revision-2";
  draft.payload.notes = "changed";
  await queue.runNow({case_id: "S21-001", expected_revision: "revision-1", confirm: true, payload: {notes: "a"}});
  assert.equal(calls.length, 1);
  assert.equal(calls[0].case_id, "S21-001");
  assert.equal(calls[0].expected_revision, "revision-1");
  assert.equal(calls[0].confirm, true);
});

test("single-key shortcuts are suppressed for form controls", () => {
  for (const tagName of ["INPUT", "TEXTAREA", "SELECT"]) {
    assert.equal(C.isTextEntry({tagName, isContentEditable: false}), true);
  }
  assert.equal(C.isTextEntry({tagName: "BUTTON", isContentEditable: false}), false);
  assert.equal(C.isTextEntry({tagName: "DIV", isContentEditable: true}), true);
});

test("mutation queue serializes an in-flight save and a newer snapshot", async () => {
  const order = [];
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const queue = new C.MutationQueue(async task => {
    order.push(`start-${task.id}`);
    if (task.id === 1) await gate;
    order.push(`end-${task.id}`);
  });
  queue.schedule({id: 1}, 10_000);
  const first = queue.flush();
  queue.schedule({id: 2}, 10_000);
  const second = queue.flush();
  release();
  await Promise.all([first, second]);
  assert.deepEqual(order, ["start-1", "end-1", "start-2", "end-2"]);
});

(async () => {
  let failures = 0;
  for (const row of tests) {
    try {
      await row.body();
      process.stdout.write(`ok - ${row.name}\n`);
    } catch (error) {
      failures += 1;
      process.stderr.write(`not ok - ${row.name}\n${error.stack}\n`);
    }
  }
  process.exitCode = failures ? 1 : 0;
})();
