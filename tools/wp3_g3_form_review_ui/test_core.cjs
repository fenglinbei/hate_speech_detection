"use strict";

const assert = require("assert").strict;
const G = require("./core.js");

const tests = [];
function test(name, body) { tests.push({name, body}); }

const item = {
  item_id: "g3form-one",
  surface: "永远滴神",
  canonical: "永远的神",
  proposed_family: "phonetic_variant",
  phonetic_scan_enabled: true,
  evidence_ids: ["ev-one"],
};

const evidence = [{
  evidence_id: "ev-one",
  source_id: "source-one",
  component_id: "component-one",
  relation_contract: "single-quote-surface-and-canonical/v2",
  quote: "“永远滴神”是“永远的神”的谐音写法。",
}];

const proposalDecision = {
  action: "accept",
  surface: item.surface,
  canonical: item.canonical,
  family: item.proposed_family,
  phonetic_scan_enabled: item.phonetic_scan_enabled,
  evidence_ids: [...item.evidence_ids],
  notes: "",
};

test("1-4 map exactly to accept, edit, reject and defer", () => {
  assert.equal(G.actionForKey("1"), "accept");
  assert.equal(G.actionForKey("2"), "edit");
  assert.equal(G.actionForKey("3"), "reject");
  assert.equal(G.actionForKey("4"), "defer");
  for (const key of ["0", "5", "a"]) assert.equal(G.actionForKey(key), null);
});

test("non-edit decisions restore every frozen proposal field", () => {
  const edited = {
    ...proposalDecision,
    action: "edit",
    surface: "永远的神啊",
    family: "known_variant",
    phonetic_scan_enabled: false,
    evidence_ids: [],
    notes: "keep this note",
  };
  for (const action of ["accept", "reject", "defer"]) {
    const decision = G.decisionForAction(action, item, edited);
    assert.equal(decision.action, action);
    assert.deepEqual(
      {
        surface: decision.surface,
        canonical: decision.canonical,
        family: decision.family,
        phonetic_scan_enabled: decision.phonetic_scan_enabled,
        evidence_ids: decision.evidence_ids,
      },
      G.proposalFromItem(item),
    );
    assert.equal(decision.notes, "keep this note");
  }
});

test("status filters retain order and isolate confirmed defer", () => {
  const summaries = [
    {item_id: "one", status: "draft", action: "defer"},
    {item_id: "two", status: "confirmed", action: "accept"},
    {item_id: "three", status: "confirmed", action: "defer"},
  ];
  const order = ["three", "missing", "one", "two", "three"];
  assert.deepEqual(G.visibleItemQueue(summaries, order, "all"), ["three", "one", "two"]);
  assert.deepEqual(G.visibleItemQueue(summaries, order, "open"), ["one"]);
  assert.deepEqual(G.visibleItemQueue(summaries, order, "complete"), ["three", "two"]);
  assert.deepEqual(G.visibleItemQueue(summaries, order, "defer"), ["three"]);
});

test("next unfinished item wraps and skips confirmed decisions", () => {
  const summaries = [
    {item_id: "one", status: "draft"},
    {item_id: "two", status: "confirmed"},
    {item_id: "three", status: "draft"},
  ];
  const queue = ["three", "two", "one"];
  assert.equal(G.nextUnfinishedItemId(queue, summaries, "three"), "one");
  assert.equal(G.nextUnfinishedItemId(queue, summaries, "one"), "three");
  assert.equal(
    G.nextUnfinishedItemId(queue, summaries.map(row => ({...row, status: "confirmed"})), "one"),
    null,
  );
});

test("local search covers form, family, publisher and multiple terms", () => {
  const summaries = [{
    item_id: "g3form-one",
    surface: "永远滴神",
    canonical: "永远的神",
    proposed_family: "phonetic_variant",
    action: "defer",
    publishers: ["公开材料甲"],
    source_roles: ["primary"],
  }];
  assert.deepEqual(G.searchItemIds(summaries, "滴神", "literal"), ["g3form-one"]);
  assert.deepEqual(G.searchItemIds(summaries, "公开 phonetic", "all_terms"), ["g3form-one"]);
  assert.deepEqual(G.searchItemIds(summaries, "永滴神", "fuzzy"), ["g3form-one"]);
  assert.deepEqual(G.searchItemIds(summaries, "不存在", "literal"), []);
});

test("accept is valid only when frozen evidence replays both forms", () => {
  assert.deepEqual(G.validateDecision(item, evidence, proposalDecision), {});
  const missingReplay = [{...evidence[0], quote: "只出现永远滴神"}];
  assert.ok(G.validateDecision(item, missingReplay, proposalDecision).proposal);
});

test("edit must change a field and still replay in one frozen component", () => {
  const unchanged = {...proposalDecision, action: "edit"};
  assert.ok(G.validateDecision(item, evidence, unchanged).proposal);
  const edited = {
    ...unchanged,
    family: "orthographic_variant",
    notes: "family 修订",
  };
  assert.deepEqual(G.validateDecision(item, evidence, edited), {});

  const twoEvidenceItem = {...item, evidence_ids: ["ev-one", "ev-two"]};
  const twoEvidence = [
    evidence[0],
    {...evidence[0], evidence_id: "ev-two", source_id: "source-two"},
  ];
  const crossSource = {...edited, evidence_ids: ["ev-one", "ev-two"]};
  assert.ok(G.validateDecision(twoEvidenceItem, twoEvidence, crossSource).evidence_ids);
});

test("reject and defer cannot smuggle edited proposal fields", () => {
  for (const action of ["reject", "defer"]) {
    const decision = {...proposalDecision, action, surface: "另一个形式"};
    assert.ok(G.validateDecision(item, evidence, decision).proposal);
  }
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
