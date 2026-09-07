(function initPairedReviewCore(global) {
  "use strict";
  const Common = global.ReviewCore || (typeof require === "function" ? require("../wp3_candidate_review_ui/core.js") : null);
  const CONDITIONS = [
    {id: "C0", short: "0", label: "仅查询"},
    {id: "CLnew", short: "SG", label: "词典 · 含类别"},
    {id: "CD", short: "D", label: "仅示例"},
    {id: "CLDnew", short: "SGD", label: "含类别词典 + 示例"},
    {id: "CLnewNoCat", short: "S", label: "词典 · 去类别"},
    {id: "CLDnewNoCat", short: "SD", label: "去类别词典 + 示例"},
  ];
  const RESOURCE_LABELS = {
    ambiguity_stance: "语义与立场",
    definition_fit: "词典义项适配",
    category_relation: "类别与对象关系",
    demo_correspondence: "示例对应",
    stage1_resource_notes: "其他资源观察",
  };
  const ASSESSMENT_LABELS = {
    gold_dispute: "Gold 判断理由",
    stage2_candidate_explanation: "候选解释",
    alternative_explanation: "替代解释",
    falsifiable_followup: "下一步输入对照",
    patching_defer_reason: "先核验 / 暂缓的原因",
    ai_comparison: "看过 AI 后的补充",
  };
  const DISPOSITIONS = {
    input_control: "进入输入对照", verify_first: "先核验再使用", defer: "暂缓使用",
  };
  const GOLD_LABELS = {agree: "认可", dispute: "有争议", uncertain: "信息不足"};
  const TASK_LABELS = {hate: "仇恨判断", group: "目标群体"};
  const LABELS = {hate: "仇恨", "non-hate": "非仇恨", Racism: "种族", Region: "地域", LGBTQ: "性少数", Sexism: "性别", others: "其他"};

  function emptyNotes() {
    return {
      resources: {...Object.fromEntries(Object.keys(RESOURCE_LABELS).map(key => [key, ""])), prior_ai_exposure: "unspecified"},
      assessment: {...Object.fromEntries(Object.keys(ASSESSMENT_LABELS).map(key => [key, ""])), gold_verdict: "", disposition: ""},
    };
  }
  function notesFromReview(review) {
    return Common.clone({resources: review.resources, assessment: review.assessment});
  }
  function labels(value) {
    if (Array.isArray(value)) return value.length ? value.map(item => LABELS[item] || item).join("、") : "无目标群体 ∅";
    return LABELS[value] || value || "—";
  }
  function number(value, signed = false) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return (signed && value > 0 ? "+" : "") + value.toFixed(3);
  }
  function goldMargin(conditionRow) {
    return conditionRow.readouts["answer_sum/gold/best_nongold_margin"];
  }
  // Search semantics follow the existing exploratory input-review workbench.
  function matches(item, query, mode) {
    const text = String(query || "").trim().toLocaleLowerCase("zh-CN");
    if (!text) return true;
    const fields = [item.item_id, item.query_preview, TASK_LABELS[item.focus_task]].map(value => String(value || "").toLocaleLowerCase("zh-CN"));
    if (mode === "all_terms") return text.split(/\s+/u).every(term => fields.some(value => value.includes(term)));
    if (mode === "fuzzy") return fields.some(value => {
      let cursor = 0;
      const letters = Array.from(text.replace(/\s+/gu, ""));
      for (const character of value.replace(/\s+/gu, "")) if (character === letters[cursor]) cursor++;
      return cursor === letters.length;
    });
    return fields.some(value => value.includes(text));
  }
  function visibleQueue(items, {scope = "initial", filter = "all", query = "", mode = "literal"} = {}) {
    return items.filter(item => {
      if (scope === "initial" && !item.initial_batch) return false;
      if (filter === "open" && item.status === "confirmed") return false;
      if (filter === "complete" && item.status !== "confirmed") return false;
      if (["verify_first", "defer"].includes(filter) && (item.status !== "confirmed" || item.disposition !== filter)) return false;
      return matches(item, query, mode);
    }).map(item => item.item_id);
  }
  function nextUnfinished(queue, items, currentId) {
    return Common.nextUnfinishedCaseId(queue, items.map(row => ({case_id: row.item_id, complete: row.status === "confirmed"})), currentId);
  }
  function validate(notes, action) {
    const errors = {};
    for (const [key, label] of Object.entries(RESOURCE_LABELS)) {
      const value = notes.resources[key];
      if (typeof value !== "string" || value.length > 1000) errors[key] = label + "最多填写 1000 字。";
      if (action === "reveal" && key !== "stage1_resource_notes" && !value.trim()) errors[key] = "请记录" + label + "；不确定也可以明确写出。";
    }
    for (const [key, label] of Object.entries(ASSESSMENT_LABELS)) {
      const value = notes.assessment[key];
      if (typeof value !== "string" || value.length > 2000) errors[key] = label + "最多填写 2000 字。";
    }
    if (action === "confirm" || action === "reveal_ai") {
      const row = notes.assessment;
      if (!GOLD_LABELS[row.gold_verdict]) errors.gold_verdict = "请选择对原 Gold 的判断。";
      if (["dispute", "uncertain"].includes(row.gold_verdict) && !row.gold_dispute.trim()) errors.gold_dispute = "请写明争议或信息不足的理由。";
      for (const key of ["stage2_candidate_explanation", "alternative_explanation"]) {
        if (!row[key].trim()) errors[key] = "请记录" + ASSESSMENT_LABELS[key] + "；无法判断时可明确说明。";
      }
      if (!DISPOSITIONS[row.disposition]) errors.disposition = "请选择下一阶段用途。";
      if (row.disposition === "input_control" && !row.falsifiable_followup.trim()) errors.falsifiable_followup = "请写明改什么、保持什么，以及观察什么。";
      if (["verify_first", "defer"].includes(row.disposition) && !row.patching_defer_reason.trim()) errors.patching_defer_reason = "请写明需要先核验或暂缓的原因。";
    }
    return errors;
  }
  const api = {
    CONDITIONS, RESOURCE_LABELS, ASSESSMENT_LABELS, DISPOSITIONS, GOLD_LABELS, TASK_LABELS,
    emptyNotes, notesFromReview, labels, number, goldMargin, matches, visibleQueue, nextUnfinished, validate,
  };
  global.PairedReviewCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof globalThis === "undefined" ? this : globalThis));
