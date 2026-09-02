(function initPilotInputAuditCore(global) {
  "use strict";

  const Common = global.ReviewCore || (
    typeof require === "function"
      ? require("../wp3_candidate_review_ui/core.js")
      : null
  );
  if (!Common) throw new Error("shared review core is unavailable");

  const DISPOSITION_LABELS = Object.freeze({
    accept: "接受",
    reject: "驳回",
    defer: "暂缓",
  });
  const DISPOSITION_BY_KEY = Object.freeze({"1": "accept", "2": "reject", "3": "defer"});
  const BINARY_LABELS = Object.freeze({pass: "通过", fail: "不通过"});
  const DEFINITION_LABELS = Object.freeze({good: "良好", usable: "可用", poor: "较差"});
  const TAG_LABELS = Object.freeze({
    quote: "引用/转述",
    negation: "否定",
    counterspeech: "反仇恨/反歧视",
    reclaimed: "自称/重领",
    irony: "反讽",
    discussion: "讨论/解释",
  });
  const LEX_DIMENSION_SHORTCUTS = Object.freeze({
    KeyQ: Object.freeze({field: "relevance", value: "pass"}),
    KeyA: Object.freeze({field: "relevance", value: "fail"}),
    KeyW: Object.freeze({field: "boundary", value: "pass"}),
    KeyS: Object.freeze({field: "boundary", value: "fail"}),
    KeyE: Object.freeze({field: "definition_quality", value: "good"}),
    KeyD: Object.freeze({field: "definition_quality", value: "usable"}),
    KeyC: Object.freeze({field: "definition_quality", value: "poor"}),
    KeyR: Object.freeze({field: "sense_fit", value: "pass"}),
    KeyF: Object.freeze({field: "sense_fit", value: "fail"}),
    KeyT: Object.freeze({field: "swap_incompatibility", value: "pass"}),
    KeyG: Object.freeze({field: "swap_incompatibility", value: "fail"}),
  });
  const NO_HIT_DIMENSION_SHORTCUTS = Object.freeze({
    KeyQ: Object.freeze({field: "no_hit_verified", value: "pass"}),
    KeyA: Object.freeze({field: "no_hit_verified", value: "fail"}),
  });
  const TAG_SHORTCUTS = Object.freeze({
    Digit1: "quote",
    Digit2: "negation",
    Digit3: "counterspeech",
    Digit4: "reclaimed",
    Digit5: "irony",
    Digit6: "discussion",
  });
  const TAG_SHORTCUT_LABELS = Object.freeze({
    quote: "⇧1",
    negation: "⇧2",
    counterspeech: "⇧3",
    reclaimed: "⇧4",
    irony: "⇧5",
    discussion: "⇧6",
  });
  const FILTERS = new Set(["all", "open", "complete", "reject", "defer"]);

  function defaultDecision() {
    return {
      disposition: "defer",
      relevance: null,
      boundary: null,
      definition_quality: null,
      sense_fit: null,
      swap_incompatibility: null,
      no_hit_verified: null,
      pragmatic_tags: [],
      notes: "",
    };
  }

  function decisionFields(stored = {}) {
    const base = defaultDecision();
    for (const key of Object.keys(base)) {
      if (Object.prototype.hasOwnProperty.call(stored, key)) base[key] = Common.clone(stored[key]);
    }
    return base;
  }

  function dispositionForKey(key) {
    return DISPOSITION_BY_KEY[String(key)] || null;
  }

  function dimensionShortcutForCode(auditKind, code) {
    const shortcuts = auditKind === "no_hit"
      ? NO_HIT_DIMENSION_SHORTCUTS
      : LEX_DIMENSION_SHORTCUTS;
    return shortcuts[String(code)] || null;
  }

  function tagShortcutForCode(code, shiftKey) {
    return shiftKey ? TAG_SHORTCUTS[String(code)] || null : null;
  }

  function normalizedSearchFields(summary) {
    return [
      summary.item_id,
      summary.query_preview,
      summary.audit_kind,
      summary.disposition,
      ...(summary.terms || []),
    ].map(value => String(value || "").toLocaleLowerCase("zh-CN"));
  }

  function fuzzySubsequence(haystack, needle) {
    const source = Array.from(haystack.replace(/\s+/gu, ""));
    const query = Array.from(needle.replace(/\s+/gu, ""));
    if (!query.length) return true;
    let cursor = 0;
    for (const character of source) {
      if (character === query[cursor]) cursor += 1;
      if (cursor === query.length) return true;
    }
    return false;
  }

  function itemMatches(summary, query, mode = "literal") {
    const normalized = String(query || "").trim().toLocaleLowerCase("zh-CN");
    if (!normalized) return true;
    const fields = normalizedSearchFields(summary);
    if (mode === "all_terms") {
      const terms = [...new Set(normalized.split(/\s+/u).filter(Boolean))];
      return terms.every(term => fields.some(field => field.includes(term)));
    }
    if (mode === "fuzzy") {
      return fields.some(field => field.includes(normalized) || fuzzySubsequence(field, normalized));
    }
    return fields.some(field => field.includes(normalized));
  }

  function searchItemIds(summaries, query, mode = "literal") {
    return (Array.isArray(summaries) ? summaries : [])
      .filter(summary => itemMatches(summary, query, mode))
      .map(summary => summary.item_id);
  }

  function visibleItemQueue(summaries, orderedItemIds = null, filter = "all") {
    const rows = Array.isArray(summaries) ? summaries : [];
    const selectedFilter = FILTERS.has(filter) ? filter : "all";
    const byId = new Map(rows.map(summary => [summary.item_id, summary]));
    const source = Array.isArray(orderedItemIds) ? orderedItemIds : rows.map(row => row.item_id);
    const result = [];
    const seen = new Set();
    for (const itemId of source) {
      const summary = byId.get(itemId);
      if (!summary || seen.has(itemId)) continue;
      const confirmed = summary.status === "confirmed";
      const matches = selectedFilter === "open"
        ? !confirmed
        : selectedFilter === "complete"
          ? confirmed
          : selectedFilter === "reject"
            ? confirmed && summary.disposition === "reject"
            : selectedFilter === "defer"
              ? confirmed && summary.disposition === "defer"
              : true;
      if (!matches) continue;
      seen.add(itemId);
      result.push(itemId);
    }
    return result;
  }

  function nextUnfinishedItemId(queue, summaries, currentItemId = null) {
    if (!Array.isArray(queue) || !queue.length) return null;
    const byId = new Map((summaries || []).map(row => [row.item_id, row]));
    const currentIndex = queue.indexOf(currentItemId);
    const start = currentIndex >= 0 ? currentIndex + 1 : 0;
    for (let offset = 0; offset < queue.length; offset += 1) {
      const itemId = queue[(start + offset) % queue.length];
      const summary = byId.get(itemId);
      if (!summary || summary.status !== "confirmed") return itemId;
    }
    return null;
  }

  function validateDecision(item, decision, {confirm = false} = {}) {
    const errors = {};
    if (!Object.prototype.hasOwnProperty.call(DISPOSITION_LABELS, decision.disposition)) {
      errors.disposition = "请选择接受、驳回或暂缓";
    }
    const binary = new Set([null, "pass", "fail"]);
    for (const field of ["relevance", "boundary", "sense_fit", "swap_incompatibility", "no_hit_verified"]) {
      if (!binary.has(decision[field])) errors[field] = "请选择通过或不通过";
    }
    if (![null, "good", "usable", "poor"].includes(decision.definition_quality)) {
      errors.definition_quality = "请选择定义质量";
    }
    const tags = Array.isArray(decision.pragmatic_tags) ? decision.pragmatic_tags : [];
    if (new Set(tags).size !== tags.length || tags.some(tag => !TAG_LABELS[tag])) {
      errors.pragmatic_tags = "语用标签无效";
    }
    const active = confirm && decision.disposition !== "defer";
    let failed = false;
    if (item.audit_kind === "lex_hit") {
      for (const field of ["relevance", "boundary", "sense_fit", "swap_incompatibility"]) {
        if (active && decision[field] === null) errors[field] = "确认前必须完成此项";
        if (decision[field] === "fail") failed = true;
      }
      if (active && decision.definition_quality === null) errors.definition_quality = "确认前必须完成此项";
      if (decision.definition_quality === "poor") failed = true;
      if (decision.no_hit_verified !== null) errors.no_hit_verified = "命中项不能填写 no-hit 核验";
    } else {
      for (const field of ["relevance", "boundary", "sense_fit", "swap_incompatibility", "definition_quality"]) {
        if (decision[field] !== null) errors[field] = "no-hit 项不填写词典维度";
      }
      if (active && decision.no_hit_verified === null) errors.no_hit_verified = "确认前必须完成 no-hit 核验";
      failed = decision.no_hit_verified === "fail";
    }
    if (confirm && decision.disposition === "accept" && failed) {
      errors.disposition = "存在不通过/较差维度，不能接受";
    }
    if (confirm && decision.disposition === "reject" && !failed) {
      errors.disposition = "驳回必须至少标记一个不通过/较差维度";
    }
    if (confirm && decision.disposition === "reject" && !String(decision.notes || "").trim()) {
      errors.notes = "驳回时请简要说明原因";
    }
    return errors;
  }

  const api = {
    BINARY_LABELS,
    DEFINITION_LABELS,
    DISPOSITION_BY_KEY,
    DISPOSITION_LABELS,
    LEX_DIMENSION_SHORTCUTS,
    NO_HIT_DIMENSION_SHORTCUTS,
    TAG_LABELS,
    TAG_SHORTCUT_LABELS,
    TAG_SHORTCUTS,
    decisionFields,
    defaultDecision,
    dimensionShortcutForCode,
    dispositionForKey,
    itemMatches,
    nextUnfinishedItemId,
    searchItemIds,
    tagShortcutForCode,
    validateDecision,
    visibleItemQueue,
  };
  global.PilotInputAuditCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof globalThis === "undefined" ? this : globalThis));
