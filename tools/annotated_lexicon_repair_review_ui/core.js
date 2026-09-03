(function initSpanGoldCore(global) {
  "use strict";

  const Common = global.ReviewCore || (
    typeof require === "function"
      ? require("../wp3_candidate_review_ui/core.js")
      : null
  );
  if (!Common) throw new Error("shared review core is unavailable");

  const COHORT_LABELS = Object.freeze({
    repair_target: "修复目标",
    accepted_control: "接受对照",
    approved_adjudication: "已复议对照",
  });
  const FILTERS = new Set(["all", "open", "complete", ...Object.keys(COHORT_LABELS)]);

  function defaultDecision(item) {
    const candidateActions = {};
    for (const candidate of item.candidates || []) candidateActions[candidate.candidate_id] = null;
    return {candidate_actions: candidateActions, additional_spans: [], notes: ""};
  }

  function decisionFields(item, stored = {}) {
    const result = defaultDecision(item);
    for (const candidateId of Object.keys(result.candidate_actions)) {
      if (stored.candidate_actions && Object.prototype.hasOwnProperty.call(stored.candidate_actions, candidateId)) {
        result.candidate_actions[candidateId] = stored.candidate_actions[candidateId];
      }
    }
    if (Array.isArray(stored.additional_spans)) result.additional_spans = Common.clone(stored.additional_spans);
    result.notes = String(stored.notes || "");
    return result;
  }

  function serializeAdditionalSpans(rows) {
    return (rows || []).map(row => [row.start, row.end, row.surface, row.reason].join(" | ")).join("\n");
  }

  function parseAdditionalSpans(text) {
    const rows = [];
    const errors = [];
    for (const [index, rawLine] of String(text || "").split(/\r?\n/u).entries()) {
      const line = rawLine.trim();
      if (!line) continue;
      const fields = line.split("|").map(value => value.trim());
      if (fields.length !== 4) {
        errors.push(`第 ${index + 1} 行必须是 start | end | surface | reason`);
        continue;
      }
      const start = Number(fields[0]);
      const end = Number(fields[1]);
      if (!Number.isInteger(start) || !Number.isInteger(end)) {
        errors.push(`第 ${index + 1} 行的 start/end 必须是整数`);
        continue;
      }
      rows.push({start, end, surface: fields[2], reason: fields[3]});
    }
    return {rows, errors};
  }

  function validateDecision(item, decision, {confirm = false} = {}) {
    const errors = {};
    const candidateIds = (item.candidates || []).map(row => row.candidate_id);
    if (!decision.candidate_actions || Object.keys(decision.candidate_actions).length !== candidateIds.length) {
      errors.candidate_actions = "候选清单不一致，请刷新页面";
    } else {
      const undecided = candidateIds.filter(candidateId => !["keep", "drop"].includes(decision.candidate_actions[candidateId]));
      if (confirm && undecided.length) errors.candidate_actions = `还有 ${undecided.length} 个候选未决定`;
    }
    if (!Array.isArray(decision.additional_spans)) errors.additional_spans = "额外 span 格式无效";
    if (String(decision.notes || "").length > 2000) errors.notes = "备注不能超过 2000 字";

    const codepoints = Array.from(String(item.query_content || ""));
    const expected = [];
    for (const candidate of item.candidates || []) {
      if (decision.candidate_actions[candidate.candidate_id] === "keep") {
        expected.push({start: candidate.span[0], end: candidate.span[1], surface: candidate.surface});
      }
    }
    for (const [index, row] of (decision.additional_spans || []).entries()) {
      if (!Number.isInteger(row.start) || !Number.isInteger(row.end) || row.start < 0 || row.start >= row.end || row.end > codepoints.length) {
        errors.additional_spans = `额外 span 第 ${index + 1} 行边界无效`;
        continue;
      }
      if (codepoints.slice(row.start, row.end).join("") !== row.surface) {
        errors.additional_spans = `额外 span 第 ${index + 1} 行文本与边界不一致`;
      }
      if (!String(row.reason || "").trim()) errors.additional_spans = `额外 span 第 ${index + 1} 行缺少理由`;
      expected.push(row);
    }
    expected.sort((left, right) => left.start - right.start || left.end - right.end);
    for (let index = 1; index < expected.length; index += 1) {
      if (expected[index - 1].end > expected[index].start) {
        errors.expected_spans = "最终保留的 span 不能重叠";
        break;
      }
    }
    return errors;
  }

  function normalizedFields(summary) {
    return [
      summary.item_id,
      summary.source_item_id,
      summary.query_preview,
      summary.cohort,
      ...(summary.surfaces || []),
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
    const fields = normalizedFields(summary);
    if (mode === "all_terms") {
      return normalized.split(/\s+/u).filter(Boolean).every(term => fields.some(field => field.includes(term)));
    }
    if (mode === "fuzzy") return fields.some(field => field.includes(normalized) || fuzzySubsequence(field, normalized));
    return fields.some(field => field.includes(normalized));
  }

  function visibleItemQueue(summaries, query = "", mode = "literal", filter = "all") {
    const selected = FILTERS.has(filter) ? filter : "all";
    return (summaries || []).filter(summary => {
      if (!itemMatches(summary, query, mode)) return false;
      if (selected === "open") return summary.status !== "confirmed";
      if (selected === "complete") return summary.status === "confirmed";
      if (Object.prototype.hasOwnProperty.call(COHORT_LABELS, selected)) return summary.cohort === selected;
      return true;
    }).map(summary => summary.item_id);
  }

  function nextUnfinishedItemId(queue, summaries, currentItemId = null) {
    if (!queue.length) return null;
    const byId = new Map((summaries || []).map(summary => [summary.item_id, summary]));
    const currentIndex = queue.indexOf(currentItemId);
    const start = currentIndex >= 0 ? currentIndex + 1 : 0;
    for (let offset = 0; offset < queue.length; offset += 1) {
      const itemId = queue[(start + offset) % queue.length];
      const summary = byId.get(itemId);
      if (!summary || summary.status !== "confirmed") return itemId;
    }
    return null;
  }

  const api = {
    COHORT_LABELS,
    decisionFields,
    defaultDecision,
    itemMatches,
    nextUnfinishedItemId,
    parseAdditionalSpans,
    serializeAdditionalSpans,
    validateDecision,
    visibleItemQueue,
  };
  global.SpanGoldCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof globalThis === "undefined" ? this : globalThis));
