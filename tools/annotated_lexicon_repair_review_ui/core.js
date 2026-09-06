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

  // DOM Range offsets count UTF-16 units; the persisted protocol counts Unicode
  // codepoints. Never locate an occurrence with indexOf(surface): it may repeat.
  function spanFromUtf16Offsets(text, startOffset, endOffset) {
    if (!Number.isInteger(startOffset) || !Number.isInteger(endOffset) ||
        startOffset < 0 || startOffset >= endOffset || endOffset > text.length) return null;
    const splitsSurrogate = offset => offset > 0 && offset < text.length &&
      /[\uD800-\uDBFF]/u.test(text[offset - 1]) && /[\uDC00-\uDFFF]/u.test(text[offset]);
    if (splitsSurrogate(startOffset) || splitsSurrogate(endOffset)) return null;
    return {
      start: Array.from(text.slice(0, startOffset)).length,
      end: Array.from(text.slice(0, endOffset)).length,
      surface: text.slice(startOffset, endOffset),
    };
  }

  function selectedQuerySpan(root, selection) {
    if (!selection || selection.rangeCount !== 1 || selection.isCollapsed) return null;
    const range = selection.getRangeAt(0);
    if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) return null;
    const prefix = range.cloneRange();
    prefix.selectNodeContents(root);
    prefix.setEnd(range.startContainer, range.startOffset);
    const startOffset = prefix.toString().length;
    prefix.setEnd(range.endContainer, range.endOffset);
    const span = spanFromUtf16Offsets(root.textContent, startOffset, prefix.toString().length);
    return span && span.surface.trim() && span.surface === range.toString() ? span : null;
  }

  function additionalSpanError(item, decision, span, editingIndex = -1) {
    if ((item.candidates || []).some(candidate => candidate.span[0] === span.start && candidate.span[1] === span.end)) {
      return "此位置已列为候选，请直接决定该 occurrence，不要重复补充。";
    }
    const rows = decision.additional_spans.filter((row, index) => index !== editingIndex);
    if (rows.some(row => row.start === span.start && row.end === span.end)) return "此位置已经补充，可在额外 span 列表中修改理由。";
    if (rows.length >= 32) return "每个 case 最多补充 32 个 span。";
    if (rows.some(row => row.start < span.end && span.start < row.end)) return "此片段与已补充的 span 重叠，请先调整或移除原补充项。";
    if ((item.candidates || []).some(candidate => decision.candidate_actions[candidate.candidate_id] === "keep" &&
        candidate.span[0] < span.end && span.start < candidate.span[1])) {
      return "此片段与已保留候选重叠，请先删除不应保留的候选。";
    }
    return "";
  }

  function reviewShortcut(event, {dialogOpen = false, busy = false, loaded = true, locked = false} = {}) {
    if (event.defaultPrevented || event.isComposing || event.keyCode === 229 || dialogOpen || busy || !loaded || event.altKey) return null;
    const key = String(event.key || "").toLowerCase();
    if (event.ctrlKey || event.metaKey) {
      if (event.shiftKey || event.repeat || locked) return null;
      return key === "s" ? "save" : key === "enter" ? "confirm" : null;
    }
    if (Common.isTextEntry(event.target)) return null;
    if (event.shiftKey && key !== "?" && key !== "q" && key !== "a") return null;
    const action = {
      "1": "keep-all", "2": "drop-all", "3": "clear-all",
      arrowup: "previous-candidate", arrowdown: "next-candidate",
      q: "keep", a: "drop", "[": "previous-item", "]": "next-item", "?": "help",
    }[key] || null;
    if (event.repeat && !["previous-candidate", "next-candidate"].includes(action)) return null;
    if (locked && ["keep-all", "drop-all", "clear-all", "keep", "drop"].includes(action)) return null;
    return action;
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
    const additional = Array.isArray(decision.additional_spans) ? decision.additional_spans : [];
    if (!Array.isArray(decision.additional_spans) || additional.length > 32) errors.additional_spans = "额外 span 必须是最多 32 项的列表";
    if (String(decision.notes || "").length > 2000) errors.notes = "备注不能超过 2000 字";

    const codepoints = Array.from(String(item.query_content || ""));
    const expected = [];
    for (const candidate of item.candidates || []) {
      if (decision.candidate_actions[candidate.candidate_id] === "keep") {
        expected.push({start: candidate.span[0], end: candidate.span[1], surface: candidate.surface});
      }
    }
    for (const [index, row] of additional.entries()) {
      if (!Number.isInteger(row.start) || !Number.isInteger(row.end) || row.start < 0 || row.start >= row.end || row.end > codepoints.length) {
        errors.additional_spans = `额外 span 第 ${index + 1} 行边界无效`;
        continue;
      }
      if (codepoints.slice(row.start, row.end).join("") !== row.surface) {
        errors.additional_spans = `额外 span 第 ${index + 1} 行文本与边界不一致`;
      }
      if (!String(row.reason || "").trim()) errors.additional_spans = `额外 span 第 ${index + 1} 行缺少理由`;
      if (Array.from(String(row.reason || "").trim()).length > 500) errors.additional_spans = `额外 span 第 ${index + 1} 行理由不能超过 500 字`;
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
    additionalSpanError,
    decisionFields,
    defaultDecision,
    itemMatches,
    nextUnfinishedItemId,
    reviewShortcut,
    selectedQuerySpan,
    spanFromUtf16Offsets,
    validateDecision,
    visibleItemQueue,
  };
  global.SpanGoldCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof globalThis === "undefined" ? this : globalThis));
