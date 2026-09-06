(function initG3ReviewCore(global) {
  "use strict";

  const Common = global.ReviewCore || (
    typeof require === "function"
      ? require("../wp3_candidate_review_ui/core.js")
      : null
  );
  if (!Common) throw new Error("shared review core is unavailable");

  const ACTION_LABELS = Object.freeze({
    accept: "接受",
    edit: "修订",
    reject: "驳回",
    defer: "暂缓",
  });

  const ACTION_BY_KEY = Object.freeze({
    "1": "accept",
    "2": "edit",
    "3": "reject",
    "4": "defer",
  });

  const FAMILY_LABELS = Object.freeze({
    known_variant: "已知书写变体",
    phonetic_variant: "语音形式变体",
    orthographic_variant: "字形形式变体",
  });

  const FILTERS = new Set(["all", "open", "complete", "defer"]);

  function actionForKey(key) {
    return ACTION_BY_KEY[String(key)] || null;
  }

  function proposalFromItem(item) {
    return {
      surface: String(item.surface || ""),
      canonical: String(item.canonical || ""),
      family: item.proposed_family,
      phonetic_scan_enabled: Boolean(item.phonetic_scan_enabled),
      evidence_ids: [...(item.evidence_ids || [])],
    };
  }

  function decisionFields(decision) {
    return {
      surface: String(decision.surface || ""),
      canonical: String(decision.canonical || ""),
      family: decision.family,
      phonetic_scan_enabled: Boolean(decision.phonetic_scan_enabled),
      evidence_ids: [...(decision.evidence_ids || [])],
    };
  }

  function sameStringArray(left, right) {
    return Array.isArray(left) && Array.isArray(right) &&
      left.length === right.length && left.every((value, index) => value === right[index]);
  }

  function sameProposal(left, right) {
    return left.surface === right.surface &&
      left.canonical === right.canonical &&
      left.family === right.family &&
      left.phonetic_scan_enabled === right.phonetic_scan_enabled &&
      sameStringArray(left.evidence_ids, right.evidence_ids);
  }

  function decisionForAction(action, item, current = {}) {
    if (!Object.prototype.hasOwnProperty.call(ACTION_LABELS, action)) {
      return Common.clone(current);
    }
    const next = {
      action,
      ...decisionFields(current),
      notes: String(current.notes || ""),
    };
    if (action !== "edit") Object.assign(next, proposalFromItem(item));
    return next;
  }

  function normalizedSearchFields(summary) {
    return [
      summary.item_id,
      summary.surface,
      summary.canonical,
      summary.proposed_family,
      summary.action,
      ...(summary.publishers || []),
      ...(summary.source_roles || []),
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
    const source = Array.isArray(orderedItemIds)
      ? orderedItemIds
      : rows.map(summary => summary.item_id);
    const seen = new Set();
    const queue = [];
    for (const itemId of source) {
      const summary = byId.get(itemId);
      if (!summary || seen.has(itemId)) continue;
      const confirmed = summary.status === "confirmed";
      const matches = selectedFilter === "open"
        ? !confirmed
        : selectedFilter === "complete"
          ? confirmed
          : selectedFilter === "defer"
            ? confirmed && summary.action === "defer"
            : true;
      if (!matches) continue;
      seen.add(itemId);
      queue.push(itemId);
    }
    return queue;
  }

  function nextUnfinishedItemId(queue, summaries, currentItemId = null) {
    const itemIds = Array.isArray(queue) ? queue : [];
    if (!itemIds.length) return null;
    const byId = new Map(
      (Array.isArray(summaries) ? summaries : [])
        .map(summary => [summary.item_id, summary]),
    );
    const currentIndex = itemIds.indexOf(currentItemId);
    const start = currentIndex >= 0 ? currentIndex + 1 : 0;
    for (let offset = 0; offset < itemIds.length; offset += 1) {
      const itemId = itemIds[(start + offset) % itemIds.length];
      const summary = byId.get(itemId);
      if (summary && summary.status !== "confirmed") return itemId;
    }
    return null;
  }

  function validateDecision(item, evidence, decision) {
    const errors = {};
    if (!Object.prototype.hasOwnProperty.call(ACTION_LABELS, decision.action)) {
      errors.action = "请选择接受、修订、驳回或暂缓";
    }
    const surface = String(decision.surface || "");
    const canonical = String(decision.canonical || "");
    if (!surface.trim()) errors.surface = "surface 不能为空";
    else if (surface !== surface.trim()) errors.surface = "surface 不能包含首尾空格";
    if (!canonical.trim()) errors.canonical = "canonical 不能为空";
    else if (canonical !== canonical.trim()) errors.canonical = "canonical 不能包含首尾空格";
    if (surface.trim() && canonical.trim() && surface.trim() === canonical.trim()) {
      errors.canonical = "surface 必须与 canonical 不同";
    }
    if (!Object.prototype.hasOwnProperty.call(FAMILY_LABELS, decision.family)) {
      errors.family = "请选择有效的关系 family";
    }

    const allowedEvidence = new Set((item.evidence_ids || []).map(String));
    const evidenceIds = Array.isArray(decision.evidence_ids)
      ? decision.evidence_ids.map(String)
      : [];
    if (!evidenceIds.length) {
      errors.evidence_ids = "至少选择一条冻结证据";
    } else if (
      new Set(evidenceIds).size !== evidenceIds.length ||
      evidenceIds.some(evidenceId => !allowedEvidence.has(evidenceId))
    ) {
      errors.evidence_ids = "证据选择包含重复项或不属于当前审核项";
    }

    const proposed = proposalFromItem(item);
    const selected = {
      surface: surface.trim(),
      canonical: canonical.trim(),
      family: decision.family,
      phonetic_scan_enabled: Boolean(decision.phonetic_scan_enabled),
      evidence_ids: evidenceIds,
    };
    if (["accept", "reject", "defer"].includes(decision.action) && !sameProposal(selected, proposed)) {
      errors.proposal = `${ACTION_LABELS[decision.action]}必须原样保留提案字段`;
    }
    if (decision.action === "edit" && sameProposal(selected, proposed)) {
      errors.proposal = "修订必须至少改变一个提案字段";
    }

    if (["accept", "edit"].includes(decision.action) && !errors.evidence_ids) {
      const evidenceById = new Map(
        (Array.isArray(evidence) ? evidence : []).map(row => [row.evidence_id, row]),
      );
      const selectedEvidence = evidenceIds.map(evidenceId => evidenceById.get(evidenceId));
      if (selectedEvidence.some(row => !row)) {
        errors.evidence_ids = "冻结证据内容缺失，请刷新页面";
      } else {
        const sourceIds = new Set(selectedEvidence.map(row => row.source_id));
        if (sourceIds.size !== 1) {
          errors.evidence_ids = "一次接受或修订只能采用同一公开来源";
        }
        const componentIds = new Set(
          selectedEvidence
            .filter(row => row.relation_contract === "single-quote-surface-and-canonical/v2")
            .map(row => row.component_id),
        );
        if (componentIds.size > 1 || componentIds.has(undefined) || componentIds.has(null)) {
          errors.evidence_ids = "v2 关系证据必须来自同一冻结 component";
        }
        const quotes = selectedEvidence.map(row => String(row.quote || ""));
        const strictSingleQuote = selectedEvidence.some(
          row => row.relation_contract === "single-quote-surface-and-canonical/v2",
        );
        const replays = strictSingleQuote
          ? quotes.some(quote => quote.includes(selected.surface) && quote.includes(selected.canonical))
          : quotes.some(quote => quote.includes(selected.surface)) &&
            quotes.some(quote => quote.includes(selected.canonical));
        if (!replays) {
          errors.proposal = "surface 与 canonical 必须能在所选冻结证据中重放";
        }
      }
    }
    return errors;
  }

  const api = {
    ACTION_BY_KEY,
    ACTION_LABELS,
    FAMILY_LABELS,
    actionForKey,
    decisionForAction,
    itemMatches,
    nextUnfinishedItemId,
    proposalFromItem,
    sameProposal,
    searchItemIds,
    validateDecision,
    visibleItemQueue,
  };

  global.G3ReviewCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof globalThis === "undefined" ? this : globalThis));
