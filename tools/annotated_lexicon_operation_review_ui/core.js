(function initOperationCore(global) {
  "use strict";
  const Common = global.ReviewCore || (typeof require === "function" ? require("../wp3_candidate_review_ui/core.js") : null);
  if (!Common) throw new Error("shared review core is unavailable");
  const CATEGORIES = Object.freeze(["Racism", "Sexism", "LGBTQ", "Region", "others"]);
  const KINDS = Object.freeze({existing: "已有词条修复", duplicate_merge: "重复组合并", new_entry: "必要新增词条"});
  const RESOLUTIONS = Object.freeze({approve: "批准提案", revise: "修改后采用", defer: "暂缓", reject: "不采用提案"});
  const TARGETS = Object.freeze({left: "左侧上下文", right: "右侧上下文", surface: "候选词面", context: "左文＋词面＋右文"});
  const FLAGS = Object.freeze({sense_unresolved: "义项未定", empty_definition_blocks_approval: "空释义将阻止采用", no_sense_level_disambiguation: "不含逐处义项消歧", duplicate_merge: "重复组合并", candidate_eligibility_rule_draft: "候选资格规则待审", multiple_senses: "多义项", negative_only_evidence: "目前仅有反例", policy_draft_not_semantic_proof: "规则提案不是语义证明", semantic_boundary_review: "语义边界待审", context_window_limitation: "注意上下文窗口限制", policy_unresolved: "资格规则未决", known_regression_until_policy_resolved: "规则解决前存在已知回归", event_link_not_established: "事件关联尚未确定", user_adjudication_required: "需要人工复议", new_surface_draft: "新词面提案", variant_link_not_approved: "变体归属尚未批准", reviewer_semantic_confirmation: "需人工核实语义", raw_surface_must_remain_unchanged: "保留原词面字符", regional_stereotype_not_fact: "地域刻板印象并非事实", abbreviation_expansion_not_asserted: "缩写展开尚未确定", no_absolute_neutrality: "不可绝对认定中性", category_needs_review: "类别待核实", ordinary_reference_possible: "可能为普通指代", span_gold_change_requires_adjudication: "改变冻结span需复议"});
  function defaultDecision(item) { return {resolution: null, entry: Common.clone(item.proposed_entry || null), notes: ""}; }
  function decisionFields(item, stored) {
    const result = defaultDecision(item);
    if (stored) {
      result.resolution = stored.resolution || null;
      if (Object.prototype.hasOwnProperty.call(stored, "entry")) result.entry = Common.clone(stored.entry);
      result.notes = String(stored.notes || "");
    }
    return result;
  }
  function canonical(value) {
    if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
    if (value && typeof value === "object") return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
    return JSON.stringify(value);
  }
  function entriesEqual(a, b) { return canonical(a) === canonical(b); }
  function isLocked(status) { return ["confirmed", "deferred"].includes(status); }
  function reviewShortcut(event, {dialogOpen = false, busy = false, loaded = true, locked = false} = {}) {
    if (event.defaultPrevented || event.isComposing || event.keyCode === 229 || dialogOpen || busy || !loaded || event.altKey || event.repeat) return null;
    const key = String(event.key || "").toLowerCase();
    if (event.ctrlKey || event.metaKey) {
      if (event.shiftKey || locked) return null;
      return key === "s" ? "save" : key === "enter" ? "confirm" : null;
    }
    if (Common.isTextEntry(event.target) || (event.shiftKey && key !== "?")) return null;
    const action = {"1": "approve", "2": "revise", "3": "defer", "4": "reject", "[": "previous-item", "]": "next-item", "?": "help"}[key] || null;
    return locked && Object.prototype.hasOwnProperty.call(RESOLUTIONS, action) ? null : action;
  }
  function validateDecision(item, decision, {confirm = false} = {}) {
    const errors = {};
    if (decision.resolution !== null && !Object.prototype.hasOwnProperty.call(RESOLUTIONS, decision.resolution)) errors.resolution = "未知的审核决定";
    if (confirm && !decision.resolution) errors.resolution = "请选择本条的审核决定";
    if (Array.from(String(decision.notes || "")).length > 2000) errors.notes = "备注不能超过 2000 字";
    if (confirm && ["defer", "reject"].includes(decision.resolution) && !String(decision.notes || "").trim()) errors.notes = "暂缓或不采用提案必须说明理由";
    const accepted = ["approve", "revise"].includes(decision.resolution);
    if (confirm && accepted && (item.open_questions || []).length && !String(decision.notes || "").trim()) errors.notes = "本项存在待判断问题，采用时请备注人工判断或核实依据";
    const entry = decision.entry;
    if (confirm && accepted && !entry) errors.entry = "采用提案需要完整词条";
    if (!entry) return errors;
    if (entry.term !== item.term || (item.proposed_entry && entry.lexicon_id !== item.proposed_entry.lexicon_id)) errors.entry = "词条 term 和稳定 ID 不可修改";
    if (!Array.isArray(entry.variants) || entry.variants.some(row => typeof row !== "string" || !row.trim())) errors.variants = "变体须为非空词面列表";
    else if (new Set(entry.variants).size !== entry.variants.length) errors.variants = "变体不能重复";
    if (!Array.isArray(entry.senses)) errors.senses = "义项须为列表";
    else {
      const ids = entry.senses.map(sense => sense.sense_id);
      if (ids.some(id => !id) || new Set(ids).size !== ids.length) errors.senses = "义项 ID 不得为空或重复";
      if (confirm && accepted && !entry.senses.length) errors.senses = "至少需要一个完整义项；无法确定时请选择暂缓";
      for (const [index, sense] of entry.senses.entries()) {
        if (typeof sense.definition !== "string") errors.senses = `义项 ${index + 1} 的释义应为文本`;
        if (confirm && accepted && !String(sense.definition || "").trim()) errors.senses = `义项 ${index + 1} 释义为空，不能确认采用；无法确定时请选择暂缓`;
        if (!Array.isArray(sense.categories) || sense.categories.some(category => !CATEGORIES.includes(category))) errors.senses = `义项 ${index + 1} 的类别无效`;
        else if (confirm && accepted && !sense.categories.length) errors.senses = `义项 ${index + 1} 至少选择一个词典类别`;
      }
    }
    const policy = entry.match_policy;
    if (confirm && accepted && (item.flags || []).includes("policy_unresolved") && entriesEqual(policy, item.proposed_entry.match_policy)) errors.policy = "本项资格规则尚未解决，请制定匹配条件后修订采用，或暂缓；不能原样采用未验证规则";
    if (!policy || !Array.isArray(policy.require_any) || !Array.isArray(policy.exclude_any)) errors.policy = "匹配条件须包含 require_any / exclude_any 列表";
    else {
      const rules = [...policy.require_any, ...policy.exclude_any];
      if (rules.length > 8) errors.policy = "每个词条最多 8 条上下文规则";
      if (new Set(rules.map(rule => rule.rule_id)).size !== rules.length) errors.policy = "规则 ID 不能重复";
      for (const rule of rules) {
        if (!rule.rule_id || !Object.prototype.hasOwnProperty.call(TARGETS, rule.target)) errors.policy = "规则 ID 或 target 无效";
        if (typeof rule.pattern !== "string" || Array.from(rule.pattern).length > 256) errors.policy = "规则 pattern 必须是最多 256 字的文本";
        if (confirm && accepted && !String(rule.pattern || "").trim()) errors.policy = "不能采用空正则；删除空规则或选择暂缓";
      }
    }
    if (confirm && decision.resolution === "approve" && !entriesEqual(entry, item.proposed_entry)) errors.resolution = "词条已修改，请选择“修改后采用”，或明确恢复原提案后批准";
    return errors;
  }
  function itemMatches(summary, query, mode = "literal") {
    const needle = String(query || "").trim().toLocaleLowerCase("zh-CN");
    const fields = [summary.item_id, summary.term, summary.query_preview, summary.operation_kind, KINDS[summary.operation_kind], ...(summary.flags || [])].map(value => String(value || "").toLocaleLowerCase("zh-CN"));
    if (!needle) return true;
    if (mode === "all_terms") return needle.split(/\s+/u).every(term => fields.some(field => field.includes(term)));
    if (mode === "fuzzy") return fields.some(field => {
      let cursor = 0;
      const characters = Array.from(needle.replace(/\s+/gu, ""));
      for (const character of Array.from(field)) if (character === characters[cursor]) cursor += 1;
      return cursor === characters.length;
    });
    return fields.some(field => field.includes(needle));
  }
  function visibleItemQueue(summaries, query = "", mode = "literal", filter = "all") {
    return (summaries || []).filter(row => itemMatches(row, query, mode) && (
      filter === "open" ? !isLocked(row.status) : filter === "complete" ? row.status === "confirmed" :
      filter === "deferred" ? row.status === "deferred" : Object.prototype.hasOwnProperty.call(KINDS, filter) ? row.operation_kind === filter : true
    )).map(row => row.item_id);
  }
  function nextUnfinishedItemId(queue, summaries, currentId = null) {
    const byId = new Map((summaries || []).map(row => [row.item_id, row]));
    const start = Math.max(0, queue.indexOf(currentId) + 1);
    for (let offset = 0; offset < queue.length; offset += 1) {
      const id = queue[(start + offset) % queue.length];
      if (!byId.has(id) || !isLocked(byId.get(id).status)) return id;
    }
    return null;
  }
  // The protocol uses Unicode codepoints, not UTF-16 offsets. Rendering never
  // interprets evidence or definitions as markup, including untrusted snippets.
  function evidenceSegments(query, expectedSpans) {
    const chars = Array.from(String(query || ""));
    const spans = (expectedSpans || []).filter(row => Array.isArray(row.span) && row.span.length === 2 && Number.isInteger(row.span[0]) && Number.isInteger(row.span[1]) && row.span[0] >= 0 && row.span[0] < row.span[1] && row.span[1] <= chars.length && chars.slice(row.span[0], row.span[1]).join("") === row.surface).sort((a, b) => a.span[0] - b.span[0]);
    const result = [];
    let cursor = 0;
    for (const row of spans) {
      if (row.span[0] < cursor) continue;
      if (row.span[0] > cursor) result.push({text: chars.slice(cursor, row.span[0]).join(""), highlighted: false});
      result.push({text: row.surface, highlighted: true, span: row.span});
      cursor = row.span[1];
    }
    if (cursor < chars.length) result.push({text: chars.slice(cursor).join(""), highlighted: false});
    return result;
  }
  function safeEvidenceUrl(value) {
    try { const url = new URL(value); return url.protocol === "https:" || url.protocol === "http:" ? url.href : null; }
    catch (_) { return null; }
  }
  const api = {CATEGORIES, KINDS, RESOLUTIONS, TARGETS, FLAGS, defaultDecision, decisionFields, entriesEqual, isLocked, reviewShortcut, validateDecision, itemMatches, visibleItemQueue, nextUnfinishedItemId, evidenceSegments, safeEvidenceUrl};
  global.OperationReviewCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof globalThis === "undefined" ? this : globalThis));
