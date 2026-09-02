(function initReviewCore(global) {
  "use strict";

  const ACTION_LABELS = Object.freeze({
    accept: "接受",
    trim: "缩短边界",
    expand: "扩大边界",
    split: "拆分",
    reject: "驳回",
    defer: "暂缓",
  });

  const ROUTE_LABELS = Object.freeze({
    A_candidate: "A · 稳定核心候选",
    B_candidate: "B · 需上下文候选",
    C_candidate: "C · 待核实候选",
  });

  const ROUTE_DESCRIPTIONS = Object.freeze({
    A_candidate: "有依据的核心义稳定可迁移，没有未解决的主要替代义。",
    B_candidate: "至少一个解释有依据，但多义、语用或普通义必须结合上下文。",
    C_candidate: "表达可能有解释价值，但含义、映射、变体、边界或证据仍未解决。",
  });

  const QUICK_ROUTE_BY_KEY = Object.freeze({
    "1": "A_candidate",
    "2": "B_candidate",
    "3": "C_candidate",
  });

  const SEARCH_MODE_ORDER = Object.freeze(["literal", "all_terms", "fuzzy"]);

  const SEARCH_MODE_LABELS = Object.freeze({
    literal: "连续",
    all_terms: "多词",
    fuzzy: "模糊",
  });

  const REASON_LABELS = Object.freeze({
    stable_core_candidate: "稳定核心释义候选",
    context_required: "需要结合上下文",
    evidence_required: "需要补充证据",
    transparent: "普通字面或常规组合",
    ordinary_identity_or_name: "普通身份、名称或称谓",
    generic_insult: "通用辱骂或负面词义",
    fragment: "非自足片段",
    function_word_attached: "附带功能词",
    sentence_level: "完整句、分句或临时修辞",
    wrong_boundary: "边界过长、过短或位置错误",
    substring_projection: "父短语意义误投射给子串",
    unsupported_sense: "具体含义缺少证据",
    one_off_creation: "一次性创造，无法迁移",
    ambiguous_surface: "表面形式存在歧义",
    context_polysemy: "解释依赖当前上下文",
    quoted_or_reclaimed: "引述、讨论或回收用法",
    evidence_conflict: "独立证据相互冲突",
    variant_unresolved: "变体关系尚未证实",
    non_contiguous_unresolved: "不连续边界协议未解决",
    label_derived: "来自被禁止的任务标签",
    not_fit_attested: "未在 fit 原文出现",
    duplicate: "与已有决定重复",
    no_neutral_gloss: "无法给出中性释义",
    other: "其他原因",
  });

  const REASON_GROUPS = Object.freeze([
    {
      label: "正向分流",
      values: ["stable_core_candidate", "context_required", "evidence_required"],
    },
    {
      label: "透明与排除",
      values: [
        "transparent",
        "ordinary_identity_or_name",
        "generic_insult",
        "fragment",
        "function_word_attached",
        "sentence_level",
      ],
    },
    {
      label: "边界问题",
      values: [
        "wrong_boundary",
        "substring_projection",
        "non_contiguous_unresolved",
      ],
    },
    {
      label: "歧义与证据",
      values: [
        "unsupported_sense",
        "one_off_creation",
        "ambiguous_surface",
        "context_polysemy",
        "quoted_or_reclaimed",
        "evidence_conflict",
        "variant_unresolved",
      ],
    },
    {
      label: "治理问题",
      values: [
        "label_derived",
        "not_fit_attested",
        "duplicate",
        "no_neutral_gloss",
        "other",
      ],
    },
  ]);

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function occurrenceStart(content, surface, ordinal) {
    if (!surface || !Number.isInteger(ordinal) || ordinal < 1) return -1;
    let from = 0;
    let found = -1;
    for (let count = 0; count < ordinal; count += 1) {
      found = content.indexOf(surface, from);
      if (found < 0) return -1;
      from = found + 1;
    }
    return found;
  }

  function occurrenceOrdinal(content, surface, selectedStart) {
    if (!surface || selectedStart < 0) return 0;
    let from = 0;
    let ordinal = 0;
    while (from <= selectedStart) {
      const found = content.indexOf(surface, from);
      if (found < 0 || found > selectedStart) break;
      ordinal += 1;
      if (found === selectedStart) return ordinal;
      from = found + 1;
    }
    return 0;
  }

  function codePointOffset(content, codeUnitOffset) {
    return Array.from(content.slice(0, codeUnitOffset)).length;
  }

  function resultMention(surface = "", occurrence = 1) {
    return {
      surface,
      occurrence_ordinal: occurrence,
    };
  }

  function defaultDiagnosticDecision(proposal) {
    return {
      action: "accept",
      result_mentions: [
        resultMention(proposal.surface, proposal.occurrence_ordinal),
      ],
      reason_codes: ["stable_core_candidate"],
      notes: "",
    };
  }

  function decisionForAction(action, proposal, notes = "") {
    if (action === "accept") return defaultDiagnosticDecision(proposal);
    if (action === "trim" || action === "expand") {
      return {
        action,
        result_mentions: [resultMention()],
        reason_codes: ["wrong_boundary"],
        notes,
      };
    }
    if (action === "split") {
      return {
        action,
        result_mentions: [resultMention(), resultMention()],
        reason_codes: ["wrong_boundary"],
        notes,
      };
    }
    if (action === "defer") {
      return {
        action,
        result_mentions: [],
        reason_codes: ["evidence_required"],
        notes,
      };
    }
    return {
      action: "reject",
      result_mentions: [],
      reason_codes: [],
      notes,
    };
  }

  function routeDefaults(route) {
    return {
      A_candidate: ["stable_core_candidate"],
      B_candidate: ["context_required"],
      C_candidate: ["evidence_required"],
    }[route] || [];
  }

  function quickRouteForKey(key) {
    return QUICK_ROUTE_BY_KEY[String(key)] || null;
  }

  function nextSearchMode(mode) {
    const index = SEARCH_MODE_ORDER.indexOf(mode);
    return index < 0
      ? SEARCH_MODE_ORDER[0]
      : SEARCH_MODE_ORDER[(index + 1) % SEARCH_MODE_ORDER.length];
  }

  function visibleCaseQueue(caseSummaries, orderedCaseIds = null, filter = "all") {
    const summaries = Array.isArray(caseSummaries) ? caseSummaries : [];
    const byId = new Map(summaries.map(summary => [summary.case_id, summary]));
    const source = Array.isArray(orderedCaseIds)
      ? orderedCaseIds
      : summaries.map(summary => summary.case_id);
    const seen = new Set();
    const queue = [];
    for (const caseId of source) {
      const summary = byId.get(caseId);
      if (!summary || seen.has(caseId)) continue;
      const matchesFilter = filter === "open"
        ? !summary.complete
        : filter === "complete"
          ? Boolean(summary.complete)
          : true;
      if (!matchesFilter) continue;
      seen.add(caseId);
      queue.push(caseId);
    }
    return queue;
  }

  function nextUnfinishedCaseId(queue, caseSummaries, currentCaseId = null) {
    const caseIds = Array.isArray(queue) ? queue : [];
    if (!caseIds.length) return null;
    const summaries = new Map(
      (Array.isArray(caseSummaries) ? caseSummaries : [])
        .map(summary => [summary.case_id, summary]),
    );
    const currentIndex = caseIds.indexOf(currentCaseId);
    const start = currentIndex >= 0 ? currentIndex + 1 : 0;
    for (let offset = 0; offset < caseIds.length; offset += 1) {
      const caseId = caseIds[(start + offset) % caseIds.length];
      const summary = summaries.get(caseId);
      if (summary && !summary.complete) return caseId;
    }
    return null;
  }

  function validateMention(content, mention) {
    const errors = {};
    const surface = String(mention.surface || "");
    if (!surface.trim()) errors.surface = "请输入或选择原文片段";
    else if (surface !== surface.trim()) errors.surface = "原文片段不能包含首尾空格";
    if (!Number.isInteger(Number(mention.occurrence_ordinal)) || Number(mention.occurrence_ordinal) < 1) {
      errors.occurrence_ordinal = "出现序号必须是大于 0 的整数";
    } else if (
      mention.surface &&
      occurrenceStart(content, mention.surface, Number(mention.occurrence_ordinal)) < 0
    ) {
      errors.occurrence_ordinal = "原文中找不到对应的第几次出现";
    }
    if (!mention.provisional_route) errors.provisional_route = "请选择暂定去向";
    if (!mention.reason_codes || !mention.reason_codes.length) {
      errors.reason_codes = "至少选择一个判定原因";
    }
    if (mention.reason_codes && mention.reason_codes.includes("other") && !String(mention.notes || "").trim()) {
      errors.notes = "选择“其他原因”时必须填写备注";
    }
    return errors;
  }

  function validateRaw(content, annotation) {
    const mentionErrors = annotation.mentions.map(mention => validateMention(content, mention));
    const errors = {};
    if (mentionErrors.some(row => Object.keys(row).length)) errors.mentions = mentionErrors;
    return errors;
  }

  function resolveMention(content, mention) {
    const start = occurrenceStart(
      content,
      String(mention.surface || ""),
      Number(mention.occurrence_ordinal),
    );
    return {
      start,
      end: start < 0 ? -1 : start + String(mention.surface || "").length,
    };
  }

  function validateDiagnostic(content, proposal, decision) {
    const errors = {};
    if (!Object.prototype.hasOwnProperty.call(ACTION_LABELS, decision.action)) {
      errors.action = "请选择处理操作";
    }
    if (!decision.reason_codes || !decision.reason_codes.length) {
      errors.reason_codes = "至少选择一个判定原因";
    }
    if (
      (decision.action === "defer" || (decision.reason_codes || []).includes("other")) &&
      !String(decision.notes || "").trim()
    ) {
      errors.notes = decision.action === "defer"
        ? "暂缓处理时必须说明待解决问题"
        : "选择“其他原因”时必须填写备注";
    }
    const results = decision.result_mentions || [];
    if (["accept", "trim", "expand"].includes(decision.action) && results.length !== 1) {
      errors.result_mentions = "该操作必须保留一个结果片段";
    }
    if (decision.action === "split" && results.length < 2) {
      errors.result_mentions = "拆分操作至少需要两个结果片段";
    }
    if (["reject", "defer"].includes(decision.action) && results.length) {
      errors.result_mentions = "驳回或暂缓不能保留结果片段";
    }
    const resultErrors = results.map(row => validateMention(content, {
      ...row,
      provisional_route: "A_candidate",
      reason_codes: ["wrong_boundary"],
      notes: "",
    }));
    if (resultErrors.some(row => row.surface || row.occurrence_ordinal)) {
      errors.results = resultErrors;
    }
    const proposalRange = {
      start: occurrenceStart(content, proposal.surface, proposal.occurrence_ordinal),
    };
    proposalRange.end = proposalRange.start + proposal.surface.length;
    const ranges = results.map(row => resolveMention(content, row));
    if (decision.action === "accept" && results.length === 1 &&
        (results[0].surface !== proposal.surface ||
         Number(results[0].occurrence_ordinal) !== Number(proposal.occurrence_ordinal))) {
      errors.result_mentions = "接受必须保留原提案的准确位置";
    }
    if (decision.action === "trim" && ranges.length === 1 && ranges[0].start >= 0 &&
        !(ranges[0].start >= proposalRange.start && ranges[0].end <= proposalRange.end &&
          (ranges[0].start !== proposalRange.start || ranges[0].end !== proposalRange.end))) {
      errors.result_mentions = "缩短后的片段必须严格位于原提案内部";
    }
    if (decision.action === "expand" && ranges.length === 1 && ranges[0].start >= 0 &&
        !(ranges[0].start <= proposalRange.start && ranges[0].end >= proposalRange.end &&
          (ranges[0].start !== proposalRange.start || ranges[0].end !== proposalRange.end))) {
      errors.result_mentions = "扩大后的片段必须严格包含原提案";
    }
    if (decision.action === "split" && ranges.some(range =>
      range.start >= 0 && (range.start < proposalRange.start || range.end > proposalRange.end))) {
      errors.result_mentions = "拆分结果必须全部位于原提案内部";
    }
    return errors;
  }

  function hasErrors(errors) {
    return Boolean(errors && Object.keys(errors).length);
  }

  function isTextEntry(target) {
    if (!target || !target.tagName) return false;
    return ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ||
      Boolean(target.isContentEditable);
  }

  class MutationQueue {
    constructor(execute, onState, onError) {
      this.execute = execute;
      this.onState = onState || (() => {});
      this.onError = onError || (() => {});
      this.pending = null;
      this.inflight = false;
      this.timer = null;
      this.chain = Promise.resolve();
    }

    get dirty() {
      return Boolean(this.pending || this.inflight);
    }

    schedule(task, delay = 700) {
      this.pending = clone(task);
      clearTimeout(this.timer);
      this.onState("dirty");
      this.timer = setTimeout(() => {
        this.flush().catch(error => this.onError(error));
      }, delay);
    }

    async flush() {
      clearTimeout(this.timer);
      this.timer = null;
      if (!this.pending) return this.chain;
      const task = this.pending;
      this.pending = null;
      const run = async () => {
        this.inflight = true;
        this.onState("saving");
        try {
          const result = await this.execute(task);
          this.inflight = false;
          this.onState(this.pending ? "dirty" : "saved");
          return result;
        } catch (error) {
          this.inflight = false;
          if (!this.pending) this.pending = task;
          this.onState("error");
          throw error;
        }
      };
      this.chain = this.chain.catch(() => undefined).then(run);
      return this.chain;
    }

    async runNow(task) {
      clearTimeout(this.timer);
      this.timer = null;
      this.pending = clone(task);
      return this.flush();
    }

    discard() {
      clearTimeout(this.timer);
      this.timer = null;
      this.pending = null;
      this.onState(this.inflight ? "saving" : "idle");
    }
  }

  const api = {
    ACTION_LABELS,
    QUICK_ROUTE_BY_KEY,
    ROUTE_LABELS,
    REASON_LABELS,
    REASON_GROUPS,
    ROUTE_DESCRIPTIONS,
    SEARCH_MODE_LABELS,
    SEARCH_MODE_ORDER,
    MutationQueue,
    clone,
    codePointOffset,
    decisionForAction,
    defaultDiagnosticDecision,
    hasErrors,
    isTextEntry,
    nextSearchMode,
    nextUnfinishedCaseId,
    occurrenceOrdinal,
    occurrenceStart,
    quickRouteForKey,
    routeDefaults,
    validateDiagnostic,
    validateMention,
    validateRaw,
    visibleCaseQueue,
  };

  global.ReviewCore = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof globalThis === "undefined" ? this : globalThis));
