"use strict";

const bundle = window.DUAL_MODEL_SPAN_REVIEW;
if (!bundle || !bundle.manifest || !Array.isArray(bundle.cases)) {
  throw new Error("离线检查包缺少 cases.js 或数据已损坏。");
}

const PACKAGE_ID = bundle.manifest.package_id;
const STORAGE_KEY = `dual-model-span-review:${PACKAGE_ID}`;
const DECISIONS = new Set([
  "model_a_better",
  "model_b_better",
  "both_acceptable",
  "both_wrong",
  "no_valid_span",
]);
const ISSUE_TAGS = new Set([
  "neutral_identity",
  "fragment_or_sentence",
  "missed_valid_span",
  "hallucinated_span",
  "wrong_type",
  "wrong_description",
  "other",
]);

const state = {
  selectedIndex: 0,
  filter: "all",
  reviewerId: "reviewer-1",
  annotations: {},
};

const dom = {
  reviewerId: document.getElementById("reviewer-id"),
  importButton: document.getElementById("import-button"),
  importFile: document.getElementById("import-file"),
  exportButton: document.getElementById("export-button"),
  completeCount: document.getElementById("complete-count"),
  totalCount: document.getElementById("total-count"),
  progressMessage: document.getElementById("progress-message"),
  progressTrack: document.querySelector(".progress-track"),
  progressFill: document.getElementById("progress-fill"),
  statusFilter: document.getElementById("status-filter"),
  caseList: document.getElementById("case-list"),
  emptyList: document.getElementById("empty-list"),
  reviewCard: document.getElementById("review-card"),
  casePosition: document.getElementById("case-position"),
  caseAlias: document.getElementById("case-alias"),
  caseFlags: document.getElementById("case-flags"),
  sourceLength: document.getElementById("source-length"),
  sourceContent: document.getElementById("source-content"),
  agreementSummary: document.getElementById("agreement-summary"),
  modelA: document.getElementById("model-a-output"),
  modelB: document.getElementById("model-b-output"),
  form: document.getElementById("review-form"),
  correctedSpans: document.getElementById("corrected-spans"),
  notes: document.getElementById("notes"),
  formError: document.getElementById("form-error"),
  previousButton: document.getElementById("previous-button"),
  clearButton: document.getElementById("clear-button"),
  toast: document.getElementById("toast"),
};

function selectedCase() {
  return bundle.cases[state.selectedIndex];
}

function isComplete(annotation) {
  return Boolean(annotation && DECISIONS.has(annotation.decision));
}

function showToast(message, error = false) {
  dom.toast.textContent = message;
  dom.toast.classList.toggle("error", error);
  dom.toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { dom.toast.hidden = true; }, 3200);
}

function persist() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      package_id: PACKAGE_ID,
      reviewer_id: state.reviewerId,
      selected_index: state.selectedIndex,
      annotations: state.annotations,
    }),
  );
}

function validImportedAnnotation(annotation, item) {
  return Boolean(
    annotation &&
      annotation.case_id === item.case_id &&
      DECISIONS.has(annotation.decision) &&
      Array.isArray(annotation.issue_tags) &&
      annotation.issue_tags.every((tag) => ISSUE_TAGS.has(tag)) &&
      new Set(annotation.issue_tags).size === annotation.issue_tags.length &&
      Array.isArray(annotation.corrected_spans) &&
      annotation.corrected_spans.every(
        (span) => typeof span === "string" && span && span.length <= 80 && item.content.includes(span),
      ) &&
      new Set(annotation.corrected_spans).size === annotation.corrected_spans.length &&
      typeof annotation.notes === "string" &&
      annotation.notes.length <= 2000
  );
}

function safeLoad() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved.package_id !== PACKAGE_ID || typeof saved.annotations !== "object") return;
    const byId = new Map(bundle.cases.map((item) => [item.case_id, item]));
    state.annotations = Object.fromEntries(
      Object.entries(saved.annotations).filter(([caseId, annotation]) => {
        const item = byId.get(caseId);
        return item && validImportedAnnotation(annotation, item);
      }),
    );
    if (typeof saved.reviewer_id === "string" && saved.reviewer_id.trim()) {
      state.reviewerId = saved.reviewer_id.slice(0, 100);
    }
    if (Number.isInteger(saved.selected_index)) {
      state.selectedIndex = Math.max(0, Math.min(bundle.cases.length - 1, saved.selected_index));
    }
  } catch (_error) {
    showToast("旧草稿无法读取，已从空白状态开始。", true);
  }
}

function visibleCases() {
  return bundle.cases.filter((item) => {
    const annotation = state.annotations[item.case_id];
    if (state.filter === "pending") return !isComplete(annotation);
    if (state.filter === "completed") return isComplete(annotation);
    if (state.filter === "disagreement") return item.flags.includes("cross_model_disagreement");
    return true;
  });
}

function createFlag(text, kind = "") {
  const badge = document.createElement("span");
  badge.className = `flag ${kind}`.trim();
  badge.textContent = text;
  return badge;
}

function renderFlags(item) {
  dom.caseFlags.replaceChildren();
  if (item.flags.includes("cross_model_disagreement")) {
    dom.caseFlags.appendChild(createFlag("模型 span 不一致", "risk"));
  } else {
    dom.caseFlags.appendChild(createFlag("模型 span 一致", "ok"));
  }
  if (item.flags.some((flag) => flag.endsWith("validation_error"))) {
    dom.caseFlags.appendChild(createFlag("结构告警", "risk"));
  }
}

function renderList() {
  dom.caseList.replaceChildren();
  const visible = visibleCases();
  dom.emptyList.hidden = visible.length !== 0;
  for (const item of visible) {
    const index = bundle.cases.findIndex((candidate) => candidate.case_id === item.case_id);
    const annotation = state.annotations[item.case_id];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-button";
    button.classList.toggle("is-active", index === state.selectedIndex);
    button.classList.toggle("is-complete", isComplete(annotation));
    button.classList.toggle("has-risk", item.flags.length > 0);
    button.addEventListener("click", () => {
      state.selectedIndex = index;
      persist();
      render();
      dom.reviewCard.focus({ preventScroll: true });
    });
    const number = document.createElement("span");
    number.className = "number";
    number.textContent = item.case_id.replace("DMH-", "#");
    const alias = document.createElement("span");
    alias.className = "alias";
    alias.textContent = item.blind_alias;
    const dot = document.createElement("span");
    dot.className = "case-dot";
    button.append(number, alias, dot);
    dom.caseList.appendChild(button);
  }
}

function modelTypeLabel(value) {
  if (value === "standalone_term") return "独立词条";
  if (value === "productive_stem") return "构词词干";
  return value || "未给类型";
}

function renderAnnotation(container, annotation, compact = false) {
  container.replaceChildren();
  const hasSpan = Boolean(annotation.effective_has_valid_span);
  const status = document.createElement("span");
  status.className = `model-status ${hasSpan ? "" : "none"}`.trim();
  status.textContent = hasSpan ? `提取 ${annotation.exact_surfaces.length} 个 exact span` : "无有效 exact span";
  container.appendChild(status);

  const list = document.createElement("ul");
  list.className = "span-list";
  for (const span of annotation.spans) {
    const item = document.createElement("li");
    item.className = "span-item";
    const surface = document.createElement("span");
    surface.className = "span-surface";
    surface.textContent = span.surface || "（空）";
    const meta = document.createElement("span");
    meta.className = "span-meta";
    const confidence = typeof span.confidence === "number" ? `${Math.round(span.confidence * 100)}%` : "—";
    meta.textContent = `${modelTypeLabel(span.type)} · 置信 ${confidence}`;
    item.append(surface, meta);
    if (!compact) {
      const description = document.createElement("p");
      description.className = "span-description";
      description.textContent = span.description || "（无描述）";
      item.appendChild(description);
    }
    if (span.validation_errors && span.validation_errors.length) {
      const warning = document.createElement("p");
      warning.className = "validation-warning";
      warning.textContent = `结构告警：${span.validation_errors.join("、")}`;
      item.appendChild(warning);
    }
    list.appendChild(item);
  }
  if (annotation.spans.length) container.appendChild(list);
  if (!compact) {
    const description = document.createElement("p");
    description.className = "record-description";
    description.textContent = `整条说明：${annotation.record_description || "（无）"}`;
    container.appendChild(description);
  }
  if (annotation.validation_errors && annotation.validation_errors.length) {
    const warning = document.createElement("p");
    warning.className = "validation-warning";
    warning.textContent = `输出结构告警：${annotation.validation_errors.join("、")}`;
    container.appendChild(warning);
  }
}

function selectDecision(value) {
  for (const input of dom.form.querySelectorAll('input[name="decision"]')) {
    input.checked = input.value === value;
  }
}

function fillForm(item) {
  const annotation = state.annotations[item.case_id];
  selectDecision(annotation ? annotation.decision : "");
  for (const input of dom.form.querySelectorAll('input[name="issue"]')) {
    input.checked = Boolean(annotation && annotation.issue_tags.includes(input.value));
  }
  dom.correctedSpans.value = annotation ? annotation.corrected_spans.join("\n") : "";
  dom.notes.value = annotation ? annotation.notes : "";
  dom.formError.hidden = true;
}

function renderCase() {
  const item = selectedCase();
  dom.casePosition.textContent = `${item.case_id} · ${state.selectedIndex + 1} / ${bundle.cases.length}`;
  dom.caseAlias.textContent = item.blind_alias;
  dom.sourceLength.textContent = `${Array.from(item.content).length} 字符`;
  dom.sourceContent.textContent = item.content;
  dom.agreementSummary.textContent = item.comparison.exact_span_set_match
    ? "exact span 集合一致"
    : `Jaccard ${item.comparison.span_jaccard.toFixed(2)}`;
  renderFlags(item);
  renderAnnotation(dom.modelA, item.model_a);
  renderAnnotation(dom.modelB, item.model_b);
  fillForm(item);
  dom.previousButton.disabled = state.selectedIndex === 0;
}

function renderProgress() {
  const complete = bundle.cases.filter((item) => isComplete(state.annotations[item.case_id])).length;
  const total = bundle.cases.length;
  const percent = total ? (complete / total) * 100 : 0;
  dom.completeCount.textContent = String(complete);
  dom.totalCount.textContent = String(total);
  dom.progressMessage.textContent = complete === total ? "全部完成，可导出" : complete ? `还剩 ${total - complete} 条` : "尚未开始";
  dom.progressFill.style.width = `${percent}%`;
  dom.progressTrack.setAttribute("aria-valuemax", String(total));
  dom.progressTrack.setAttribute("aria-valuenow", String(complete));
}

function render() {
  renderList();
  renderCase();
  renderProgress();
}

function collectCurrent() {
  const item = selectedCase();
  const decisionInput = dom.form.querySelector('input[name="decision"]:checked');
  if (!decisionInput || !DECISIONS.has(decisionInput.value)) {
    throw new Error("请选择一个总体判断。");
  }
  const issueTags = [...dom.form.querySelectorAll('input[name="issue"]:checked')].map(
    (input) => input.value,
  );
  const correctedSpans = dom.correctedSpans.value
    .split(/\r?\n/)
    .map((value) => value.trim())
    .filter(Boolean);
  if (new Set(correctedSpans).size !== correctedSpans.length) {
    throw new Error("corrected spans 中有重复行。");
  }
  const invalid = correctedSpans.find((span) => span.length > 80 || !item.content.includes(span));
  if (invalid) {
    throw new Error(`corrected span 不是原文逐字子串：${invalid}`);
  }
  return {
    case_id: item.case_id,
    decision: decisionInput.value,
    issue_tags: issueTags,
    corrected_spans: correctedSpans,
    notes: dom.notes.value.trim(),
  };
}

function nextIndex() {
  for (let offset = 1; offset <= bundle.cases.length; offset += 1) {
    const index = (state.selectedIndex + offset) % bundle.cases.length;
    if (!isComplete(state.annotations[bundle.cases[index].case_id])) return index;
  }
  return Math.min(bundle.cases.length - 1, state.selectedIndex + 1);
}

function saveAndNext(event) {
  event.preventDefault();
  try {
    const annotation = collectCurrent();
    state.annotations[annotation.case_id] = annotation;
    state.selectedIndex = nextIndex();
    persist();
    render();
    showToast("已保存，转到下一条待检查 case。");
    dom.reviewCard.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    dom.formError.textContent = error.message;
    dom.formError.hidden = false;
  }
}

function clearCurrent() {
  const item = selectedCase();
  if (!state.annotations[item.case_id] && !dom.form.querySelector('input[name="decision"]:checked')) return;
  delete state.annotations[item.case_id];
  persist();
  fillForm(item);
  renderList();
  renderProgress();
  showToast("已清除此条判断。");
}

function exportAnnotations() {
  const reviewerId = dom.reviewerId.value.trim();
  if (!reviewerId) {
    showToast("请填写标注者 ID。", true);
    dom.reviewerId.focus();
    return;
  }
  const annotations = bundle.cases
    .map((item) => state.annotations[item.case_id])
    .filter(isComplete);
  const output = {
    schema_version: "dual-model-span-human-review/v1",
    package_id: PACKAGE_ID,
    reviewer_id: reviewerId,
    annotations,
  };
  const blob = new Blob([JSON.stringify(output, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `dual_model_span_review_${annotations.length}of${bundle.cases.length}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  showToast(`已导出 ${annotations.length} / ${bundle.cases.length} 条。`);
}

async function importAnnotations(file) {
  try {
    const payload = JSON.parse(await file.text());
    if (
      payload.schema_version !== "dual-model-span-human-review/v1" ||
      payload.package_id !== PACKAGE_ID ||
      !Array.isArray(payload.annotations)
    ) {
      throw new Error("文件不属于当前 package。");
    }
    const byId = new Map(bundle.cases.map((item) => [item.case_id, item]));
    const annotations = {};
    for (const annotation of payload.annotations) {
      const item = byId.get(annotation.case_id);
      if (!item || !validImportedAnnotation(annotation, item)) {
        throw new Error(`annotation 无效：${annotation && annotation.case_id ? annotation.case_id : "未知 case"}`);
      }
      if (annotations[annotation.case_id]) throw new Error(`case 重复：${annotation.case_id}`);
      annotations[annotation.case_id] = annotation;
    }
    state.annotations = annotations;
    if (typeof payload.reviewer_id === "string" && payload.reviewer_id.trim()) {
      state.reviewerId = payload.reviewer_id.slice(0, 100);
      dom.reviewerId.value = state.reviewerId;
    }
    persist();
    render();
    showToast(`已导入 ${Object.keys(annotations).length} 条。`);
  } catch (error) {
    showToast(`导入失败：${error.message}`, true);
  } finally {
    dom.importFile.value = "";
  }
}

dom.form.addEventListener("submit", saveAndNext);
dom.previousButton.addEventListener("click", () => {
  if (state.selectedIndex > 0) {
    state.selectedIndex -= 1;
    persist();
    render();
  }
});
dom.clearButton.addEventListener("click", clearCurrent);
dom.statusFilter.addEventListener("change", () => {
  state.filter = dom.statusFilter.value;
  renderList();
});
dom.reviewerId.addEventListener("change", () => {
  const value = dom.reviewerId.value.trim();
  if (value) {
    state.reviewerId = value.slice(0, 100);
    persist();
  }
});
dom.exportButton.addEventListener("click", exportAnnotations);
dom.importButton.addEventListener("click", () => dom.importFile.click());
dom.importFile.addEventListener("change", () => {
  if (dom.importFile.files && dom.importFile.files[0]) importAnnotations(dom.importFile.files[0]);
});

safeLoad();
dom.reviewerId.value = state.reviewerId;
render();
