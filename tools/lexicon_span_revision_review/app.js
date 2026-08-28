"use strict";

const bundle = window.LEXICON_SPAN_REVISION_PACKAGE;
if (!bundle || !bundle.manifest || !Array.isArray(bundle.cases)) {
  throw new Error("离线 span 修订数据缺失或损坏。");
}

const PACKAGE_ID = bundle.manifest.package_id;
const STORAGE_KEY = `lexicon-span-revision:${PACKAGE_ID}`;
const DECISIONS = new Set(["no_valid_span", "corrected_spans"]);
const SCOPES = new Set(["standalone_group_term", "productive_stem"]);

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
  backupButton: document.getElementById("backup-button"),
  exportButton: document.getElementById("export-button"),
  completeCount: document.getElementById("complete-count"),
  totalCount: document.getElementById("total-count"),
  progressFill: document.getElementById("progress-fill"),
  progressTrack: document.querySelector(".progress-track"),
  progressMessage: document.getElementById("progress-message"),
  statusFilter: document.getElementById("status-filter"),
  caseList: document.getElementById("case-list"),
  emptyList: document.getElementById("empty-list"),
  reviewCard: document.getElementById("review-card"),
  casePosition: document.getElementById("case-position"),
  candidateTerm: document.getElementById("candidate-term"),
  sourceForm: document.getElementById("source-form"),
  caseStatus: document.getElementById("case-status"),
  contextCount: document.getElementById("context-count"),
  contexts: document.getElementById("contexts"),
  form: document.getElementById("revision-form"),
  spanEditor: document.getElementById("span-editor"),
  spanRows: document.getElementById("span-rows"),
  spanRowTemplate: document.getElementById("span-row-template"),
  addSpanButton: document.getElementById("add-span-button"),
  notes: document.getElementById("notes"),
  notesCount: document.getElementById("notes-count"),
  formError: document.getElementById("form-error"),
  previousButton: document.getElementById("previous-button"),
  clearButton: document.getElementById("clear-button"),
  toast: document.getElementById("toast"),
};

function selectedCase() {
  return bundle.cases[state.selectedIndex];
}

function selectedDecision() {
  const input = dom.form.querySelector('input[name="decision"]:checked');
  return input ? input.value : "";
}

function selectDecision(value) {
  for (const input of dom.form.querySelectorAll('input[name="decision"]')) {
    input.checked = input.value === value;
  }
}

function isComplete(annotation) {
  return Boolean(annotation && DECISIONS.has(annotation.decision));
}

function spanMatchesCase(span, item) {
  if (
    !span ||
    typeof span.surface !== "string" ||
    !span.surface ||
    span.surface.length > 80 ||
    !SCOPES.has(span.scope) ||
    !Number.isInteger(span.context_index) ||
    span.context_index < 0 ||
    span.context_index >= item.contexts.length
  ) {
    return false;
  }
  const context = item.contexts[span.context_index];
  return (
    span.context_record_id === context.record_id &&
    context.content.includes(span.surface)
  );
}

function validAnnotation(annotation, item) {
  if (
    !annotation ||
    annotation.case_id !== item.case_id ||
    annotation.source_case_id !== item.source_case_id ||
    annotation.original_term !== item.original_term ||
    !DECISIONS.has(annotation.decision) ||
    !Array.isArray(annotation.spans) ||
    typeof annotation.notes !== "string" ||
    annotation.notes.length > 1000 ||
    typeof annotation.saved_at !== "string"
  ) {
    return false;
  }
  if (annotation.decision === "no_valid_span") {
    return annotation.spans.length === 0;
  }
  if (annotation.spans.length === 0 || !annotation.spans.every((span) => spanMatchesCase(span, item))) {
    return false;
  }
  return new Set(annotation.spans.map((span) => span.surface)).size === annotation.spans.length;
}

function showToast(message, isError = false) {
  dom.toast.textContent = message;
  dom.toast.classList.toggle("is-error", isError);
  dom.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    dom.toast.hidden = true;
  }, 3200);
}

function safeLoad() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved.package_id !== PACKAGE_ID || typeof saved.annotations !== "object") return;
    state.reviewerId = typeof saved.reviewer_id === "string" ? saved.reviewer_id : "reviewer-1";
    const caseById = new Map(bundle.cases.map((item) => [item.case_id, item]));
    state.annotations = Object.fromEntries(
      Object.entries(saved.annotations || {}).filter(([caseId, annotation]) => {
        const item = caseById.get(caseId);
        return item && validAnnotation(annotation, item);
      }),
    );
    if (Number.isInteger(saved.selected_index)) {
      state.selectedIndex = Math.max(0, Math.min(bundle.cases.length - 1, saved.selected_index));
    }
  } catch (_error) {
    showToast("浏览器中的旧草稿无法读取，已从空白状态开始。", true);
  }
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

function visibleCases() {
  return bundle.cases.filter((item) => {
    const annotation = state.annotations[item.case_id];
    if (state.filter === "pending") return !isComplete(annotation);
    if (state.filter === "complete") return isComplete(annotation);
    if (state.filter === "corrected") return annotation && annotation.decision === "corrected_spans";
    if (state.filter === "none") return annotation && annotation.decision === "no_valid_span";
    return true;
  });
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
    button.addEventListener("click", () => {
      state.selectedIndex = index;
      persist();
      render();
      dom.reviewCard.focus({ preventScroll: true });
    });
    const number = document.createElement("span");
    number.textContent = item.case_id.replace("LSR49-", "#");
    const term = document.createElement("strong");
    term.textContent = item.original_term;
    const status = document.createElement("span");
    status.className = "case-dot";
    status.textContent = isComplete(annotation) ? "已完成" : "待处理";
    button.append(number, term, status);
    dom.caseList.appendChild(button);
  }
}

function contextOptionText(context, index) {
  const compact = context.content.replace(/\s+/g, " ").slice(0, 32);
  return `上下文 ${index + 1} · ${compact}${context.content.length > 32 ? "…" : ""}`;
}

function updateSpanMatch(row) {
  const item = selectedCase();
  const contextIndex = Number(row.querySelector(".span-context").value);
  const surface = row.querySelector(".span-surface").value.trim();
  const status = row.querySelector(".span-match-status");
  if (!surface) {
    status.textContent = "等待填写 span";
    status.dataset.match = "pending";
    return;
  }
  const context = item.contexts[contextIndex];
  const matches = Boolean(context && context.content.includes(surface));
  status.textContent = matches ? "已在所选原文中逐字匹配" : "未在所选原文中找到完全匹配";
  status.dataset.match = matches ? "yes" : "no";
}

function addSpanRow(value = {}) {
  const item = selectedCase();
  const fragment = dom.spanRowTemplate.content.cloneNode(true);
  const row = fragment.querySelector(".span-row");
  const contextSelect = row.querySelector(".span-context");
  for (const [index, context] of item.contexts.entries()) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = contextOptionText(context, index);
    contextSelect.appendChild(option);
  }
  contextSelect.value = String(Number.isInteger(value.context_index) ? value.context_index : 0);
  row.querySelector(".span-surface").value = value.surface || "";
  row.querySelector(".span-scope").value = value.scope || "standalone_group_term";
  row.querySelector(".remove-span-button").addEventListener("click", () => {
    row.remove();
    if (!dom.spanRows.children.length) addSpanRow();
  });
  for (const input of row.querySelectorAll("input, select")) {
    input.addEventListener("input", () => updateSpanMatch(row));
    input.addEventListener("change", () => updateSpanMatch(row));
  }
  dom.spanRows.appendChild(fragment);
  updateSpanMatch(dom.spanRows.lastElementChild);
}

function collectSpanRows() {
  const item = selectedCase();
  return [...dom.spanRows.querySelectorAll(".span-row")].map((row) => {
    const contextIndex = Number(row.querySelector(".span-context").value);
    return {
      surface: row.querySelector(".span-surface").value.trim(),
      scope: row.querySelector(".span-scope").value,
      context_index: contextIndex,
      context_record_id: item.contexts[contextIndex]
        ? item.contexts[contextIndex].record_id
        : "",
    };
  });
}

function syncDecisionEditor() {
  const corrected = selectedDecision() === "corrected_spans";
  dom.spanEditor.hidden = !corrected;
  if (corrected && !dom.spanRows.children.length) addSpanRow();
}

function renderCase() {
  const item = selectedCase();
  const annotation = state.annotations[item.case_id] || {};
  dom.casePosition.textContent = `${item.case_id} · 来源 ${item.source_case_id} · ${state.selectedIndex + 1}/${bundle.cases.length}`;
  dom.candidateTerm.textContent = item.original_term;
  dom.sourceForm.textContent = item.source_surface_form === "fragment" ? "上一轮：碎片" : "上一轮：句子";
  dom.caseStatus.textContent = isComplete(annotation) ? "已完成" : "未完成";
  dom.caseStatus.classList.toggle("is-complete", isComplete(annotation));
  dom.contextCount.textContent = `${item.contexts.length} 条`;
  dom.contexts.replaceChildren();
  for (const [index, context] of item.contexts.entries()) {
    const block = document.createElement("blockquote");
    block.dataset.contextIndex = String(index);
    const label = document.createElement("span");
    label.className = "context-label";
    label.textContent = `上下文 ${index + 1}`;
    const content = document.createElement("p");
    content.textContent = context.content;
    block.append(label, content);
    dom.contexts.appendChild(block);
  }
  selectDecision(annotation.decision || "");
  dom.spanRows.replaceChildren();
  for (const span of annotation.spans || []) addSpanRow(span);
  dom.notes.value = annotation.notes || "";
  dom.notesCount.textContent = String(dom.notes.value.length);
  dom.formError.hidden = true;
  dom.previousButton.disabled = state.selectedIndex === 0;
  syncDecisionEditor();
}

function renderProgress() {
  const complete = bundle.cases.filter((item) => isComplete(state.annotations[item.case_id])).length;
  const total = bundle.cases.length;
  dom.completeCount.textContent = String(complete);
  dom.totalCount.textContent = String(total);
  dom.progressFill.style.width = `${(complete / total) * 100}%`;
  dom.progressTrack.setAttribute("aria-valuemax", String(total));
  dom.progressTrack.setAttribute("aria-valuenow", String(complete));
  dom.exportButton.disabled = complete !== total;
  dom.progressMessage.textContent =
    complete === total
      ? "49 条已完成，可以导出完整结果。"
      : `还剩 ${total - complete} 条；已保存结果保留在当前浏览器。`;
}

function render() {
  dom.reviewerId.value = state.reviewerId;
  dom.statusFilter.value = state.filter;
  renderProgress();
  renderList();
  renderCase();
}

function nextPending() {
  for (let offset = 1; offset <= bundle.cases.length; offset += 1) {
    const index = (state.selectedIndex + offset) % bundle.cases.length;
    if (!isComplete(state.annotations[bundle.cases[index].case_id])) {
      state.selectedIndex = index;
      return;
    }
  }
  state.selectedIndex = Math.min(state.selectedIndex + 1, bundle.cases.length - 1);
}

function resultPayload(status) {
  const annotations = bundle.cases
    .map((item) => state.annotations[item.case_id])
    .filter((item) => isComplete(item));
  const decisions = annotations.reduce(
    (counts, item) => {
      counts[item.decision] += 1;
      counts.corrected_span_count += item.spans.length;
      return counts;
    },
    { no_valid_span: 0, corrected_spans: 0, corrected_span_count: 0 },
  );
  return {
    schema_version: "stage1-lexicon-span-revision-annotation/v1",
    package_id: PACKAGE_ID,
    source_package_id: bundle.manifest.source.parent_package_id,
    reviewer_id: state.reviewerId.trim(),
    reviewer_count: 1,
    review_status: status,
    exported_at: new Date().toISOString(),
    progress: { complete: annotations.length, total: bundle.cases.length },
    summary: decisions,
    annotations,
  };
}

function downloadJson(filename, value) {
  const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportResult(requireComplete) {
  state.reviewerId = dom.reviewerId.value.trim();
  if (!state.reviewerId) {
    showToast("请先填写标注者代号。", true);
    dom.reviewerId.focus();
    return;
  }
  const complete = bundle.cases.filter((item) => isComplete(state.annotations[item.case_id])).length;
  if (requireComplete && complete !== bundle.cases.length) {
    showToast("尚未完成全部 49 条。", true);
    return;
  }
  persist();
  downloadJson(
    requireComplete ? "lexicon_span_revision_annotations.json" : "lexicon_span_revision_annotations.backup.json",
    resultPayload(requireComplete ? "complete" : "partial"),
  );
}

function importResult(value) {
  if (!value || value.package_id !== PACKAGE_ID || !Array.isArray(value.annotations)) {
    throw new Error("结果文件与当前 package ID 不匹配。");
  }
  const caseById = new Map(bundle.cases.map((item) => [item.case_id, item]));
  const imported = {};
  for (const annotation of value.annotations) {
    const item = caseById.get(annotation.case_id);
    if (!item || !validAnnotation(annotation, item) || imported[annotation.case_id]) {
      throw new Error(`结果中包含无效标注：${annotation.case_id || "unknown"}`);
    }
    imported[annotation.case_id] = annotation;
  }
  state.annotations = imported;
  state.reviewerId = typeof value.reviewer_id === "string" ? value.reviewer_id : "reviewer-1";
  state.selectedIndex = bundle.cases.findIndex((item) => !isComplete(imported[item.case_id]));
  if (state.selectedIndex < 0) state.selectedIndex = 0;
  persist();
  render();
  showToast(`已导入 ${Object.keys(imported).length} 条修订。`);
}

dom.form.addEventListener("change", (event) => {
  if (event.target && event.target.name === "decision") syncDecisionEditor();
});
dom.addSpanButton.addEventListener("click", () => addSpanRow());
dom.notes.addEventListener("input", () => {
  dom.notesCount.textContent = String(dom.notes.value.length);
});
dom.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const item = selectedCase();
  const decision = selectedDecision();
  dom.formError.hidden = true;
  if (!DECISIONS.has(decision)) {
    dom.formError.textContent = "请选择“无有效词条”或“填写正确 span”。";
    dom.formError.hidden = false;
    return;
  }
  const spans = decision === "corrected_spans" ? collectSpanRows() : [];
  if (decision === "corrected_spans") {
    if (!spans.length || spans.some((span) => !spanMatchesCase(span, item))) {
      dom.formError.textContent = "每个正确 span 都必须非空，并逐字出现在所选上下文中。";
      dom.formError.hidden = false;
      return;
    }
    if (new Set(spans.map((span) => span.surface)).size !== spans.length) {
      dom.formError.textContent = "同一个 surface 不要重复提交；保留一个证据上下文即可。";
      dom.formError.hidden = false;
      return;
    }
  }
  state.annotations[item.case_id] = {
    case_id: item.case_id,
    source_case_id: item.source_case_id,
    original_term: item.original_term,
    decision,
    spans,
    notes: dom.notes.value.trim(),
    saved_at: new Date().toISOString(),
  };
  nextPending();
  persist();
  render();
  showToast(`${item.case_id} 已保存。`);
});
dom.previousButton.addEventListener("click", () => {
  state.selectedIndex = Math.max(0, state.selectedIndex - 1);
  persist();
  render();
});
dom.clearButton.addEventListener("click", () => {
  const item = selectedCase();
  if (!window.confirm(`清除 ${item.case_id} 的已保存修订？`)) return;
  delete state.annotations[item.case_id];
  persist();
  render();
});
dom.statusFilter.addEventListener("change", () => {
  state.filter = dom.statusFilter.value;
  renderList();
});
dom.reviewerId.addEventListener("change", () => {
  state.reviewerId = dom.reviewerId.value.trim() || "reviewer-1";
  persist();
  dom.reviewerId.value = state.reviewerId;
});
dom.backupButton.addEventListener("click", () => exportResult(false));
dom.exportButton.addEventListener("click", () => exportResult(true));
dom.importButton.addEventListener("click", () => dom.importFile.click());
dom.importFile.addEventListener("change", async () => {
  const [file] = dom.importFile.files;
  if (!file) return;
  try {
    importResult(JSON.parse(await file.text()));
  } catch (error) {
    showToast(error instanceof Error ? error.message : "导入失败。", true);
  } finally {
    dom.importFile.value = "";
  }
});
document.addEventListener("keydown", (event) => {
  if (!event.altKey) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    dom.previousButton.click();
  }
  if (event.key === "ArrowRight") {
    event.preventDefault();
    state.selectedIndex = Math.min(bundle.cases.length - 1, state.selectedIndex + 1);
    persist();
    render();
  }
});

safeLoad();
render();
