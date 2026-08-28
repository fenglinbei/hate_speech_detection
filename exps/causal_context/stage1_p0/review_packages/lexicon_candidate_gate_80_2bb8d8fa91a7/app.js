"use strict";

const bundle = window.LEXICON_REVIEW_PACKAGE;
if (!bundle || !bundle.manifest || !Array.isArray(bundle.cases)) {
  throw new Error("离线标注数据缺失或损坏。");
}

const PACKAGE_ID = bundle.manifest.package_id;
const STORAGE_KEY = `lexicon-candidate-review:${PACKAGE_ID}`;
const SURFACE_FORMS = new Set(["complete", "fragment", "sentence", "uncertain"]);
const TERM_SCOPES = new Set([
  "standalone_group_term",
  "productive_stem",
  "phrase_only",
  "generic_abuse",
  "behavior_or_phenomenon",
  "non_hateful_or_other",
  "context_fragment",
  "uncertain",
]);
const CONFIDENCE_VALUES = new Set(["high", "medium", "low"]);
const ELIGIBLE_SCOPES = new Set(["standalone_group_term", "productive_stem"]);
const NEGATIVE_SCOPES = new Set([
  "phrase_only",
  "generic_abuse",
  "behavior_or_phenomenon",
  "non_hateful_or_other",
  "context_fragment",
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
  caseStatus: document.getElementById("case-status"),
  contextCount: document.getElementById("context-count"),
  contexts: document.getElementById("contexts"),
  form: document.getElementById("annotation-form"),
  notes: document.getElementById("notes"),
  notesCount: document.getElementById("notes-count"),
  derivedResult: document.getElementById("derived-result"),
  formError: document.getElementById("form-error"),
  previousButton: document.getElementById("previous-button"),
  clearButton: document.getElementById("clear-button"),
  toast: document.getElementById("toast"),
};

function isComplete(annotation) {
  return Boolean(
    annotation &&
      annotation.surface_form &&
      annotation.term_scope &&
      annotation.confidence,
  );
}

function containsUncertain(annotation) {
  return Boolean(
    annotation &&
      (annotation.surface_form === "uncertain" ||
        annotation.term_scope === "uncertain" ||
        annotation.confidence === "low"),
  );
}

function deriveEligibility(surfaceForm, termScope) {
  if (surfaceForm === "fragment" || surfaceForm === "sentence" || termScope === "context_fragment") {
    return "no";
  }
  if (surfaceForm === "complete" && ELIGIBLE_SCOPES.has(termScope)) {
    return "yes";
  }
  if (surfaceForm === "complete" && NEGATIVE_SCOPES.has(termScope)) {
    return "no";
  }
  return "uncertain";
}

function fieldsAreConsistent(surfaceForm, termScope) {
  if (surfaceForm === "fragment" || surfaceForm === "sentence") {
    return termScope === "context_fragment" || termScope === "uncertain";
  }
  if (surfaceForm === "complete" && termScope === "context_fragment") {
    return false;
  }
  return true;
}

function validImportedAnnotation(annotation, item) {
  return Boolean(
    annotation &&
      annotation.case_id === item.case_id &&
      annotation.term === item.term &&
      SURFACE_FORMS.has(annotation.surface_form) &&
      TERM_SCOPES.has(annotation.term_scope) &&
      CONFIDENCE_VALUES.has(annotation.confidence) &&
      typeof annotation.notes === "string" &&
      annotation.notes.length <= 1000 &&
      typeof annotation.saved_at === "string" &&
      fieldsAreConsistent(annotation.surface_form, annotation.term_scope) &&
      annotation.provider_eligible ===
        deriveEligibility(annotation.surface_form, annotation.term_scope),
  );
}

function safeLoad() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved.package_id !== PACKAGE_ID || typeof saved.annotations !== "object") return;
    state.reviewerId = typeof saved.reviewer_id === "string" ? saved.reviewer_id : "reviewer-1";
    state.annotations = saved.annotations || {};
    if (Number.isInteger(saved.selected_index)) {
      state.selectedIndex = Math.max(0, Math.min(bundle.cases.length - 1, saved.selected_index));
    }
  } catch (_error) {
    showToast("浏览器中的旧草稿无法读取，已从空白状态开始。", true);
  }
}

function persist() {
  const value = {
    package_id: PACKAGE_ID,
    reviewer_id: state.reviewerId,
    selected_index: state.selectedIndex,
    annotations: state.annotations,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
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

function selectedCase() {
  return bundle.cases[state.selectedIndex];
}

function radioValue(name) {
  const input = dom.form.querySelector(`input[name="${name}"]:checked`);
  return input ? input.value : "";
}

function selectRadio(name, value) {
  for (const input of dom.form.querySelectorAll(`input[name="${name}"]`)) {
    input.checked = input.value === value;
  }
}

function currentDraft() {
  return {
    surface_form: radioValue("surface_form"),
    term_scope: radioValue("term_scope"),
    confidence: radioValue("confidence"),
    notes: dom.notes.value.trim(),
  };
}

function renderDerived() {
  const draft = currentDraft();
  const result = deriveEligibility(draft.surface_form, draft.term_scope);
  const labels = {
    yes: "建议进入供应商阶段",
    no: "不建议进入供应商阶段",
    uncertain: "当前结果：不确定",
  };
  dom.derivedResult.textContent = labels[result];
  dom.derivedResult.dataset.result = result;
}

function visibleCases() {
  return bundle.cases.filter((item) => {
    const annotation = state.annotations[item.case_id];
    if (state.filter === "pending") return !isComplete(annotation);
    if (state.filter === "complete") return isComplete(annotation);
    if (state.filter === "uncertain") return containsUncertain(annotation);
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
    number.textContent = item.case_id.replace("LCG80-", "#");
    const term = document.createElement("strong");
    term.textContent = item.term;
    const status = document.createElement("span");
    status.className = "case-dot";
    status.textContent = isComplete(annotation) ? "已完成" : "待处理";
    button.append(number, term, status);
    dom.caseList.appendChild(button);
  }
}

function renderCase() {
  const item = selectedCase();
  const annotation = state.annotations[item.case_id] || {};
  dom.casePosition.textContent = `${item.case_id} · ${state.selectedIndex + 1}/${bundle.cases.length}`;
  dom.candidateTerm.textContent = item.term;
  dom.caseStatus.textContent = isComplete(annotation) ? "已完成" : "未完成";
  dom.caseStatus.classList.toggle("is-complete", isComplete(annotation));
  dom.contextCount.textContent = `${item.contexts.length} 条`;
  dom.contexts.replaceChildren();
  for (const [index, context] of item.contexts.entries()) {
    const block = document.createElement("blockquote");
    const label = document.createElement("span");
    label.className = "context-label";
    label.textContent = `上下文 ${index + 1}`;
    const content = document.createElement("p");
    content.textContent = context.content;
    block.append(label, content);
    dom.contexts.appendChild(block);
  }
  selectRadio("surface_form", annotation.surface_form || "");
  selectRadio("term_scope", annotation.term_scope || "");
  selectRadio("confidence", annotation.confidence || "");
  dom.notes.value = annotation.notes || "";
  dom.notesCount.textContent = String(dom.notes.value.length);
  dom.formError.hidden = true;
  dom.previousButton.disabled = state.selectedIndex === 0;
  renderDerived();
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
      ? "80 条已完成，可以导出完整结果。"
      : `还剩 ${total - complete} 条；草稿保存在当前浏览器。`;
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
  const eligibility = annotations.reduce(
    (counts, item) => {
      counts[item.provider_eligible] += 1;
      return counts;
    },
    { yes: 0, no: 0, uncertain: 0 },
  );
  return {
    schema_version: "stage1-lexicon-candidate-annotation/v1",
    package_id: PACKAGE_ID,
    reviewer_id: state.reviewerId.trim(),
    reviewer_count: 1,
    review_status: status,
    exported_at: new Date().toISOString(),
    progress: { complete: annotations.length, total: bundle.cases.length },
    summary: { provider_eligible: eligibility },
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
    showToast("尚未完成全部 80 条。", true);
    return;
  }
  persist();
  downloadJson(
    requireComplete ? "lexicon_candidate_annotations.json" : "lexicon_candidate_annotations.backup.json",
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
    if (!item || !validImportedAnnotation(annotation, item)) {
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
  showToast(`已导入 ${Object.keys(imported).length} 条标注。`);
}

dom.form.addEventListener("change", renderDerived);
dom.notes.addEventListener("input", () => {
  dom.notesCount.textContent = String(dom.notes.value.length);
});
dom.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const item = selectedCase();
  const draft = currentDraft();
  if (!draft.surface_form || !draft.term_scope || !draft.confidence) {
    dom.formError.textContent = "请完成形式、scope 和判断信心三个字段。";
    dom.formError.hidden = false;
    return;
  }
  if (!fieldsAreConsistent(draft.surface_form, draft.term_scope)) {
    dom.formError.textContent = "形式与 scope 不一致：碎片/句子应选择“上下文碎片”；完整词条不能选择“上下文碎片”。";
    dom.formError.hidden = false;
    return;
  }
  const providerEligible = deriveEligibility(draft.surface_form, draft.term_scope);
  state.annotations[item.case_id] = {
    case_id: item.case_id,
    term: item.term,
    surface_form: draft.surface_form,
    term_scope: draft.term_scope,
    provider_eligible: providerEligible,
    confidence: draft.confidence,
    notes: draft.notes,
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
  if (!window.confirm(`清除 ${item.case_id} 的已保存标注？`)) return;
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
