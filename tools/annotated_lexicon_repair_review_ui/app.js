"use strict";

const Common = window.ReviewCore;
const Gold = window.SpanGoldCore;

const ui = {
  sidebar: document.querySelector("#item-sidebar"),
  sidebarToggle: document.querySelector("#sidebar-toggle"),
  sidebarClose: document.querySelector("#sidebar-close"),
  sidebarBackdrop: document.querySelector("#sidebar-backdrop"),
  progressText: document.querySelector("#top-progress-text"),
  progressBar: document.querySelector("#progress-bar"),
  sidebarProgress: document.querySelector("#sidebar-progress"),
  saveState: document.querySelector("#save-state"),
  exportSnapshot: document.querySelector("#export-snapshot"),
  itemSearch: document.querySelector("#item-search"),
  searchMode: document.querySelector("#search-mode"),
  searchStatus: document.querySelector("#search-status"),
  itemFilters: document.querySelector("#item-filters"),
  itemList: document.querySelector("#item-list"),
  itemEmpty: document.querySelector("#item-empty"),
  itemPosition: document.querySelector("#item-position"),
  itemId: document.querySelector("#item-id"),
  previousItem: document.querySelector("#previous-item"),
  nextItem: document.querySelector("#next-item"),
  cohort: document.querySelector("#cohort"),
  queryContent: document.querySelector("#query-content"),
  selectionToolbar: document.querySelector("#selection-toolbar"),
  selectionSurface: document.querySelector("#selection-surface"),
  selectionError: document.querySelector("#selection-error"),
  confirmSelection: document.querySelector("#confirm-selection"),
  priorDisposition: document.querySelector("#prior-disposition"),
  priorAudit: document.querySelector("#prior-audit"),
  candidateStatus: document.querySelector("#candidate-status"),
  activeCandidateStatus: document.querySelector("#active-candidate-status"),
  candidateList: document.querySelector("#candidate-list"),
  zeroCandidate: document.querySelector("#zero-candidate"),
  decisionTitle: document.querySelector("#decision-title"),
  decisionStatus: document.querySelector("#decision-status"),
  keepAll: document.querySelector("#keep-all"),
  dropAll: document.querySelector("#drop-all"),
  clearAll: document.querySelector("#clear-all"),
  candidateError: document.querySelector("#candidate-error"),
  additionalList: document.querySelector("#additional-list"),
  additionalCount: document.querySelector("#additional-count"),
  additionalEmpty: document.querySelector("#additional-empty"),
  selectInQuery: document.querySelector("#select-in-query"),
  additionalDialog: document.querySelector("#additional-dialog"),
  additionalForm: document.querySelector("#additional-form"),
  additionalDialogTitle: document.querySelector("#additional-dialog-title"),
  additionalSurface: document.querySelector("#additional-surface"),
  additionalPosition: document.querySelector("#additional-position"),
  additionalContext: document.querySelector("#additional-context"),
  additionalReason: document.querySelector("#additional-reason"),
  additionalDialogError: document.querySelector("#additional-dialog-error"),
  cancelAdditional: document.querySelector("#cancel-additional"),
  submitAdditional: document.querySelector("#submit-additional"),
  additionalError: document.querySelector("#additional-error"),
  notes: document.querySelector("#notes"),
  notesError: document.querySelector("#notes-error"),
  saveDraft: document.querySelector("#save-draft"),
  confirmNext: document.querySelector("#confirm-next"),
  lockedCard: document.querySelector("#locked-card"),
  reopen: document.querySelector("#reopen"),
  requestError: document.querySelector("#request-error"),
  requestErrorMessage: document.querySelector("#request-error-message"),
  requestErrorDetails: document.querySelector("#request-error-details"),
  guidelines: document.querySelector("#guidelines"),
  guidelineDialog: document.querySelector("#guideline-dialog"),
};

let bootstrap = null;
let current = null;
let draft = null;
let activeCandidateIndex = 0;
let selectedSpan = null;
let pendingAdditional = null;
let visibleItemIds = [];
let itemFilter = "all";
let searchMode = "literal";
let dirty = false;
let busy = false;
let saveTimer = null;
let saveChain = Promise.resolve();
const AUTOSAVE_DELAY_MS = 5000;

const clientInstanceId = (() => {
  if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
  const bytes = new Uint8Array(16);
  window.crypto.getRandomValues(bytes);
  return [...bytes].map(value => value.toString(16).padStart(2, "0")).join("");
})();
const clientHeaders = {"X-Review-Client-Instance": clientInstanceId};

class ApiError extends Error {
  constructor(message, status, technical) {
    super(message);
    this.status = status;
    this.technical = technical;
  }
}

async function parseError(response) {
  const payload = await response.json().catch(() => ({}));
  const friendly = {
    403: "页面会话已失效，请刷新后重试。",
    404: "审核项不存在。",
    409: "审核状态已在另一个页面发生变化。",
    422: "当前 span 决定未通过校验。",
    500: "审核服务发生内部错误。",
  }[response.status] || "请求失败。";
  return new ApiError(friendly, response.status, payload.error || `HTTP ${response.status}`);
}

async function getJson(path) {
  const response = await fetch(path, {cache: "no-store", headers: clientHeaders});
  if (!response.ok) throw await parseError(response);
  return response.json();
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json", ...clientHeaders},
    body: JSON.stringify({...payload, session_token: bootstrap.session_token}),
  });
  if (!response.ok) throw await parseError(response);
  return response;
}

function setSaveState(state, text = "") {
  const defaults = {idle: "尚未修改", dirty: "有未保存修改", saving: "正在保存…", saved: "已保存", error: "保存失败"};
  ui.saveState.dataset.state = state;
  ui.saveState.textContent = text || defaults[state] || state;
}

function showError(message, technical = "") {
  ui.requestErrorMessage.textContent = message;
  ui.requestErrorDetails.textContent = technical || message;
  ui.requestError.classList.remove("hidden");
}

function clearError() {
  ui.requestError.classList.add("hidden");
  ui.requestErrorDetails.textContent = "";
}

function summaryFor(itemId) {
  return bootstrap && bootstrap.items.find(row => row.item_id === itemId) || null;
}

function updateSummary(summary) {
  const index = bootstrap.items.findIndex(row => row.item_id === summary.item_id);
  if (index >= 0) bootstrap.items[index] = summary;
}

function renderProgress() {
  const status = bootstrap.status;
  const ratio = status.item_count ? status.confirmed_count / status.item_count : 1;
  ui.progressText.textContent = `${status.confirmed_count} / ${status.item_count} 已确认`;
  ui.sidebarProgress.textContent = `${status.confirmed_count} / ${status.item_count}`;
  ui.progressBar.style.width = `${ratio * 100}%`;
}

function refreshVisibleQueue() {
  visibleItemIds = Gold.visibleItemQueue(
    bootstrap.items,
    ui.itemSearch.value.trim(),
    searchMode,
    itemFilter,
  );
  return visibleItemIds;
}

function renderItemList() {
  refreshVisibleQueue();
  ui.itemList.replaceChildren();
  for (const itemId of visibleItemIds) {
    const summary = summaryFor(itemId);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-item";
    if (current && current.item.item_id === itemId) button.classList.add("active");
    const text = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = summary.source_item_id;
    const preview = document.createElement("small");
    preview.textContent = summary.query_preview;
    const kind = document.createElement("small");
    kind.className = "item-kind";
    kind.textContent = `${Gold.COHORT_LABELS[summary.cohort] || summary.cohort} · ${summary.candidate_count} 候选`;
    text.append(title, preview, kind);
    const dot = document.createElement("span");
    dot.className = `case-state-dot ${summary.status === "confirmed" ? "complete" : ""}`;
    button.append(text, dot);
    button.addEventListener("click", () => navigateTo(itemId));
    ui.itemList.append(button);
  }
  ui.itemEmpty.classList.toggle("hidden", visibleItemIds.length > 0);
  ui.searchStatus.textContent = `显示 ${visibleItemIds.length} / ${bootstrap.items.length} 条`;
}

function isLocked() {
  return Boolean(current && current.decision.status === "confirmed");
}

function buildPriorAudit(item) {
  ui.priorAudit.replaceChildren();
  const labels = {
    notes: "备注",
    relevance: "相关性",
    boundary: "边界",
    definition_quality: "定义",
    sense_fit: "词义",
    no_hit_verified: "遗漏核验",
  };
  ui.priorDisposition.textContent = item.prior_audit.disposition;
  for (const key of Object.keys(labels)) {
    const value = item.prior_audit[key];
    if (value === undefined || value === null || value === "") continue;
    const term = document.createElement("dt");
    term.textContent = labels[key];
    const description = document.createElement("dd");
    description.textContent = String(value);
    ui.priorAudit.append(term, description);
  }
}

function buildCandidateCard(candidate, index) {
  const card = document.createElement("article");
  card.className = "candidate-card";
  card.dataset.candidateId = candidate.candidate_id;
  card.dataset.action = draft.candidate_actions[candidate.candidate_id] || "";
  card.setAttribute("aria-label", `候选 ${index + 1}：${candidate.surface}，位置 ${candidate.span[0]} 至 ${candidate.span[1]}`);
  card.addEventListener("click", () => selectCandidate(index));
  card.addEventListener("focusin", () => selectCandidate(index));

  const head = document.createElement("div");
  head.className = "candidate-head";
  const surface = document.createElement("strong");
  surface.textContent = `“${candidate.surface}”`;
  const span = document.createElement("code");
  span.textContent = `[${candidate.span[0]}, ${candidate.span[1]})`;
  head.append(surface, span);

  const body = document.createElement("div");
  body.className = "candidate-body";
  const tags = document.createElement("div");
  tags.className = "source-tags";
  for (const sourceType of candidate.source_types) {
    const tag = document.createElement("span");
    tag.className = `source-tag ${sourceType === "known_omission" ? "omission" : ""}`;
    tag.textContent = sourceType === "known_omission" ? "已知遗漏" : "旧 matcher 命中";
    tags.append(tag);
  }
  body.append(tags);
  for (const hit of candidate.source_hits) {
    const source = document.createElement("div");
    source.className = "candidate-source";
    const title = document.createElement("strong");
    title.textContent = `${hit.term} · ${hit.category} · ${hit.lexicon_id}`;
    const definition = document.createElement("p");
    definition.textContent = hit.definition;
    source.append(title, definition);
    body.append(source);
  }
  const actions = document.createElement("div");
  actions.className = "candidate-actions";
  for (const [action, label] of [["keep", "保留该 span"], ["drop", "删除该 span"]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.action = action;
    button.textContent = `${label} · ${action === "keep" ? "Q" : "A"}`;
    button.classList.toggle("active", draft.candidate_actions[candidate.candidate_id] === action);
    button.disabled = busy || isLocked();
    button.addEventListener("click", () => {
      selectCandidate(index);
      setCandidateAction(action);
    });
    actions.append(button);
  }
  body.append(actions);
  card.append(head, body);
  return card;
}

function buildCandidateList() {
  ui.candidateList.replaceChildren();
  current.item.candidates.forEach((candidate, index) => ui.candidateList.append(buildCandidateCard(candidate, index)));
  ui.zeroCandidate.classList.toggle("hidden", current.item.candidates.length > 0);
  refreshCandidateState();
}

function refreshCandidateState() {
  const candidates = current.item.candidates;
  activeCandidateIndex = Math.max(0, Math.min(activeCandidateIndex, candidates.length - 1));
  ui.candidateList.querySelectorAll(".candidate-card").forEach((card, index) => {
    const active = index === activeCandidateIndex;
    const action = draft.candidate_actions[card.dataset.candidateId];
    card.dataset.action = action || "";
    card.classList.toggle("is-current", active);
    card.tabIndex = active ? 0 : -1;
    card.setAttribute("aria-current", String(active));
    card.querySelectorAll("button[data-action]").forEach(button => {
      const pressed = button.dataset.action === action;
      button.classList.toggle("active", pressed);
      button.setAttribute("aria-pressed", String(pressed));
    });
  });
  const values = Object.values(draft.candidate_actions);
  const decided = values.filter(Boolean).length;
  const kept = values.filter(value => value === "keep").length;
  ui.candidateStatus.textContent = `${decided} / ${values.length} 已决定；保留 ${kept}`;
  const active = candidates[activeCandidateIndex];
  ui.activeCandidateStatus.textContent = active
    ? `当前 ${activeCandidateIndex + 1} / ${candidates.length}：“${active.surface}” · ↑ ↓ 切换 · Q 保留 / A 删除`
    : "没有候选 occurrence；如有遗漏，请在原文中划词补充。";
  renderQuery();
}

function selectCandidate(index, {focus = false} = {}) {
  if (!current || busy || !current.item.candidates.length) return;
  activeCandidateIndex = Math.max(0, Math.min(index, current.item.candidates.length - 1));
  refreshCandidateState();
  if (focus) {
    const card = ui.candidateList.children[activeCandidateIndex];
    card.focus({preventScroll: true});
    card.scrollIntoView({block: "nearest"});
  }
}

function setCandidateAction(action) {
  if (!current || busy || isLocked()) return;
  const candidate = current.item.candidates[activeCandidateIndex];
  if (!candidate) return;
  draft.candidate_actions[candidate.candidate_id] = action;
  refreshCandidateState();
  markDirty();
}

function renderQuery() {
  const item = current.item;
  const candidate = item.candidates[activeCandidateIndex];
  const candidateId = candidate ? candidate.candidate_id : "";
  if (ui.queryContent.dataset.itemId === item.item_id && ui.queryContent.dataset.candidateId === candidateId) return;
  ui.queryContent.dataset.itemId = item.item_id;
  ui.queryContent.dataset.candidateId = candidateId;
  ui.queryContent.replaceChildren();
  const text = Array.from(item.query_content);
  if (candidate) {
    const mark = document.createElement("mark");
    mark.textContent = text.slice(candidate.span[0], candidate.span[1]).join("");
    ui.queryContent.append(document.createTextNode(text.slice(0, candidate.span[0]).join("")), mark,
      document.createTextNode(text.slice(candidate.span[1]).join("")));
  } else ui.queryContent.textContent = item.query_content;
  selectedSpan = null;
  ui.selectionToolbar.classList.add("hidden");
}

function captureSelection() {
  if (!current || busy || isLocked() || ui.additionalDialog.open || ui.guidelineDialog.open) return;
  const span = Gold.selectedQuerySpan(ui.queryContent, window.getSelection());
  selectedSpan = span ? {...span, itemId: current.item.item_id} : null;
  ui.selectionToolbar.classList.toggle("hidden", !span);
  ui.selectionSurface.textContent = span ? `“${span.surface}”` : "";
  ui.selectionError.textContent = span ? Gold.additionalSpanError(current.item, draft, span) : "";
  syncControls();
}

function renderAdditionalList() {
  ui.additionalList.replaceChildren();
  ui.additionalCount.textContent = draft.additional_spans.length;
  ui.additionalEmpty.classList.toggle("hidden", draft.additional_spans.length > 0);
  draft.additional_spans.forEach((row, index) => {
    const card = document.createElement("article");
    card.className = "additional-card";
    const head = document.createElement("div");
    head.className = "additional-card-head";
    const surface = document.createElement("strong");
    surface.textContent = `“${row.surface}”`;
    const position = document.createElement("code");
    position.textContent = `[${row.start}, ${row.end})`;
    const reason = document.createElement("p");
    reason.textContent = row.reason;
    const actions = document.createElement("div");
    actions.className = "additional-actions";
    const edit = document.createElement("button");
    edit.type = "button";
    edit.textContent = "修改理由";
    edit.addEventListener("click", () => openAdditionalDialog(row, index));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "移除";
    remove.addEventListener("click", () => {
      if (busy || isLocked()) return;
      draft.additional_spans.splice(index, 1);
      renderAdditionalList();
      syncControls();
      markDirty();
    });
    head.append(surface, position);
    actions.append(edit, remove);
    card.append(head, reason, actions);
    ui.additionalList.append(card);
  });
}

function openAdditionalDialog(span, editingIndex = -1) {
  if (!current || busy || isLocked() || !span || (span.itemId && span.itemId !== current.item.item_id)) return;
  pendingAdditional = {start: span.start, end: span.end, surface: span.surface, editingIndex, itemId: current.item.item_id};
  window.clearTimeout(saveTimer);
  const characters = Array.from(current.item.query_content);
  ui.additionalSurface.textContent = `“${span.surface}”`;
  ui.additionalPosition.textContent = `[${span.start}, ${span.end}) · 位置已自动计算`;
  ui.additionalContext.textContent = `${span.start > 16 ? "…" : ""}${characters.slice(Math.max(0, span.start - 16), span.start).join("")}【${span.surface}】${characters.slice(span.end, span.end + 16).join("")}${span.end + 16 < characters.length ? "…" : ""}`;
  ui.additionalReason.value = span.reason || "";
  ui.additionalDialogError.classList.add("hidden");
  ui.additionalDialogTitle.textContent = editingIndex < 0 ? "加入额外期望 span" : "修改补充理由";
  ui.submitAdditional.textContent = editingIndex < 0 ? "加入本 case" : "保存理由";
  ui.additionalDialog.showModal();
  ui.additionalReason.focus();
}

function submitAdditional(event) {
  event.preventDefault();
  if (!pendingAdditional || !current || busy || isLocked() || pendingAdditional.itemId !== current.item.item_id) return;
  const {start, end, surface, editingIndex} = pendingAdditional;
  const reason = ui.additionalReason.value.trim();
  const error = !reason ? "请填写此处需要解释的理由。"
    : Array.from(reason).length > 500 ? "理由不能超过 500 字。"
    : Gold.additionalSpanError(current.item, draft, pendingAdditional, editingIndex);
  if (error) {
    ui.additionalDialogError.textContent = error;
    ui.additionalDialogError.classList.remove("hidden");
    return;
  }
  const row = {start, end, surface, reason};
  if (editingIndex < 0) draft.additional_spans.push(row);
  else draft.additional_spans[editingIndex] = row;
  ui.additionalDialog.close();
  selectedSpan = null;
  ui.selectionToolbar.classList.add("hidden");
  renderAdditionalList();
  syncControls();
  markDirty();
}

function validationErrors(confirm = false) {
  return Gold.validateDecision(current.item, draft, {confirm});
}

function renderErrors(confirm = false) {
  const errors = validationErrors(confirm);
  ui.candidateError.textContent = errors.candidate_actions || errors.expected_spans || "";
  ui.candidateError.classList.toggle("hidden", !ui.candidateError.textContent);
  ui.additionalError.textContent = errors.additional_spans || "";
  ui.additionalError.classList.toggle("hidden", !ui.additionalError.textContent);
  ui.notesError.textContent = errors.notes || "";
  ui.notesError.classList.toggle("hidden", !ui.notesError.textContent);
  return errors;
}

function syncControls() {
  const disabled = busy || !current;
  const locked = isLocked();
  for (const control of [ui.saveDraft, ui.confirmNext, ui.keepAll, ui.dropAll, ui.clearAll, ui.selectInQuery, ui.additionalReason, ui.submitAdditional, ui.notes]) control.disabled = disabled || locked;
  ui.confirmSelection.disabled = disabled || locked || !selectedSpan || Boolean(ui.selectionError.textContent);
  ui.additionalList.querySelectorAll("button").forEach(control => { control.disabled = disabled || locked; });
  ui.candidateList.querySelectorAll("button[data-action]").forEach(control => {
    control.disabled = disabled || locked;
  });
  ui.reopen.disabled = disabled || !locked;
  const position = current ? visibleItemIds.indexOf(current.item.item_id) : -1;
  ui.previousItem.disabled = busy || position <= 0;
  ui.nextItem.disabled = busy || position < 0 || position >= visibleItemIds.length - 1;
}

function renderCurrent() {
  const item = current.item;
  const locked = isLocked();
  const position = visibleItemIds.indexOf(item.item_id);
  ui.itemPosition.textContent = position >= 0 ? `${position + 1} / ${visibleItemIds.length}` : "—";
  ui.itemId.textContent = item.source_item_id;
  ui.cohort.textContent = Gold.COHORT_LABELS[item.cohort] || item.cohort;
  renderQuery();
  buildPriorAudit(item);
  buildCandidateList();
  ui.decisionTitle.textContent = locked ? "已确认" : "待确认";
  ui.decisionStatus.textContent = locked ? "已确认" : "草稿";
  ui.decisionStatus.dataset.state = locked ? "confirmed" : "draft";
  renderAdditionalList();
  ui.notes.value = draft.notes;
  ui.lockedCard.classList.toggle("hidden", !locked);
  renderErrors(false);
  syncControls();
  renderItemList();
}

function markDirty() {
  if (!current || isLocked() || busy) return;
  dirty = true;
  setSaveState("dirty", "有未保存修改 · 5 秒后自动保存");
  renderErrors(false);
  window.clearTimeout(saveTimer);
  if (!ui.additionalDialog.open) saveTimer = window.setTimeout(() => enqueueSave(false), AUTOSAVE_DELAY_MS);
}

function applyMutation(result) {
  bootstrap.revision = result.revision;
  bootstrap.status = result.status;
  updateSummary(result.item_summary);
  current.revision = result.revision;
  current.decision = result.decision;
  draft = Gold.decisionFields(current.item, result.decision);
  dirty = false;
  renderProgress();
  renderCurrent();
  setSaveState("saved");
  clearError();
}

async function resolveConflict(error) {
  const localDraft = Common.clone(draft);
  const latest = await getJson("/api/bootstrap");
  const replay = window.confirm("服务器版本已经更新。\n\n确定：载入服务器版本\n取消：保留本地草稿并以最新 revision 重新保存");
  bootstrap = latest;
  renderProgress();
  renderItemList();
  if (replay) {
    await loadItem(current.item.item_id, {skipFlush: true});
    showError("已载入服务器版本", error.technical);
  } else {
    current.revision = latest.revision;
    draft = localDraft;
    dirty = true;
    renderCurrent();
    setSaveState("dirty", "保留本地草稿 · 请重新保存");
    showError("已保留本地草稿，请核对后重新保存。", error.technical);
  }
}

async function performSave(confirm) {
  if (!current || isLocked() || ui.additionalDialog.open) return false;
  const errors = renderErrors(confirm);
  if (Common.hasErrors(errors)) {
    setSaveState("error", "请先修正审核字段");
    return false;
  }
  busy = true;
  syncControls();
  setSaveState("saving");
  try {
    const response = await postJson("/api/save", {
      expected_revision: bootstrap.revision,
      item_id: current.item.item_id,
      decision: draft,
      confirm,
    });
    applyMutation(await response.json());
    return true;
  } catch (error) {
    setSaveState("error");
    if (error.status === 409) await resolveConflict(error);
    else showError(error.message, error.technical);
    return false;
  } finally {
    busy = false;
    syncControls();
  }
}

function enqueueSave(confirm) {
  window.clearTimeout(saveTimer);
  saveTimer = null;
  saveChain = saveChain.catch(() => undefined).then(() => performSave(confirm));
  return saveChain;
}

async function loadItem(itemId, {skipFlush = false} = {}) {
  if (!skipFlush && dirty && current && !isLocked()) {
    const saved = await enqueueSave(false);
    if (!saved) return;
  }
  busy = true;
  syncControls();
  try {
    current = await getJson(`/api/items/${encodeURIComponent(itemId)}`);
    bootstrap.revision = current.revision;
    draft = Gold.decisionFields(current.item, current.decision);
    activeCandidateIndex = 0;
    selectedSpan = null;
    dirty = false;
    setSaveState("idle");
    clearError();
    renderCurrent();
  } catch (error) {
    showError(error.message, error.technical);
  } finally {
    busy = false;
    syncControls();
  }
}

async function navigateTo(itemId) {
  if (busy || ui.additionalDialog.open || !itemId || (current && current.item.item_id === itemId)) return;
  await loadItem(itemId);
  ui.sidebar.classList.remove("open");
  ui.sidebarBackdrop.classList.add("hidden");
}

async function navigateOffset(offset) {
  const index = current ? visibleItemIds.indexOf(current.item.item_id) : -1;
  const target = visibleItemIds[index + offset];
  if (target) await navigateTo(target);
}

async function confirmAndContinue() {
  if (!current || busy || isLocked() || ui.additionalDialog.open) return;
  const itemId = current.item.item_id;
  const saved = await enqueueSave(true);
  if (!saved) return;
  const next = Gold.nextUnfinishedItemId(visibleItemIds, bootstrap.items, itemId);
  if (next) await loadItem(next, {skipFlush: true});
}

async function reopenCurrent() {
  const reason = window.prompt("请输入重新打开该决定的理由：", "更正 span 决定");
  if (!reason || !reason.trim()) return;
  busy = true;
  syncControls();
  try {
    const response = await postJson("/api/reopen", {
      expected_revision: bootstrap.revision,
      item_id: current.item.item_id,
      reason: reason.trim(),
    });
    applyMutation(await response.json());
  } catch (error) {
    if (error.status === 409) await resolveConflict(error);
    else showError(error.message, error.technical);
  } finally {
    busy = false;
    syncControls();
  }
}

async function exportSnapshot() {
  try {
    const response = await postJson("/api/export", {expected_revision: bootstrap.revision});
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `span-gold-review-${bootstrap.revision.slice(0, 12)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    if (error.status === 409) await resolveConflict(error);
    else showError(error.message, error.technical);
  }
}

function setAllCandidates(value) {
  if (!current || busy || isLocked()) return;
  for (const candidateId of Object.keys(draft.candidate_actions)) draft.candidate_actions[candidateId] = value;
  refreshCandidateState();
  markDirty();
}

function bindEvents() {
  ui.itemSearch.addEventListener("input", () => { renderItemList(); renderCurrent(); });
  ui.searchMode.addEventListener("click", () => {
    searchMode = Common.nextSearchMode(searchMode);
    ui.searchMode.textContent = `${Common.SEARCH_MODE_LABELS[searchMode]} ↻`;
    renderItemList();
  });
  ui.itemFilters.addEventListener("click", event => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    itemFilter = button.dataset.filter;
    ui.itemFilters.querySelectorAll("button").forEach(row => row.classList.toggle("active", row === button));
    renderItemList();
  });
  ui.keepAll.addEventListener("click", () => setAllCandidates("keep"));
  ui.dropAll.addEventListener("click", () => setAllCandidates("drop"));
  ui.clearAll.addEventListener("click", () => setAllCandidates(null));
  document.addEventListener("selectionchange", captureSelection);
  ui.confirmSelection.addEventListener("pointerdown", event => event.preventDefault());
  ui.confirmSelection.addEventListener("click", () => openAdditionalDialog(selectedSpan));
  ui.selectInQuery.addEventListener("click", () => {
    ui.queryContent.scrollIntoView({block: "center", behavior: "smooth"});
    ui.queryContent.focus({preventScroll: true});
  });
  ui.additionalForm.addEventListener("submit", submitAdditional);
  ui.cancelAdditional.addEventListener("click", () => ui.additionalDialog.close());
  ui.additionalDialog.addEventListener("close", () => {
    pendingAdditional = null;
    if (dirty) markDirty();
  });
  ui.notes.addEventListener("input", () => { draft.notes = ui.notes.value; markDirty(); });
  ui.saveDraft.addEventListener("click", () => enqueueSave(false));
  ui.confirmNext.addEventListener("click", confirmAndContinue);
  ui.reopen.addEventListener("click", reopenCurrent);
  ui.previousItem.addEventListener("click", () => navigateOffset(-1));
  ui.nextItem.addEventListener("click", () => navigateOffset(1));
  ui.exportSnapshot.addEventListener("click", exportSnapshot);
  ui.guidelines.addEventListener("click", () => ui.guidelineDialog.showModal());
  ui.sidebarToggle.addEventListener("click", () => { ui.sidebar.classList.add("open"); ui.sidebarBackdrop.classList.remove("hidden"); });
  for (const control of [ui.sidebarClose, ui.sidebarBackdrop]) control.addEventListener("click", () => { ui.sidebar.classList.remove("open"); ui.sidebarBackdrop.classList.add("hidden"); });
  window.addEventListener("beforeunload", event => {
    if (dirty || busy || ui.additionalDialog.open) { event.preventDefault(); event.returnValue = ""; }
  });
  window.addEventListener("keydown", event => {
    const action = Gold.reviewShortcut(event, {
      dialogOpen: Boolean(document.querySelector("dialog[open]")),
      loaded: Boolean(current), busy, locked: isLocked(),
    });
    if (!action) return;
    event.preventDefault();
    const handlers = {
      "keep-all": () => setAllCandidates("keep"),
      "drop-all": () => setAllCandidates("drop"),
      "clear-all": () => setAllCandidates(null),
      "previous-candidate": () => selectCandidate(activeCandidateIndex - 1, {focus: true}),
      "next-candidate": () => selectCandidate(activeCandidateIndex + 1, {focus: true}),
      keep: () => setCandidateAction("keep"), drop: () => setCandidateAction("drop"),
      "previous-item": () => navigateOffset(-1), "next-item": () => navigateOffset(1),
      save: () => enqueueSave(false), confirm: confirmAndContinue,
      help: () => ui.guidelineDialog.showModal(),
    };
    handlers[action]();
  });
}

async function start() {
  bindEvents();
  try {
    bootstrap = await getJson("/api/bootstrap");
    renderProgress();
    renderItemList();
    const first = Gold.nextUnfinishedItemId(visibleItemIds, bootstrap.items) || visibleItemIds[0];
    if (first) await loadItem(first, {skipFlush: true});
  } catch (error) {
    showError(error.message, error.technical);
  }
}

start();
