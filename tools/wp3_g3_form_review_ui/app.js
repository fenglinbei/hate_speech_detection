"use strict";

const C = window.ReviewCore;
const G = window.G3ReviewCore;

const ui = {
  sidebarToggle: document.querySelector("#sidebar-toggle"),
  sidebar: document.querySelector("#item-sidebar"),
  sidebarClose: document.querySelector("#sidebar-close"),
  sidebarBackdrop: document.querySelector("#sidebar-backdrop"),
  progressText: document.querySelector("#top-progress-text"),
  progressBar: document.querySelector("#progress-bar"),
  sidebarProgress: document.querySelector("#sidebar-progress"),
  saveState: document.querySelector("#save-state"),
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
  relationFamily: document.querySelector("#relation-family"),
  relationSurface: document.querySelector("#relation-surface"),
  relationCanonical: document.querySelector("#relation-canonical"),
  relationMeta: document.querySelector("#relation-meta"),
  evidenceStatus: document.querySelector("#evidence-status"),
  evidenceList: document.querySelector("#evidence-list"),
  decisionTitle: document.querySelector("#decision-title"),
  decisionStatus: document.querySelector("#decision-status"),
  saveDraft: document.querySelector("#save-draft"),
  confirmNext: document.querySelector("#confirm-next"),
  requestError: document.querySelector("#request-error"),
  requestErrorMessage: document.querySelector("#request-error-message"),
  requestErrorDetails: document.querySelector("#request-error-details"),
  decisionEditor: document.querySelector("#decision-editor"),
  actionGrid: document.querySelector("#action-grid"),
  actionError: document.querySelector("#action-error"),
  proposalFields: document.querySelector("#proposal-fields"),
  proposalModeHint: document.querySelector("#proposal-mode-hint"),
  surface: document.querySelector("#surface"),
  surfaceError: document.querySelector("#surface-error"),
  canonical: document.querySelector("#canonical"),
  canonicalError: document.querySelector("#canonical-error"),
  family: document.querySelector("#family"),
  familyError: document.querySelector("#family-error"),
  scan: document.querySelector("#scan"),
  proposalError: document.querySelector("#proposal-error"),
  evidenceChecks: document.querySelector("#evidence-checks"),
  evidenceError: document.querySelector("#evidence-error"),
  notes: document.querySelector("#notes"),
  lockedCard: document.querySelector("#locked-card"),
  reopen: document.querySelector("#reopen"),
  actionDialog: document.querySelector("#action-dialog"),
  dialogTitle: document.querySelector("#dialog-title"),
  dialogMessage: document.querySelector("#dialog-message"),
  dialogInput: document.querySelector("#dialog-input"),
  dialogInputError: document.querySelector("#dialog-input-error"),
  conflictDialog: document.querySelector("#conflict-dialog"),
  conflictDetails: document.querySelector("#conflict-details"),
  guidelines: document.querySelector("#guidelines"),
  guidelineDialog: document.querySelector("#guideline-dialog"),
  shortcuts: document.querySelector("#shortcuts"),
  shortcutDialog: document.querySelector("#shortcut-dialog"),
};

class ApiError extends Error {
  constructor(message, status, technical) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.technical = technical;
  }
}

let bootstrap = null;
let current = null;
let draft = null;
let currentErrors = {};
let visibleItemIds = [];
let itemFilter = "all";
let searchMode = "literal";
let searchTimer = null;
let localDirty = false;
let editVersion = 0;
let loadSequence = 0;
let saveBusy = false;
let confirmBusy = false;
let handlingConflict = false;
const clientInstanceId = (() => {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  const bytes = new Uint8Array(16);
  window.crypto.getRandomValues(bytes);
  return [...bytes].map(value => value.toString(16).padStart(2, "0")).join("");
})();
const clientHeaders = {"X-Review-Client-Instance": clientInstanceId};

function nowText() {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date());
}

function decisionDraft(stored) {
  return {
    action: stored.action,
    surface: stored.surface,
    canonical: stored.canonical,
    family: stored.family,
    phonetic_scan_enabled: Boolean(stored.phonetic_scan_enabled),
    evidence_ids: [...stored.evidence_ids],
    notes: stored.notes || "",
  };
}

function setSaveState(state, message = "") {
  const defaults = {
    idle: "尚未修改",
    dirty: "有未保存修改",
    saving: "正在保存…",
    saved: `已保存 ${nowText()}`,
    error: "保存失败 · 等待处理",
  };
  ui.saveState.dataset.state = state;
  ui.saveState.textContent = message || defaults[state] || state;
  saveBusy = state === "saving";
  if (state !== "error") ui.saveState.removeAttribute("title");
  if (state === "saved") clearRequestError();
  syncControls();
}

function showRequestError(message, technical = "") {
  ui.requestErrorMessage.textContent = message;
  ui.requestErrorDetails.textContent = technical || message;
  ui.requestError.classList.remove("hidden");
}

function clearRequestError() {
  ui.requestError.classList.add("hidden");
  ui.requestErrorDetails.textContent = "";
}

function setButtonBusy(button, busy, text = "") {
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = text || button.textContent;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    delete button.dataset.originalText;
  }
}

async function parseError(response) {
  const payload = await response.json().catch(() => ({}));
  const technical = payload.error || `HTTP ${response.status}`;
  const friendly = {
    403: "当前登录或页面会话已经失效，请刷新页面后重试。",
    404: "请求的关系审核项不存在。",
    409: "审核数据已在其他页面发生变化。",
    422: "当前决定未通过校验，请检查关系字段和冻结证据。",
    500: "审核服务发生内部错误。",
  }[response.status] || "请求失败，请稍后重试。";
  return new ApiError(friendly, response.status, technical);
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
  return response.json();
}

function summaryIndex(itemId) {
  return bootstrap.items.findIndex(row => row.item_id === itemId);
}

function summaryFor(itemId) {
  const index = summaryIndex(itemId);
  return index < 0 ? null : bootstrap.items[index];
}

function updateSummary(summary) {
  const index = summaryIndex(summary.item_id);
  if (index >= 0) bootstrap.items[index] = summary;
}

function renderProgress() {
  if (!bootstrap) return;
  const status = bootstrap.status;
  const ratio = status.item_count ? status.confirmed_count / status.item_count : 1;
  const defer = status.deferred_count ? `；defer ${status.deferred_count}` : "";
  ui.progressText.textContent = `${status.confirmed_count} / ${status.item_count} 已确认${defer}`;
  ui.progressBar.style.width = `${ratio * 100}%`;
  ui.sidebarProgress.textContent = `${status.confirmed_count} / ${status.item_count}`;
}

function refreshVisibleQueue() {
  if (!bootstrap) {
    visibleItemIds = [];
    return visibleItemIds;
  }
  const query = ui.itemSearch.value.trim();
  const ordered = query ? G.searchItemIds(bootstrap.items, query, searchMode) : null;
  visibleItemIds = G.visibleItemQueue(bootstrap.items, ordered, itemFilter);
  return visibleItemIds;
}

function renderSearchMode() {
  const label = C.SEARCH_MODE_LABELS[searchMode] || searchMode;
  const next = C.nextSearchMode(searchMode);
  ui.searchMode.textContent = `${label} ↻`;
  ui.searchMode.title = `点击切换为${C.SEARCH_MODE_LABELS[next]}模式`;
  ui.itemSearch.placeholder = {
    literal: "输入形式、标准形式、来源或编号",
    all_terms: "用空格分隔多个关键词",
    fuzzy: "输入允许跳字匹配的形式或来源",
  }[searchMode];
}

function updateSearchStatus() {
  if (!bootstrap) return;
  const query = ui.itemSearch.value.trim();
  if (!query) {
    ui.searchStatus.textContent = itemFilter === "all"
      ? `全部 ${visibleItemIds.length} 条`
      : `筛选后 ${visibleItemIds.length} 条`;
  } else {
    ui.searchStatus.textContent = visibleItemIds.length
      ? `${visibleItemIds.length} 条匹配结果`
      : "无匹配结果";
  }
}

function revealActiveItem() {
  if (!current) return;
  window.requestAnimationFrame(() => {
    const button = ui.itemList.querySelector(`[data-item-id="${CSS.escape(current.item.item_id)}"]`);
    if (button) button.scrollIntoView({block: "nearest"});
  });
}

function renderItemList({revealCurrent = false} = {}) {
  if (!bootstrap) return;
  const previousScrollTop = ui.itemList.scrollTop;
  refreshVisibleQueue();
  ui.itemList.replaceChildren();
  for (const itemId of visibleItemIds) {
    const summary = summaryFor(itemId);
    if (!summary) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-item";
    button.dataset.itemId = itemId;
    button.dataset.action = summary.action;
    if (current && current.item.item_id === itemId) button.classList.add("active");

    const title = document.createElement("strong");
    title.textContent = `${summary.surface} → ${summary.canonical}`;
    const dot = document.createElement("span");
    dot.className = "case-state-dot" + (summary.status === "confirmed" ? " complete" : "");
    const source = document.createElement("small");
    const publishers = summary.publishers && summary.publishers.length
      ? summary.publishers.join(" / ")
      : summary.item_id;
    source.textContent = `${publishers} · ${summary.proposed_family}`;
    const action = document.createElement("small");
    action.className = "case-action-label";
    action.textContent = summary.status === "confirmed"
      ? G.ACTION_LABELS[summary.action]
      : "待确认";
    button.append(title, dot, source, action);
    button.disabled = saveBusy || confirmBusy;
    button.addEventListener("click", () => loadItemById(itemId));
    ui.itemList.append(button);
  }
  ui.itemEmpty.textContent = ui.itemSearch.value.trim()
    ? "没有匹配当前搜索和筛选条件的关系审核项"
    : "没有匹配当前筛选条件的关系审核项";
  ui.itemEmpty.classList.toggle("hidden", visibleItemIds.length !== 0);
  updateSearchStatus();
  syncControls();
  if (revealCurrent) revealActiveItem();
  else ui.itemList.scrollTop = previousScrollTop;
}

function addMetaChip(text) {
  if (!text) return;
  const chip = document.createElement("span");
  chip.className = "meta-chip";
  chip.textContent = text;
  ui.relationMeta.append(chip);
}

function renderRelation() {
  const item = current.item;
  ui.itemId.textContent = item.item_id;
  const globalIndex = bootstrap.items.findIndex(row => row.item_id === item.item_id);
  ui.itemPosition.textContent = `${globalIndex + 1} / ${bootstrap.items.length}`;
  ui.relationSurface.textContent = item.surface;
  ui.relationCanonical.textContent = item.canonical;
  ui.relationFamily.textContent = G.FAMILY_LABELS[item.proposed_family] || item.proposed_family;
  ui.relationMeta.replaceChildren();
  addMetaChip(`family=${item.proposed_family}`);
  addMetaChip(`phonetic_scan=${item.phonetic_scan_enabled ? "on" : "off"}`);
  addMetaChip(`${item.evidence_ids.length} 条冻结证据`);
  for (const publisher of [...new Set(current.evidence.map(row => row.publisher))]) {
    addMetaChip(`来源：${publisher}`);
  }
}

function renderEvidence() {
  ui.evidenceStatus.textContent = `${current.evidence.length} 条证据 · 仅使用本地冻结快照`;
  ui.evidenceList.replaceChildren();
  ui.evidenceChecks.replaceChildren();
  for (const row of current.evidence) {
    const article = document.createElement("article");
    article.className = "evidence-item";
    article.dataset.evidenceCard = row.evidence_id;
    const head = document.createElement("div");
    head.className = "evidence-head";
    const publisher = document.createElement("strong");
    publisher.textContent = row.publisher;
    const evidenceId = document.createElement("code");
    evidenceId.textContent = row.evidence_id;
    head.append(publisher, evidenceId);
    const quote = document.createElement("blockquote");
    quote.textContent = row.quote;
    const note = document.createElement("p");
    note.className = "evidence-note";
    note.textContent = row.relation_note;
    const provenance = document.createElement("p");
    provenance.className = "provenance";
    const parts = [];
    if (row.source_role) parts.push(`source_role=${row.source_role}`);
    if (row.acquisition_mode) parts.push(`acquisition=${row.acquisition_mode}`);
    if (row.component_id) parts.push(`component=${row.component_id}`);
    if (row.relation_contract) parts.push(`contract=${row.relation_contract}`);
    provenance.textContent = parts.join(" · ");
    article.append(head, quote, note, provenance);
    ui.evidenceList.append(article);

    const label = document.createElement("label");
    label.className = "evidence-check";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.dataset.evidence = row.evidence_id;
    const text = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = row.evidence_id;
    const small = document.createElement("small");
    small.textContent = `${row.publisher}${row.component_id ? ` · ${row.component_id}` : ""}`;
    text.append(strong, small);
    label.append(checkbox, text);
    ui.evidenceChecks.append(label);
  }
}

function setFieldError(node, message) {
  node.textContent = message || "";
  node.classList.toggle("hidden", !message);
}

function validateCurrent() {
  if (!current || !draft) return false;
  currentErrors = G.validateDecision(current.item, current.evidence, draft);
  return !C.hasErrors(currentErrors);
}

function renderValidationErrors() {
  setFieldError(ui.actionError, currentErrors.action);
  setFieldError(ui.surfaceError, currentErrors.surface);
  setFieldError(ui.canonicalError, currentErrors.canonical);
  setFieldError(ui.familyError, currentErrors.family);
  setFieldError(ui.proposalError, currentErrors.proposal);
  setFieldError(ui.evidenceError, currentErrors.evidence_ids);
}

function isLocked() {
  return Boolean(current && current.decision.status === "confirmed");
}

function updateEvidenceHighlights() {
  const selected = new Set(draft ? draft.evidence_ids : []);
  ui.evidenceList.querySelectorAll("[data-evidence-card]").forEach(card => {
    card.classList.toggle("selected", selected.has(card.dataset.evidenceCard));
  });
}

function renderDecisionControls() {
  if (!current || !draft) return;
  const locked = isLocked();
  const editing = draft.action === "edit";
  ui.decisionTitle.textContent = G.ACTION_LABELS[draft.action] || "等待选择";
  ui.decisionStatus.textContent = locked ? "已确认" : "草稿";
  ui.decisionStatus.dataset.state = locked ? "confirmed" : "draft";
  ui.actionGrid.querySelectorAll("[data-action]").forEach(button => {
    button.classList.toggle("active", button.dataset.action === draft.action);
  });
  ui.surface.value = draft.surface;
  ui.canonical.value = draft.canonical;
  ui.family.value = draft.family;
  ui.scan.checked = draft.phonetic_scan_enabled;
  ui.notes.value = draft.notes;
  ui.evidenceChecks.querySelectorAll("[data-evidence]").forEach(input => {
    input.checked = draft.evidence_ids.includes(input.dataset.evidence);
  });
  ui.proposalFields.classList.toggle("readonly", !editing);
  ui.proposalModeHint.textContent = editing
    ? "修订内容必须在同一冻结证据中重放"
    : "接受、驳回和暂缓须原样保留";
  ui.lockedCard.classList.toggle("hidden", !locked);
  validateCurrent();
  renderValidationErrors();
  updateEvidenceHighlights();
  syncControls();
}

function renderCurrent() {
  renderRelation();
  renderEvidence();
  renderDecisionControls();
  renderItemList({revealCurrent: true});
}

function syncControls() {
  const available = Boolean(current && draft);
  const locked = isLocked();
  const navigationBusy = saveBusy || confirmBusy;
  const editing = available && draft.action === "edit" && !locked;
  ui.saveDraft.disabled = !available || locked || navigationBusy;
  ui.confirmNext.disabled = !available || locked || navigationBusy;
  ui.actionGrid.querySelectorAll("button").forEach(button => {
    button.disabled = !available || locked || navigationBusy;
  });
  for (const input of [ui.surface, ui.canonical, ui.family, ui.scan]) {
    input.disabled = !editing || navigationBusy;
  }
  ui.notes.disabled = !available || locked || navigationBusy;
  ui.evidenceChecks.querySelectorAll("[data-evidence]").forEach(input => {
    input.disabled = !editing || navigationBusy;
    input.closest(".evidence-check").classList.toggle("disabled", input.disabled);
  });
  ui.reopen.disabled = !available || !locked || navigationBusy;
  const position = available ? visibleItemIds.indexOf(current.item.item_id) : -1;
  ui.previousItem.disabled = navigationBusy || position <= 0;
  ui.nextItem.disabled = navigationBusy || position < 0 || position >= visibleItemIds.length - 1;
  ui.itemList.querySelectorAll(".case-item").forEach(button => {
    button.disabled = navigationBusy;
  });
  ui.itemSearch.disabled = navigationBusy;
  ui.searchMode.disabled = navigationBusy;
  ui.itemFilters.querySelectorAll("button").forEach(button => {
    button.disabled = navigationBusy;
  });
}

function setConfirmationLock(locked) {
  confirmBusy = locked;
  ui.decisionEditor.inert = locked;
  syncControls();
}

function updateDraftFromForm() {
  if (!draft) return;
  draft.surface = ui.surface.value;
  draft.canonical = ui.canonical.value;
  draft.family = ui.family.value;
  draft.phonetic_scan_enabled = ui.scan.checked;
  draft.evidence_ids = [...ui.evidenceChecks.querySelectorAll("[data-evidence]:checked")]
    .map(input => input.dataset.evidence);
  draft.notes = ui.notes.value;
}

function makeTask(confirm) {
  return {
    version: editVersion,
    item_id: current.item.item_id,
    expected_revision: bootstrap.revision,
    decision: {
      action: draft.action,
      surface: String(draft.surface || "").trim(),
      canonical: String(draft.canonical || "").trim(),
      family: draft.family,
      phonetic_scan_enabled: Boolean(draft.phonetic_scan_enabled),
      evidence_ids: [...draft.evidence_ids],
      notes: draft.notes || "",
    },
    confirm,
  };
}

function applyMutation(task, result) {
  if (saveQueue.pending && saveQueue.pending.expected_revision === task.expected_revision) {
    saveQueue.pending.expected_revision = result.revision;
  }
  bootstrap.revision = result.revision;
  bootstrap.status = result.status;
  updateSummary(result.item_summary);
  if (current && current.item.item_id === task.item_id) {
    current.revision = result.revision;
    current.decision = result.decision;
    if (task.version === editVersion) {
      localDirty = false;
      draft = decisionDraft(result.decision);
    }
  }
  renderProgress();
  renderItemList({revealCurrent: true});
}

async function executeMutation(task) {
  // A task may have waited behind a slower autosave. Rebase it to the latest
  // revision observed by this page immediately before it is sent. A change
  // made by another page still conflicts because this page has not observed it.
  task.expected_revision = bootstrap.revision;
  const result = await postJson("/api/save", {
    expected_revision: task.expected_revision,
    item_id: task.item_id,
    decision: task.decision,
    confirm: task.confirm,
  });
  applyMutation(task, result);
  return result;
}

const saveQueue = new C.MutationQueue(
  executeMutation,
  state => {
    if (state === "saved" && localDirty) setSaveState("dirty");
    else setSaveState(state);
  },
  error => handleMutationError(error),
);

function markDirty() {
  if (confirmBusy || !current || isLocked()) return;
  localDirty = true;
  editVersion += 1;
  const valid = validateCurrent();
  renderValidationErrors();
  updateEvidenceHighlights();
  setSaveState("dirty");
  if (valid) {
    saveQueue.schedule(makeTask(false), 700);
  } else if (saveQueue.pending) {
    saveQueue.discard();
    setSaveState("dirty");
  }
}

async function flushChanges() {
  if (!localDirty && !saveQueue.dirty) return true;
  if (!validateCurrent()) {
    renderValidationErrors();
    setSaveState("error", "请先修正表单");
    return false;
  }
  if (!saveQueue.pending && !saveQueue.inflight && localDirty) {
    saveQueue.schedule(makeTask(false), 0);
  }
  try {
    await saveQueue.flush();
    return true;
  } catch (error) {
    await handleMutationError(error);
    return false;
  }
}

async function handleMutationError(error) {
  setSaveState("error", error.message);
  ui.saveState.title = error.technical || error.message;
  showRequestError(error.message, error.technical);
  if (error.status === 409 && !handlingConflict) await resolveConflict(error);
}

async function resolveConflict(error) {
  handlingConflict = true;
  const task = saveQueue.pending ? C.clone(saveQueue.pending) : null;
  const itemId = current && current.item.item_id;
  ui.conflictDetails.textContent = error.technical || error.message;
  ui.conflictDialog.showModal();
  const choice = await new Promise(resolve => {
    ui.conflictDialog.addEventListener(
      "close",
      () => resolve(ui.conflictDialog.returnValue),
      {once: true},
    );
  });
  try {
    saveQueue.discard();
    bootstrap = await getJson("/api/bootstrap");
    renderProgress();
    renderItemList();
    if (choice === "retry" && task) {
      task.expected_revision = bootstrap.revision;
      await saveQueue.runNow(task);
      if (current && current.item.item_id === task.item_id) renderDecisionControls();
      setSaveState("saved");
    } else if (itemId && summaryFor(itemId)) {
      current = await getJson(`/api/items/${encodeURIComponent(itemId)}`);
      draft = decisionDraft(current.decision);
      localDirty = false;
      currentErrors = {};
      renderCurrent();
      setSaveState("saved", "已采用服务器版本");
    }
  } catch (retryError) {
    setSaveState("error", retryError.message);
    showRequestError(retryError.message, retryError.technical);
  } finally {
    handlingConflict = false;
  }
}

async function loadItemById(itemId, {skipFlush = false, force = false} = {}) {
  if (!bootstrap || !summaryFor(itemId)) return false;
  if (!force && current && current.item.item_id === itemId) {
    closeSidebar();
    return true;
  }
  if (!skipFlush && !(await flushChanges())) return false;
  const sequence = ++loadSequence;
  try {
    let row = await getJson(`/api/items/${encodeURIComponent(itemId)}`);
    if (sequence !== loadSequence) return false;
    if (row.revision !== bootstrap.revision) {
      bootstrap = await getJson("/api/bootstrap");
      if (sequence !== loadSequence) return false;
      renderProgress();
      renderItemList();
      row = await getJson(`/api/items/${encodeURIComponent(itemId)}`);
      if (sequence !== loadSequence) return false;
    }
    saveQueue.discard();
    current = row;
    draft = decisionDraft(row.decision);
    localDirty = false;
    currentErrors = {};
    renderCurrent();
    setSaveState(isLocked() ? "saved" : "idle", isLocked() ? "已确认并锁定" : "尚未修改");
    closeSidebar();
    return true;
  } catch (error) {
    setSaveState("error", error.message);
    showRequestError(error.message, error.technical);
    return false;
  }
}

function nextOpenItemId(currentItemId = current && current.item.item_id, queue = visibleItemIds) {
  return G.nextUnfinishedItemId(queue, bootstrap.items, currentItemId);
}

async function navigateVisibleItems(offset) {
  if (!current || !visibleItemIds.length || saveBusy || confirmBusy) return false;
  const index = visibleItemIds.indexOf(current.item.item_id);
  const target = index < 0 ? 0 : index + offset;
  if (target < 0 || target >= visibleItemIds.length) return false;
  return loadItemById(visibleItemIds[target]);
}

async function saveCurrentDraft() {
  if (saveBusy || confirmBusy || !current || isLocked()) return;
  updateDraftFromForm();
  if (!validateCurrent()) {
    renderValidationErrors();
    setSaveState("error", "请先修正表单");
    return;
  }
  if (!localDirty && !saveQueue.dirty) {
    setSaveState("saved", "草稿没有新修改");
    return;
  }
  localDirty = true;
  editVersion += 1;
  try {
    await saveQueue.runNow(makeTask(false));
    renderDecisionControls();
  } catch (error) {
    await handleMutationError(error);
  }
}

async function confirmCurrent() {
  if (saveBusy || confirmBusy || !current || isLocked()) return;
  updateDraftFromForm();
  if (!validateCurrent()) {
    renderValidationErrors();
    setSaveState("error", "请先修正表单");
    return;
  }
  localDirty = true;
  editVersion += 1;
  const queueBeforeConfirm = [...visibleItemIds];
  const itemId = current.item.item_id;
  const task = makeTask(true);
  setConfirmationLock(true);
  setButtonBusy(ui.confirmNext, true, "正在确认…");
  try {
    await saveQueue.runNow(task);
    renderDecisionControls();
    const nextItemId = nextOpenItemId(itemId, queueBeforeConfirm);
    if (nextItemId && nextItemId !== itemId) {
      await loadItemById(nextItemId, {skipFlush: true});
    } else {
      renderCurrent();
      setSaveState("saved", "已确认；当前结果中没有其他待确认项");
    }
  } catch (error) {
    await handleMutationError(error);
  } finally {
    setConfirmationLock(false);
    setButtonBusy(ui.confirmNext, false);
    syncControls();
  }
}

function chooseAction(action) {
  if (!current || !draft || isLocked() || confirmBusy || saveBusy) return;
  if (!Object.prototype.hasOwnProperty.call(G.ACTION_LABELS, action)) return;
  draft = G.decisionForAction(action, current.item, draft);
  currentErrors = {};
  renderDecisionControls();
  markDirty();
}

function openReopenDialog() {
  ui.dialogTitle.textContent = "重新打开关系决定";
  ui.dialogMessage.textContent = "该操作会写入 amendment 审计记录。请输入本次修改原因。";
  ui.dialogInput.value = "";
  ui.dialogInputError.classList.add("hidden");
  return new Promise(resolve => {
    const onSubmit = event => {
      if (
        event.submitter &&
        event.submitter.value === "confirm" &&
        !ui.dialogInput.value.trim()
      ) {
        event.preventDefault();
        ui.dialogInputError.classList.remove("hidden");
        ui.dialogInput.focus();
      }
    };
    const onClose = () => {
      ui.actionDialog.removeEventListener("submit", onSubmit);
      resolve(
        ui.actionDialog.returnValue === "confirm"
          ? ui.dialogInput.value.trim()
          : null,
      );
    };
    ui.actionDialog.addEventListener("submit", onSubmit);
    ui.actionDialog.addEventListener("close", onClose, {once: true});
    ui.actionDialog.showModal();
    window.setTimeout(() => ui.dialogInput.focus(), 0);
  });
}

async function reopenCurrent() {
  if (!current || !isLocked() || saveBusy || confirmBusy) return;
  const reason = await openReopenDialog();
  if (!reason) return;
  setSaveState("saving", "正在重新打开…");
  try {
    const result = await postJson("/api/reopen", {
      expected_revision: bootstrap.revision,
      item_id: current.item.item_id,
      reason,
    });
    bootstrap.revision = result.revision;
    bootstrap.status = result.status;
    updateSummary(result.item_summary);
    current.revision = result.revision;
    current.decision = result.decision;
    draft = decisionDraft(result.decision);
    localDirty = false;
    currentErrors = {};
    renderProgress();
    renderCurrent();
    setSaveState("saved", "已重新打开并记录修改原因");
  } catch (error) {
    await handleMutationError(error);
  }
}

async function applyQueueChange() {
  renderItemList();
  if (!current || visibleItemIds.includes(current.item.item_id) || !visibleItemIds.length) {
    revealActiveItem();
    return;
  }
  await loadItemById(visibleItemIds[0]);
}

function scheduleSearch() {
  window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(() => {
    applyQueueChange().catch(error => showRequestError(error.message, error.technical));
  }, 180);
}

function cycleSearchMode() {
  searchMode = C.nextSearchMode(searchMode);
  renderSearchMode();
  applyQueueChange().catch(error => showRequestError(error.message, error.technical));
}

function openSidebar() {
  ui.sidebar.classList.add("open");
  ui.sidebarBackdrop.classList.remove("hidden");
}

function closeSidebar() {
  ui.sidebar.classList.remove("open");
  ui.sidebarBackdrop.classList.add("hidden");
}

function handleKeydown(event) {
  if (saveBusy || confirmBusy || document.querySelector("dialog[open]")) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveCurrentDraft();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    confirmCurrent();
    return;
  }
  if (C.isTextEntry(event.target)) return;
  if (event.key === "?") {
    event.preventDefault();
    ui.shortcutDialog.showModal();
    return;
  }
  if (event.key === "[") {
    event.preventDefault();
    navigateVisibleItems(-1);
    return;
  }
  if (event.key === "]") {
    event.preventDefault();
    navigateVisibleItems(1);
    return;
  }
  const action = G.actionForKey(event.key);
  if (action) {
    event.preventDefault();
    chooseAction(action);
  }
}

async function initialize() {
  try {
    bootstrap = await getJson("/api/bootstrap");
    renderSearchMode();
    renderProgress();
    renderItemList();
    const firstItemId = nextOpenItemId(null) || visibleItemIds[0];
    if (firstItemId) await loadItemById(firstItemId, {skipFlush: true});
  } catch (error) {
    setSaveState("error", error.message);
    showRequestError(error.message, error.technical);
  }
}

ui.sidebarToggle.addEventListener("click", openSidebar);
ui.sidebarClose.addEventListener("click", closeSidebar);
ui.sidebarBackdrop.addEventListener("click", closeSidebar);
ui.itemSearch.addEventListener("input", scheduleSearch);
ui.searchMode.addEventListener("click", cycleSearchMode);
ui.itemFilters.addEventListener("click", event => {
  const button = event.target.closest("button[data-filter]");
  if (!button) return;
  itemFilter = button.dataset.filter;
  ui.itemFilters.querySelectorAll("button").forEach(row => {
    row.classList.toggle("active", row.dataset.filter === itemFilter);
  });
  applyQueueChange().catch(error => showRequestError(error.message, error.technical));
});
ui.previousItem.addEventListener("click", () => navigateVisibleItems(-1));
ui.nextItem.addEventListener("click", () => navigateVisibleItems(1));
ui.actionGrid.addEventListener("click", event => {
  const button = event.target.closest("button[data-action]");
  if (button) chooseAction(button.dataset.action);
});
for (const node of [ui.surface, ui.canonical, ui.family, ui.scan, ui.notes, ui.evidenceChecks]) {
  node.addEventListener("input", () => {
    updateDraftFromForm();
    markDirty();
  });
}
ui.saveDraft.addEventListener("click", saveCurrentDraft);
ui.confirmNext.addEventListener("click", confirmCurrent);
ui.reopen.addEventListener("click", reopenCurrent);
ui.shortcuts.addEventListener("click", () => ui.shortcutDialog.showModal());
ui.guidelines.addEventListener("click", () => ui.guidelineDialog.showModal());
document.addEventListener("keydown", handleKeydown);
window.addEventListener("beforeunload", event => {
  if (!localDirty && !saveQueue.dirty) return;
  event.preventDefault();
  event.returnValue = "";
});

if (window.visualViewport) {
  const updateKeyboardState = () => {
    const keyboardOpen = window.innerWidth <= 767 &&
      window.visualViewport.height < window.innerHeight * 0.78;
    document.body.classList.toggle("keyboard-open", keyboardOpen);
  };
  window.visualViewport.addEventListener("resize", updateKeyboardState);
  window.visualViewport.addEventListener("scroll", updateKeyboardState);
}

initialize();
