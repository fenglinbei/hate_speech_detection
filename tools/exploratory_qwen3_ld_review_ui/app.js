"use strict";

const Common = window.ReviewCore;
const Audit = window.PilotInputAuditCore;

const ui = {
  sidebar: document.querySelector("#item-sidebar"),
  sidebarToggle: document.querySelector("#sidebar-toggle"),
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
  auditKind: document.querySelector("#audit-kind"),
  queryContent: document.querySelector("#query-content"),
  hitStatus: document.querySelector("#hit-status"),
  hitList: document.querySelector("#hit-list"),
  noHitCard: document.querySelector("#no-hit-card"),
  decisionTitle: document.querySelector("#decision-title"),
  decisionStatus: document.querySelector("#decision-status"),
  dispositionGrid: document.querySelector("#disposition-grid"),
  dispositionError: document.querySelector("#disposition-error"),
  lexDimensions: document.querySelector("#lex-dimensions"),
  noHitDimension: document.querySelector("#no-hit-dimension"),
  tagGrid: document.querySelector("#tag-grid"),
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
  shortcuts: document.querySelector("#shortcuts"),
  guidelineDialog: document.querySelector("#guideline-dialog"),
  shortcutDialog: document.querySelector("#shortcut-dialog"),
};

let bootstrap = null;
let current = null;
let draft = null;
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
    422: "当前审核字段未通过校验。",
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
  return response.json();
}

function setSaveState(state, text = "") {
  const defaults = {
    idle: "尚未修改",
    dirty: "有未保存修改",
    saving: "正在保存…",
    saved: "已保存",
    error: "保存失败",
  };
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
  const extra = status.deferred_count ? `；暂缓 ${status.deferred_count}` : "";
  ui.progressText.textContent = `${status.confirmed_count} / ${status.item_count} 已确认${extra}`;
  ui.sidebarProgress.textContent = `${status.confirmed_count} / ${status.item_count}`;
  ui.progressBar.style.width = `${ratio * 100}%`;
}

function refreshVisibleQueue() {
  const query = ui.itemSearch.value.trim();
  const ordered = query ? Audit.searchItemIds(bootstrap.items, query, searchMode) : null;
  visibleItemIds = Audit.visibleItemQueue(bootstrap.items, ordered, itemFilter);
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
    const title = document.createElement("strong");
    title.textContent = summary.terms && summary.terms.length ? summary.terms.join(" / ") : "无精确命中";
    const dot = document.createElement("span");
    dot.className = `case-state-dot${summary.status === "confirmed" ? " complete" : ""}`;
    const snippet = document.createElement("small");
    snippet.className = "case-match-snippet";
    snippet.textContent = summary.query_preview;
    const meta = document.createElement("small");
    meta.className = "item-kind";
    meta.textContent = summary.status === "confirmed"
      ? `${summary.audit_kind} · ${Audit.DISPOSITION_LABELS[summary.disposition]}`
      : `${summary.audit_kind} · 待确认`;
    button.append(title, dot, snippet, meta);
    button.addEventListener("click", () => loadItem(itemId));
    ui.itemList.append(button);
  }
  ui.itemEmpty.classList.toggle("hidden", visibleItemIds.length !== 0);
  ui.searchStatus.textContent = `${visibleItemIds.length} 条${ui.itemSearch.value.trim() ? "匹配" : "可见"}`;
  syncControls();
}

function renderSearchMode() {
  const labels = Common.SEARCH_MODE_LABELS || {literal: "连续", all_terms: "多词", fuzzy: "模糊"};
  ui.searchMode.textContent = `${labels[searchMode] || searchMode} ↻`;
}

function hitField(label, content) {
  const row = document.createElement("div");
  row.className = "hit-field";
  const name = document.createElement("span");
  name.textContent = label;
  const value = document.createElement("div");
  if (content instanceof Node) value.append(content);
  else value.textContent = content;
  row.append(name, value);
  return row;
}

function renderHits() {
  const item = current.item;
  ui.hitList.replaceChildren();
  ui.hitStatus.textContent = item.hit_count ? `${item.hit_count} 条冻结命中` : "0 条命中";
  ui.noHitCard.classList.toggle("hidden", item.audit_kind !== "no_hit");
  for (const [index, hit] of item.lexicon_hits.entries()) {
    const card = document.createElement("article");
    card.className = "hit-card";
    const head = document.createElement("div");
    head.className = "hit-head";
    const title = document.createElement("strong");
    title.textContent = `${index + 1}. ${hit.term}`;
    const category = document.createElement("span");
    category.className = "badge";
    category.textContent = hit.category;
    head.append(title, category);
    const body = document.createElement("div");
    body.className = "hit-body";
    const spans = document.createElement("div");
    for (const span of hit.match_spans) {
      const chip = document.createElement("span");
      chip.className = "span-chip";
      chip.textContent = `[${span[0]}, ${span[1]})`;
      spans.append(chip);
    }
    body.append(hitField("命中 span", spans));
    body.append(hitField("当前定义", hit.definition));
    const swap = document.createElement("div");
    swap.className = "swap-box";
    const donor = hit.definition_swap_donor;
    const strong = document.createElement("strong");
    strong.textContent = `DefinitionSwap donor · ${donor.term} · ${donor.category}`;
    const paragraph = document.createElement("p");
    paragraph.textContent = donor.definition;
    paragraph.style.margin = "4px 0 0";
    swap.append(strong, paragraph);
    body.append(hitField("预注册 swap", swap));
    card.append(head, body);
    ui.hitList.append(card);
  }
}

function renderSource() {
  const item = current.item;
  const index = bootstrap.items.findIndex(row => row.item_id === item.item_id);
  ui.itemPosition.textContent = `${index + 1} / ${bootstrap.items.length}`;
  ui.itemId.textContent = item.item_id;
  ui.auditKind.textContent = item.audit_kind === "lex_hit" ? "LEX-HIT" : "NO-HIT";
  ui.queryContent.textContent = item.query_content;
  renderHits();
}

function setFieldError(node, message) {
  node.textContent = message || "";
  node.classList.toggle("hidden", !message);
}

function validationErrors(confirm = false) {
  return current && draft ? Audit.validateDecision(current.item, draft, {confirm}) : {};
}

function renderErrors(errors = {}) {
  setFieldError(ui.dispositionError, errors.disposition);
  setFieldError(ui.notesError, errors.notes);
  document.querySelectorAll(".dimension-row[data-field]").forEach(row => {
    setFieldError(row.querySelector(".field-error"), errors[row.dataset.field]);
  });
}

function isLocked() {
  return Boolean(current && current.decision.status === "confirmed");
}

function renderDecision() {
  const locked = isLocked();
  ui.decisionTitle.textContent = Audit.DISPOSITION_LABELS[draft.disposition] || "等待选择";
  ui.decisionStatus.textContent = locked ? "已确认" : "草稿";
  ui.decisionStatus.dataset.state = locked ? "confirmed" : "draft";
  ui.dispositionGrid.querySelectorAll("[data-disposition]").forEach(button => {
    button.classList.toggle("active", button.dataset.disposition === draft.disposition);
  });
  document.querySelectorAll(".dimension-row[data-field]").forEach(row => {
    row.querySelectorAll("button[data-value]").forEach(button => {
      button.classList.toggle("active", draft[row.dataset.field] === button.dataset.value);
    });
  });
  ui.lexDimensions.classList.toggle("hidden", current.item.audit_kind !== "lex_hit");
  ui.noHitDimension.classList.toggle("hidden", current.item.audit_kind !== "no_hit");
  ui.notes.value = draft.notes;
  ui.tagGrid.querySelectorAll("input[data-tag]").forEach(input => {
    input.checked = draft.pragmatic_tags.includes(input.dataset.tag);
  });
  ui.lockedCard.classList.toggle("hidden", !locked);
  renderErrors();
  syncControls();
}

function buildTagGrid() {
  ui.tagGrid.replaceChildren();
  for (const tag of bootstrap.pragmatic_tags) {
    const label = document.createElement("label");
    label.className = "tag-choice";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.tag = tag;
    input.setAttribute("aria-keyshortcuts", `Shift+${Audit.TAG_SHORTCUT_LABELS[tag].replace("⇧", "")}`);
    const text = document.createElement("span");
    text.textContent = Audit.TAG_LABELS[tag] || tag;
    const shortcut = document.createElement("kbd");
    shortcut.textContent = Audit.TAG_SHORTCUT_LABELS[tag];
    label.append(input, text, shortcut);
    ui.tagGrid.append(label);
  }
}

function syncControls() {
  const disabled = busy || !current;
  const locked = isLocked();
  ui.saveDraft.disabled = disabled || locked;
  ui.confirmNext.disabled = disabled || locked;
  ui.dispositionGrid.querySelectorAll("button").forEach(button => { button.disabled = disabled || locked; });
  document.querySelectorAll(".dimension-row button, #tag-grid input, #notes").forEach(input => {
    input.disabled = disabled || locked;
  });
  ui.reopen.disabled = disabled || !locked;
  const position = current ? visibleItemIds.indexOf(current.item.item_id) : -1;
  ui.previousItem.disabled = busy || position <= 0;
  ui.nextItem.disabled = busy || position < 0 || position >= visibleItemIds.length - 1;
}

function markDirty() {
  if (!current || isLocked() || busy) return;
  dirty = true;
  setSaveState("dirty", "有未保存修改 · 5 秒后自动保存");
  renderErrors(validationErrors(false));
  window.clearTimeout(saveTimer);
  saveTimer = window.setTimeout(() => enqueueSave(false), AUTOSAVE_DELAY_MS);
}

function applyMutation(result) {
  bootstrap.revision = result.revision;
  bootstrap.status = result.status;
  updateSummary(result.item_summary);
  current.revision = result.revision;
  current.decision = result.decision;
  draft = Audit.decisionFields(result.decision);
  dirty = false;
  renderProgress();
  renderItemList();
  renderDecision();
  setSaveState("saved");
  clearError();
}

async function performSave(confirm) {
  if (!current || isLocked()) return false;
  const errors = validationErrors(confirm);
  if (Common.hasErrors(errors)) {
    renderErrors(errors);
    setSaveState("error", "请先完成必填审核项");
    return false;
  }
  busy = true;
  syncControls();
  setSaveState("saving");
  try {
    const result = await postJson("/api/save", {
      expected_revision: bootstrap.revision,
      item_id: current.item.item_id,
      decision: Common.clone(draft),
      confirm,
    });
    applyMutation(result);
    return true;
  } catch (error) {
    setSaveState("error", error.message);
    showError(error.message, error.technical);
    if (error.status === 409) {
      bootstrap = await getJson("/api/bootstrap");
      const itemId = current.item.item_id;
      current = await getJson(`/api/items/${encodeURIComponent(itemId)}`);
      draft = Audit.decisionFields(current.decision);
      dirty = false;
      renderProgress();
      renderItemList();
      renderSource();
      renderDecision();
    }
    return false;
  } finally {
    busy = false;
    syncControls();
  }
}

function enqueueSave(confirm) {
  window.clearTimeout(saveTimer);
  saveTimer = null;
  saveChain = saveChain.then(() => performSave(confirm));
  return saveChain;
}

async function flushDraft() {
  window.clearTimeout(saveTimer);
  saveTimer = null;
  await saveChain;
  if (dirty) return performSave(false);
  return true;
}

async function loadItem(itemId, {skipFlush = false} = {}) {
  if (!summaryFor(itemId) || current && current.item.item_id === itemId) return;
  if (!skipFlush && !(await flushDraft())) return;
  busy = true;
  syncControls();
  try {
    current = await getJson(`/api/items/${encodeURIComponent(itemId)}`);
    if (current.revision !== bootstrap.revision) {
      bootstrap = await getJson("/api/bootstrap");
      current = await getJson(`/api/items/${encodeURIComponent(itemId)}`);
    }
    draft = Audit.decisionFields(current.decision);
    dirty = false;
    buildTagGrid();
    renderSource();
    renderDecision();
    renderItemList();
    setSaveState(isLocked() ? "saved" : "idle", isLocked() ? "已确认并锁定" : "尚未修改");
    closeSidebar();
  } catch (error) {
    setSaveState("error", error.message);
    showError(error.message, error.technical);
  } finally {
    busy = false;
    syncControls();
  }
}

async function confirmCurrent() {
  window.clearTimeout(saveTimer);
  saveTimer = null;
  await saveChain;
  const currentId = current.item.item_id;
  if (!(await performSave(true))) return;
  const nextId = Audit.nextUnfinishedItemId(visibleItemIds, bootstrap.items, currentId);
  if (nextId && nextId !== currentId) await loadItem(nextId, {skipFlush: true});
}

async function navigate(offset) {
  if (!current || busy) return;
  const index = visibleItemIds.indexOf(current.item.item_id);
  const target = visibleItemIds[index + offset];
  if (target) await loadItem(target);
}

function chooseDisposition(value) {
  if (!draft || isLocked()) return;
  draft.disposition = value;
  renderDecision();
  markDirty();
}

function chooseDimension(field, value) {
  if (!draft || isLocked()) return;
  draft[field] = draft[field] === value ? null : value;
  renderDecision();
  markDirty();
}

function setDimensionFromShortcut(field, value) {
  if (!draft || isLocked()) return;
  draft[field] = value;
  renderDecision();
  markDirty();
}

function togglePragmaticTag(tag) {
  if (!draft || isLocked() || !bootstrap.pragmatic_tags.includes(tag)) return;
  const selected = new Set(draft.pragmatic_tags);
  if (selected.has(tag)) selected.delete(tag);
  else selected.add(tag);
  draft.pragmatic_tags = bootstrap.pragmatic_tags.filter(value => selected.has(value));
  renderDecision();
  markDirty();
}

async function reopenCurrent() {
  if (!current || !isLocked()) return;
  const reason = window.prompt("请输入重新打开该决定的原因：", "");
  if (!reason || !reason.trim()) return;
  busy = true;
  syncControls();
  try {
    const result = await postJson("/api/reopen", {
      expected_revision: bootstrap.revision,
      item_id: current.item.item_id,
      reason: reason.trim(),
    });
    applyMutation(result);
  } catch (error) {
    showError(error.message, error.technical);
  } finally {
    busy = false;
    syncControls();
  }
}

function applyQueueChange() {
  renderItemList();
  if (current && !visibleItemIds.includes(current.item.item_id) && visibleItemIds.length) {
    loadItem(visibleItemIds[0]);
  }
}

function openSidebar() {
  ui.sidebar.classList.add("open");
  ui.sidebarBackdrop.classList.remove("hidden");
}

function closeSidebar() {
  ui.sidebar.classList.remove("open");
  ui.sidebarBackdrop.classList.add("hidden");
}

async function initialize() {
  try {
    bootstrap = await getJson("/api/bootstrap");
    renderSearchMode();
    renderProgress();
    renderItemList();
    const first = Audit.nextUnfinishedItemId(visibleItemIds, bootstrap.items) || visibleItemIds[0];
    if (first) await loadItem(first, {skipFlush: true});
  } catch (error) {
    setSaveState("error", error.message);
    showError(error.message, error.technical);
  }
}

ui.dispositionGrid.addEventListener("click", event => {
  const button = event.target.closest("button[data-disposition]");
  if (button) chooseDisposition(button.dataset.disposition);
});
document.querySelectorAll(".dimension-row[data-field]").forEach(row => {
  row.addEventListener("click", event => {
    const button = event.target.closest("button[data-value]");
    if (button) chooseDimension(row.dataset.field, button.dataset.value);
  });
});
ui.tagGrid.addEventListener("change", () => {
  draft.pragmatic_tags = [...ui.tagGrid.querySelectorAll("input[data-tag]:checked")].map(input => input.dataset.tag);
  markDirty();
});
ui.notes.addEventListener("input", () => { draft.notes = ui.notes.value; markDirty(); });
ui.saveDraft.addEventListener("click", () => enqueueSave(false));
ui.confirmNext.addEventListener("click", confirmCurrent);
ui.reopen.addEventListener("click", reopenCurrent);
ui.previousItem.addEventListener("click", () => navigate(-1));
ui.nextItem.addEventListener("click", () => navigate(1));
ui.itemSearch.addEventListener("input", applyQueueChange);
ui.searchMode.addEventListener("click", () => {
  searchMode = Common.nextSearchMode(searchMode);
  renderSearchMode();
  applyQueueChange();
});
ui.itemFilters.addEventListener("click", event => {
  const button = event.target.closest("button[data-filter]");
  if (!button) return;
  itemFilter = button.dataset.filter;
  ui.itemFilters.querySelectorAll("button").forEach(row => row.classList.toggle("active", row === button));
  applyQueueChange();
});
ui.sidebarToggle.addEventListener("click", openSidebar);
ui.sidebarClose.addEventListener("click", closeSidebar);
ui.sidebarBackdrop.addEventListener("click", closeSidebar);
ui.guidelines.addEventListener("click", () => ui.guidelineDialog.showModal());
ui.shortcuts.addEventListener("click", () => ui.shortcutDialog.showModal());

document.addEventListener("keydown", event => {
  if (busy || document.querySelector("dialog[open]")) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    enqueueSave(false);
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    confirmCurrent();
    return;
  }
  if (Common.isTextEntry(event.target)) return;
  if (event.repeat || event.altKey) return;
  const tag = Audit.tagShortcutForCode(event.code, event.shiftKey);
  if (tag) { event.preventDefault(); togglePragmaticTag(tag); return; }
  if (!event.shiftKey) {
    const dimension = Audit.dimensionShortcutForCode(current && current.item.audit_kind, event.code);
    if (dimension) {
      event.preventDefault();
      setDimensionFromShortcut(dimension.field, dimension.value);
      return;
    }
  }
  if (event.key === "[") { event.preventDefault(); navigate(-1); return; }
  if (event.key === "]") { event.preventDefault(); navigate(1); return; }
  const disposition = Audit.dispositionForKey(event.key);
  if (disposition) { event.preventDefault(); chooseDisposition(disposition); }
});

initialize();
