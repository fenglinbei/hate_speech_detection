"use strict";

const C = window.ReviewCore;

const ui = {
  phaseTitle: document.querySelector("#phase-title"),
  phaseBadge: document.querySelector("#phase-badge"),
  progressText: document.querySelector("#top-progress-text"),
  progressBar: document.querySelector("#progress-bar"),
  saveState: document.querySelector("#save-state"),
  sidebar: document.querySelector("#case-sidebar"),
  sidebarToggle: document.querySelector("#sidebar-toggle"),
  sidebarClose: document.querySelector("#sidebar-close"),
  sidebarBackdrop: document.querySelector("#sidebar-backdrop"),
  sidebarProgress: document.querySelector("#sidebar-progress"),
  caseSearch: document.querySelector("#case-search"),
  searchMode: document.querySelector("#search-mode"),
  searchStatus: document.querySelector("#search-status"),
  caseFilters: document.querySelector("#case-filters"),
  caseList: document.querySelector("#case-list"),
  caseEmpty: document.querySelector("#case-empty"),
  previousCase: document.querySelector("#previous-case"),
  nextCase: document.querySelector("#next-case"),
  casePosition: document.querySelector("#case-position"),
  caseAlias: document.querySelector("#case-alias"),
  sourceHint: document.querySelector("#source-hint"),
  content: document.querySelector("#content"),
  selectionBar: document.querySelector("#selection-bar"),
  selectionPreview: document.querySelector("#selection-preview"),
  quickRouteActions: document.querySelector("#quick-route-actions"),
  useSelection: document.querySelector("#use-selection"),
  manualMention: document.querySelector("#manual-mention"),
  itemStripKicker: document.querySelector("#item-strip-kicker"),
  itemStripStatus: document.querySelector("#item-strip-status"),
  itemStripCard: document.querySelector("#item-strip-card"),
  itemStripBody: document.querySelector("#item-strip-body"),
  itemStrip: document.querySelector("#item-strip"),
  itemEmpty: document.querySelector("#item-empty"),
  proposalSheetToggle: document.querySelector("#proposal-sheet-toggle"),
  proposalSheetClose: document.querySelector("#proposal-sheet-close"),
  proposalNav: document.querySelector("#proposal-nav"),
  previousProposal: document.querySelector("#previous-proposal"),
  nextProposal: document.querySelector("#next-proposal"),
  proposalPosition: document.querySelector("#proposal-position"),
  decisionKicker: document.querySelector("#decision-kicker"),
  decisionTitle: document.querySelector("#decision-title"),
  decisionStatus: document.querySelector("#decision-status"),
  decisionEditor: document.querySelector("#decision-editor"),
  saveDraft: document.querySelector("#save-draft"),
  confirmNext: document.querySelector("#confirm-next"),
  lockRaw: document.querySelector("#lock-raw"),
  exportButton: document.querySelector("#export"),
  topActionMenu: document.querySelector("#top-action-menu"),
  shortcuts: document.querySelector("#shortcuts"),
  guidelines: document.querySelector("#guidelines"),
  guidelineDialog: document.querySelector("#guideline-dialog"),
  requestError: document.querySelector("#request-error"),
  requestErrorMessage: document.querySelector("#request-error-message"),
  requestErrorDetails: document.querySelector("#request-error-details"),
  actionDialog: document.querySelector("#action-dialog"),
  dialogTitle: document.querySelector("#dialog-title"),
  dialogMessage: document.querySelector("#dialog-message"),
  dialogInputWrap: document.querySelector("#dialog-input-wrap"),
  dialogInput: document.querySelector("#dialog-input"),
  dialogInputError: document.querySelector("#dialog-input-error"),
  dialogConfirm: document.querySelector("#dialog-confirm"),
  conflictDialog: document.querySelector("#conflict-dialog"),
  conflictDetails: document.querySelector("#conflict-details"),
  shortcutDialog: document.querySelector("#shortcut-dialog"),
};

class ApiError extends Error {
  constructor(message, status, technical = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.technical = technical || message;
  }
}

let bootstrap = null;
let caseIndex = 0;
let activeCase = null;
let activeMentionIndex = null;
let activeProposalIndex = 0;
let rawDraft = null;
let diagnosticDraft = null;
let selectionCandidate = null;
let caseFilter = "all";
let searchMode = "literal";
let searchMatches = null;
let searchMatchById = new Map();
let visibleCaseIds = [];
let searchTimer = null;
let searchController = null;
let searchSequence = 0;
let localDirty = false;
let editVersion = 0;
let currentErrors = {};
let loadSequence = 0;
let handlingConflict = false;
let saveBusy = false;
let confirmBusy = false;
let resumeSearchAfterConfirm = false;
const caseCache = new Map();

function nowText() {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date());
}

function setSaveState(state, message = "") {
  const defaults = {
    idle: "尚未修改",
    dirty: "有未保存修改",
    saving: "正在保存…",
    saved: `已保存 ${nowText()}`,
    error: "保存失败 · 等待重试",
  };
  ui.saveState.dataset.state = state;
  ui.saveState.textContent = message || defaults[state] || state;
  saveBusy = state === "saving";
  ui.saveDraft.disabled = saveBusy || confirmBusy || !activeCase;
  ui.confirmNext.disabled = saveBusy || confirmBusy || !activeCase;
  ui.exportButton.disabled = saveBusy;
  ui.lockRaw.disabled = saveBusy;
  if (state !== "error") ui.saveState.removeAttribute("title");
  if (state === "saved") clearRequestError();
  syncNavigationButtons();
  if (!saveBusy) syncActionButtons();
}

function syncActionButtons() {
  if (!activeCase) return;
  let locked = false;
  if (mode() !== "diagnostic") {
    locked = activeCase.raw_annotation.status === "confirmed";
  } else {
    const proposal = activeProposal();
    const decision = proposal && activeCase.diagnostic_decisions[proposal.proposal_id];
    locked = !proposal || Boolean(decision && decision.status === "confirmed");
  }
  ui.saveDraft.disabled = saveBusy || confirmBusy || locked;
  ui.confirmNext.disabled = saveBusy || confirmBusy || locked;
}

function syncNavigationButtons() {
  const navigationBusy = saveBusy || confirmBusy;
  const visibleIndex = activeCase
    ? visibleCaseIds.indexOf(activeCase.case_id)
    : -1;
  ui.previousCase.disabled = navigationBusy || visibleIndex <= 0;
  ui.nextCase.disabled = navigationBusy || visibleIndex < 0 || visibleIndex >= visibleCaseIds.length - 1;
  const proposalCount = activeCase && activeCase.proposals ? activeCase.proposals.length : 0;
  ui.previousProposal.disabled = navigationBusy || activeProposalIndex <= 0;
  ui.nextProposal.disabled = navigationBusy || activeProposalIndex >= proposalCount - 1;
  ui.caseList.querySelectorAll(".case-item").forEach(button => {
    button.disabled = navigationBusy;
  });
}

function setConfirmationLock(locked) {
  confirmBusy = locked;
  ui.decisionEditor.inert = locked;
  ui.content.inert = locked;
  ui.selectionBar.inert = locked;
  ui.itemStripCard.inert = locked;
  ui.caseSearch.disabled = locked;
  ui.searchMode.disabled = locked;
  ui.caseFilters.querySelectorAll("button").forEach(button => {
    button.disabled = locked;
  });
  ui.manualMention.disabled = locked;
  if (locked && (searchTimer !== null || searchController !== null)) {
    resumeSearchAfterConfirm = true;
    clearTimeout(searchTimer);
    searchTimer = null;
    searchSequence += 1;
    if (searchController) searchController.abort();
    searchController = null;
  }
  syncNavigationButtons();
  syncActionButtons();
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

function setButtonBusy(button, busy, busyText) {
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = busyText;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
  }
}

async function parseError(response) {
  const payload = await response.json().catch(() => ({}));
  const technical = payload.error || `HTTP ${response.status}`;
  const friendly = {
    403: "当前登录或页面会话已经失效，请刷新页面后重试。",
    404: "请求的审核条目不存在。",
    409: "审核数据已在其他页面发生变化。",
    422: "当前标注未通过校验，请检查表单内容。",
    500: "审核服务发生内部错误。",
  }[response.status] || "请求失败，请稍后重试。";
  return new ApiError(friendly, response.status, technical);
}

async function getJson(path) {
  const response = await fetch(path, {cache: "no-store"});
  if (!response.ok) throw await parseError(response);
  return response.json();
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({...payload, session_token: bootstrap.session_token}),
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

async function postBlob(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({...payload, session_token: bootstrap.session_token}),
  });
  if (!response.ok) throw await parseError(response);
  return response.blob();
}

function summaryIndex(caseId) {
  return bootstrap.case_summaries.findIndex(row => row.case_id === caseId);
}

function summaryFor(caseId) {
  return bootstrap.case_summaries[summaryIndex(caseId)];
}

function updateSummary(summary) {
  const index = summaryIndex(summary.case_id);
  if (index >= 0) bootstrap.case_summaries[index] = summary;
}

function refreshVisibleCaseQueue() {
  if (!bootstrap) {
    visibleCaseIds = [];
    return visibleCaseIds;
  }
  const ordered = searchMatches === null
    ? null
    : searchMatches.map(match => match.case_id);
  visibleCaseIds = C.visibleCaseQueue(bootstrap.case_summaries, ordered, caseFilter);
  return visibleCaseIds;
}

function revealActiveCase() {
  if (!activeCase) return;
  window.requestAnimationFrame(() => {
    const button = [...ui.caseList.querySelectorAll(".case-item")]
      .find(row => row.dataset.caseId === activeCase.case_id);
    if (button) button.scrollIntoView({block: "center", behavior: "auto"});
  });
}

function cacheCase(caseRow) {
  caseCache.delete(caseRow.case_id);
  caseCache.set(caseRow.case_id, caseRow);
  while (caseCache.size > 5) {
    const oldest = caseCache.keys().next().value;
    if (oldest === (activeCase && activeCase.case_id)) {
      const value = caseCache.get(oldest);
      caseCache.delete(oldest);
      caseCache.set(oldest, value);
      continue;
    }
    caseCache.delete(oldest);
  }
}

function mode() {
  if (!bootstrap || !activeCase) return "raw";
  if (bootstrap.phase === "raw") return "raw";
  return activeCase.raw_annotation.status === "confirmed" ? "diagnostic" : "raw-repair";
}

function activeProposal() {
  return activeCase && activeCase.proposals
    ? activeCase.proposals[activeProposalIndex]
    : null;
}

function makeRawTask(confirm) {
  return {
    kind: "raw",
    version: editVersion,
    case_id: activeCase.case_id,
    expected_revision: bootstrap.revision,
    annotation: {
      needs_explanation: rawDraft.mentions.length > 0,
      mentions: rawDraft.mentions.map(mention => ({
        surface: String(mention.surface || "").trim(),
        occurrence_ordinal: Number(mention.occurrence_ordinal),
        provisional_route: mention.provisional_route,
        reason_codes: [...(mention.reason_codes || [])],
        notes: mention.notes || "",
      })),
      notes: rawDraft.notes || "",
    },
    confirm,
  };
}

function makeDiagnosticTask(confirm) {
  const proposal = activeProposal();
  return {
    kind: "diagnostic",
    version: editVersion,
    case_id: activeCase.case_id,
    proposal_id: proposal.proposal_id,
    expected_revision: bootstrap.revision,
    decision: {
      action: diagnosticDraft.action,
      result_mentions: (diagnosticDraft.result_mentions || []).map(mention => ({
        surface: String(mention.surface || "").trim(),
        occurrence_ordinal: Number(mention.occurrence_ordinal),
      })),
      reason_codes: [...(diagnosticDraft.reason_codes || [])],
      notes: diagnosticDraft.notes || "",
    },
    confirm,
  };
}

function applyMutation(task, result) {
  if (
    saveQueue.pending &&
    saveQueue.pending.expected_revision === task.expected_revision
  ) {
    saveQueue.pending.expected_revision = result.revision;
  }
  bootstrap.revision = result.revision;
  bootstrap.phase = result.phase;
  bootstrap.status = result.status;
  updateSummary(result.case_summary);
  const cached = caseCache.get(task.case_id);
  if (cached && task.kind === "raw") cached.raw_annotation = result.raw_annotation;
  if (cached && task.kind === "diagnostic") {
    cached.diagnostic_decisions[task.proposal_id] = result.diagnostic_decision;
  }
  if (activeCase && activeCase.case_id === task.case_id) {
    if (task.kind === "raw") activeCase.raw_annotation = result.raw_annotation;
    if (task.kind === "diagnostic") {
      activeCase.diagnostic_decisions[task.proposal_id] = result.diagnostic_decision;
    }
  }
  if (task.version === editVersion) localDirty = false;
  renderProgress();
  renderCaseList({revealCurrent: true});
}

async function executeMutation(task) {
  const common = {expected_revision: task.expected_revision};
  let result;
  if (task.kind === "raw") {
    result = await postJson("/api/raw", {
      ...common,
      case_id: task.case_id,
      annotation: task.annotation,
      confirm: task.confirm,
    });
  } else {
    result = await postJson("/api/diagnostic", {
      ...common,
      case_id: task.case_id,
      proposal_id: task.proposal_id,
      decision: task.decision,
      confirm: task.confirm,
    });
  }
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

function validateCurrent() {
  if (!activeCase) return false;
  if (mode() !== "diagnostic") {
    currentErrors = C.validateRaw(activeCase.content, rawDraft);
  } else {
    currentErrors = C.validateDiagnostic(
      activeCase.content,
      activeProposal(),
      diagnosticDraft,
    );
  }
  return !C.hasErrors(currentErrors);
}

function markDirty() {
  if (confirmBusy) return;
  localDirty = true;
  editVersion += 1;
  const valid = validateCurrent();
  setSaveState("dirty");
  if (valid) {
    const task = mode() === "diagnostic"
      ? makeDiagnosticTask(false)
      : makeRawTask(false);
    saveQueue.schedule(task, 700);
  } else if (saveQueue.pending) {
    saveQueue.discard();
    setSaveState("dirty");
  }
}

async function flushChanges() {
  if (!localDirty && !saveQueue.dirty) return true;
  if (!validateCurrent()) {
    renderDecisionEditor();
    setSaveState("error", "请先修正表单");
    return false;
  }
  if (!saveQueue.pending && localDirty) {
    saveQueue.schedule(
      mode() === "diagnostic" ? makeDiagnosticTask(false) : makeRawTask(false),
      0,
    );
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
  const activeCaseId = activeCase && activeCase.case_id;
  ui.conflictDetails.textContent = error.technical || error.message;
  ui.conflictDialog.showModal();
  const choice = await new Promise(resolve => {
    ui.conflictDialog.addEventListener("close", () => resolve(ui.conflictDialog.returnValue), {once: true});
  });
  try {
    saveQueue.discard();
    const latest = await getJson("/api/bootstrap");
    bootstrap = latest;
    caseCache.clear();
    if (choice === "retry" && task) {
      task.expected_revision = bootstrap.revision;
      await saveQueue.runNow(task);
      setSaveState("saved");
    } else {
      localDirty = false;
      refreshVisibleCaseQueue();
      const targetCaseId = activeCaseId && summaryIndex(activeCaseId) >= 0
        ? activeCaseId
        : visibleCaseIds[0];
      if (targetCaseId) {
        await loadCaseById(targetCaseId, {skipFlush: true, force: true});
      }
      setSaveState("saved", "已采用服务器版本");
    }
  } catch (retryError) {
    setSaveState("error", retryError.message);
  } finally {
    handlingConflict = false;
  }
}

function openActionDialog({title, message, confirmLabel = "确认", input = false, danger = false}) {
  ui.dialogTitle.textContent = title;
  ui.dialogMessage.textContent = message;
  ui.dialogConfirm.textContent = confirmLabel;
  ui.dialogConfirm.className = danger ? "danger-button" : "primary-button";
  ui.dialogInputWrap.classList.toggle("hidden", !input);
  ui.dialogInputError.classList.add("hidden");
  ui.dialogInput.value = "";
  return new Promise(resolve => {
    const onSubmit = event => {
      if (
        event.submitter &&
        event.submitter.value === "confirm" &&
        input &&
        !ui.dialogInput.value.trim()
      ) {
        event.preventDefault();
        ui.dialogInputError.classList.remove("hidden");
        ui.dialogInput.focus();
      }
    };
    const onClose = () => {
      ui.actionDialog.removeEventListener("submit", onSubmit);
      resolve(ui.actionDialog.returnValue === "confirm" ? ui.dialogInput.value.trim() || true : null);
    };
    ui.actionDialog.addEventListener("submit", onSubmit);
    ui.actionDialog.addEventListener("close", onClose, {once: true});
    ui.actionDialog.showModal();
    if (input) setTimeout(() => ui.dialogInput.focus(), 0);
  });
}

function renderProgress() {
  if (!bootstrap) return;
  const key = bootstrap.phase === "raw" ? "raw" : "diagnostic";
  const status = bootstrap.status[key];
  const ratio = status.total ? status.confirmed / status.total : 1;
  ui.phaseTitle.textContent = bootstrap.phase === "raw"
    ? "阶段 A：原文独立标注"
    : "阶段 B：匿名提案诊断";
  ui.phaseBadge.textContent = bootstrap.phase === "raw" ? "RAW" : "DIAGNOSTIC";
  ui.progressText.textContent = `${status.confirmed} / ${status.total} 已确认`;
  ui.progressBar.style.width = `${ratio * 100}%`;
  ui.sidebarProgress.textContent = `${status.confirmed} / ${status.total}`;
  ui.lockRaw.classList.toggle(
    "hidden",
    bootstrap.phase !== "raw" || status.confirmed !== status.total,
  );
}

function appendMatchSnippet(container, match) {
  if (!match || !match.snippet) return;
  const snippet = document.createElement("small");
  snippet.className = "case-match-snippet";
  const characters = Array.from(match.snippet);
  const start = Number(match.match_start);
  const end = Number(match.match_end);
  if (Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end > start && end <= characters.length) {
    snippet.append(document.createTextNode(characters.slice(0, start).join("")));
    const mark = document.createElement("mark");
    mark.textContent = characters.slice(start, end).join("");
    snippet.append(mark, document.createTextNode(characters.slice(end).join("")));
  } else {
    snippet.textContent = match.snippet;
  }
  container.append(snippet);
}

function updateSearchStatus() {
  if (!bootstrap) return;
  ui.searchStatus.removeAttribute("title");
  const query = ui.caseSearch.value.trim();
  if (!query) {
    ui.searchStatus.textContent = caseFilter === "all"
      ? `全部 ${visibleCaseIds.length} 条`
      : `筛选后 ${visibleCaseIds.length} 条`;
    return;
  }
  const matches = searchMatches ? searchMatches.length : 0;
  if (matches === 0) {
    ui.searchStatus.textContent = "无匹配结果";
    return;
  }
  ui.searchStatus.textContent = caseFilter === "all"
    ? `${matches} 条搜索结果`
    : `显示 ${visibleCaseIds.length} / 命中 ${matches} 条`;
}

function renderCaseList({revealCurrent = false} = {}) {
  if (!bootstrap) return;
  const previousScrollTop = ui.caseList.scrollTop;
  refreshVisibleCaseQueue();
  ui.caseList.replaceChildren();
  visibleCaseIds.forEach(caseId => {
    const summary = summaryFor(caseId);
    if (!summary) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-item";
    button.dataset.caseId = caseId;
    if (activeCase && activeCase.case_id === caseId) button.classList.add("active");
    const id = document.createElement("strong");
    id.textContent = summary.case_id;
    const dot = document.createElement("span");
    dot.className = "case-state-dot" + (summary.complete ? " complete" : "");
    const alias = document.createElement("small");
    alias.textContent = summary.blind_alias;
    const count = document.createElement("small");
    count.textContent = bootstrap.phase === "raw"
      ? (summary.complete ? "已确认" : "待确认")
      : `${summary.diagnostic.confirmed} / ${summary.diagnostic.total}`;
    button.append(id, dot, alias, count);
    appendMatchSnippet(button, searchMatchById.get(caseId));
    button.disabled = saveBusy;
    button.addEventListener("click", () => loadCaseById(caseId));
    ui.caseList.append(button);
  });
  ui.caseEmpty.textContent = ui.caseSearch.value.trim()
    ? "没有匹配当前搜索和筛选条件的 Case"
    : "没有匹配当前筛选条件的 Case";
  ui.caseEmpty.classList.toggle("hidden", visibleCaseIds.length !== 0);
  updateSearchStatus();
  syncNavigationButtons();
  if (revealCurrent) revealActiveCase();
  else ui.caseList.scrollTop = previousScrollTop;
}

function renderSearchMode() {
  const label = C.SEARCH_MODE_LABELS[searchMode] || searchMode;
  const next = C.nextSearchMode(searchMode);
  ui.searchMode.textContent = `${label} ↻`;
  ui.searchMode.title = `点击切换为${C.SEARCH_MODE_LABELS[next]}模式`;
  ui.caseSearch.placeholder = {
    literal: "输入编号、别名或连续文本",
    all_terms: "用空格分隔多个关键词",
    fuzzy: "输入允许少量差异的表达",
  }[searchMode];
}

async function applyVisibleQueueChange(searchGeneration = null) {
  if (searchGeneration !== null && searchGeneration !== searchSequence) return false;
  renderCaseList();
  if (searchGeneration !== null && searchGeneration !== searchSequence) return false;
  if (!activeCase || visibleCaseIds.includes(activeCase.case_id)) {
    revealActiveCase();
    return true;
  }
  if (!visibleCaseIds.length) return true;
  const isCurrent = searchGeneration === null
    ? null
    : () => searchGeneration === searchSequence;
  return loadCaseById(visibleCaseIds[0], {isCurrent});
}

async function runCaseSearch() {
  clearTimeout(searchTimer);
  searchTimer = null;
  const query = ui.caseSearch.value.trim();
  const sequence = ++searchSequence;
  if (searchController) searchController.abort();
  searchController = null;
  if (!query) {
    searchMatches = null;
    searchMatchById = new Map();
    await applyVisibleQueueChange(sequence);
    return;
  }
  const controller = new AbortController();
  searchController = controller;
  ui.searchStatus.textContent = "搜索中…";
  try {
    const params = new URLSearchParams({q: query, mode: searchMode});
    const response = await fetch(`/api/cases/search?${params}`, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw await parseError(response);
    const payload = await response.json();
    if (sequence !== searchSequence) return;
    searchMatches = Array.isArray(payload.matches) ? payload.matches : [];
    searchMatchById = new Map(searchMatches.map(match => [match.case_id, match]));
    await applyVisibleQueueChange(sequence);
  } catch (error) {
    if (error.name === "AbortError" || sequence !== searchSequence) return;
    ui.searchStatus.textContent = "搜索失败 · 请重试";
    ui.searchStatus.title = error.technical || error.message;
  } finally {
    if (searchController === controller) searchController = null;
  }
}

function scheduleCaseSearch() {
  clearTimeout(searchTimer);
  searchSequence += 1;
  if (searchController) searchController.abort();
  searchController = null;
  ui.searchStatus.textContent = "搜索中…";
  searchTimer = window.setTimeout(runCaseSearch, 300);
}

async function changeCaseFilter(nextFilter) {
  caseFilter = nextFilter;
  ui.caseFilters.querySelectorAll("button").forEach(row => {
    row.classList.toggle("active", row.dataset.filter === caseFilter);
  });
  await applyVisibleQueueChange();
}

function cycleSearchMode() {
  searchMode = C.nextSearchMode(searchMode);
  renderSearchMode();
  runCaseSearch();
}

function openSidebar() {
  ui.sidebar.classList.add("open");
  ui.sidebarBackdrop.classList.remove("hidden");
}

function closeSidebar() {
  ui.sidebar.classList.remove("open");
  ui.sidebarBackdrop.classList.add("hidden");
}

function caseStatusText(summary) {
  if (bootstrap.phase === "raw") return summary.complete ? "已确认" : "待确认";
  return `${summary.diagnostic.confirmed} / ${summary.diagnostic.total} 个提案已确认`;
}

function codePointRangeForMention(mention) {
  if (Number.isInteger(mention.start) && Number.isInteger(mention.end)) {
    return {start: mention.start, end: mention.end};
  }
  const unitStart = C.occurrenceStart(
    activeCase.content,
    mention.surface,
    Number(mention.occurrence_ordinal),
  );
  if (unitStart < 0) return {start: -1, end: -1};
  return {
    start: C.codePointOffset(activeCase.content, unitStart),
    end: C.codePointOffset(activeCase.content, unitStart + mention.surface.length),
  };
}

function renderHighlightedContent() {
  ui.content.replaceChildren();
  if (!activeCase) return;
  const characters = Array.from(activeCase.content);
  const flags = characters.map(() => "");
  if (mode() !== "diagnostic") {
    rawDraft.mentions.forEach((mention, index) => {
      const range = codePointRangeForMention(mention);
      if (range.start < 0) return;
      for (let cursor = range.start; cursor < range.end; cursor += 1) {
        flags[cursor] = index === activeMentionIndex ? "active-highlight" : "mention-highlight";
      }
    });
  } else {
    const proposal = activeProposal();
    if (proposal) {
      for (let cursor = proposal.start; cursor < proposal.end; cursor += 1) {
        flags[cursor] = "active-highlight";
      }
    }
  }
  let cursor = 0;
  while (cursor < characters.length) {
    const className = flags[cursor];
    let end = cursor + 1;
    while (end < characters.length && flags[end] === className) end += 1;
    const node = className ? document.createElement("mark") : document.createElement("span");
    if (className) node.className = className;
    node.textContent = characters.slice(cursor, end).join("");
    ui.content.append(node);
    cursor = end;
  }
}

function clearSelectionCandidate() {
  selectionCandidate = null;
  ui.selectionBar.classList.add("hidden");
}

function captureSelection() {
  if (confirmBusy || !activeCase) return;
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || !selection.rangeCount) {
    clearSelectionCandidate();
    return;
  }
  const range = selection.getRangeAt(0);
  const ancestor = range.commonAncestorContainer.nodeType === Node.TEXT_NODE
    ? range.commonAncestorContainer.parentElement
    : range.commonAncestorContainer;
  if (!ancestor || !ui.content.contains(ancestor)) {
    clearSelectionCandidate();
    return;
  }
  const preceding = range.cloneRange();
  preceding.selectNodeContents(ui.content);
  preceding.setEnd(range.startContainer, range.startOffset);
  const rawSurface = range.toString();
  const leading = rawSurface.length - rawSurface.trimStart().length;
  const surface = rawSurface.trim();
  const start = preceding.toString().length + leading;
  const ordinal = C.occurrenceOrdinal(activeCase.content, surface, start);
  if (!surface || !ordinal) {
    clearSelectionCandidate();
    return;
  }
  if (mode() === "diagnostic" && !["trim", "expand", "split"].includes(diagnosticDraft.action)) {
    clearSelectionCandidate();
    return;
  }
  if (mode() !== "diagnostic" && activeCase.raw_annotation.status === "confirmed") {
    clearSelectionCandidate();
    return;
  }
  selectionCandidate = {surface, occurrence_ordinal: ordinal};
  ui.selectionPreview.textContent = `${surface} · 第 ${ordinal} 次出现`;
  ui.quickRouteActions.classList.toggle("hidden", mode() === "diagnostic");
  ui.useSelection.classList.toggle("hidden", mode() !== "diagnostic");
  ui.useSelection.textContent = mode() === "diagnostic" && diagnosticDraft.action === "split"
    ? "添加拆分片段"
    : mode() === "diagnostic"
      ? "用作修订片段"
      : "添加为 A 档标注";
  ui.selectionBar.classList.remove("hidden");
}

function createField(label, input, error, helper = "") {
  const wrapper = document.createElement("label");
  wrapper.className = "field";
  const title = document.createElement("span");
  title.textContent = label;
  wrapper.append(title, input);
  if (helper) {
    const note = document.createElement("small");
    note.className = "helper";
    note.textContent = helper;
    wrapper.append(note);
  }
  if (error) {
    const message = document.createElement("small");
    message.className = "field-error";
    message.textContent = error;
    wrapper.append(message);
  }
  return wrapper;
}

function createReasonPicker(selected, onChange, error) {
  const wrapper = document.createElement("div");
  const sheet = document.createElement("details");
  sheet.className = "reason-sheet";
  sheet.open = Boolean(error) || !window.matchMedia("(max-width: 767px)").matches;
  const summary = document.createElement("summary");
  const summaryLabel = document.createElement("strong");
  summaryLabel.textContent = "选择判定原因";
  const summaryCount = document.createElement("span");
  const updateSummary = values => {
    summaryCount.textContent = values.length
      ? `已选择 ${values.length} 项`
      : "尚未选择";
  };
  updateSummary(selected);
  summary.append(summaryLabel, summaryCount);
  const picker = document.createElement("div");
  picker.className = "reason-picker";
  const search = document.createElement("input");
  search.className = "reason-search";
  search.placeholder = "搜索中文含义或英文代码";
  search.autocomplete = "off";
  const list = document.createElement("div");
  list.className = "reason-list";
  const allowed = new Set(bootstrap.reason_codes);
  for (const group of C.REASON_GROUPS) {
    const title = document.createElement("p");
    title.className = "reason-group-title";
    title.textContent = group.label;
    list.append(title);
    for (const value of group.values.filter(item => allowed.has(item))) {
      const option = document.createElement("label");
      option.className = "reason-option";
      option.dataset.search = `${C.REASON_LABELS[value]} ${value}`.toLowerCase();
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = selected.includes(value);
      const copy = document.createElement("span");
      const chinese = document.createElement("strong");
      chinese.textContent = C.REASON_LABELS[value] || value;
      const code = document.createElement("small");
      code.textContent = value;
      copy.append(chinese, code);
      option.append(checkbox, copy);
      checkbox.addEventListener("change", () => {
        const next = [...list.querySelectorAll(".reason-option input:checked")]
          .map(input => input.closest(".reason-option").dataset.value);
        onChange(next);
        updateSummary(next);
        const oldError = wrapper.querySelector(".field-error");
        if (oldError) oldError.remove();
        if (window.matchMedia("(max-width: 767px)").matches) {
          sheet.open = false;
        }
      });
      option.dataset.value = value;
      list.append(option);
    }
  }
  search.addEventListener("input", () => {
    const term = search.value.trim().toLowerCase();
    list.querySelectorAll(".reason-option").forEach(option => {
      option.classList.toggle("hidden", Boolean(term) && !option.dataset.search.includes(term));
    });
    list.querySelectorAll(".reason-group-title").forEach(title => {
      let sibling = title.nextElementSibling;
      let hasVisible = false;
      while (sibling && !sibling.classList.contains("reason-group-title")) {
        if (!sibling.classList.contains("hidden")) hasVisible = true;
        sibling = sibling.nextElementSibling;
      }
      title.classList.toggle("hidden", !hasVisible);
    });
  });
  const done = document.createElement("button");
  done.type = "button";
  done.className = "primary-button reason-sheet-done";
  done.textContent = "完成原因选择";
  done.addEventListener("click", () => {
    sheet.open = false;
  });
  picker.append(search, list, done);
  sheet.append(summary, picker);
  wrapper.append(sheet);
  if (error) {
    const message = document.createElement("small");
    message.className = "field-error";
    message.textContent = error;
    wrapper.append(message);
  }
  return wrapper;
}

function renderItemStrip() {
  ui.itemStrip.replaceChildren();
  if (!activeCase) return;
  const currentMode = mode();
  ui.itemStripCard.classList.toggle("diagnostic-mode", currentMode === "diagnostic");
  ui.proposalSheetToggle.classList.toggle("hidden", currentMode !== "diagnostic");
  if (currentMode !== "diagnostic") {
    ui.itemStripCard.classList.remove("sheet-open");
    ui.itemStripKicker.textContent = currentMode === "raw-repair" ? "正在修订原文标注" : "本条原文标注";
    ui.itemStripStatus.textContent = `${rawDraft.mentions.length} 个 mention`;
    ui.proposalNav.classList.add("hidden");
    rawDraft.mentions.forEach((mention, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "item-chip";
      if (index === activeMentionIndex) button.classList.add("active");
      button.textContent = mention.surface || `未完成标注 ${index + 1}`;
      const code = document.createElement("small");
      code.className = "code-label";
      code.textContent = `#${mention.occurrence_ordinal || "?"} · ${mention.provisional_route || "未选择"}`;
      button.append(code);
      button.addEventListener("click", () => {
        activeMentionIndex = index;
        currentErrors = {};
        renderCurrentCase();
      });
      ui.itemStrip.append(button);
    });
    ui.itemEmpty.textContent = "尚未添加标注。请选择原文，或将本条标记为无可标注表达。";
    ui.itemEmpty.classList.toggle("hidden", rawDraft.mentions.length !== 0);
    return;
  }

  ui.itemStripKicker.textContent = "匿名提案序列";
  const proposals = activeCase.proposals || [];
  ui.proposalSheetToggle.classList.toggle("hidden", proposals.length === 0);
  const summary = summaryFor(activeCase.case_id);
  ui.itemStripStatus.textContent = `${summary.diagnostic.confirmed} / ${summary.diagnostic.total} 已确认`;
  ui.proposalNav.classList.remove("hidden");
  ui.proposalPosition.textContent = proposals.length
    ? `${activeProposalIndex + 1} / ${proposals.length}`
    : "0 / 0";
  ui.previousProposal.disabled = saveBusy || activeProposalIndex <= 0;
  ui.nextProposal.disabled = saveBusy || activeProposalIndex >= proposals.length - 1;
  proposals.forEach((proposal, index) => {
    const decision = activeCase.diagnostic_decisions[proposal.proposal_id];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "item-chip";
    if (index === activeProposalIndex) button.classList.add("active");
    if (decision) button.classList.add(decision.status === "confirmed" ? "confirmed" : "draft");
    const label = document.createElement("span");
    label.textContent = proposal.surface;
    const code = document.createElement("small");
    code.className = "code-label";
    code.textContent = `#${index + 1} · 第 ${proposal.occurrence_ordinal} 次`;
    button.append(label, code);
    button.addEventListener("click", () => selectProposal(index));
    ui.itemStrip.append(button);
  });
  ui.itemEmpty.textContent = "本条没有匿名提案，无需诊断。";
  ui.itemEmpty.classList.toggle("hidden", proposals.length !== 0);
}

function renderRawEditor() {
  const locked = activeCase.raw_annotation.status === "confirmed";
  ui.decisionKicker.textContent = mode() === "raw-repair" ? "原文标注修订" : "原文独立标注";
  ui.decisionTitle.textContent = activeMentionIndex === null
    ? "本条整体决定"
    : `Mention ${activeMentionIndex + 1}`;
  ui.decisionStatus.dataset.state = locked ? "confirmed" : "draft";
  ui.decisionStatus.textContent = locked ? "已确认" : "待确认";
  ui.saveDraft.disabled = locked;
  ui.confirmNext.disabled = locked;

  if (locked) {
    const card = document.createElement("div");
    card.className = "locked-card";
    const text = document.createElement("p");
    text.textContent = rawDraft.mentions.length
      ? `本条已锁定，共包含 ${rawDraft.mentions.length} 个 mention。`
      : "本条已锁定为无可标注表达。";
    const reopen = document.createElement("button");
    reopen.type = "button";
    reopen.className = "danger-button";
    reopen.textContent = "重新打开原文标注";
    reopen.addEventListener("click", () => reopenItem("raw"));
    card.append(text, reopen);
    ui.decisionEditor.append(card);
    return;
  }

  if (activeMentionIndex !== null && rawDraft.mentions[activeMentionIndex]) {
    const mention = rawDraft.mentions[activeMentionIndex];
    const mentionErrors = (currentErrors.mentions || [])[activeMentionIndex] || {};
    const fields = document.createElement("section");
    fields.className = "editor-section";
    const title = document.createElement("h3");
    title.textContent = "原文位置";
    const surface = document.createElement("input");
    surface.value = mention.surface || "";
    surface.maxLength = 80;
    surface.placeholder = "精确复制原文片段";
    surface.addEventListener("input", () => {
      mention.surface = surface.value;
      delete mention.start;
      delete mention.end;
      markDirty();
      renderHighlightedContent();
      renderItemStrip();
    });
    const ordinal = document.createElement("input");
    ordinal.type = "number";
    ordinal.min = "1";
    ordinal.value = mention.occurrence_ordinal || 1;
    ordinal.addEventListener("input", () => {
      mention.occurrence_ordinal = Number(ordinal.value);
      delete mention.start;
      delete mention.end;
      markDirty();
      renderHighlightedContent();
      renderItemStrip();
    });
    fields.append(
      title,
      createField("原文片段", surface, mentionErrors.surface),
      createField("出现序号", ordinal, mentionErrors.occurrence_ordinal, "同一字符串重复出现时，从 1 开始计数"),
    );
    ui.decisionEditor.append(fields);

    const routeSection = document.createElement("section");
    routeSection.className = "editor-section";
    const routeTitle = document.createElement("h3");
    routeTitle.textContent = "暂定去向";
    const routeHint = document.createElement("p");
    routeHint.className = "helper route-hint";
    routeHint.textContent = "先排除透明词、碎片和通用辱骂；不能把 C 当作不确定垃圾桶。";
    const routeGrid = document.createElement("div");
    routeGrid.className = "route-grid";
    bootstrap.provisional_routes.forEach(route => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "route-choice" + (mention.provisional_route === route ? " active" : "");
      const label = document.createElement("strong");
      label.textContent = C.ROUTE_LABELS[route] || route;
      const code = document.createElement("small");
      code.textContent = route;
      const description = document.createElement("span");
      description.className = "route-description";
      description.textContent = C.ROUTE_DESCRIPTIONS[route] || "";
      button.append(label, description, code);
      button.addEventListener("click", () => {
        mention.provisional_route = route;
        mention.reason_codes = C.routeDefaults(route);
        markDirty();
        renderDecisionEditor();
        renderItemStrip();
      });
      routeGrid.append(button);
    });
    const guide = document.createElement("button");
    guide.type = "button";
    guide.className = "quiet-button route-guide-button";
    guide.textContent = "查看完整 A / B / C 收录口径";
    guide.addEventListener("click", showGuidelines);
    routeSection.append(routeTitle, routeHint, routeGrid, guide);
    ui.decisionEditor.append(routeSection);

    const reasons = document.createElement("section");
    reasons.className = "editor-section";
    const reasonTitle = document.createElement("h3");
    reasonTitle.textContent = "判定原因";
    reasons.append(
      reasonTitle,
      createReasonPicker(mention.reason_codes || [], values => {
        mention.reason_codes = values;
        markDirty();
      }, mentionErrors.reason_codes),
    );
    ui.decisionEditor.append(reasons);

    const notes = document.createElement("textarea");
    notes.value = mention.notes || "";
    notes.maxLength = 2000;
    notes.placeholder = "可选；选择“其他原因”时必填";
    notes.addEventListener("input", () => {
      mention.notes = notes.value;
      markDirty();
    });
    const notesSection = document.createElement("section");
    notesSection.className = "editor-section";
    notesSection.append(createField("Mention 备注", notes, mentionErrors.notes));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger-button";
    remove.textContent = "删除这个 Mention";
    remove.addEventListener("click", () => {
      rawDraft.mentions.splice(activeMentionIndex, 1);
      rawDraft.needs_explanation = rawDraft.mentions.length > 0;
      activeMentionIndex = rawDraft.mentions.length
        ? Math.min(activeMentionIndex, rawDraft.mentions.length - 1)
        : null;
      markDirty();
      renderCurrentCase();
    });
    const dangerSection = document.createElement("div");
    dangerSection.className = "mention-danger";
    dangerSection.append(remove);
    ui.decisionEditor.append(notesSection, dangerSection);
  } else {
    const info = document.createElement("div");
    info.className = "info-card";
    info.textContent = "在左侧原文中选择文字即可添加标注；如果本条不包含需要解释的表达，可明确标记为无可标注表达。";
    const none = document.createElement("button");
    none.type = "button";
    none.className = "secondary-button no-mention-button";
    none.textContent = "本条无可标注表达";
    none.addEventListener("click", markNoMention);
    ui.decisionEditor.append(info, none);
  }

  const caseNotes = document.createElement("textarea");
  caseNotes.value = rawDraft.notes || "";
  caseNotes.maxLength = 4000;
  caseNotes.placeholder = "记录整条 Case 的补充说明（可选）";
  caseNotes.addEventListener("input", () => {
    rawDraft.notes = caseNotes.value;
    markDirty();
  });
  const notesSection = document.createElement("section");
  notesSection.className = "editor-section";
  notesSection.append(createField("Case 备注", caseNotes, null));
  ui.decisionEditor.append(notesSection);
}

function resultRow(result, index, errors) {
  const row = document.createElement("div");
  row.className = "result-row";
  const surface = document.createElement("input");
  surface.value = result.surface || "";
  surface.maxLength = 80;
  surface.placeholder = "原文片段";
  surface.disabled = diagnosticDraft.action === "accept";
  surface.addEventListener("input", () => {
    result.surface = surface.value;
    markDirty();
  });
  const ordinal = document.createElement("input");
  ordinal.type = "number";
  ordinal.min = "1";
  ordinal.value = result.occurrence_ordinal || 1;
  ordinal.disabled = diagnosticDraft.action === "accept";
  ordinal.addEventListener("input", () => {
    result.occurrence_ordinal = Number(ordinal.value);
    markDirty();
  });
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "icon-button";
  remove.textContent = "删除";
  remove.disabled = diagnosticDraft.action !== "split";
  remove.addEventListener("click", () => {
    diagnosticDraft.result_mentions.splice(index, 1);
    markDirty();
    renderDecisionEditor();
  });
  row.append(
    createField("原文片段", surface, errors && errors.surface),
    createField("第几次", ordinal, errors && errors.occurrence_ordinal),
    remove,
  );
  return row;
}

function renderDiagnosticEditor() {
  const proposal = activeProposal();
  if (!proposal) {
    ui.decisionKicker.textContent = "匿名提案诊断";
    ui.decisionTitle.textContent = "无需处理";
    ui.decisionStatus.textContent = "已完成";
    ui.decisionStatus.dataset.state = "confirmed";
    ui.saveDraft.disabled = true;
    ui.confirmNext.disabled = true;
    const empty = document.createElement("div");
    empty.className = "info-card";
    empty.textContent = "本条 Case 没有匿名提案。";
    ui.decisionEditor.append(empty);
    return;
  }
  const saved = activeCase.diagnostic_decisions[proposal.proposal_id];
  const locked = saved && saved.status === "confirmed";
  ui.decisionKicker.textContent = "匿名提案诊断";
  ui.decisionTitle.textContent = proposal.surface;
  ui.decisionStatus.dataset.state = locked ? "confirmed" : saved ? "draft" : "open";
  ui.decisionStatus.textContent = locked ? "已确认" : saved ? "草稿" : "待处理";
  ui.saveDraft.disabled = locked;
  ui.confirmNext.disabled = locked;

  const rawReopen = document.createElement("button");
  rawReopen.type = "button";
  rawReopen.className = "quiet-button";
  rawReopen.textContent = "修改原文标注";
  rawReopen.addEventListener("click", () => reopenItem("raw"));

  if (locked) {
    const card = document.createElement("div");
    card.className = "locked-card";
    const text = document.createElement("p");
    text.textContent = `已确认：${C.ACTION_LABELS[saved.action] || saved.action}`;
    const reopen = document.createElement("button");
    reopen.type = "button";
    reopen.className = "danger-button";
    reopen.textContent = "重新打开这个决定";
    reopen.addEventListener("click", () => reopenItem("diagnostic", proposal.proposal_id));
    card.append(text, reopen);
    ui.decisionEditor.append(card, rawReopen);
    return;
  }

  const actionSection = document.createElement("section");
  actionSection.className = "editor-section";
  const actionTitle = document.createElement("h3");
  actionTitle.textContent = "处理操作";
  const actionGrid = document.createElement("div");
  actionGrid.className = "action-grid";
  bootstrap.proposal_actions.forEach(action => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "action-choice" + (diagnosticDraft.action === action ? " active" : "");
    const label = document.createElement("span");
    label.textContent = C.ACTION_LABELS[action] || action;
    const code = document.createElement("small");
    code.textContent = action;
    button.append(label, code);
    button.addEventListener("click", () => changeDiagnosticAction(action));
    actionGrid.append(button);
  });
  actionSection.append(actionTitle, actionGrid);
  if (currentErrors.action) {
    const error = document.createElement("small");
    error.className = "field-error";
    error.textContent = currentErrors.action;
    actionSection.append(error);
  }
  ui.decisionEditor.append(actionSection);

  if (!["reject", "defer"].includes(diagnosticDraft.action)) {
    const resultSection = document.createElement("section");
    resultSection.className = "editor-section";
    const resultTitle = document.createElement("h3");
    resultTitle.textContent = diagnosticDraft.action === "accept" ? "保留的原文片段" : "修订后的原文片段";
    const helper = document.createElement("p");
    helper.className = "helper";
    helper.textContent = diagnosticDraft.action === "accept"
      ? "接受操作固定保留原提案。"
      : "建议直接在原文中选择片段；也可以手动输入作为兜底。";
    const list = document.createElement("div");
    list.className = "result-list";
    diagnosticDraft.result_mentions.forEach((result, index) => {
      list.append(resultRow(result, index, (currentErrors.results || [])[index]));
    });
    resultSection.append(resultTitle, helper, list);
    if (currentErrors.result_mentions) {
      const error = document.createElement("small");
      error.className = "field-error";
      error.textContent = currentErrors.result_mentions;
      resultSection.append(error);
    }
    if (diagnosticDraft.action === "split") {
      const add = document.createElement("button");
      add.type = "button";
      add.className = "quiet-button";
      add.textContent = "添加拆分项";
      add.addEventListener("click", () => {
        diagnosticDraft.result_mentions.push({surface: "", occurrence_ordinal: 1});
        markDirty();
        renderDecisionEditor();
      });
      resultSection.append(add);
    }
    ui.decisionEditor.append(resultSection);
  }

  const reasonSection = document.createElement("section");
  reasonSection.className = "editor-section";
  const reasonTitle = document.createElement("h3");
  reasonTitle.textContent = "判定原因";
  reasonSection.append(
    reasonTitle,
    createReasonPicker(diagnosticDraft.reason_codes || [], values => {
      diagnosticDraft.reason_codes = values;
      markDirty();
    }, currentErrors.reason_codes),
  );
  ui.decisionEditor.append(reasonSection);

  const notes = document.createElement("textarea");
  notes.value = diagnosticDraft.notes || "";
  notes.maxLength = 2000;
  notes.placeholder = diagnosticDraft.action === "defer"
    ? "请说明仍需解决的问题（必填）"
    : "补充审核说明（可选）";
  notes.addEventListener("input", () => {
    diagnosticDraft.notes = notes.value;
    markDirty();
  });
  const notesSection = document.createElement("section");
  notesSection.className = "editor-section";
  notesSection.append(createField("备注", notes, currentErrors.notes));
  ui.decisionEditor.append(notesSection, rawReopen);
}

function renderDecisionEditor() {
  ui.decisionEditor.replaceChildren();
  if (!activeCase) return;
  if (mode() === "diagnostic") renderDiagnosticEditor();
  else renderRawEditor();
}

function renderCurrentCase({revealInSidebar = false} = {}) {
  if (!activeCase) return;
  const summary = summaryFor(activeCase.case_id);
  refreshVisibleCaseQueue();
  const visibleIndex = visibleCaseIds.indexOf(activeCase.case_id);
  const conditional = Boolean(ui.caseSearch.value.trim()) || caseFilter !== "all";
  ui.casePosition.textContent = conditional && visibleIndex >= 0
    ? `结果 ${visibleIndex + 1} / ${visibleCaseIds.length} · 全部 ${caseIndex + 1} / ${bootstrap.case_summaries.length} · ${activeCase.case_id}`
    : `${caseIndex + 1} / ${bootstrap.case_summaries.length} · ${activeCase.case_id}`;
  ui.caseAlias.textContent = activeCase.blind_alias;
  ui.sourceHint.textContent = mode() === "diagnostic"
    ? "当前匿名提案已在原文中高亮；修订边界时可直接选择文字"
    : "选择需要解释的原文片段，再直接选择 A / B / C 暂定去向";
  ui.manualMention.classList.toggle("hidden", mode() === "diagnostic" || activeCase.raw_annotation.status === "confirmed");
  clearSelectionCandidate();
  renderHighlightedContent();
  renderItemStrip();
  renderDecisionEditor();
  renderProgress();
  renderCaseList({revealCurrent: revealInSidebar});
  ui.itemStripStatus.title = caseStatusText(summary);
}

function initializeCaseDrafts() {
  rawDraft = {
    needs_explanation: Boolean(activeCase.raw_annotation.needs_explanation),
    mentions: C.clone(activeCase.raw_annotation.mentions || []),
    notes: activeCase.raw_annotation.notes || "",
  };
  activeMentionIndex = rawDraft.mentions.length ? 0 : null;
  if (bootstrap.phase === "diagnostic") {
    const proposals = activeCase.proposals || [];
    const firstOpen = proposals.findIndex(proposal => {
      const decision = activeCase.diagnostic_decisions[proposal.proposal_id];
      return !decision || decision.status !== "confirmed";
    });
    activeProposalIndex = firstOpen >= 0 ? firstOpen : 0;
    const proposal = proposals[activeProposalIndex];
    const saved = proposal && activeCase.diagnostic_decisions[proposal.proposal_id];
    diagnosticDraft = proposal
      ? C.clone(saved || C.defaultDiagnosticDecision(proposal))
      : null;
    if (diagnosticDraft) delete diagnosticDraft.status;
    if (diagnosticDraft) delete diagnosticDraft.case_id;
    if (diagnosticDraft) delete diagnosticDraft.proposal_id;
  } else {
    activeProposalIndex = 0;
    diagnosticDraft = null;
  }
  localDirty = false;
  currentErrors = {};
  setSaveState("idle");
}

async function requestCase(caseId, force = false) {
  if (!force && caseCache.has(caseId)) return caseCache.get(caseId);
  const response = await getJson(`/api/cases/${encodeURIComponent(caseId)}`);
  if (response.phase !== bootstrap.phase || response.revision !== bootstrap.revision) {
    bootstrap = await getJson("/api/bootstrap");
    caseCache.clear();
    const refreshed = await getJson(`/api/cases/${encodeURIComponent(caseId)}`);
    cacheCase(refreshed.case);
    return refreshed.case;
  }
  updateSummary(response.case_summary);
  cacheCase(response.case);
  return response.case;
}

async function loadCaseById(caseId, {skipFlush = false, force = false, isCurrent = null} = {}) {
  const index = summaryIndex(caseId);
  if (!bootstrap || index < 0) return false;
  if (isCurrent && !isCurrent()) return false;
  if (!skipFlush && !(await flushChanges())) return false;
  if (isCurrent && !isCurrent()) return false;
  const sequence = ++loadSequence;
  const summary = bootstrap.case_summaries[index];
  ui.caseAlias.textContent = "正在读取 Case…";
  try {
    const caseRow = await requestCase(summary.case_id, force);
    if (sequence !== loadSequence || (isCurrent && !isCurrent())) return false;
    caseIndex = index;
    activeCase = caseRow;
    initializeCaseDrafts();
    renderCurrentCase({revealInSidebar: true});
    closeSidebar();
    prefetchNextOpen();
    return true;
  } catch (error) {
    setSaveState("error", error.message);
    showRequestError(error.message, error.technical);
    ui.decisionEditor.replaceChildren();
    const card = document.createElement("div");
    card.className = "error-card";
    card.textContent = error.message;
    ui.decisionEditor.append(card);
    return false;
  }
}

function nextOpenCaseId(currentCaseId = activeCase && activeCase.case_id, queue = visibleCaseIds) {
  return C.nextUnfinishedCaseId(queue, bootstrap.case_summaries, currentCaseId);
}

function showVisibleQueueComplete() {
  if (ui.caseSearch.value.trim()) {
    ui.searchStatus.textContent = "当前结果已全部确认";
  } else if (caseFilter !== "all") {
    ui.searchStatus.textContent = "当前筛选结果已全部确认";
  }
}

function navigateVisibleCases(offset) {
  if (!activeCase || !visibleCaseIds.length) return false;
  const current = visibleCaseIds.indexOf(activeCase.case_id);
  const target = current < 0 ? 0 : current + offset;
  if (target < 0 || target >= visibleCaseIds.length) return false;
  return loadCaseById(visibleCaseIds[target]);
}

function prefetchNextOpen() {
  const caseId = nextOpenCaseId();
  if (!caseId) return;
  if (caseCache.has(caseId)) return;
  const revision = bootstrap.revision;
  window.setTimeout(async () => {
    try {
      const response = await getJson(`/api/cases/${encodeURIComponent(caseId)}`);
      if (bootstrap.revision === revision && response.revision === revision) cacheCase(response.case);
    } catch (_) {
      // Prefetch failure must never interrupt active review.
    }
  }, 120);
}

async function selectProposal(index) {
  if (confirmBusy || !activeCase || mode() !== "diagnostic" || index === activeProposalIndex) return;
  if (!(await flushChanges())) return;
  activeProposalIndex = Math.max(0, Math.min(index, activeCase.proposals.length - 1));
  const proposal = activeProposal();
  const saved = activeCase.diagnostic_decisions[proposal.proposal_id];
  diagnosticDraft = C.clone(saved || C.defaultDiagnosticDecision(proposal));
  delete diagnosticDraft.status;
  delete diagnosticDraft.case_id;
  delete diagnosticDraft.proposal_id;
  localDirty = false;
  currentErrors = {};
  setSaveState("idle");
  ui.itemStripCard.classList.remove("sheet-open");
  renderCurrentCase();
}

function addRawMention(mention, route = "A_candidate") {
  if (confirmBusy || !activeCase || activeCase.raw_annotation.status === "confirmed") return;
  const provisionalRoute = bootstrap.provisional_routes.includes(route)
    ? route
    : "A_candidate";
  rawDraft.mentions.push({
    surface: mention.surface || "",
    occurrence_ordinal: Number(mention.occurrence_ordinal) || 1,
    provisional_route: provisionalRoute,
    reason_codes: C.routeDefaults(provisionalRoute),
    notes: "",
  });
  rawDraft.needs_explanation = true;
  activeMentionIndex = rawDraft.mentions.length - 1;
  markDirty();
  renderCurrentCase();
}

function useSelectionForRoute(route) {
  if (
    confirmBusy || !selectionCandidate ||
    mode() === "diagnostic" ||
    activeCase.raw_annotation.status === "confirmed"
  ) return;
  const selected = C.clone(selectionCandidate);
  const selection = window.getSelection();
  if (selection) selection.removeAllRanges();
  clearSelectionCandidate();
  addRawMention(selected, route);
}

async function markNoMention() {
  if (confirmBusy) return;
  if (rawDraft.mentions.length) {
    const confirmed = await openActionDialog({
      title: "清除已有 Mention？",
      message: `这会删除当前 ${rawDraft.mentions.length} 个 mention，并把本条标记为无可标注表达。`,
      confirmLabel: "清除并继续",
      danger: true,
    });
    if (!confirmed) return;
  }
  rawDraft.mentions = [];
  rawDraft.needs_explanation = false;
  activeMentionIndex = null;
  markDirty();
  renderCurrentCase();
}

function useSelection() {
  if (confirmBusy || !selectionCandidate) return;
  const selected = C.clone(selectionCandidate);
  window.getSelection().removeAllRanges();
  clearSelectionCandidate();
  if (mode() !== "diagnostic") {
    addRawMention(selected, "A_candidate");
    return;
  }
  if (diagnosticDraft.action === "split") {
    const empty = diagnosticDraft.result_mentions.find(row => !row.surface);
    if (empty) Object.assign(empty, selected);
    else diagnosticDraft.result_mentions.push(selected);
  } else if (["trim", "expand"].includes(diagnosticDraft.action)) {
    diagnosticDraft.result_mentions = [selected];
  }
  markDirty();
  renderCurrentCase();
}

function changeDiagnosticAction(action) {
  if (confirmBusy || !activeProposal() || diagnosticDraft.action === action) return;
  diagnosticDraft = C.decisionForAction(action, activeProposal(), diagnosticDraft.notes || "");
  currentErrors = {};
  markDirty();
  renderCurrentCase();
}

async function saveCurrentDraft() {
  if (confirmBusy || !activeCase || (mode() !== "diagnostic" && activeCase.raw_annotation.status === "confirmed")) return;
  if (mode() === "diagnostic") {
    const saved = activeCase.diagnostic_decisions[activeProposal().proposal_id];
    if (saved && saved.status === "confirmed") return;
  }
  if (!validateCurrent()) {
    renderDecisionEditor();
    setSaveState("error", "请先修正表单");
    return;
  }
  localDirty = true;
  editVersion += 1;
  const task = mode() === "diagnostic" ? makeDiagnosticTask(false) : makeRawTask(false);
  try {
    await saveQueue.runNow(task);
    renderItemStrip();
    renderDecisionEditor();
  } catch (error) {
    await handleMutationError(error);
  }
}

async function confirmCurrent() {
  if (confirmBusy || !activeCase) return;
  if (mode() !== "diagnostic" && activeCase.raw_annotation.status === "confirmed") return;
  if (mode() === "diagnostic") {
    const saved = activeCase.diagnostic_decisions[activeProposal().proposal_id];
    if (saved && saved.status === "confirmed") return;
  }
  if (!validateCurrent()) {
    renderDecisionEditor();
    setSaveState("error", "请先修正表单");
    return;
  }
  localDirty = true;
  editVersion += 1;
  const queueBeforeConfirm = [...visibleCaseIds];
  const task = mode() === "diagnostic" ? makeDiagnosticTask(true) : makeRawTask(true);
  setConfirmationLock(true);
  setButtonBusy(ui.confirmNext, true, "正在确认…");
  try {
    await saveQueue.runNow(task);
    if (task.kind === "raw") {
      rawDraft = {
        needs_explanation: activeCase.raw_annotation.needs_explanation,
        mentions: C.clone(activeCase.raw_annotation.mentions),
        notes: activeCase.raw_annotation.notes,
      };
      if (bootstrap.phase === "diagnostic") {
        const refreshed = await requestCase(activeCase.case_id, true);
        activeCase = refreshed;
        initializeCaseDrafts();
        renderCurrentCase();
      } else {
        const nextCaseId = nextOpenCaseId(task.case_id, queueBeforeConfirm);
        if (nextCaseId && nextCaseId !== task.case_id) {
          await loadCaseById(nextCaseId, {skipFlush: true});
        } else {
          renderCurrentCase({revealInSidebar: true});
          showVisibleQueueComplete();
        }
      }
      return;
    }
    const nextProposal = activeCase.proposals.findIndex((proposal, index) => {
      if (index <= activeProposalIndex) return false;
      const decision = activeCase.diagnostic_decisions[proposal.proposal_id];
      return !decision || decision.status !== "confirmed";
    });
    const fallbackProposal = activeCase.proposals.findIndex(proposal => {
      const decision = activeCase.diagnostic_decisions[proposal.proposal_id];
      return !decision || decision.status !== "confirmed";
    });
    const target = nextProposal >= 0 ? nextProposal : fallbackProposal;
    if (target >= 0) {
      activeProposalIndex = target;
      const proposal = activeProposal();
      diagnosticDraft = C.clone(
        activeCase.diagnostic_decisions[proposal.proposal_id] ||
        C.defaultDiagnosticDecision(proposal),
      );
      delete diagnosticDraft.status;
      delete diagnosticDraft.case_id;
      delete diagnosticDraft.proposal_id;
      localDirty = false;
      currentErrors = {};
      renderCurrentCase();
    } else {
      const nextCaseId = nextOpenCaseId(task.case_id, queueBeforeConfirm);
      if (nextCaseId && nextCaseId !== task.case_id) {
        await loadCaseById(nextCaseId, {skipFlush: true});
      } else {
        renderCurrentCase({revealInSidebar: true});
        showVisibleQueueComplete();
      }
    }
  } catch (error) {
    await handleMutationError(error);
  } finally {
    setConfirmationLock(false);
    setButtonBusy(ui.confirmNext, false);
    syncActionButtons();
    if (resumeSearchAfterConfirm) {
      resumeSearchAfterConfirm = false;
      runCaseSearch();
    }
  }
}

async function reopenItem(scope, proposalId = null) {
  if (!(await flushChanges())) return;
  const reason = await openActionDialog({
    title: scope === "raw" ? "重新打开原文标注" : "重新打开诊断决定",
    message: "该操作会写入 amendment 审计记录。请输入本次修改原因。",
    confirmLabel: "重新打开",
    input: true,
    danger: true,
  });
  if (!reason) return;
  setSaveState("saving", "正在重新打开…");
  try {
    const result = await postJson("/api/reopen", {
      expected_revision: bootstrap.revision,
      scope,
      case_id: activeCase.case_id,
      proposal_id: proposalId,
      reason,
    });
    bootstrap.revision = result.revision;
    bootstrap.phase = result.phase;
    bootstrap.status = result.status;
    updateSummary(result.case_summary);
    activeCase = result.case;
    cacheCase(activeCase);
    initializeCaseDrafts();
    if (scope === "diagnostic") {
      activeProposalIndex = activeCase.proposals.findIndex(row => row.proposal_id === proposalId);
      const proposal = activeProposal();
      diagnosticDraft = C.clone(activeCase.diagnostic_decisions[proposalId] || C.defaultDiagnosticDecision(proposal));
      delete diagnosticDraft.status;
      delete diagnosticDraft.case_id;
      delete diagnosticDraft.proposal_id;
    }
    setSaveState("saved", "已重新打开并记录修改原因");
    renderCurrentCase();
  } catch (error) {
    await handleMutationError(error);
  }
}

async function lockRawPhase() {
  if (!(await flushChanges())) return;
  const status = bootstrap.status.raw;
  const confirmed = await openActionDialog({
    title: "锁定阶段 A 并揭示匿名提案？",
    message: `当前 ${status.confirmed} / ${status.total} 条均已确认。锁定后将进入阶段 B，此顺序不可逆。`,
    confirmLabel: "锁定并揭示提案",
    danger: true,
  });
  if (!confirmed) return;
  ui.topActionMenu.open = false;
  setButtonBusy(ui.lockRaw, true, "正在锁定…");
  setSaveState("saving", "正在锁定阶段 A…");
  try {
    const activeCaseId = activeCase && activeCase.case_id;
    bootstrap = await postJson("/api/lock-raw", {expected_revision: bootstrap.revision});
    caseCache.clear();
    renderProgress();
    refreshVisibleCaseQueue();
    renderCaseList();
    const targetCaseId = activeCaseId && visibleCaseIds.includes(activeCaseId)
      ? activeCaseId
      : visibleCaseIds[0] || activeCaseId;
    if (targetCaseId) {
      await loadCaseById(targetCaseId, {skipFlush: true, force: true});
    }
    setSaveState("saved", "阶段 A 已锁定，匿名提案已揭示");
  } catch (error) {
    await handleMutationError(error);
  } finally {
    setButtonBusy(ui.lockRaw, false);
  }
}

async function exportReview() {
  if (!(await flushChanges())) return;
  setButtonBusy(ui.exportButton, true, "正在导出…");
  ui.topActionMenu.open = false;
  try {
    const blob = await postBlob("/api/export", {expected_revision: bootstrap.revision});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "wp3-s21-development-review.zip";
    anchor.click();
    URL.revokeObjectURL(url);
    setSaveState("saved", "审核包已导出");
  } catch (error) {
    await handleMutationError(error);
  } finally {
    setButtonBusy(ui.exportButton, false);
  }
}

function showShortcuts() {
  ui.shortcutDialog.showModal();
}

function showGuidelines() {
  ui.guidelineDialog.showModal();
}

function handleKeydown(event) {
  if (confirmBusy || document.querySelector("dialog[open]")) return;
  if (C.isTextEntry(event.target)) {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveCurrentDraft();
    } else if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      confirmCurrent();
    }
    return;
  }
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
  if (event.key === "?") {
    event.preventDefault();
    showShortcuts();
    return;
  }
  if (event.key === "[") {
    event.preventDefault();
    navigateVisibleCases(-1);
    return;
  }
  if (event.key === "]") {
    event.preventDefault();
    navigateVisibleCases(1);
    return;
  }
  if (mode() !== "diagnostic") {
    const route = C.quickRouteForKey(event.key);
    if (route && selectionCandidate) {
      event.preventDefault();
      useSelectionForRoute(route);
    }
    return;
  }
  if (event.key.toLowerCase() === "j") {
    event.preventDefault();
    selectProposal(activeProposalIndex - 1);
  } else if (event.key.toLowerCase() === "k") {
    event.preventDefault();
    selectProposal(activeProposalIndex + 1);
  }
}

async function initialize() {
  try {
    bootstrap = await getJson("/api/bootstrap");
    searchMode = "literal";
    searchMatches = null;
    searchMatchById = new Map();
    renderSearchMode();
    renderProgress();
    refreshVisibleCaseQueue();
    renderCaseList();
    const firstCaseId = nextOpenCaseId(null) || visibleCaseIds[0];
    if (firstCaseId) await loadCaseById(firstCaseId, {skipFlush: true});
  } catch (error) {
    setSaveState("error", error.message);
    showRequestError(error.message, error.technical);
    ui.decisionEditor.replaceChildren();
    const card = document.createElement("div");
    card.className = "error-card";
    card.textContent = error.message;
    ui.decisionEditor.append(card);
  }
}

ui.sidebarToggle.addEventListener("click", openSidebar);
ui.sidebarClose.addEventListener("click", closeSidebar);
ui.sidebarBackdrop.addEventListener("click", closeSidebar);
ui.caseSearch.addEventListener("input", scheduleCaseSearch);
ui.searchMode.addEventListener("click", cycleSearchMode);
ui.caseFilters.addEventListener("click", event => {
  const button = event.target.closest("button[data-filter]");
  if (!button) return;
  changeCaseFilter(button.dataset.filter);
});
ui.previousCase.addEventListener("click", () => navigateVisibleCases(-1));
ui.nextCase.addEventListener("click", () => navigateVisibleCases(1));
ui.previousProposal.addEventListener("click", () => selectProposal(activeProposalIndex - 1));
ui.nextProposal.addEventListener("click", () => selectProposal(activeProposalIndex + 1));
ui.proposalSheetToggle.addEventListener("click", () => {
  ui.itemStripCard.classList.add("sheet-open");
});
ui.proposalSheetClose.addEventListener("click", () => {
  ui.itemStripCard.classList.remove("sheet-open");
});
ui.manualMention.addEventListener("click", () => addRawMention({surface: "", occurrence_ordinal: 1}));
ui.quickRouteActions.addEventListener("click", event => {
  const button = event.target.closest("button[data-route]");
  if (button) useSelectionForRoute(button.dataset.route);
});
ui.useSelection.addEventListener("click", useSelection);
ui.content.addEventListener("mouseup", () => setTimeout(captureSelection, 0));
ui.content.addEventListener("touchend", () => setTimeout(captureSelection, 80));
ui.saveDraft.addEventListener("click", saveCurrentDraft);
ui.confirmNext.addEventListener("click", confirmCurrent);
ui.lockRaw.addEventListener("click", lockRawPhase);
ui.exportButton.addEventListener("click", exportReview);
ui.shortcuts.addEventListener("click", showShortcuts);
ui.guidelines.addEventListener("click", showGuidelines);
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
