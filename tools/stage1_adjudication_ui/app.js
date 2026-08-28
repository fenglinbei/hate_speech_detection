"use strict";

const CONTEXT_FIELDS = Object.freeze([
  ["field", "异常字段"],
  ["observed_type", "当前类型"],
  ["allowed_correction_types", "允许修正类型"],
]);

const GROUP_VALUES = Object.freeze([
  "Racism",
  "Region",
  "LGBTQ",
  "Sexism",
  "others",
  "non-hate",
]);

const ISSUE_LABELS = Object.freeze({
  "group-hate-atypical": "组合复核",
  "group-hate-conflict": "组合冲突",
  "hateful-null-sentinel": "NULL 标签",
  "non-string-quad-field": "非字符串字段",
});

const ISSUE_GUIDANCE = Object.freeze({
  "group-hate-atypical":
    "可接受当前独立标签组合，也可修正。若修正，non-hate 分组只能配 non-hate；其他具体分组必须配 hate。",
  "group-hate-conflict":
    "必须修正 targeted_group、hateful 或二者，并消除组合冲突：non-hate 只能互相配对，具体分组只能配 hate。",
  "hateful-null-sentinel":
    "必须把 legacy NULL hateful 标签替换为可用标签；当前具体分组要求 hateful 为 hate。",
  "non-string-quad-field":
    "必须把异常字段修正为非空 NFC 字符串或明确的 JSON null；理由代码需与最终值类型一致。",
});

const REASON_LABELS = Object.freeze({
  "valid-independent-label-combination": "接受独立标签组合",
  "correct-source-label": "修正源标签",
  "resolve-group-hate-conflict": "消除 group–hate 冲突",
  "replace-legacy-null": "替换 legacy NULL",
  "coerce-numeric-annotation": "将数值标注转为字符串",
  "set-explicit-null": "设为明确 JSON null",
});

const app = {
  state: null,
  selected: null,
  pendingSubmission: null,
  loading: false,
  submitting: false,
  filters: {
    scope: "all",
    status: "all",
    type: "all",
  },
};

const dom = {
  connectionStatus: document.getElementById("connection-status"),
  reloadButton: document.getElementById("reload-button"),
  exportButton: document.getElementById("export-button"),
  totalComplete: document.getElementById("total-complete"),
  totalCount: document.getElementById("total-count"),
  totalProgressBar: document.getElementById("total-progress-bar"),
  scopeProgress: document.getElementById("scope-progress"),
  scopeTabs: document.getElementById("scope-tabs"),
  statusFilter: document.getElementById("status-filter"),
  typeFilter: document.getElementById("type-filter"),
  visibleCount: document.getElementById("visible-count"),
  issueList: document.getElementById("issue-list"),
  emptyFilter: document.getElementById("empty-filter"),
  loadingState: document.getElementById("loading-state"),
  errorState: document.getElementById("error-state"),
  errorMessage: document.getElementById("error-message"),
  retryButton: document.getElementById("retry-button"),
  noSelectionState: document.getElementById("no-selection-state"),
  emptyNextButton: document.getElementById("empty-next-button"),
  issueCard: document.getElementById("issue-card"),
  issuePosition: document.getElementById("issue-position"),
  issueAlias: document.getElementById("issue-alias"),
  issueCode: document.getElementById("issue-code"),
  issueStatus: document.getElementById("issue-status"),
  ruleSummary: document.getElementById("rule-summary"),
  contextContentWrap: document.getElementById("context-content-wrap"),
  contextContent: document.getElementById("context-content"),
  contextMetadata: document.getElementById("context-metadata"),
  tupleBeforeWrap: document.getElementById("tuple-before-wrap"),
  tupleBefore: document.getElementById("tuple-before"),
  decisionForm: document.getElementById("decision-form"),
  acceptedOption: document.getElementById("accepted-option"),
  acceptedInput: document.getElementById("decision-accepted"),
  correctedInput: document.getElementById("decision-corrected"),
  acceptUnavailable: document.getElementById("accept-unavailable"),
  editSection: document.getElementById("edit-section"),
  editControls: document.getElementById("edit-controls"),
  previewSection: document.getElementById("preview-section"),
  beforePreview: document.getElementById("before-preview"),
  afterPreview: document.getElementById("after-preview"),
  reasonCode: document.getElementById("reason-code"),
  reason: document.getElementById("reason"),
  reasonLength: document.getElementById("reason-length"),
  validationSummary: document.getElementById("validation-summary"),
  draftStatus: document.getElementById("draft-status"),
  clearDraftButton: document.getElementById("clear-draft-button"),
  nextPendingButton: document.getElementById("next-pending-button"),
  confirmButton: document.getElementById("confirm-button"),
  completedNotice: document.getElementById("completed-notice"),
  completedTime: document.getElementById("completed-time"),
  completedNextButton: document.getElementById("completed-next-button"),
  confirmDialog: document.getElementById("confirm-dialog"),
  modalClose: document.getElementById("modal-close"),
  confirmSummary: document.getElementById("confirm-summary"),
  modalError: document.getElementById("modal-error"),
  modalSubmit: document.getElementById("modal-submit"),
  toastRegion: document.getElementById("toast-region"),
};

function createElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== undefined) {
    element.textContent = String(text);
  }
  return element;
}

function safeJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (_error) {
    return String(value);
  }
}

function issueIsComplete(issue) {
  return Boolean(issue && issue.current && issue.current.decision);
}

function issueHasDraft(issue) {
  return Boolean(issue && !issueIsComplete(issue) && loadDraft(issue));
}

function flattenedIssues() {
  if (!app.state) {
    return [];
  }
  const result = [];
  for (const scope of app.state.scopes) {
    for (const issue of scope.issues) {
      result.push({ scope, issue });
    }
  }
  return result;
}

function visibleIssues() {
  return flattenedIssues().filter(({ scope, issue }) => {
    if (app.filters.scope !== "all") {
      const scopeMatches =
        scope.scope === app.filters.scope || issue.issue_kind === app.filters.scope;
      if (!scopeMatches) {
        return false;
      }
    }
    if (app.filters.type !== "all" && issue.issue_code !== app.filters.type) {
      return false;
    }
    if (app.filters.status === "pending" && issueIsComplete(issue)) {
      return false;
    }
    if (app.filters.status === "complete" && !issueIsComplete(issue)) {
      return false;
    }
    if (app.filters.status === "draft" && !issueHasDraft(issue)) {
      return false;
    }
    return true;
  });
}

function entryIsVisible(entry) {
  if (!entry) {
    return false;
  }
  return visibleIssues().some(
    ({ scope, issue }) =>
      scope.scope === entry.scope.scope && issue.alias === entry.issue.alias,
  );
}

function syncFilterControls() {
  dom.statusFilter.value = app.filters.status;
  dom.typeFilter.value = app.filters.type;
  for (const tab of dom.scopeTabs.querySelectorAll("[data-scope-filter]")) {
    const active = tab.dataset.scopeFilter === app.filters.scope;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-pressed", String(active));
  }
}

function ensureEntryVisible(entry) {
  if (!entry || entryIsVisible(entry)) {
    return;
  }
  app.filters.scope = "all";
  app.filters.type = "all";
  if (app.filters.status === "complete" && !issueIsComplete(entry.issue)) {
    app.filters.status = "all";
  }
  if (app.filters.status === "pending" && issueIsComplete(entry.issue)) {
    app.filters.status = "all";
  }
  if (app.filters.status === "draft" && !issueHasDraft(entry.issue)) {
    app.filters.status = "all";
  }
  syncFilterControls();
}

function selectedEntry() {
  if (!app.selected) {
    return null;
  }
  return (
    flattenedIssues().find(
      ({ scope, issue }) =>
        scope.scope === app.selected.scope && issue.alias === app.selected.alias,
    ) || null
  );
}

function selectEntry(entry, options = {}) {
  if (!entry) {
    app.selected = null;
  } else {
    app.selected = { scope: entry.scope.scope, alias: entry.issue.alias };
  }
  renderIssueList();
  renderSelectedIssue();
  if (options.focusCard && entry) {
    dom.issueAlias.focus({ preventScroll: true });
    const reduceMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    dom.issueCard.scrollIntoView({
      behavior: reduceMotion ? "auto" : "smooth",
      block: "start",
    });
  }
}

function chooseInitialSelection() {
  const existing = selectedEntry();
  if (existing) {
    return existing;
  }
  const entries = visibleIssues();
  return entries.find(({ issue }) => !issueIsComplete(issue)) || entries[0] || null;
}

function setConnection(kind, message) {
  dom.connectionStatus.classList.remove("is-online", "is-error");
  if (kind === "online") {
    dom.connectionStatus.classList.add("is-online");
  }
  if (kind === "error") {
    dom.connectionStatus.classList.add("is-error");
  }
  dom.connectionStatus.textContent = message;
}

function showToast(message, kind = "info") {
  const toast = createElement("div", "toast", message);
  toast.setAttribute("role", kind === "error" ? "alert" : "status");
  if (kind === "error") {
    toast.classList.add("is-error");
  }
  if (kind === "success") {
    toast.classList.add("is-success");
  }
  dom.toastRegion.appendChild(toast);
  window.setTimeout(() => {
    toast.remove();
  }, 4800);
}

function assertStateShape(value) {
  if (!value || typeof value !== "object") {
    throw new Error("状态响应不是对象。");
  }
  if (
    typeof value.session_token !== "string" ||
    !value.session_token ||
    typeof value.workspace_fingerprint !== "string" ||
    !value.workspace_fingerprint ||
    !Array.isArray(value.scopes)
  ) {
    throw new Error("状态响应缺少必要字段。");
  }
  for (const scope of value.scopes) {
    if (
      !scope ||
      typeof scope.scope !== "string" ||
      typeof scope.revision !== "string" ||
      !Array.isArray(scope.issues)
    ) {
      throw new Error("分组状态格式无效。");
    }
  }
  return value;
}

async function responseMessage(response, fallback) {
  try {
    const value = await response.json();
    if (value && typeof value.error === "string" && value.error) {
      return value.error;
    }
    if (value && typeof value.message === "string" && value.message) {
      return value.message;
    }
  } catch (_error) {
    // The caller supplies a safe fallback for non-JSON errors.
  }
  return fallback;
}

async function loadState(options = {}) {
  if (app.loading) {
    return;
  }
  app.loading = true;
  dom.reloadButton.disabled = true;
  if (!app.state) {
    dom.loadingState.hidden = false;
    dom.errorState.hidden = true;
    dom.noSelectionState.hidden = true;
    dom.issueCard.hidden = true;
  }
  setConnection("loading", "正在同步…");
  try {
    const response = await fetch("/api/state", {
      method: "GET",
      cache: "no-store",
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(await responseMessage(response, `载入失败（${response.status}）`));
    }
    const nextState = assertStateShape(await response.json());
    app.state = nextState;
    setConnection("online", "工作区已同步");
    renderAll();
    const initial = chooseInitialSelection();
    ensureEntryVisible(initial);
    selectEntry(initial);
    if (options.conflict) {
      showToast("工作区已更新，未提交草稿仍保留，请重新核对。", "error");
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "发生未知错误。";
    setConnection("error", "连接失败");
    dom.loadingState.hidden = true;
    dom.issueCard.hidden = true;
    dom.noSelectionState.hidden = true;
    dom.errorState.hidden = false;
    dom.errorMessage.textContent = message;
  } finally {
    app.loading = false;
    dom.reloadButton.disabled = false;
  }
}

function renderAll() {
  dom.loadingState.hidden = true;
  dom.errorState.hidden = true;
  renderProgress();
  renderIssueList();
  dom.exportButton.disabled = !app.state || !app.state.all_complete || app.submitting;
}

function renderProgress() {
  const complete = Number((app.state && app.state.complete_count) || 0);
  const total = Number((app.state && app.state.total_count) || 0);
  dom.totalComplete.textContent = String(complete);
  dom.totalCount.textContent = String(total);
  const percent = total > 0 ? Math.min(100, Math.max(0, (complete / total) * 100)) : 0;
  dom.totalProgressBar.style.width = `${percent}%`;
  if (dom.totalProgressBar.parentElement) {
    dom.totalProgressBar.parentElement.setAttribute("aria-valuenow", String(Math.round(percent)));
    dom.totalProgressBar.parentElement.setAttribute("aria-valuetext", `已完成 ${complete} / ${total}`);
  }

  dom.scopeProgress.replaceChildren();
  for (const scope of (app.state && app.state.scopes) || []) {
    const item = createElement("div", "scope-progress-item");
    const title = createElement("span", "", scope.title || scope.scope);
    const counts = createElement(
      "strong",
      "",
      `${Number(scope.complete_count || 0)} / ${Number(scope.row_count || 0)}`,
    );
    item.append(title, counts);
    dom.scopeProgress.appendChild(item);
  }
}

function renderIssueList() {
  dom.issueList.replaceChildren();
  const entries = visibleIssues();
  dom.visibleCount.textContent = `${entries.length} 条`;
  dom.emptyFilter.hidden = entries.length !== 0;

  for (const entry of entries) {
    const { scope, issue } = entry;
    const button = createElement("button", "issue-list-button");
    button.type = "button";
    button.dataset.alias = issue.alias;
    const statusLabel = issueIsComplete(issue)
      ? "已完成"
      : issueHasDraft(issue)
        ? "有草稿"
        : "待处理";
    const selected =
      app.selected &&
      app.selected.scope === scope.scope &&
      app.selected.alias === issue.alias;
    if (selected) {
      button.classList.add("is-selected");
      button.setAttribute("aria-current", "true");
    }
    button.setAttribute(
      "aria-label",
      `${issue.ordinal}，${issue.alias}，${statusLabel}`,
    );
    button.addEventListener("click", () => selectEntry(entry, { focusCard: true }));

    const order = createElement("span", "issue-order", String(issue.ordinal));
    const main = createElement("span", "issue-list-main");
    const alias = createElement("strong", "", issue.alias);
    const detail = createElement(
      "small",
      "",
      `${ISSUE_LABELS[issue.issue_code] || issue.issue_code} · ${statusLabel}`,
    );
    main.append(alias, detail);
    const status = createElement("span", "list-status");
    status.setAttribute("aria-hidden", "true");
    if (issueIsComplete(issue)) {
      status.classList.add("is-complete");
    } else if (issueHasDraft(issue)) {
      status.classList.add("is-draft");
    }
    button.append(order, main, status);
    dom.issueList.appendChild(button);
  }
}

function refreshIssueListDraftMarker(issue) {
  for (const button of dom.issueList.querySelectorAll(".issue-list-button")) {
    if (button.dataset.alias !== issue.alias) {
      continue;
    }
    const draft = issueHasDraft(issue);
    const status = button.querySelector(".list-status");
    if (status) {
      status.classList.toggle("is-draft", draft);
    }
    button.setAttribute(
      "aria-label",
      `${issue.ordinal}，${issue.alias}，${draft ? "有草稿" : "待处理"}`,
    );
    const detail = button.querySelector(".issue-list-main small");
    if (detail) {
      detail.textContent = `${ISSUE_LABELS[issue.issue_code] || issue.issue_code} · ${draft ? "有草稿" : "待处理"}`;
    }
  }
}

function safeContextValue(value) {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).join("、");
  }
  return safeJson(value);
}

function renderContext(issue) {
  const context = issue.context && typeof issue.context === "object" ? issue.context : {};
  const content = typeof context.content === "string" ? context.content : "";
  dom.contextContentWrap.hidden = !content;
  dom.contextContent.textContent = content;

  dom.contextMetadata.replaceChildren();
  for (const [key, label] of CONTEXT_FIELDS) {
    if (!Object.prototype.hasOwnProperty.call(context, key)) {
      continue;
    }
    const group = createElement("div");
    const term = createElement("dt", "", label);
    const description = createElement("dd", "", safeContextValue(context[key]));
    group.append(term, description);
    dom.contextMetadata.appendChild(group);
  }

  const hasTuple =
    context.tuple_before && typeof context.tuple_before === "object";
  dom.tupleBeforeWrap.hidden = !hasTuple;
  dom.tupleBefore.textContent = hasTuple ? safeJson(context.tuple_before) : "";
}

function reasonCodeOptions(issue, decision) {
  const source = issue.reason_codes;
  let values = [];
  if (Array.isArray(source)) {
    values = source;
  } else if (source && typeof source === "object" && Array.isArray(source[decision])) {
    values = source[decision];
  }
  return values
    .map((value) => {
      if (typeof value === "string") {
        return {
          value,
          label: REASON_LABELS[value] ? `${REASON_LABELS[value]} · ${value}` : value,
        };
      }
      if (value && typeof value === "object") {
        const code = value.value || value.code;
        if (typeof code === "string" && code) {
          return {
            value: code,
            label: typeof value.label === "string" && value.label ? value.label : code,
          };
        }
      }
      return null;
    })
    .filter(Boolean);
}

function renderReasonCodes(issue, desiredValue = "") {
  const decision = currentDecision();
  const options = reasonCodeOptions(issue, decision);
  dom.reasonCode.replaceChildren();
  const placeholder = createElement("option", "", decision ? "请选择" : "请先选择裁决");
  placeholder.value = "";
  dom.reasonCode.appendChild(placeholder);
  for (const item of options) {
    const option = createElement("option", "", item.label);
    option.value = item.value;
    dom.reasonCode.appendChild(option);
  }
  const exists = options.some((item) => item.value === desiredValue);
  dom.reasonCode.value = exists ? desiredValue : "";
  dom.reasonCode.disabled = !decision || issueIsComplete(issue);
}

function fieldFromPath(path) {
  const parts = String(path).split("/").filter(Boolean);
  return parts[parts.length - 1] || "value";
}

function setValueControlsEnabled(root, enabled) {
  for (const control of root.querySelectorAll("input, select, textarea")) {
    if (!control.classList.contains("edit-enable")) {
      control.disabled = !enabled;
    }
  }
  if (enabled) {
    for (const nullable of root.querySelectorAll('[data-value-kind="nullable-text"]')) {
      const nullControl = nullable.querySelector(".null-value");
      const textControl = nullable.querySelector(".text-value");
      if (nullControl && nullControl.checked && textControl) {
        textControl.disabled = true;
      }
    }
  }
}

function createSelectControl(values, currentValue) {
  const select = createElement("select", "edit-value");
  select.dataset.valueKind = "select";
  const placeholder = createElement("option", "", "请选择修正值");
  placeholder.value = "";
  select.appendChild(placeholder);
  for (const value of values) {
    const option = createElement("option", "", value);
    option.value = value;
    select.appendChild(option);
  }
  if (typeof currentValue === "string" && values.includes(currentValue)) {
    select.value = currentValue;
  }
  return select;
}

function createGroupControl(currentValue) {
  const container = createElement("div", "group-options");
  container.dataset.valueKind = "groups";
  container.setAttribute("role", "group");
  container.setAttribute("aria-label", "新的 targeted_group 值");
  const selected = new Set(
    typeof currentValue === "string"
      ? currentValue.split(",").map((value) => value.trim()).filter(Boolean)
      : [],
  );
  for (const value of GROUP_VALUES) {
    const label = createElement("label");
    const checkbox = createElement("input", "group-value");
    checkbox.type = "checkbox";
    checkbox.value = value;
    checkbox.checked = selected.has(value);
    const text = createElement("span", "", value);
    label.append(checkbox, text);
    container.appendChild(label);
  }
  container.addEventListener("change", (event) => {
    if (!(event.target instanceof HTMLInputElement) || !event.target.checked) {
      return;
    }
    const boxes = [...container.querySelectorAll(".group-value")];
    if (event.target.value === "non-hate") {
      for (const box of boxes) {
        if (box !== event.target) {
          box.checked = false;
        }
      }
    } else {
      const nonHate = boxes.find((box) => box.value === "non-hate");
      if (nonHate) {
        nonHate.checked = false;
      }
    }
  });
  return container;
}

function createNullableTextControl(field, currentValue) {
  const container = createElement("div", "edit-value-wrap");
  container.dataset.valueKind = "nullable-text";
  const nullWrap = createElement("div", "null-toggle");
  const nullLabel = createElement("label");
  const nullInput = createElement("input", "null-value");
  nullInput.type = "checkbox";
  nullInput.checked = currentValue === null;
  const nullText = createElement("span", "", "设为 JSON null");
  nullLabel.append(nullInput, nullText);
  nullWrap.appendChild(nullLabel);
  const textInput = createElement("input", "text-value");
  textInput.type = "text";
  textInput.autocomplete = "off";
  textInput.placeholder = `输入新的 ${field} 字符串`;
  textInput.setAttribute("aria-label", `新的 ${field} 字符串`);
  if (typeof currentValue === "string") {
    textInput.value = currentValue;
  }
  textInput.disabled = nullInput.checked;
  nullInput.addEventListener("change", () => {
    textInput.disabled = nullInput.checked || !isEditRootEnabled(container.closest(".edit-control"));
    if (!textInput.disabled) {
      textInput.focus();
    }
  });
  container.append(nullWrap, textInput);
  return container;
}

function createPlainTextControl(field, currentValue) {
  const input = createElement("input", "edit-value text-value");
  input.dataset.valueKind = "text";
  input.type = "text";
  input.autocomplete = "off";
  input.placeholder = `输入新的 ${field}`;
  if (typeof currentValue === "string") {
    input.value = currentValue;
  }
  return input;
}

function createJsonControl(currentValue) {
  const textarea = createElement("textarea", "edit-value json-value");
  textarea.dataset.valueKind = "json";
  textarea.rows = 3;
  textarea.spellcheck = false;
  textarea.placeholder = "输入合法 JSON 值";
  if (currentValue !== undefined) {
    textarea.value = JSON.stringify(currentValue);
  }
  return textarea;
}

function createValueControl(path, currentValue, issue) {
  const field = fieldFromPath(path);
  if (field === "hateful") {
    const values = issue.issue_code === "hateful-null-sentinel"
      ? ["hate"]
      : ["hate", "non-hate"];
    const control = createSelectControl(values, currentValue);
    control.setAttribute("aria-label", "新的 hateful 值");
    return control;
  }
  if (field === "targeted_group") {
    return createGroupControl(currentValue);
  }
  if (field === "target" || field === "argument") {
    return createNullableTextControl(field, currentValue);
  }
  if (field === "id") {
    const control = createPlainTextControl(field, currentValue);
    control.setAttribute("aria-label", `新的 ${field} 值`);
    return control;
  }
  const control = createJsonControl(currentValue);
  control.setAttribute("aria-label", `新的 ${field} JSON 值`);
  return control;
}

function isEditRootEnabled(root) {
  const toggle = root && root.querySelector(".edit-enable");
  return Boolean(toggle && toggle.checked);
}

function renderEditControls(issue, edits = []) {
  dom.editControls.replaceChildren();
  const editsByPath = new Map(
    (Array.isArray(edits) ? edits : [])
      .filter((edit) => edit && typeof edit.json_pointer === "string")
      .map((edit) => [edit.json_pointer, edit]),
  );
  const paths = Array.isArray(issue.allowed_edit_paths) ? issue.allowed_edit_paths : [];

  for (const path of paths) {
    const existing = editsByPath.get(path);
    const root = createElement("div", "edit-control");
    root.dataset.path = path;

    const toggleLabel = createElement("label", "edit-toggle");
    const enabled = createElement("input", "edit-enable");
    enabled.type = "checkbox";
    enabled.checked = Boolean(existing) || paths.length === 1;
    const pathText = createElement("span", "edit-path", path);
    toggleLabel.append(enabled, pathText);

    const valueWrap = createElement("div", "edit-value-wrap");
    const valueLabel = createElement("span", "", `新的 ${fieldFromPath(path)} 值`);
    const valueControl = createValueControl(path, existing && existing.value, issue);
    valueWrap.append(valueLabel, valueControl);
    root.append(toggleLabel, valueWrap);
    dom.editControls.appendChild(root);

    setValueControlsEnabled(root, enabled.checked);
    enabled.addEventListener("change", () => {
      setValueControlsEnabled(root, enabled.checked);
      updatePreview();
      saveCurrentDraft();
    });
  }
}

function currentDecision() {
  const selected = dom.decisionForm.querySelector('input[name="decision"]:checked');
  return selected ? selected.value : "";
}

function readEditValue(root, strict) {
  const path = root.dataset.path || "";
  const field = fieldFromPath(path);
  const select = root.querySelector('[data-value-kind="select"]');
  if (select) {
    if (strict && !select.value) {
      throw new Error(`${field} 尚未选择修正值。`);
    }
    return select.value;
  }

  const groups = root.querySelector('[data-value-kind="groups"]');
  if (groups) {
    const selected = [...groups.querySelectorAll(".group-value:checked")].map(
      (input) => input.value,
    );
    if (strict && selected.length === 0) {
      throw new Error("targeted_group 至少选择一项。");
    }
    if (strict && selected.includes("non-hate") && selected.length > 1) {
      throw new Error("non-hate 不能与其他分组同时选择。");
    }
    return GROUP_VALUES.filter((value) => selected.includes(value)).join(", ");
  }

  const nullable = root.querySelector('[data-value-kind="nullable-text"]');
  if (nullable) {
    const nullControl = nullable.querySelector(".null-value");
    const isNull = nullControl && nullControl.checked;
    if (isNull) {
      return null;
    }
    const input = nullable.querySelector(".text-value");
    const value = (input && input.value) || "";
    if (strict && (!value || value !== value.trim())) {
      throw new Error(`${field} 必须是非空且无首尾空白的字符串。`);
    }
    const normalized = value.normalize("NFC");
    if (strict && value !== normalized) {
      throw new Error(`${field} 必须使用 NFC 规范字符串。`);
    }
    if (strict && value === "NULL") {
      throw new Error(`${field} 不能使用字符串 “NULL”；如需空值请勾选 JSON null。`);
    }
    return value;
  }

  const text = root.querySelector('[data-value-kind="text"]');
  if (text) {
    const value = text.value;
    if (strict && field === "id" && !/^[1-9][0-9]*$/.test(value)) {
      throw new Error("id 必须是非零开头的十进制字符串。");
    }
    if (strict && !value) {
      throw new Error(`${field} 不能为空。`);
    }
    return value;
  }

  const json = root.querySelector('[data-value-kind="json"]');
  if (json) {
    if (!json.value.trim()) {
      if (strict) {
        throw new Error(`${field} 尚未填写 JSON 值。`);
      }
      return "";
    }
    try {
      return JSON.parse(json.value);
    } catch (_error) {
      if (strict) {
        throw new Error(`${field} 不是合法 JSON。`);
      }
      return json.value;
    }
  }
  if (strict) {
    throw new Error(`无法读取 ${path} 的修正值。`);
  }
  return "";
}

function collectEdits(strict = false) {
  const edits = [];
  for (const root of dom.editControls.querySelectorAll(".edit-control")) {
    if (!isEditRootEnabled(root)) {
      continue;
    }
    edits.push({
      json_pointer: root.dataset.path,
      value: readEditValue(root, strict),
    });
  }
  return edits;
}

function draftKey(issue) {
  if (!app.state || !issue) {
    return "";
  }
  return `stage1-adjudication-draft:v1:${app.state.workspace_fingerprint}:${issue.alias}`;
}

function storageRead(storage, key) {
  try {
    const raw = storage.getItem(key);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch (_error) {
    return null;
  }
}

function loadDraft(issue) {
  const key = draftKey(issue);
  if (!key) {
    return null;
  }
  return storageRead(window.localStorage, key) || storageRead(window.sessionStorage, key);
}

function saveDraft(issue, value) {
  const key = draftKey(issue);
  if (!key) {
    return;
  }
  const serialized = JSON.stringify(value);
  let saved = false;
  for (const storage of [window.localStorage, window.sessionStorage]) {
    try {
      storage.setItem(key, serialized);
      saved = true;
    } catch (_error) {
      // A second storage area can still preserve the draft.
    }
  }
  dom.draftStatus.textContent = saved
    ? "草稿已保存在此浏览器中，提交成功后会自动清除。"
    : "浏览器未允许保存草稿，请勿关闭本页。";
  dom.clearDraftButton.disabled = !saved;
  refreshIssueListDraftMarker(issue);
}

function clearDraft(issue) {
  const key = draftKey(issue);
  if (!key) {
    return;
  }
  for (const storage of [window.localStorage, window.sessionStorage]) {
    try {
      storage.removeItem(key);
    } catch (_error) {
      // Clearing the other storage area is still useful.
    }
  }
  refreshIssueListDraftMarker(issue);
}

function saveCurrentDraft() {
  const entry = selectedEntry();
  if (!entry || issueIsComplete(entry.issue)) {
    return;
  }
  saveDraft(entry.issue, {
    decision: currentDecision(),
    edits: collectEdits(false),
    reason_code: dom.reasonCode.value,
    reason: dom.reason.value,
  });
}

function setDecision(decision, options = {}) {
  const entry = selectedEntry();
  if (!entry) {
    return;
  }
  const issue = entry.issue;
  if (decision === "accepted" && !issue.accept_allowed) {
    return;
  }
  dom.acceptedInput.checked = decision === "accepted";
  dom.correctedInput.checked = decision === "corrected";
  dom.editSection.hidden = decision !== "corrected";
  dom.previewSection.hidden = decision !== "corrected";
  const desiredReason =
    options.reasonCode !== undefined ? options.reasonCode : dom.reasonCode.value;
  renderReasonCodes(issue, desiredReason);
  syncEditControlsForDecision(issue, decision);
  updatePreview();
  hideValidationSummary();
  if (!options.skipDraft) {
    saveCurrentDraft();
  }
}

function syncEditControlsForDecision(issue, decision) {
  const corrected = decision === "corrected" && !issueIsComplete(issue);
  for (const root of dom.editControls.querySelectorAll(".edit-control")) {
    const enabled = root.querySelector(".edit-enable");
    if (enabled) {
      enabled.disabled = !corrected;
    }
    setValueControlsEnabled(
      root,
      corrected && Boolean(enabled && enabled.checked),
    );
  }
}

function previewObjects(issue, edits) {
  const context = issue.context && typeof issue.context === "object" ? issue.context : {};
  const before =
    context.tuple_before && typeof context.tuple_before === "object"
      ? JSON.parse(JSON.stringify(context.tuple_before))
      : {};
  const after = JSON.parse(JSON.stringify(before));
  for (const edit of edits) {
    const field = fieldFromPath(edit.json_pointer);
    after[field] = edit.value;
  }
  return { before, after };
}

function valuesEqual(left, right) {
  return safeJson(left) === safeJson(right);
}

function semanticCorrectionErrors(issue, edits, reasonCode) {
  const errors = [];
  const context = issue.context && typeof issue.context === "object" ? issue.context : {};
  const before = context.tuple_before && typeof context.tuple_before === "object"
    ? context.tuple_before
    : {};
  for (const edit of edits) {
    const field = fieldFromPath(edit.json_pointer);
    if (Object.prototype.hasOwnProperty.call(before, field) && valuesEqual(before[field], edit.value)) {
      errors.push(`${field} 的修正值与原值相同，不能提交 no-op。`);
    }
  }

  if (issue.issue_kind === "group-hate") {
    const after = previewObjects(issue, edits).after;
    const groups = typeof after.targeted_group === "string"
      ? after.targeted_group.split(",").map((value) => value.trim()).filter(Boolean)
      : [];
    const expectedHateful = groups.length === 1 && groups[0] === "non-hate"
      ? "non-hate"
      : "hate";
    if (after.hateful !== expectedHateful) {
      errors.push(
        `修正后仍有组合冲突：当前 targeted_group 要求 hateful 为 ${expectedHateful}。`,
      );
    }
  }

  if (issue.issue_kind === "field-type" && edits.length === 1) {
    const value = edits[0].value;
    if (value === null && reasonCode !== "set-explicit-null") {
      errors.push("JSON null 必须使用理由代码 set-explicit-null。");
    }
    if (typeof value === "string" && reasonCode !== "coerce-numeric-annotation") {
      errors.push("字符串修正必须使用理由代码 coerce-numeric-annotation。");
    }
  }
  return errors;
}

function syncFieldReasonCode() {
  const entry = selectedEntry();
  if (
    !entry ||
    entry.issue.issue_kind !== "field-type" ||
    currentDecision() !== "corrected"
  ) {
    return;
  }
  const edits = collectEdits(false);
  if (edits.length !== 1) {
    return;
  }
  const desired = edits[0].value === null
    ? "set-explicit-null"
    : typeof edits[0].value === "string"
      ? "coerce-numeric-annotation"
      : "";
  const allowed = reasonCodeOptions(entry.issue, "corrected").map((item) => item.value);
  if (desired && allowed.includes(desired)) {
    dom.reasonCode.value = desired;
  }
}

function updatePreview() {
  const entry = selectedEntry();
  if (!entry || currentDecision() !== "corrected") {
    dom.previewSection.hidden = true;
    return;
  }
  dom.previewSection.hidden = false;
  const preview = previewObjects(entry.issue, collectEdits(false));
  dom.beforePreview.textContent = safeJson(preview.before);
  dom.afterPreview.textContent = safeJson(preview.after);
}

function setFormDisabled(disabled) {
  for (const control of dom.decisionForm.querySelectorAll("input, select, textarea, button")) {
    control.disabled = disabled;
  }
}

function renderSelectedIssue() {
  dom.loadingState.hidden = true;
  dom.errorState.hidden = true;
  const entry = selectedEntry();
  if (!entry) {
    dom.issueCard.hidden = true;
    dom.noSelectionState.hidden = false;
    return;
  }
  dom.noSelectionState.hidden = true;
  dom.issueCard.hidden = false;
  const { scope, issue } = entry;
  const complete = issueIsComplete(issue);

  dom.issuePosition.textContent = `${scope.title || scope.scope} · ${issue.ordinal} / ${scope.row_count}`;
  dom.issueAlias.textContent = issue.alias;
  dom.issueAlias.tabIndex = -1;
  dom.issueCode.textContent = `${ISSUE_LABELS[issue.issue_code] || "裁决异常"} · ${issue.issue_code}`;
  dom.ruleSummary.textContent = ISSUE_GUIDANCE[issue.issue_code] || "请按冻结规则完成本条裁决。";
  dom.issueStatus.textContent = complete ? "已完成" : "待处理";
  dom.issueStatus.classList.toggle("is-complete", complete);
  renderContext(issue);

  dom.decisionForm.reset();
  dom.acceptedOption.hidden = !issue.accept_allowed;
  dom.acceptedInput.disabled = !issue.accept_allowed;
  dom.acceptUnavailable.hidden = Boolean(issue.accept_allowed);
  hideValidationSummary();

  const storedDraft = complete ? null : loadDraft(issue);
  const source = storedDraft || issue.current || {};
  renderEditControls(issue, Array.isArray(source.edits) ? source.edits : []);
  dom.reason.value = typeof source.reason === "string" ? source.reason : "";
  dom.reasonLength.textContent = String(dom.reason.value.length);
  setDecision(typeof source.decision === "string" ? source.decision : "", {
    reasonCode: typeof source.reason_code === "string" ? source.reason_code : "",
    skipDraft: true,
  });

  dom.completedNotice.hidden = !complete;
  dom.completedNotice.style.display = complete ? "flex" : "none";
  dom.completedTime.textContent = complete && issue.current.reviewed_at
    ? `提交时间：${formatDateTime(issue.current.reviewed_at)}`
    : "";
  setFormDisabled(complete);
  dom.clearDraftButton.disabled = complete || !storedDraft;
  dom.acceptedInput.disabled = complete || !issue.accept_allowed;
  dom.reasonCode.disabled = complete || !currentDecision();
  syncEditControlsForDecision(issue, currentDecision());
  dom.draftStatus.textContent = storedDraft
    ? "已恢复此工作区与条目的浏览器草稿。"
    : complete
      ? ""
      : "尚未产生本地草稿。";
  updatePreview();
}

function formatDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function showValidationSummary(errors) {
  dom.validationSummary.replaceChildren();
  const list = createElement("ul");
  for (const error of errors) {
    list.appendChild(createElement("li", "", error));
  }
  dom.validationSummary.appendChild(list);
  dom.validationSummary.hidden = false;
  dom.validationSummary.focus({ preventScroll: true });
}

function hideValidationSummary() {
  dom.validationSummary.hidden = true;
  dom.validationSummary.replaceChildren();
}

function buildSubmission() {
  const entry = selectedEntry();
  if (!entry) {
    throw new Error("尚未选择条目。");
  }
  const { scope, issue } = entry;
  const decision = currentDecision();
  const errors = [];
  if (!decision) {
    errors.push("请选择“接受原标注”或“修正标注”。");
  }
  if (decision === "accepted" && !issue.accept_allowed) {
    errors.push("此条目不允许接受原标注。");
  }

  let edits = [];
  if (decision === "corrected") {
    try {
      edits = collectEdits(true);
      if (edits.length === 0) {
        errors.push("修正标注时至少填写一项字段修正。");
      }
    } catch (error) {
      errors.push(error instanceof Error ? error.message : "字段修正无效。");
    }
  }

  const reasonCode = dom.reasonCode.value;
  const validCodes = reasonCodeOptions(issue, decision).map((item) => item.value);
  if (!reasonCode || !validCodes.includes(reasonCode)) {
    errors.push("请选择与当前裁决匹配的理由代码。");
  }
  if (decision === "corrected" && edits.length > 0) {
    errors.push(...semanticCorrectionErrors(issue, edits, reasonCode));
  }
  const reason = dom.reason.value.trim();
  if (!reason) {
    errors.push("请填写裁决理由。");
  }

  if (errors.length > 0) {
    showValidationSummary(errors);
    return null;
  }
  hideValidationSummary();
  return {
    session_token: app.state.session_token,
    scope: scope.scope,
    alias: issue.alias,
    revision: scope.revision,
    decision,
    edits: decision === "corrected" ? edits : [],
    reason_code: reasonCode,
    reason,
  };
}

function appendSummaryRow(label, value) {
  const term = createElement("dt", "", label);
  const description = createElement("dd", "", value);
  dom.confirmSummary.append(term, description);
}

function showConfirmation(submission) {
  app.pendingSubmission = submission;
  dom.modalError.hidden = true;
  dom.modalError.textContent = "";
  dom.confirmSummary.replaceChildren();
  appendSummaryRow("盲化编号", submission.alias);
  appendSummaryRow(
    "裁决",
    submission.decision === "accepted" ? "接受原标注" : "修正标注",
  );
  appendSummaryRow(
    "修正",
    submission.edits.length > 0 ? safeJson(submission.edits) : "无",
  );
  appendSummaryRow("理由代码", submission.reason_code);
  appendSummaryRow("裁决理由", submission.reason);
  dom.modalSubmit.disabled = false;
  dom.modalSubmit.textContent = "确认提交";
  dom.confirmDialog.showModal();
  dom.modalSubmit.focus();
}

async function submitDecision() {
  if (!app.pendingSubmission || app.submitting) {
    return;
  }
  const submission = app.pendingSubmission;
  const originalEntry = selectedEntry();
  app.submitting = true;
  dom.modalSubmit.disabled = true;
  dom.modalSubmit.textContent = "正在提交…";
  try {
    const response = await fetch("/api/decision", {
      method: "POST",
      cache: "no-store",
      credentials: "same-origin",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(submission),
    });
    if (response.status === 409) {
      const conflictMessage = await responseMessage(
        response,
        "工作区已发生变化，请重新核对后提交。",
      );
      dom.confirmDialog.close();
      app.pendingSubmission = null;
      await loadState();
      showToast(conflictMessage, "error");
      return;
    }
    if (!response.ok) {
      throw new Error(await responseMessage(response, `提交失败（${response.status}）`));
    }
    const nextState = assertStateShape(await response.json());
    if (originalEntry) {
      clearDraft(originalEntry.issue);
    }
    app.state = nextState;
    dom.confirmDialog.close();
    app.pendingSubmission = null;
    setConnection("online", "工作区已同步");
    renderAll();
    const next = findNextPending(originalEntry);
    const refreshedOriginal = flattenedIssues().find(
      ({ scope, issue }) =>
        scope.scope === submission.scope && issue.alias === submission.alias,
    );
    const destination = next || refreshedOriginal || chooseInitialSelection();
    ensureEntryVisible(destination);
    selectEntry(destination);
    showToast(
      next
        ? `${submission.alias} 已标记完成；已切换到下一条待处理。`
        : `${submission.alias} 已标记完成。`,
      "success",
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "提交失败。";
    dom.modalError.textContent = message;
    dom.modalError.hidden = false;
    showToast(message, "error");
    dom.modalSubmit.disabled = false;
    dom.modalSubmit.textContent = "重新提交";
  } finally {
    app.submitting = false;
    dom.exportButton.disabled = !app.state || !app.state.all_complete;
  }
}

function findNextPending(currentEntry = selectedEntry()) {
  let entries = visibleIssues();
  if (!entries.some(({ issue }) => !issueIsComplete(issue))) {
    entries = flattenedIssues();
  }
  if (entries.length === 0) {
    return null;
  }
  const currentIndex = currentEntry
    ? entries.findIndex(
        ({ scope, issue }) =>
          scope.scope === currentEntry.scope.scope && issue.alias === currentEntry.issue.alias,
      )
    : -1;
  for (let offset = 1; offset <= entries.length; offset += 1) {
    const index = (Math.max(currentIndex, -1) + offset) % entries.length;
    if (!issueIsComplete(entries[index].issue)) {
      return entries[index];
    }
  }
  return null;
}

function goToNextPending() {
  const next = findNextPending();
  if (next) {
    ensureEntryVisible(next);
    selectEntry(next, { focusCard: true });
  } else {
    showToast(
      app.state && app.state.all_complete
        ? "所有条目均已完成，可以导出结果。"
        : "当前没有可见的待处理条目。",
    );
  }
}

function navigateVisible(direction) {
  const entries = visibleIssues();
  if (entries.length === 0) {
    return;
  }
  const current = selectedEntry();
  const currentIndex = current
    ? entries.findIndex(
        ({ scope, issue }) =>
          scope.scope === current.scope.scope && issue.alias === current.issue.alias,
      )
    : -1;
  const start = currentIndex >= 0 ? currentIndex : 0;
  const nextIndex = (start + direction + entries.length) % entries.length;
  selectEntry(entries[nextIndex], { focusCard: true });
}

function filenameFromDisposition(value) {
  if (!value) {
    return "stage1-adjudication-results.zip";
  }
  const utf8 = value.match(/filename\*=UTF-8''([^;]+)/i);
  const basic = value.match(/filename="?([^";]+)"?/i);
  const raw = utf8 ? decodeURIComponent(utf8[1]) : basic ? basic[1] : "";
  const clean = raw.replace(/[^a-zA-Z0-9._-]/g, "_");
  return clean.toLowerCase().endsWith(".zip") ? clean : "stage1-adjudication-results.zip";
}

async function exportResults() {
  if (!app.state || !app.state.all_complete || app.submitting) {
    showToast("所有条目完成后才能导出。", "error");
    return;
  }
  app.submitting = true;
  dom.exportButton.disabled = true;
  dom.exportButton.textContent = "正在导出…";
  try {
    const revisions = {};
    for (const scope of app.state.scopes) {
      revisions[scope.scope] = scope.revision;
    }
    const response = await fetch("/api/export", {
      method: "POST",
      cache: "no-store",
      credentials: "same-origin",
      headers: {
        Accept: "application/zip",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_token: app.state.session_token,
        workspace_fingerprint: app.state.workspace_fingerprint,
        revisions,
      }),
    });
    if (response.status === 409) {
      const conflictMessage = await responseMessage(
        response,
        "工作区已发生变化，请重新核对后导出。",
      );
      await loadState();
      showToast(conflictMessage, "error");
      return;
    }
    if (!response.ok) {
      throw new Error(await responseMessage(response, `导出失败（${response.status}）`));
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filenameFromDisposition(response.headers.get("Content-Disposition"));
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("结果 ZIP 已生成。", "success");
  } catch (error) {
    showToast(error instanceof Error ? error.message : "导出失败。", "error");
  } finally {
    app.submitting = false;
    dom.exportButton.textContent = "导出结果 ZIP";
    dom.exportButton.disabled = !app.state || !app.state.all_complete;
  }
}

function applyFilters() {
  const initial = chooseInitialSelection();
  const entries = visibleIssues();
  const selectedVisible = initial
    ? entries.some(
        ({ scope, issue }) =>
          scope.scope === initial.scope.scope && issue.alias === initial.issue.alias,
      )
    : false;
  if (!selectedVisible) {
    app.selected = null;
  }
  renderIssueList();
  selectEntry(selectedVisible ? initial : chooseInitialSelection());
}

function clearCurrentDraft() {
  const entry = selectedEntry();
  if (!entry || issueIsComplete(entry.issue) || !issueHasDraft(entry.issue)) {
    return;
  }
  clearDraft(entry.issue);
  if (app.filters.status === "draft") {
    applyFilters();
  } else {
    renderSelectedIssue();
  }
  showToast("本条浏览器草稿已清除。", "success");
}

function onScopeTab(event) {
  const button = event.target.closest("[data-scope-filter]");
  if (!button) {
    return;
  }
  app.filters.scope = button.dataset.scopeFilter;
  syncFilterControls();
  applyFilters();
}

function onFormChange(event) {
  if (event.target.name === "decision") {
    setDecision(event.target.value);
    syncFieldReasonCode();
    saveCurrentDraft();
    if (event.target.value === "corrected") {
      const first = dom.editControls.querySelector("input:not(:disabled), select:not(:disabled), textarea:not(:disabled)");
      if (first) {
        first.focus();
      }
    }
    return;
  }
  syncFieldReasonCode();
  updatePreview();
  hideValidationSummary();
  saveCurrentDraft();
}

function onFormInput(event) {
  if (event.target === dom.reason) {
    dom.reasonLength.textContent = String(dom.reason.value.length);
  }
  syncFieldReasonCode();
  updatePreview();
  saveCurrentDraft();
}

function onFormSubmit(event) {
  event.preventDefault();
  const entry = selectedEntry();
  if (!entry || issueIsComplete(entry.issue)) {
    return;
  }
  const submission = buildSubmission();
  if (submission) {
    showConfirmation(submission);
  }
}

function onKeyboardShortcut(event) {
  if (event.key === "Escape" && dom.confirmDialog.open) {
    event.preventDefault();
    dom.confirmDialog.close();
    app.pendingSubmission = null;
    return;
  }
  if (dom.confirmDialog.open) {
    return;
  }
  const target = event.target;
  const typingTarget = target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    (target instanceof HTMLElement && target.isContentEditable);
  const confirmShortcut = (event.ctrlKey || event.metaKey) && event.key === "Enter";
  if (typingTarget && !confirmShortcut) {
    return;
  }
  if (event.altKey && event.key === "ArrowLeft") {
    event.preventDefault();
    navigateVisible(-1);
    return;
  }
  if (event.altKey && event.key === "ArrowRight") {
    event.preventDefault();
    navigateVisible(1);
    return;
  }
  if (event.altKey && event.key.toLowerCase() === "n") {
    event.preventDefault();
    goToNextPending();
    return;
  }
  const entry = selectedEntry();
  const editable = entry && !issueIsComplete(entry.issue);
  if (editable && event.altKey && event.key.toLowerCase() === "c") {
    event.preventDefault();
    dom.correctedInput.focus();
    setDecision("corrected");
    return;
  }
  if (
    editable &&
    entry.issue.accept_allowed &&
    event.altKey &&
    event.key.toLowerCase() === "a"
  ) {
    event.preventDefault();
    dom.acceptedInput.focus();
    setDecision("accepted");
    return;
  }
  if (editable && (event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    dom.decisionForm.requestSubmit();
  }
}

dom.scopeTabs.addEventListener("click", onScopeTab);
dom.statusFilter.addEventListener("change", () => {
  app.filters.status = dom.statusFilter.value;
  applyFilters();
});
dom.typeFilter.addEventListener("change", () => {
  app.filters.type = dom.typeFilter.value;
  applyFilters();
});
dom.reloadButton.addEventListener("click", () => loadState());
dom.retryButton.addEventListener("click", () => loadState());
dom.exportButton.addEventListener("click", exportResults);
dom.emptyNextButton.addEventListener("click", goToNextPending);
dom.nextPendingButton.addEventListener("click", goToNextPending);
dom.clearDraftButton.addEventListener("click", clearCurrentDraft);
dom.completedNextButton.addEventListener("click", goToNextPending);
dom.decisionForm.addEventListener("change", onFormChange);
dom.decisionForm.addEventListener("input", onFormInput);
dom.decisionForm.addEventListener("submit", onFormSubmit);
dom.modalClose.addEventListener("click", () => {
  app.pendingSubmission = null;
});
dom.confirmDialog.addEventListener("cancel", () => {
  app.pendingSubmission = null;
});
dom.confirmDialog.addEventListener("close", () => {
  app.pendingSubmission = null;
  dom.modalError.hidden = true;
  dom.modalError.textContent = "";
});
dom.modalSubmit.addEventListener("click", submitDecision);
document.addEventListener("keydown", onKeyboardShortcut);

loadState();
