"use strict";

const Common = window.ReviewCore;
const Ops = window.OperationReviewCore;
const ui = {};
for (const id of ["sidebar-toggle", "sidebar-close", "sidebar-backdrop", "item-sidebar", "top-progress-text", "progress-bar", "sidebar-progress", "save-state", "export-snapshot", "item-search", "search-mode", "search-status", "item-filters", "item-list", "item-empty", "item-position", "item-id", "stable-id", "previous-item", "next-item", "frame-warnings", "operation-kind", "rationale", "flags", "open-questions-card", "open-questions", "source-entries", "evidence-list", "evidence-count", "external-card", "external-evidence", "decision-title", "decision-status", "save-draft", "confirm-next", "resolution-options", "resolution-hint", "senses-list", "add-sense", "variants", "policy-details", "rules-count", "require-rules", "exclude-rules", "notes", "locked-card", "locked-message", "reopen", "request-error", "request-error-message", "request-error-details", "recover-local", "guidelines", "guideline-dialog"]) ui[id] = document.getElementById(id);
let bootstrap = null, current = null, draft = null, conflictDraft = null;
let visibleItemIds = [], itemFilter = "all", searchMode = "literal";
let dirty = false, busy = false, saveTimer = null;
const AUTOSAVE_DELAY_MS = 5000;
function randomId(prefix) {
  const bytes = new Uint8Array(8);
  window.crypto.getRandomValues(bytes);
  return prefix + Array.from(bytes).map(value => value.toString(16).padStart(2, "0")).join("");
}
const clientHeaders = {"X-Review-Client-Instance": randomId("operation-client-")};

class ApiError extends Error {
  constructor(message, status, technical) { super(message); this.status = status; this.technical = technical; }
}
async function parseError(response) {
  const payload = await response.json().catch(() => ({}));
  const friendly = {403: "页面会话已失效，请刷新后重试。", 404: "审核项不存在。", 409: "审核状态已在另一页面发生变化。", 422: "当前词典修复决定未通过校验。", 500: "审核服务发生内部错误。"}[response.status] || "请求失败。";
  return new ApiError(friendly, response.status, typeof payload.error === "string" ? payload.error : JSON.stringify(payload.error || `HTTP ${response.status}`));
}
async function getJson(path) {
  const response = await fetch(path, {cache: "no-store", headers: clientHeaders});
  if (!response.ok) throw await parseError(response);
  return response.json();
}
async function postJson(path, payload) {
  const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json", ...clientHeaders}, body: JSON.stringify({...payload, session_token: bootstrap.session_token})});
  if (!response.ok) throw await parseError(response);
  return response;
}
function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
}
function button(text, onClick, className = "quiet-button") {
  const result = node("button", text, className);
  result.type = "button";
  result.addEventListener("click", onClick);
  return result;
}
function isLocked() { return Boolean(current && Ops.isLocked(current.decision.status)); }
function dialogOpen() { return Boolean(document.querySelector("dialog[open]")); }
function setSaveState(state, text) {
  ui["save-state"].dataset.state = state;
  ui["save-state"].textContent = text || {idle: "尚未修改", dirty: "有未保存修改 · 5 秒后自动保存", saving: "正在保存…", saved: "已保存", error: "保存失败"}[state];
}
function showError(message, technical) {
  ui["request-error-message"].textContent = message;
  ui["request-error-details"].textContent = technical || "";
  ui["request-error"].classList.remove("hidden");
  ui["recover-local"].classList.toggle("hidden", !conflictDraft);
}
function clearError() {
  if (conflictDraft) return;
  ui["request-error"].classList.add("hidden");
  ui["request-error-details"].textContent = "";
}
function renderProgress() {
  const status = bootstrap.status;
  ui["top-progress-text"].textContent = `${status.confirmed_count} / ${status.item_count} 已确认 · ${status.deferred_count || 0} 暂缓`;
  ui["sidebar-progress"].textContent = `${status.confirmed_count} / ${status.item_count}`;
  ui["progress-bar"].style.width = `${status.item_count ? status.confirmed_count / status.item_count * 100 : 100}%`;
  ui["frame-warnings"].replaceChildren();
  for (const warning of bootstrap.warnings || []) ui["frame-warnings"].append(node("p", warning));
  ui["frame-warnings"].classList.toggle("hidden", !(bootstrap.warnings || []).length);
}
function renderItemList() {
  if (!bootstrap) return;
  visibleItemIds = Ops.visibleItemQueue(bootstrap.items, ui["item-search"].value, searchMode, itemFilter);
  ui["item-list"].replaceChildren();
  for (const id of visibleItemIds) {
    const summary = bootstrap.items.find(row => row.item_id === id);
    const control = button("", () => navigateTo(id), "case-item");
    const text = node("span");
    text.append(node("strong", summary.term), node("small", summary.query_preview || summary.item_id), node("small", `${Ops.KINDS[summary.operation_kind] || summary.operation_kind} · ${summary.status === "deferred" ? "暂缓" : summary.status === "confirmed" ? "已确认" : "待确认"}`, "item-kind"));
    control.append(text, node("span", "", `case-state-dot ${summary.status}`));
    control.classList.toggle("active", Boolean(current && current.item.item_id === id));
    control.disabled = busy;
    ui["item-list"].append(control);
  }
  ui["item-empty"].classList.toggle("hidden", visibleItemIds.length > 0);
  ui["search-status"].textContent = `显示 ${visibleItemIds.length} / ${bootstrap.items.length} 条`;
  updatePosition();
}
function updatePosition() {
  const index = current ? visibleItemIds.indexOf(current.item.item_id) : -1;
  ui["item-position"].textContent = index >= 0 ? `${index + 1} / ${visibleItemIds.length}` : "当前项不在筛选结果中";
  ui["previous-item"].disabled = busy || index <= 0;
  ui["next-item"].disabled = busy || index < 0 || index >= visibleItemIds.length - 1;
}
function renderEvidence() {
  const item = current.item;
  ui["source-entries"].replaceChildren();
  for (const entry of item.source_entries || []) {
    const card = node("article", undefined, "source-entry");
    card.append(node("strong", `${entry.term} · ${Array.isArray(entry.category) ? entry.category.join(" / ") : entry.category} · 原始行 ${entry.source_row_index}`), node("p", entry.definition || "（原释义为空）"));
    ui["source-entries"].append(card);
  }
  if (!(item.source_entries || []).length) ui["source-entries"].append(node("p", "本项为必要新增词条，没有原词典记录。", "section-hint"));
  ui["evidence-list"].replaceChildren();
  ui["evidence-count"].textContent = (item.evidence || []).length;
  for (const evidence of item.evidence || []) {
    const card = node("article", undefined, "evidence-card");
    card.append(node("div", evidence.source_item_id, "evidence-head"));
    const query = node("blockquote", undefined, "query-content");
    for (const segment of Ops.evidenceSegments(evidence.query_content, evidence.expected_spans)) {
      const part = node(segment.highlighted ? "mark" : "span", segment.text);
      if (segment.highlighted) part.title = `已冻结 span [${segment.span[0]}, ${segment.span[1]})`;
      query.append(part);
    }
    card.append(query);
    const reasonedSpans = (evidence.expected_spans || []).filter(row => row.reason);
    if (reasonedSpans.length) {
      const reasons = node("div", undefined, "evidence-notes");
      for (const row of reasonedSpans) reasons.append(node("p", `保留“${row.surface}” [${row.span[0]}, ${row.span[1]})：${row.reason}`));
      card.append(reasons);
    }
    if ((evidence.dropped_candidates || []).length) {
      const dropped = node("div", undefined, "dropped-candidates");
      dropped.append(node("strong", "已删除旧候选（反例）"));
      for (const row of evidence.dropped_candidates) dropped.append(node("span", `“${row.surface}” [${row.span[0]}, ${row.span[1]})`));
      card.append(dropped);
    }
    if (evidence.review_notes) card.append(node("p", evidence.review_notes, "evidence-notes"));
    ui["evidence-list"].append(card);
  }
  if (!(item.evidence || []).length) ui["evidence-list"].append(node("p", "本项暂无直接查询证据；请依据原始记录与明确来源审核，勿臆造用法。", "section-hint"));
  ui["external-evidence"].replaceChildren();
  for (const evidence of item.external_evidence || []) {
    const card = node("article", undefined, "source-entry");
    const url = Ops.safeEvidenceUrl(evidence.url);
    const title = node(url ? "a" : "strong", evidence.title || "外部证据");
    if (url) { title.href = url; title.target = "_blank"; title.rel = "noopener noreferrer"; }
    card.append(title, node("p", evidence.note || ""));
    ui["external-evidence"].append(card);
  }
  ui["external-card"].classList.toggle("hidden", !(item.external_evidence || []).length);
}
function markDirty() {
  if (!current || isLocked() || busy) return;
  dirty = true;
  setSaveState("dirty");
  renderErrors(false);
  window.clearTimeout(saveTimer);
  if (!dialogOpen()) saveTimer = window.setTimeout(() => save(false), AUTOSAVE_DELAY_MS);
}
function entryChanged() {
  draft.resolution = "revise";
  renderResolution();
  markDirty();
}
function inputField(label, value, onInput, {rows = 3, className = "", readOnly = false} = {}) {
  const wrapper = node("label", undefined, "field");
  const control = node("textarea");
  control.rows = rows;
  control.value = value || "";
  control.className = className;
  control.readOnly = readOnly;
  control.addEventListener("input", () => { if (!busy && !isLocked()) onInput(control.value); });
  wrapper.append(node("span", label), control);
  return wrapper;
}
function renderSenses() {
  ui["senses-list"].replaceChildren();
  if (!draft.entry) { ui["senses-list"].append(node("p", "当前无词条内容。请选择暂缓 / 不采用，或恢复提案后编辑。", "section-hint")); return; }
  draft.entry.senses.forEach((sense, index) => {
    const card = node("article", undefined, "sense-card");
    card.dataset.senseId = sense.sense_id;
    const head = node("div", undefined, "sense-head");
    head.append(node("code", `义项 ${index + 1} · ${sense.sense_id}`), button("移除义项", () => {
      if (busy || isLocked()) return;
      if (!window.confirm(`移除义项 ${index + 1}？其余义项的稳定 ID 不变。`)) return;
      draft.entry.senses.splice(index, 1); renderSenses(); entryChanged(); syncControls();
    }, "remove-button"));
    card.append(head, inputField(`义项 ${index + 1} 的释义`, sense.definition, value => { sense.definition = value; entryChanged(); }, {className: "sense-definition"}));
    const categories = node("fieldset", undefined, "category-options");
    categories.append(node("legend", "义项类别（可多选，不是句子标签）"));
    for (const category of Ops.CATEGORIES) {
      const label = node("label");
      const checkbox = node("input");
      checkbox.type = "checkbox";
      checkbox.value = category;
      checkbox.checked = (sense.categories || []).includes(category);
      checkbox.addEventListener("change", () => {
        if (busy || isLocked()) return;
        sense.categories = Ops.CATEGORIES.filter(value => value === category ? checkbox.checked : sense.categories.includes(value));
        entryChanged();
      });
      label.append(checkbox, node("span", category)); categories.append(label);
    }
    card.append(categories); ui["senses-list"].append(card);
  });
}
function renderRules() {
  let count = 0;
  for (const group of ["require_any", "exclude_any"]) {
    const container = ui[group === "require_any" ? "require-rules" : "exclude-rules"];
    container.replaceChildren();
    const rows = draft.entry ? draft.entry.match_policy[group] : [];
    count += rows.length;
    rows.forEach((rule, index) => {
      const card = node("div", undefined, "rule-editor");
      const head = node("div", undefined, "rule-head");
      head.append(node("code", rule.rule_id), button("移除规则", () => {
        if (busy || isLocked()) return;
        draft.entry.match_policy[group].splice(index, 1); renderRules(); entryChanged(); syncControls();
      }, "remove-button"));
      const targetLabel = node("label", undefined, "field");
      const select = node("select");
      select.setAttribute("aria-label", `规则 ${index + 1} target`);
      for (const target of Object.keys(Ops.TARGETS)) { const option = node("option", `${target} · ${Ops.TARGETS[target]}`); option.value = target; select.append(option); }
      select.value = rule.target;
      select.addEventListener("change", () => { if (!busy && !isLocked()) { rule.target = select.value; entryChanged(); } });
      targetLabel.append(node("span", "作用对象 target"), select);
      card.append(head, targetLabel, inputField("正则 pattern（后端校验）", rule.pattern, value => { rule.pattern = value; entryChanged(); }, {rows: 2}));
      container.append(card);
    });
    if (!rows.length) container.append(node("p", group === "require_any" ? "无额外准入限制。" : "无额外排除条件。", "section-hint"));
  }
  ui["rules-count"].textContent = count;
}
function renderResolution() {
  for (const control of ui["resolution-options"].querySelectorAll("button")) {
    const selected = control.dataset.resolution === draft.resolution;
    control.classList.toggle("active", selected);
    control.setAttribute("aria-pressed", String(selected));
  }
  ui["resolution-hint"].textContent = {
    approve: "原样采用提案；确认前仍会校验义项完整性与匹配规则。",
    revise: "将采用右侧编辑后的词条；保留稳定 ID，不代表逐处义项已消歧。",
    defer: "请说明缺少什么证据。确认后列为暂缓，不计入冻结完成。",
    reject: "仅不采用此次提案，不删除原词条；可能遗留的 span 覆盖冲突仍会阻断后续。",
  }[draft.resolution] || "选择决定不会自动确认；修改词条将自动切为“修改后采用”。";
}
function renderErrors(confirm) {
  const errors = Ops.validateDecision(current.item, draft, {confirm});
  for (const key of ["resolution", "senses", "variants", "policy", "notes", "entry"]) {
    const control = document.getElementById(`${key}-error`);
    control.textContent = errors[key] || "";
    control.classList.toggle("hidden", !errors[key]);
  }
  return errors;
}
function syncControls() {
  const disabled = busy || !current, locked = isLocked();
  document.querySelectorAll(".audit-editor input, .audit-editor textarea, .audit-editor select, .audit-editor button, .decision-actions button").forEach(control => { control.disabled = disabled || locked; });
  ui.reopen.disabled = disabled || !locked;
  ui["add-sense"].disabled = disabled || locked || !draft || !draft.entry;
  ui.variants.disabled = disabled || locked || !draft || !draft.entry;
  document.querySelectorAll("[data-add-rule]").forEach(control => { control.disabled = disabled || locked || !draft || !draft.entry; });
  ui["export-snapshot"].disabled = busy || !bootstrap;
  ui["recover-local"].disabled = disabled || locked || !conflictDraft || conflictDraft.item_id !== current.item.item_id;
  ui["item-list"].querySelectorAll("button").forEach(control => { control.disabled = busy; });
  updatePosition();
}
function renderCurrent() {
  const item = current.item;
  ui["item-id"].textContent = item.term;
  ui["stable-id"].textContent = `${item.item_id} · ${draft.entry ? draft.entry.lexicon_id : "无词条 ID"}`;
  ui["operation-kind"].textContent = Ops.KINDS[item.operation_kind] || item.operation_kind;
  ui.rationale.textContent = item.rationale || "";
  ui.flags.replaceChildren();
  for (const flag of item.flags || []) { const tag = node("span", Ops.FLAGS[flag] || flag, "source-tag"); tag.title = flag; ui.flags.append(tag); }
  ui["open-questions"].replaceChildren();
  for (const question of item.open_questions || []) ui["open-questions"].append(node("li", question));
  ui["open-questions-card"].classList.toggle("hidden", !(item.open_questions || []).length);
  ui["decision-title"].textContent = item.term;
  const status = current.decision.status || "draft";
  ui["decision-status"].dataset.state = status;
  ui["decision-status"].textContent = {confirmed: "已确认", deferred: "暂缓 · 未完成", draft: "草稿"}[status] || status;
  ui["locked-card"].classList.toggle("hidden", !isLocked());
  ui["locked-message"].textContent = status === "deferred" ? "本条已暂缓并锁定，尚未完成。" : "该决定已确认并锁定。";
  renderEvidence(); renderSenses(); renderRules(); renderResolution();
  ui.variants.value = draft.entry ? draft.entry.variants.join("\n") : "";
  ui.notes.value = draft.notes;
  renderErrors(false); renderItemList(); syncControls();
}
function setResolution(resolution) {
  if (!current || busy || isLocked()) return;
  if (resolution === "approve" && !Ops.entriesEqual(draft.entry, current.item.proposed_entry)) {
    if (!window.confirm("词条内容已修改。“批准提案”必须原样采用提案。是否放弃当前词条编辑并恢复原提案？审核备注会保留。")) return;
    draft.entry = Common.clone(current.item.proposed_entry);
    renderSenses(); renderRules(); ui.variants.value = draft.entry ? draft.entry.variants.join("\n") : "";
  }
  draft.resolution = resolution; renderResolution(); markDirty(); syncControls();
}
function applyMutation(result) {
  bootstrap.revision = result.revision;
  bootstrap.status = result.status;
  const index = bootstrap.items.findIndex(row => row.item_id === result.item_summary.item_id);
  if (index >= 0) bootstrap.items[index] = result.item_summary;
  current.revision = result.revision; current.decision = result.decision;
  draft = Ops.decisionFields(current.item, result.decision);
  dirty = false;
  renderProgress(); renderCurrent(); setSaveState("saved"); clearError();
}
async function resolveConflict(error) {
  if (dirty) conflictDraft = {item_id: current.item.item_id, decision: Common.clone(draft)};
  bootstrap = await getJson("/api/bootstrap");
  current = await getJson(`/api/items/${encodeURIComponent(current.item.item_id)}`);
  bootstrap.revision = current.revision;
  draft = Ops.decisionFields(current.item, current.decision);
  dirty = false;
  renderProgress(); renderCurrent();
  setSaveState("error", "版本冲突 · 已载入服务器状态");
  showError("服务器状态已更新，未自动覆盖。", `${error.technical || ""}\n${conflictDraft ? "本地草稿已保留。请核对服务器决定；如已锁定须先重新打开，才能恢复本地草稿。" : "请核对服务器最新决定后继续。"}`);
}
async function save(confirm) {
  window.clearTimeout(saveTimer);
  if (!current || busy || isLocked() || dialogOpen()) return false;
  if (Common.hasErrors(renderErrors(confirm))) { setSaveState("error", "请先修正审核字段"); return false; }
  busy = true; syncControls(); setSaveState("saving");
  try {
    const response = await postJson("/api/save", {expected_revision: bootstrap.revision, item_id: current.item.item_id, decision: draft, confirm});
    applyMutation(await response.json()); return true;
  } catch (error) {
    setSaveState("error");
    try { if (error.status === 409) await resolveConflict(error); else showError(error.message, error.technical); }
    catch (refreshError) { showError("版本冲突后刷新失败，本地草稿仍保留。", refreshError.message); }
    return false;
  } finally { busy = false; syncControls(); }
}
async function loadItem(id, {skipFlush = false} = {}) {
  if (!skipFlush && dirty && !(await save(false))) return false;
  window.clearTimeout(saveTimer);
  busy = true; syncControls();
  try {
    const itemResponse = await getJson(`/api/items/${encodeURIComponent(id)}`);
    current = itemResponse; bootstrap.revision = current.revision;
    draft = Ops.decisionFields(current.item, current.decision); dirty = false;
    if (current.item_summary) {
      const index = bootstrap.items.findIndex(row => row.item_id === id);
      if (index >= 0) bootstrap.items[index] = current.item_summary;
    }
    renderCurrent(); setSaveState("idle"); clearError(); return true;
  } catch (error) { showError(error.message, error.technical); return false; }
  finally { busy = false; syncControls(); }
}
async function navigateTo(id) {
  if (!id || busy || dialogOpen() || (current && current.item.item_id === id)) return;
  if (await loadItem(id)) { ui["item-sidebar"].classList.remove("open"); ui["sidebar-backdrop"].classList.add("hidden"); }
}
function navigateOffset(offset) {
  const index = current ? visibleItemIds.indexOf(current.item.item_id) : -1;
  if (index >= 0) navigateTo(visibleItemIds[index + offset]);
}
async function confirmAndContinue() {
  if (!current || busy || isLocked() || dialogOpen()) return;
  const queue = visibleItemIds.slice(), id = current.item.item_id;
  if (!(await save(true))) return;
  const next = Ops.nextUnfinishedItemId(queue, bootstrap.items, id);
  if (next) await loadItem(next, {skipFlush: true});
  else if ((bootstrap.status.deferred_count || 0) > 0) setSaveState("saved", "当前队列已处理 · 仍有暂缓项未完成");
}
async function reopenCurrent() {
  if (!current || busy || !isLocked()) return;
  const reason = window.prompt("请输入重新打开此修复决定的理由：", "");
  if (!reason || !reason.trim()) return;
  busy = true; syncControls();
  try {
    const response = await postJson("/api/reopen", {expected_revision: bootstrap.revision, item_id: current.item.item_id, reason: reason.trim()});
    applyMutation(await response.json());
  } catch (error) { if (error.status === 409) await resolveConflict(error); else showError(error.message, error.technical); }
  finally { busy = false; syncControls(); }
}
async function exportSnapshot() {
  if (!bootstrap || busy) return;
  if (dirty && !(await save(false))) return;
  busy = true; syncControls();
  try {
    const response = await postJson("/api/export", {expected_revision: bootstrap.revision});
    const url = URL.createObjectURL(await response.blob());
    const anchor = node("a"); anchor.href = url; anchor.download = `repair-operation-review-${bootstrap.revision.slice(0, 12)}.json`;
    anchor.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { if (error.status === 409 && current) await resolveConflict(error); else showError(error.message, error.technical); }
  finally { busy = false; syncControls(); }
}
function openHelp() { window.clearTimeout(saveTimer); ui["guideline-dialog"].showModal(); }
function bindEvents() {
  ui["item-search"].addEventListener("input", renderItemList);
  ui["search-mode"].addEventListener("click", () => { searchMode = Common.nextSearchMode(searchMode); ui["search-mode"].textContent = `${Common.SEARCH_MODE_LABELS[searchMode]} ↻`; renderItemList(); });
  ui["item-filters"].addEventListener("click", event => {
    const control = event.target.closest("button[data-filter]");
    if (!control) return;
    itemFilter = control.dataset.filter;
    ui["item-filters"].querySelectorAll("button").forEach(row => row.classList.toggle("active", row === control));
    renderItemList();
  });
  ui["resolution-options"].addEventListener("click", event => { const control = event.target.closest("button[data-resolution]"); if (control) setResolution(control.dataset.resolution); });
  ui["add-sense"].addEventListener("click", () => {
    if (!current || busy || isLocked() || !draft.entry) return;
    draft.entry.senses.push({sense_id: randomId("review-sense-"), definition: "", categories: []});
    renderSenses(); entryChanged(); syncControls();
  });
  ui.variants.addEventListener("input", () => {
    if (!current || busy || isLocked() || !draft.entry) return;
    draft.entry.variants = ui.variants.value.split(/\r?\n/u).map(row => row.trim()).filter(Boolean); entryChanged();
  });
  document.querySelectorAll("[data-add-rule]").forEach(control => control.addEventListener("click", () => {
    if (!current || busy || isLocked() || !draft.entry) return;
    draft.entry.match_policy[control.dataset.addRule].push({rule_id: randomId("review-rule-"), target: "context", pattern: ""});
    renderRules(); entryChanged(); syncControls();
  }));
  ui.notes.addEventListener("input", () => { if (!current || busy || isLocked()) return; draft.notes = ui.notes.value; markDirty(); });
  ui["save-draft"].addEventListener("click", () => save(false)); ui["confirm-next"].addEventListener("click", confirmAndContinue);
  ui.reopen.addEventListener("click", reopenCurrent);
  ui["recover-local"].addEventListener("click", () => {
    if (busy || isLocked() || !conflictDraft || conflictDraft.item_id !== current.item.item_id) return;
    if (!window.confirm("将本地冲突前草稿恢复到编辑区。此操作不会立即覆盖服务器；请核对后手动保存。")) return;
    draft = Common.clone(conflictDraft.decision); conflictDraft = null;
    renderCurrent(); clearError(); markDirty(); window.clearTimeout(saveTimer); setSaveState("dirty", "本地草稿已恢复 · 请核对后手动保存");
  });
  ui["previous-item"].addEventListener("click", () => navigateOffset(-1)); ui["next-item"].addEventListener("click", () => navigateOffset(1));
  ui["export-snapshot"].addEventListener("click", exportSnapshot);
  ui.guidelines.addEventListener("click", openHelp);
  ui["guideline-dialog"].addEventListener("close", () => { if (dirty) markDirty(); });
  ui["sidebar-toggle"].addEventListener("click", () => { ui["item-sidebar"].classList.add("open"); ui["sidebar-backdrop"].classList.remove("hidden"); });
  for (const id of ["sidebar-close", "sidebar-backdrop"]) ui[id].addEventListener("click", () => { ui["item-sidebar"].classList.remove("open"); ui["sidebar-backdrop"].classList.add("hidden"); });
  window.addEventListener("beforeunload", event => { if (dirty || busy || conflictDraft) { event.preventDefault(); event.returnValue = ""; } });
  window.addEventListener("keydown", event => {
    const action = Ops.reviewShortcut(event, {dialogOpen: dialogOpen(), busy, loaded: Boolean(current), locked: isLocked()});
    if (!action) return;
    event.preventDefault();
    if (Object.prototype.hasOwnProperty.call(Ops.RESOLUTIONS, action)) setResolution(action);
    else ({"previous-item": () => navigateOffset(-1), "next-item": () => navigateOffset(1), save: () => save(false), confirm: confirmAndContinue, help: openHelp})[action]();
  });
}
async function start() {
  bindEvents(); syncControls();
  try {
    bootstrap = await getJson("/api/bootstrap"); renderProgress(); renderItemList();
    const first = Ops.nextUnfinishedItemId(visibleItemIds, bootstrap.items) || visibleItemIds[0];
    if (first) await loadItem(first, {skipFlush: true});
    else { ui["item-id"].textContent = "没有审核项"; syncControls(); }
  } catch (error) { showError(error.message, error.technical); }
}
start();
