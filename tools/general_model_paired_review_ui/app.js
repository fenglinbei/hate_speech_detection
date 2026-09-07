"use strict";

const Common = window.ReviewCore;
const Pair = window.PairedReviewCore;
const ui = Object.fromEntries(Array.from(document.querySelectorAll("[id]")).map(element => [element.id, element]));
const AUTOSAVE_MS = 5000;
let bootstrap = null, current = null, draft = null, conflictDraft = null;
let dirty = false, busy = false, busyAction = "", timer = null, editVersion = 0;
let scope = "initial", filter = "all", searchMode = "literal", visibleIds = [];
let materialTab = "resources", trajectoryTask = "hate", promptRequest = 0;
let recoveryNeedsSave = false;
const placeholders = {
  ambiguity_stance: "在说谁？有引用、反驳、反讽或上下文缺失吗？",
  definition_fit: "义项与本句用法是否匹配？没有查询命中也可以写明。",
  category_relation: "类别与实际对象是否对应？哪些关系尚不明确？",
  demo_correspondence: "示例与查询哪里相关？是否有离题或标签疑问？",
  stage1_resource_notes: "补充目前观察到的资源问题。",
  gold_dispute: "有争议或信息不足时，简要写明依据。",
  stage2_candidate_explanation: "观察到什么变化？哪项资源可能解释这一变化？",
  alternative_explanation: "长度、位置、词义或示例内容能否同样解释？",
  falsifiable_followup: "改什么、保持什么、观察什么结果？例如同位置等长字段对照。",
};

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function button(text, action, className = "quiet-button") {
  const control = node("button", text, className);
  control.type = "button";
  control.addEventListener("click", action);
  return control;
}
function show(element, visible) { element.classList.toggle("hidden", !visible); }
function locked() { return Boolean(current && current.review.status === "confirmed"); }
function phaseTwo() { return Boolean(current && current.review.resources_locked_at); }
function isDialogOpen() { return Boolean(document.querySelector("dialog[open]")); }
function setSaveState(state, label) {
  ui["save-state"].dataset.state = state;
  ui["save-state"].textContent = label || {
    idle: "尚未修改", dirty: "有修改 · 5 秒后保存", saving: "正在保存…",
    saved: "已保存", error: "保存未完成",
  }[state];
}
function showError(message, details = "") {
  ui["request-error-message"].textContent = message;
  ui["request-error-details"].textContent = details;
  show(ui["request-error"], true);
  show(ui["recover-local"], Boolean(conflictDraft));
}
function clearError() {
  if (!conflictDraft) show(ui["request-error"], false);
}
async function responseError(response) {
  const body = await response.json().catch(() => ({}));
  const error = new Error(body.error || "请求失败，请稍后重试。");
  error.status = response.status;
  return error;
}
async function getJson(path) {
  const response = await fetch(path, {cache: "no-store"});
  if (!response.ok) throw await responseError(response);
  return response.json();
}
async function post(path, payload) {
  const response = await fetch(path, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({...payload, session_token: bootstrap.session_token}),
  });
  if (!response.ok) throw await responseError(response);
  return response;
}
function backupKey(key) {
  return ["paired-human-review", bootstrap.source_identity, bootstrap.reviewer_id, key].join(":");
}
function rememberPosition() {
  try { localStorage.setItem(backupKey("position"), JSON.stringify({item_id: current.item_id, scope})); } catch (_) { /* Server records remain authoritative. */ }
}
function backupDraft() {
  if (!current || !draft || !dirty) return;
  try {
    localStorage.setItem(backupKey(current.item_id), JSON.stringify({
      item_id: current.item_id, revision: bootstrap.revision,
      resources_locked_at: current.review.resources_locked_at,
      notes: draft, updated_at: new Date().toISOString(),
    }));
  } catch (_) { /* A network error still retains the in-memory draft. */ }
}
function removeBackup(key) {
  try { localStorage.removeItem(backupKey(key)); } catch (_) { /* No persisted local draft. */ }
}
function download(value, filename, mime = "application/json") {
  const url = URL.createObjectURL(value instanceof Blob ? value : new Blob([JSON.stringify(value, null, 2)], {type: mime}));
  const link = node("a");
  link.href = url; link.download = filename;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function downloadDraft() {
  if (!current) return;
  download(conflictDraft || {item_id: current.item_id, notes: draft}, "paired-review-draft-" + current.item_id + ".json");
}

function makeField(key, label, section, optional = false) {
  const wrapper = node("label", undefined, "field");
  const title = node("span", label);
  if (optional) title.append(node("small", "  可选"));
  const input = node("textarea");
  input.id = key; input.rows = key === "gold_dispute" ? 2 : 3;
  input.maxLength = section === "resources" ? 1000 : 2000;
  input.placeholder = placeholders[key] || "";
  input.autocomplete = "off";
  const error = node("small", "", "field-error hidden");
  error.dataset.error = key;
  wrapper.append(title, input, error);
  ui[key] = input;
  input.addEventListener("input", () => {
    if (!canEdit(section)) return;
    draft[section][key] = input.value;
    markDirty(); clearFieldError(key);
  });
  return wrapper;
}
function setupFields() {
  for (const [key, label] of Object.entries(Pair.RESOURCE_LABELS)) {
    const field = makeField(key, label, "resources", key === "stage1_resource_notes");
    if (key === "stage1_resource_notes") {
      const details = node("details", undefined, "optional-detail");
      details.append(node("summary", "补充资源观察（可选）"), field);
      ui["resource-fields"].append(details);
    } else ui["resource-fields"].append(field);
  }
  for (const key of ["gold_dispute", "stage2_candidate_explanation", "alternative_explanation", "falsifiable_followup"]) {
    ui["assessment-fields"].append(makeField(key, Pair.ASSESSMENT_LABELS[key], "assessment", key === "gold_dispute"));
  }
  for (const key of ["patching_defer_reason", "ai_comparison"]) {
    ui[key].addEventListener("input", () => {
      if (!canEdit("assessment")) return;
      draft.assessment[key] = ui[key].value; markDirty(); clearFieldError(key);
    });
  }
  for (const option of Pair.CONDITIONS) {
    const control = node("option", option.short + " · " + option.label);
    control.value = option.id; ui["prompt-condition"].append(control);
  }
}
function canEdit(section) {
  return Boolean(current && !locked() && (!busy || busyAction === "save") &&
    (section === "resources" ? !phaseTwo() : phaseTwo()));
}
function clearFieldError(key) {
  const element = document.querySelector('[data-error="' + key + '"]');
  if (element) show(element, false);
  if (ui[key]) ui[key].removeAttribute("aria-invalid");
}
function clearValidation() {
  document.querySelectorAll("[data-error]").forEach(element => show(element, false));
  document.querySelectorAll('[aria-invalid="true"]').forEach(element => element.removeAttribute("aria-invalid"));
}
function validate(action) {
  clearValidation();
  const errors = Pair.validate(draft, action);
  for (const [key, message] of Object.entries(errors)) {
    const element = document.querySelector('[data-error="' + key + '"]');
    if (element) { element.textContent = message; show(element, true); }
    if (ui[key]) ui[key].setAttribute("aria-invalid", "true");
  }
  if (Object.keys(errors).length) {
    const first = Object.keys(errors)[0];
    const target = ui[first] || document.querySelector('[data-error="' + first + '"]');
    target?.scrollIntoView({block: "center", behavior: "smooth"});
    target?.focus({preventScroll: true});
    return false;
  }
  return true;
}
function markDirty() {
  dirty = true; editVersion++;
  setSaveState("dirty", recoveryNeedsSave ? "已恢复草稿 · 请手动保存" : undefined);
  backupDraft(); scheduleSave(); syncControls();
}
function scheduleSave() {
  clearTimeout(timer);
  if (!dirty || busy || conflictDraft || recoveryNeedsSave || locked()) return;
  timer = setTimeout(() => { if (!busy && dirty) mutate("save"); }, AUTOSAVE_MS);
}
function syncControls() {
  const unavailable = !current || busy;
  ui["save-draft"].disabled = unavailable || locked() || Boolean(conflictDraft);
  ui["primary-action"].disabled = unavailable || locked() || Boolean(conflictDraft);
  ui["resource-editor"].disabled = !current || phaseTwo() || locked() || (busy && busyAction !== "save");
  ui["assessment-editor"].disabled = !current || !phaseTwo() || locked() || (busy && busyAction !== "save");
  ui["reopen"].disabled = unavailable;
  ui["recover-local"].disabled = busy;
  ui["export-json"].disabled = !bootstrap || busy;
  ui["export-csv"].disabled = !bootstrap || busy;
  ui["reveal-ai"].disabled = unavailable || !phaseTwo() || locked() || Boolean(conflictDraft);
  ui["open-prompt"].disabled = unavailable || !phaseTwo();
  ui["tab-trajectory"].disabled = !phaseTwo() || busy;
  ui["tab-ai"].disabled = !current || !current.review.ai_revealed_at || busy;
  ui["prior-ai-exposure"].disabled = !canEdit("resources");
  document.querySelectorAll("#item-list button").forEach(control => { control.disabled = busy; });
  updatePosition();
}
function fillForm() {
  if (!draft) return;
  for (const key of Object.keys(Pair.RESOURCE_LABELS)) ui[key].value = draft.resources[key];
  ui["prior-ai-exposure"].value = draft.resources.prior_ai_exposure;
  for (const key of Object.keys(Pair.ASSESSMENT_LABELS)) ui[key].value = draft.assessment[key];
  renderChoices();
}
function renderChoices() {
  if (!draft) return;
  for (const control of document.querySelectorAll("[data-gold]")) {
    const selected = control.dataset.gold === draft.assessment.gold_verdict;
    control.classList.toggle("active", selected); control.setAttribute("aria-pressed", String(selected));
  }
  for (const control of document.querySelectorAll("[data-disposition]")) {
    const selected = control.dataset.disposition === draft.assessment.disposition;
    control.classList.toggle("active", selected); control.setAttribute("aria-pressed", String(selected));
  }
  const needsReason = ["verify_first", "defer"].includes(draft.assessment.disposition);
  show(ui["defer-reason-field"], needsReason || Boolean(draft.assessment.patching_defer_reason));
}
function renderProgress() {
  if (!bootstrap) return;
  const status = bootstrap.status;
  const initial = scope === "initial";
  const complete = initial ? status.initial_confirmed_count : status.confirmed_count;
  const total = initial ? status.initial_count : status.item_count;
  ui["top-progress-text"].textContent = (initial ? "首批 " : "全部 ") + complete + " / " + total + " 已确认";
  ui["sidebar-progress"].textContent = (initial ? "首批 " : "全部 ") + complete + " / " + total;
  ui["progress-bar"].style.width = (total ? complete / total * 100 : 0) + "%";
  ui["reviewer-name"].textContent = bootstrap.reviewer_id;
  show(ui["batch-complete"], complete === total && total > 0);
  ui["batch-complete"].textContent = initial ? "首批 12 条已复核完成。可导出人工记录，准备下一阶段输入对照。" : "32 条 discovery 均已复核。可导出完整人工记录。";
}
function renderQueue() {
  if (!bootstrap) return;
  visibleIds = Pair.visibleQueue(bootstrap.items, {scope, filter, query: ui["item-search"].value, mode: searchMode});
  ui["item-list"].replaceChildren();
  for (const key of visibleIds) {
    const summary = bootstrap.items.find(row => row.item_id === key);
    const control = button("", () => navigateTo(key), "case-item");
    control.setAttribute("aria-label", "案例 " + key);
    const body = node("span");
    const heading = node("span", undefined, "queue-top");
    heading.append(node("strong", "#" + key), node("span", Pair.TASK_LABELS[summary.focus_task], "queue-task"));
    body.append(heading, node("span", summary.query_preview, "queue-preview"));
    const status = summary.status === "confirmed" ? "已确认 · " + Pair.DISPOSITIONS[summary.disposition] :
      summary.stage === "assessment" ? "待完成轨迹复核" : summary.status === "draft" ? "初读草稿" : "待初读";
    body.append(node("small", status, "queue-status"));
    control.append(body, node("span", "", "case-state-dot " + summary.status));
    control.classList.toggle("active", Boolean(current && current.item_id === key));
    if (current && current.item_id === key) control.setAttribute("aria-current", "true");
    ui["item-list"].append(control);
  }
  const total = scope === "initial" ? bootstrap.status.initial_count : bootstrap.status.item_count;
  ui["search-status"].textContent = "显示 " + visibleIds.length + " / " + total + " 条 · 固定顺序";
  show(ui["item-empty"], visibleIds.length === 0);
  renderProgress(); syncControls();
}
function updatePosition() {
  if (!current) {
    ui["previous-item"].disabled = true; ui["next-item"].disabled = true; return;
  }
  const index = visibleIds.indexOf(current.item_id);
  ui["item-position"].textContent = index < 0 ? "当前案例不在筛选结果中" :
    (scope === "initial" ? "首批复核" : "全部 discovery") + "  /  " + String(index + 1).padStart(2, "0") + " — " + visibleIds.length;
  ui["previous-item"].disabled = busy || index <= 0;
  ui["next-item"].disabled = busy || index < 0 || index >= visibleIds.length - 1;
}
function renderResources() {
  ui["item-id"].textContent = "案例 #" + current.item_id;
  ui["focus-task"].textContent = Pair.TASK_LABELS[current.focus_task];
  ui["query-content"].textContent = current.query;
  const lq = new Set(current.resource_ids.lq_ids);
  const both = new Set(current.resource_ids.intersection_ids);
  const entries = current.lexicon_entries;
  ui["query-resource-count"].textContent = "查询命中 " + lq.size + " 项";
  ui["lexicon-count"].textContent = entries.length + " 个词条";
  ui["lexicon-list"].replaceChildren();
  const termById = new Map(entries.map(entry => [entry.lexicon_id, entry.term]));
  for (const entry of entries) {
    const card = node("article", undefined, "lexicon-entry");
    const heading = node("div", undefined, "lexicon-heading");
    const source = both.has(entry.lexicon_id) ? "查询与示例共有" : lq.has(entry.lexicon_id) ? "查询命中 · Lq" : "示例带入 · Ld-only";
    heading.append(node("strong", entry.term), node("span", source, "origin-chip" + (lq.has(entry.lexicon_id) ? " query-origin" : "")));
    card.append(heading);
    for (const sense of entry.senses || [{definition: entry.definition, categories: [entry.category]}]) {
      const row = node("div", undefined, "sense-row");
      row.append(node("p", sense.definition));
      const categories = node("div", undefined, "sense-categories");
      categories.append(node("small", "显式类别"));
      for (const category of sense.categories || []) categories.append(node("span", Pair.labels(category) + " · " + category, "category-chip"));
      row.append(categories); card.append(row);
    }
    ui["lexicon-list"].append(card);
  }
  if (!entries.length) ui["lexicon-list"].append(node("p", "本例没有注入词典。可记录“无相关词条”，继续核对固定示例。", "empty-material"));
  ui["demo-count"].textContent = current.demonstrations.length + " 条示例";
  ui["demo-list"].replaceChildren();
  current.demonstrations.forEach((demo, index) => {
    const card = node("article", undefined, "demo-card");
    const heading = node("div", undefined, "demo-heading");
    const name = node("span");
    name.append(node("span", String(index + 1).padStart(2, "0"), "demo-number"), node("span", "  /  示例 #" + demo.id));
    heading.append(name, node("span", Pair.labels(demo.answer), "demo-answer"));
    card.append(heading, node("p", demo.content, "demo-content"));
    card.append(node("small", demo.lexicon_ids.length ? "命中：" + demo.lexicon_ids.map(id => termById.get(id) || id).join("、") : "无词典命中", "demo-terms"));
    ui["demo-list"].append(card);
  });
}
function renderTrajectory() {
  if (!current.trajectory) return;
  const trajectory = current.trajectory;
  document.querySelectorAll("[data-task]").forEach(control => {
    const selected = control.dataset.task === trajectoryTask;
    control.classList.toggle("active", selected); control.setAttribute("aria-pressed", String(selected));
  });
  ui["gold-summary"].replaceChildren(
    node("span", "原 Gold"), node("strong", Pair.labels(trajectory.gold[trajectoryTask])),
    node("span", trajectoryTask === current.focus_task ? "· 本例重点任务" : "· 补充查看"),
  );
  ui["trajectory-rows"].replaceChildren();
  const warnings = [], margins = {};
  for (const condition of Pair.CONDITIONS) {
    const value = trajectory.conditions[condition.id][trajectoryTask];
    margins[condition.short] = Pair.goldMargin(value);
    const row = node("tr");
    const name = node("td"), labels = node("span", undefined, "condition-label");
    labels.append(node("strong", condition.short), node("small", condition.label));
    name.append(labels);
    const prediction = node("td", Pair.labels(value.prediction.labels));
    const correct = node("td");
    correct.append(node("span", value.correct ? "✓ 正确" : "× 错误", "correct-pill" + (value.correct ? "" : " incorrect")));
    if (value.score_mode_sensitive) {
      prediction.append(node("small", "计分口径敏感", "sensitive-mark"));
      warnings.push(condition.short + " 的预测随计分口径改变");
    }
    if (value.prediction.within_two_epsilon || value.prediction.tied_top_count > 1) warnings.push(condition.short + " 存在并列或近并列");
    row.append(name, prediction, correct, node("td", Pair.number(margins[condition.short], true)),
      node("td", Pair.number(value.prediction.top_score_gap)), node("td", String(value.context.prompt_tokens)));
    ui["trajectory-rows"].append(row);
  }
  ui["trajectory-warnings"].replaceChildren();
  if (warnings.length) ui["trajectory-warnings"].append(node("p", warnings.join("；") + "。建议在选作干预材料前先核验。"));
  ui["paired-effects"].replaceChildren();
  const effects = [
    ["有示例时删除类别", margins.SD - margins.SGD, "SD − SGD"],
    ["去类别词典的增量", margins.SD - margins.D, "SD − D"],
    ["无示例时删除类别", margins.S - margins.SG, "S − SG"],
    ["逐查询分数交互", margins.SD - margins.S - margins.D + margins["0"], "SD − S − D + 0"],
  ];
  for (const [label, value, formula] of effects) {
    const tile = node("div", undefined, "effect-tile");
    tile.append(node("small", label), node("strong", Pair.number(value, true)), node("span", "Gold margin · " + formula));
    ui["paired-effects"].append(tile);
  }
  ui["trajectory-detail"].replaceChildren(
    node("p", "主桶：" + trajectory.primary_bucket + "；全部行为标签：" + trajectory.candidate_labels.join("、")),
    node("p", "四位轨迹（0 / S / D / SD）：" + trajectory.core_mask[trajectoryTask]),
    node("pre", JSON.stringify(trajectory.gold.extraction || trajectory.gold, null, 2)),
  );
}
function renderAi() {
  ui["ai-content"].replaceChildren();
  if (!current.ai_review) return;
  const fields = {
    ambiguity_stance: "AI 的语义与立场观察", definition_fit: "AI 的义项适配观察",
    observed_behavior: "观察到的模型行为", candidate_explanation: "候选解释",
    alternative_explanation: "替代解释", gold_review: "Gold 与未知项",
    falsifiable_followup: "建议的可证伪对照", recommended_role: "建议用途",
  };
  for (const [key, label] of Object.entries(fields)) {
    const value = current.ai_review[key];
    if (!value) continue;
    const section = node("section", undefined, "ai-section");
    section.append(node("h4", label));
    let list = null;
    if (key === "falsifiable_followup") {
      try { const parsed = JSON.parse(value); if (Array.isArray(parsed)) list = parsed; } catch (_) { /* Display the literal frozen note. */ }
    }
    if (list) { const ol = node("ol"); list.forEach(text => ol.append(node("li", String(text)))); section.append(ol); }
    else section.append(node("p", value));
    ui["ai-content"].append(section);
  }
}
function setTab(tab) {
  if (!current || (tab === "trajectory" && !phaseTwo()) || (tab === "ai" && !current.ai_review)) return;
  materialTab = tab;
  for (const key of ["resources", "trajectory", "ai"]) {
    const selected = key === tab;
    ui["tab-" + key].classList.toggle("active", selected);
    ui["tab-" + key].setAttribute("aria-selected", String(selected));
    ui["tab-" + key].tabIndex = selected ? 0 : -1;
    show(ui["panel-" + key], selected);
  }
}
function renderPhase() {
  if (!current) return;
  const revealed = phaseTwo(), confirmed = locked();
  show(ui["resource-editor"], !revealed);
  show(ui["resource-snapshot"], revealed);
  show(ui["assessment-editor"], revealed);
  show(ui["locked-card"], confirmed);
  ui["editor-phase-label"].textContent = confirmed ? "本条已完成" : revealed ? "第二步" : "第一步";
  ui["editor-title"].textContent = confirmed ? "查看我的复核记录" : revealed ? "核对解释与下一步" : "记录资源初读";
  ui["decision-status"].textContent = confirmed ? "已确认" : current.review.status === "unreviewed" ? "未开始" : "草稿";
  ui["decision-status"].dataset.state = confirmed ? "confirmed" : "draft";
  ui["primary-action"].textContent = revealed ? "确认并继续 →" : "保存初读 · 查看轨迹";
  ui["tab-trajectory"].querySelector(".tab-lock").textContent = revealed ? "" : "待初读";
  ui["tab-ai"].querySelector(".tab-lock").textContent = current.ai_review ? "" : "稍后对照";
  for (const [name, active, done] of [
    ["resources", !revealed, revealed], ["assessment", revealed && !confirmed, confirmed], ["confirmed", confirmed, false],
  ]) {
    ui["step-" + name].classList.toggle("active", active); ui["step-" + name].classList.toggle("done", done);
  }
  if (revealed) {
    ui["resource-snapshot-time"].textContent = "已于 " + new Date(current.review.resources_locked_at).toLocaleString("zh-CN") + " 保存，随后开放预测轨迹。";
    const list = node("dl");
    for (const [key, label] of Object.entries(Pair.RESOURCE_LABELS)) if (current.review.resources[key]) {
      list.append(node("dt", label), node("dd", current.review.resources[key]));
    }
    ui["resource-snapshot-notes"].replaceChildren(list);
  }
  show(ui["ai-reveal-card"], revealed && !current.review.ai_revealed_at && current.has_ai_review && !confirmed);
  show(ui["ai-comparison-field"], Boolean(current.review.ai_revealed_at));
  ui["ai-reveal-hint"].textContent = "先完成 Gold 判断、候选解释、替代解释与用途初判。";
  if (!current.has_ai_review) ui["tab-ai"].querySelector(".tab-lock").textContent = "本例暂无";
  renderChoices(); syncControls();
}
function renderCurrent() {
  renderResources(); renderTrajectory(); renderAi(); fillForm(); renderPhase(); setTab(materialTab); renderQueue();
}
function applyResult(result, preserveDraft = false) {
  bootstrap = result.bootstrap;
  current = result.current;
  if (!preserveDraft) { draft = Pair.notesFromReview(current.review); fillForm(); }
  renderTrajectory(); renderAi(); renderPhase(); renderQueue();
}
function closeSidebar() { ui["item-sidebar"].classList.remove("open"); show(ui["sidebar-backdrop"], false); }
function showSidebar() { ui["item-sidebar"].classList.add("open"); show(ui["sidebar-backdrop"], true); }

async function loadCase(key, {recover = true} = {}) {
  const data = await getJson("/api/items/" + encodeURIComponent(key));
  const token = bootstrap.session_token;
  if (data.bootstrap) bootstrap = {...data.bootstrap, session_token: token};
  else bootstrap.revision = data.revision;
  current = data; draft = Pair.notesFromReview(current.review);
  dirty = false; editVersion = 0; recoveryNeedsSave = false;
  materialTab = phaseTwo() ? "trajectory" : "resources";
  trajectoryTask = current.focus_task;
  clearValidation(); show(ui["resume-notice"], false);
  if (recover) {
    let backup = null;
    try { backup = JSON.parse(localStorage.getItem(backupKey(key)) || "null"); } catch (_) { /* Ignore corrupt browser-only backup. */ }
    if (backup && backup.notes && JSON.stringify(backup.notes) !== JSON.stringify(draft)) {
      if (!locked() && backup.revision === bootstrap.revision && backup.resources_locked_at === current.review.resources_locked_at) {
        draft = backup.notes; dirty = true; recoveryNeedsSave = true;
        ui["resume-notice"].textContent = "已恢复浏览器中的未保存草稿。检查后点击“保存草稿”。";
        show(ui["resume-notice"], true);
      } else {
        conflictDraft = backup;
        showError("找到较早的本地草稿，服务器记录已发生变化。", "当前展示服务器记录；可以下载草稿，或在核对后恢复。");
      }
    }
  }
  renderCurrent(); rememberPosition();
  setSaveState(dirty ? "dirty" : "saved", dirty ? "已恢复草稿 · 请手动保存" : current.review.status === "unreviewed" ? "等待初读" : "已载入保存记录");
}
async function navigateTo(key) {
  if (busy || !key || current?.item_id === key) return;
  if (conflictDraft) { showError("先处理当前草稿，再切换案例。", "可以下载草稿，或载入最新状态并恢复。"); return; }
  if (dirty && (!await mutate("save") || dirty)) return;
  clearTimeout(timer); busy = true; busyAction = "load"; syncControls();
  try {
    await loadCase(key); closeSidebar();
    ui["source-pane"].scrollTop = 0; ui["decision-pane"].scrollTop = 0;
    if (innerWidth <= 760) window.scrollTo({top: 0, behavior: "instant"});
  } catch (error) { showError("案例读取失败，当前记录仍保留。", error.message); }
  finally { busy = false; busyAction = ""; syncControls(); }
}
async function mutate(action, extra = {}) {
  if (!current || busy || (conflictDraft && action !== "reopen")) return false;
  if (action !== "reopen" && !validate(action)) return false;
  clearTimeout(timer); clearError();
  const submitted = Common.clone(draft), version = editVersion, key = current.item_id;
  const previousQueue = [...visibleIds];
  busy = true; busyAction = action; syncControls(); setSaveState("saving");
  let succeeded = false, nextId = null;
  try {
    const path = {save: "save", reveal: "reveal", reveal_ai: "reveal-ai", confirm: "confirm", reopen: "reopen"}[action];
    const payload = {expected_revision: bootstrap.revision, item_id: key};
    if (action === "reopen") payload.reason = extra.reason;
    else payload.notes = submitted;
    const response = await post("/api/" + path, payload);
    const result = await response.json();
    const newerEdits = action === "save" && version !== editVersion;
    applyResult(result, newerEdits);
    dirty = newerEdits; recoveryNeedsSave = false;
    if (newerEdits) backupDraft();
    else if (!conflictDraft) removeBackup(key);
    show(ui["resume-notice"], false);
    if (action === "reveal") {
      setTab("trajectory"); ui["decision-pane"].scrollTop = 0;
      if (innerWidth <= 760) ui["tab-trajectory"].scrollIntoView({block: "start", behavior: "smooth"});
      else ui["source-pane"].scrollTop = ui["source-pane"].scrollTop + ui["tab-trajectory"].getBoundingClientRect().top - ui["source-pane"].getBoundingClientRect().top - 14;
    }
    if (action === "reveal_ai") setTab("ai");
    if (action === "confirm") nextId = Pair.nextUnfinished(previousQueue, bootstrap.items, key);
    if (action === "reopen") ui["reopen-dialog"].close();
    setSaveState(newerEdits ? "dirty" : "saved", action === "confirm" ? "已确认保存" : undefined);
    succeeded = true;
  } catch (error) {
    dirty = action !== "reopen" || dirty;
    if (error.status === 409) {
      conflictDraft = {item_id: key, notes: Common.clone(draft), revision: bootstrap.revision, resources_locked_at: current.review.resources_locked_at};
      showError("另一页面更新了记录，当前草稿已保留。", error.message);
    } else showError(error.status === 422 ? "还有内容需要补充" : "保存未完成，当前草稿已保留。", error.message);
    backupDraft(); setSaveState("error");
  } finally {
    busy = false; busyAction = ""; syncControls();
    if (succeeded && dirty) scheduleSave();
  }
  if (nextId) await navigateTo(nextId);
  return succeeded;
}
async function recoverLocal() {
  if (busy || !conflictDraft) return;
  const local = Common.clone(conflictDraft);
  busy = true; busyAction = "recover"; syncControls();
  try {
    await loadCase(local.item_id, {recover: false});
    if (locked()) {
      conflictDraft = local;
      showError("服务器上的记录已确认。", "草稿仍保留；需要合并时先重新打开记录，再点击恢复。");
    } else if (JSON.stringify(local.notes.resources) !== JSON.stringify(current.review.resources) && phaseTwo()) {
      conflictDraft = null;
      try { localStorage.setItem(backupKey("preserved-" + local.item_id + "-" + Date.now()), JSON.stringify(local)); } catch (_) { /* The download also preserves this record. */ }
      download(local, "paired-review-preserved-draft-" + local.item_id + ".json");
      removeBackup(local.item_id);
      showError("另一页面已保存不同的资源初读。", "原本地草稿已下载；当前展示最新记录。初读保持原样，补充意见请写入后续复核。");
    } else {
      conflictDraft = null; draft = local.notes;
      if (phaseTwo()) draft.resources = Common.clone(current.review.resources);
      dirty = true; recoveryNeedsSave = true; editVersion++;
      clearError(); fillForm(); backupDraft();
      ui["resume-notice"].textContent = "已在最新版本上恢复你的草稿。检查内容后手动保存。";
      show(ui["resume-notice"], true); setSaveState("dirty", "已恢复草稿 · 请手动保存");
    }
  } catch (error) { conflictDraft = local; showError("无法载入最新状态，草稿仍保留。", error.message); }
  finally { busy = false; busyAction = ""; syncControls(); }
}
async function exportRecords(format) {
  if (busy || !bootstrap) return;
  if (conflictDraft) { showError("导出前先处理当前草稿。", "未保存内容可以单独下载。"); return; }
  if (dirty && (!await mutate("save") || dirty)) return;
  busy = true; busyAction = "export"; syncControls();
  try {
    const response = await post("/api/export", {expected_revision: bootstrap.revision, format});
    download(await response.blob(), "paired-human-review." + format);
    document.querySelector(".top-action-menu").open = false;
  } catch (error) { showError("导出失败，已保存记录不受影响。", error.message); }
  finally { busy = false; busyAction = ""; syncControls(); }
}
async function loadPrompt() {
  const request = ++promptRequest;
  ui["prompt-text"].textContent = "正在读取…";
  try {
    const data = await getJson("/api/prompt/" + encodeURIComponent(current.item_id) + "?task=" + ui["prompt-task"].value + "&condition=" + ui["prompt-condition"].value);
    if (request !== promptRequest) return;
    ui["prompt-text"].textContent = data.prompt_text;
    ui["prompt-meta"].textContent = "案例 #" + data.query_id + " · " + data.prompt_tokens + " tokens · 冻结提示原文";
  } catch (error) { if (request === promptRequest) ui["prompt-text"].textContent = "读取失败：" + error.message; }
}
function primaryAction() { return mutate(phaseTwo() ? "confirm" : "reveal"); }
function chooseDisposition(value) {
  if (!canEdit("assessment")) return;
  draft.assessment.disposition = value; renderChoices(); clearFieldError("disposition"); markDirty();
}
function bindEvents() {
  ui["review-form"].addEventListener("submit", event => event.preventDefault());
  ui["save-draft"].addEventListener("click", () => mutate("save"));
  ui["primary-action"].addEventListener("click", primaryAction);
  ui["reveal-ai"].addEventListener("click", () => mutate("reveal_ai"));
  ui["recover-local"].addEventListener("click", recoverLocal);
  ui["download-draft"].addEventListener("click", downloadDraft);
  ui["export-csv"].addEventListener("click", () => exportRecords("csv"));
  ui["export-json"].addEventListener("click", () => exportRecords("json"));
  ui["item-search"].addEventListener("input", renderQueue);
  ui["search-mode"].addEventListener("click", () => {
    searchMode = Common.nextSearchMode(searchMode);
    ui["search-mode"].textContent = Common.SEARCH_MODE_LABELS[searchMode] + " ↻"; renderQueue();
  });
  document.querySelectorAll("[data-scope]").forEach(control => control.addEventListener("click", () => {
    scope = control.dataset.scope;
    document.querySelectorAll("[data-scope]").forEach(item => { item.classList.toggle("active", item.dataset.scope === scope); item.setAttribute("aria-pressed", String(item.dataset.scope === scope)); });
    renderQueue(); if (current) rememberPosition();
  }));
  document.querySelectorAll("[data-filter]").forEach(control => control.addEventListener("click", () => {
    filter = control.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach(item => item.classList.toggle("active", item.dataset.filter === filter));
    renderQueue();
  }));
  ui["previous-item"].addEventListener("click", () => navigateTo(visibleIds[visibleIds.indexOf(current?.item_id) - 1]));
  ui["next-item"].addEventListener("click", () => navigateTo(visibleIds[visibleIds.indexOf(current?.item_id) + 1]));
  ui["prior-ai-exposure"].addEventListener("change", () => {
    if (!canEdit("resources")) return;
    draft.resources.prior_ai_exposure = ui["prior-ai-exposure"].value; markDirty();
  });
  document.querySelectorAll("[data-gold]").forEach(control => control.addEventListener("click", () => {
    if (!canEdit("assessment")) return;
    draft.assessment.gold_verdict = control.dataset.gold; renderChoices(); clearFieldError("gold_verdict"); markDirty();
  }));
  document.querySelectorAll("[data-disposition]").forEach(control => control.addEventListener("click", () => chooseDisposition(control.dataset.disposition)));
  document.querySelectorAll("[data-task]").forEach(control => control.addEventListener("click", () => { trajectoryTask = control.dataset.task; renderTrajectory(); }));
  for (const tab of ["resources", "trajectory", "ai"]) ui["tab-" + tab].addEventListener("click", () => setTab(tab));
  document.querySelector(".material-tabs").addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    const tabs = Array.from(document.querySelectorAll(".material-tabs button")).filter(control => !control.disabled);
    const index = tabs.indexOf(document.activeElement);
    if (index < 0) return;
    const next = tabs[(index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    event.preventDefault(); next.click(); next.focus();
  });
  ui["sidebar-toggle"].addEventListener("click", showSidebar);
  ui["sidebar-close"].addEventListener("click", closeSidebar);
  ui["sidebar-backdrop"].addEventListener("click", closeSidebar);
  ui["guidelines"].addEventListener("click", () => ui["guideline-dialog"].showModal());
  ui["open-prompt"].addEventListener("click", () => {
    ui["prompt-task"].value = trajectoryTask; ui["prompt-condition"].value = "CLDnewNoCat";
    ui["prompt-dialog"].showModal(); loadPrompt();
  });
  ui["close-prompt"].addEventListener("click", () => ui["prompt-dialog"].close());
  ui["prompt-task"].addEventListener("change", loadPrompt);
  ui["prompt-condition"].addEventListener("change", loadPrompt);
  ui["reopen"].addEventListener("click", () => { ui["reopen-reason"].value = ""; ui["reopen-dialog"].showModal(); });
  ui["cancel-reopen"].addEventListener("click", () => ui["reopen-dialog"].close());
  ui["reopen-form"].addEventListener("submit", event => {
    event.preventDefault();
    const reason = ui["reopen-reason"].value.trim();
    if (reason) mutate("reopen", {reason});
  });
  ui["mobile-materials"].addEventListener("click", () => window.scrollTo({top: 0, behavior: "smooth"}));
  ui["mobile-editor"].addEventListener("click", () => ui["decision-pane"].scrollIntoView({block: "start", behavior: "smooth"}));
  window.addEventListener("beforeunload", event => {
    if (dirty || busy || conflictDraft) { backupDraft(); event.preventDefault(); event.returnValue = ""; }
  });
  window.addEventListener("keydown", event => {
    if (event.isComposing || event.keyCode === 229 || event.repeat || isDialogOpen()) return;
    if (event.key === "Escape") { closeSidebar(); return; }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault(); if (!busy && !locked()) mutate("save"); return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault(); if (!busy && !locked()) primaryAction(); return;
    }
    if (Common.isTextEntry(event.target) || busy || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    if (["[", "]"].includes(event.key)) {
      event.preventDefault();
      const index = visibleIds.indexOf(current?.item_id);
      navigateTo(visibleIds[index + (event.key === "[" ? -1 : 1)]);
    } else if (["1", "2", "3"].includes(event.key)) chooseDisposition(["input_control", "verify_first", "defer"][Number(event.key) - 1]);
  });
}
async function start() {
  setupFields(); bindEvents(); syncControls();
  busy = true;
  try {
    bootstrap = await getJson("/api/bootstrap");
    let position = null;
    try { position = JSON.parse(localStorage.getItem(backupKey("position")) || "null"); } catch (_) { /* Start with the fixed initial batch. */ }
    if (position?.scope === "all") scope = "all";
    document.querySelectorAll("[data-scope]").forEach(control => { control.classList.toggle("active", control.dataset.scope === scope); control.setAttribute("aria-pressed", String(control.dataset.scope === scope)); });
    renderQueue();
    const previousId = position?.item_id;
    const key = bootstrap.items.some(row => row.item_id === previousId && (scope === "all" || row.initial_batch)) ? previousId :
      Pair.nextUnfinished(visibleIds, bootstrap.items, null) || visibleIds[0];
    if (key) await loadCase(key);
  } catch (error) { setSaveState("error", "连接失败"); showError("工作台读取失败。", error.message); }
  finally { busy = false; busyAction = ""; syncControls(); }
}
start();
