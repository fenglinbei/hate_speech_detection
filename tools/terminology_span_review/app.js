"use strict";

const bundle = window.TERMINOLOGY_SPAN_REVIEW;
if (!bundle || !bundle.manifest || !Array.isArray(bundle.cases)) {
  throw new Error("复审数据缺失或损坏。");
}

const packageId = bundle.manifest.package_id;
const storageKey = `terminology-span-review:${packageId}`;
const issueValues = new Set([
  "missed_span", "too_wide", "too_narrow", "ordinary_phrase",
  "sentence_fragment", "wrong_occurrence", "other",
]);
const state = { index: 0, filter: "all", reviewerId: "reviewer-1", annotations: {} };
const $ = (id) => document.getElementById(id);
const dom = {
  reviewer: $("reviewer-id"), importButton: $("import-button"), importFile: $("import-file"),
  exportButton: $("export-button"), complete: $("complete-count"), total: $("total-count"),
  fill: $("progress-fill"), message: $("progress-message"), filter: $("status-filter"),
  list: $("case-list"), card: $("review-card"), position: $("case-position"), alias: $("case-alias"),
  status: $("case-status"), length: $("source-length"), content: $("source-content"),
  focal: $("focal-box"), proposals: $("proposal-list"), form: $("review-form"),
  editor: $("span-editor"), rows: $("span-rows"), template: $("span-template"), add: $("add-span"),
  notes: $("notes"), error: $("form-error"), previous: $("previous"), clear: $("clear"), toast: $("toast"),
};

function current() { return bundle.cases[state.index]; }
function decision() { const node = dom.form.querySelector('input[name="decision"]:checked'); return node ? node.value : ""; }
function complete(annotation) { return Boolean(annotation && typeof annotation.needs_explanation === "boolean"); }
function positions(content, surface) {
  const result = []; let offset = 0;
  while (surface && offset <= content.length - surface.length) {
    const start = content.indexOf(surface, offset); if (start < 0) break;
    result.push(start); offset = start + 1;
  }
  return result;
}
function toast(message, error = false) {
  dom.toast.textContent = message; dom.toast.classList.toggle("error", error); dom.toast.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { dom.toast.hidden = true; }, 3000);
}
function persist() {
  localStorage.setItem(storageKey, JSON.stringify({
    package_id: packageId, reviewer_id: state.reviewerId, selected_index: state.index,
    annotations: state.annotations,
  }));
}
function validAnnotation(row, item) {
  if (!row || row.case_id !== item.case_id || typeof row.needs_explanation !== "boolean" ||
      !Array.isArray(row.spans) || !Array.isArray(row.issue_tags) || typeof row.notes !== "string") return false;
  if (row.needs_explanation !== (row.spans.length > 0)) return false;
  if (!row.issue_tags.every((tag) => issueValues.has(tag))) return false;
  return row.spans.every((span) => {
    if (!span || typeof span.surface !== "string" || !span.surface ||
        !Number.isInteger(span.occurrence_ordinal) || span.occurrence_ordinal < 1) return false;
    return span.occurrence_ordinal <= positions(item.content, span.surface).length;
  });
}
function loadDraft() {
  try {
    const raw = localStorage.getItem(storageKey); if (!raw) return;
    const saved = JSON.parse(raw); if (saved.package_id !== packageId) return;
    const byId = new Map(bundle.cases.map((item) => [item.case_id, item]));
    state.annotations = Object.fromEntries(Object.entries(saved.annotations || {}).filter(([id, row]) => byId.has(id) && validAnnotation(row, byId.get(id))));
    if (typeof saved.reviewer_id === "string" && saved.reviewer_id.trim()) state.reviewerId = saved.reviewer_id.slice(0, 100);
    if (Number.isInteger(saved.selected_index)) state.index = Math.max(0, Math.min(bundle.cases.length - 1, saved.selected_index));
  } catch (_error) { toast("旧草稿无法读取，已从空白状态开始。", true); }
}
function visibleCases() {
  return bundle.cases.filter((item) => {
    const row = state.annotations[item.case_id];
    if (state.filter === "pending") return !complete(row);
    if (state.filter === "complete") return complete(row);
    if (state.filter === "flagged") return row && row.issue_tags.length > 0;
    return true;
  });
}
function renderList() {
  dom.list.replaceChildren();
  for (const item of visibleCases()) {
    const index = bundle.cases.findIndex((candidate) => candidate.case_id === item.case_id);
    const button = document.createElement("button"); button.type = "button";
    button.className = `case-button${index === state.index ? " active" : ""}${complete(state.annotations[item.case_id]) ? " complete" : ""}`;
    button.textContent = `${item.case_id} · ${item.blind_alias}`;
    button.addEventListener("click", () => { state.index = index; persist(); render(); });
    dom.list.appendChild(button);
  }
}
function addSpan(value = {}) {
  const fragment = dom.template.content.cloneNode(true); const row = fragment.querySelector(".span-row");
  row.querySelector(".surface").value = value.surface || "";
  row.querySelector(".ordinal").value = value.occurrence_ordinal || 1;
  const update = () => {
    const surface = row.querySelector(".surface").value.trim(); const ordinal = Number(row.querySelector(".ordinal").value);
    const count = positions(current().content, surface).length; const ok = surface && Number.isInteger(ordinal) && ordinal >= 1 && ordinal <= count;
    const match = row.querySelector(".match"); match.textContent = surface ? (ok ? `匹配，共 ${count} 次` : `无法匹配（共 ${count} 次）`) : "等待填写";
    match.classList.toggle("bad", Boolean(surface) && !ok);
  };
  row.querySelector(".surface").addEventListener("input", update); row.querySelector(".ordinal").addEventListener("input", update);
  row.querySelector(".remove").addEventListener("click", () => { row.remove(); if (!dom.rows.children.length) addSpan(); });
  dom.rows.appendChild(fragment); update();
}
function collectSpans() {
  return [...dom.rows.querySelectorAll(".span-row")].map((row) => ({
    surface: row.querySelector(".surface").value.trim(),
    occurrence_ordinal: Number(row.querySelector(".ordinal").value),
  }));
}
function selectDecision(value) { for (const node of dom.form.querySelectorAll('input[name="decision"]')) node.checked = node.value === value; }
function renderCase() {
  const item = current(); const saved = state.annotations[item.case_id];
  dom.position.textContent = `${state.index + 1} / ${bundle.cases.length}`; dom.alias.textContent = item.blind_alias;
  dom.status.textContent = complete(saved) ? "已完成" : "未完成"; dom.status.classList.toggle("done", complete(saved));
  dom.length.textContent = `${item.content.length} 字符`; dom.content.textContent = item.content;
  dom.focal.hidden = !item.focal_proposal;
  if (item.focal_proposal) dom.focal.textContent = `本次焦点：${item.focal_proposal.surface}（第 ${item.focal_proposal.occurrence_ordinal} 次）`;
  dom.proposals.replaceChildren();
  for (const proposal of item.proposals || []) { const chip = document.createElement("span"); chip.textContent = `${proposal.surface} · #${proposal.occurrence_ordinal}`; dom.proposals.appendChild(chip); }
  selectDecision(saved ? (saved.needs_explanation ? "has_spans" : "no_span") : "");
  dom.editor.hidden = !(saved && saved.needs_explanation); dom.rows.replaceChildren();
  if (saved && saved.spans.length) for (const span of saved.spans) addSpan(span); else addSpan();
  for (const node of dom.form.querySelectorAll('input[name="issue"]')) node.checked = Boolean(saved && saved.issue_tags.includes(node.value));
  dom.notes.value = saved ? saved.notes : ""; dom.error.hidden = true;
}
function updateProgress() {
  const count = Object.values(state.annotations).filter(complete).length; dom.complete.textContent = count; dom.total.textContent = bundle.cases.length;
  dom.fill.style.width = `${(count / bundle.cases.length) * 100}%`; dom.message.textContent = count === bundle.cases.length ? "全部完成，可以导出" : `还剩 ${bundle.cases.length - count} 条`;
}
function render() { dom.reviewer.value = state.reviewerId; renderList(); renderCase(); updateProgress(); }

dom.form.addEventListener("change", (event) => { if (event.target.name === "decision") dom.editor.hidden = event.target.value !== "has_spans"; });
dom.add.addEventListener("click", () => addSpan());
dom.form.addEventListener("submit", (event) => {
  event.preventDefault(); const item = current(); const choice = decision();
  if (!choice) { dom.error.textContent = "请选择总体结论。"; dom.error.hidden = false; return; }
  const spans = choice === "has_spans" ? collectSpans() : [];
  const candidate = { case_id: item.case_id, needs_explanation: choice === "has_spans", spans,
    issue_tags: [...dom.form.querySelectorAll('input[name="issue"]:checked')].map((node) => node.value), notes: dom.notes.value };
  if (!validAnnotation(candidate, item)) { dom.error.textContent = "请检查 span 是否逐字存在、出现序号是否正确，且有 span 结论至少填写一项。"; dom.error.hidden = false; return; }
  state.annotations[item.case_id] = candidate; persist(); state.index = Math.min(bundle.cases.length - 1, state.index + 1); persist(); render();
});
dom.previous.addEventListener("click", () => { state.index = Math.max(0, state.index - 1); persist(); render(); });
dom.clear.addEventListener("click", () => { delete state.annotations[current().case_id]; persist(); render(); });
dom.filter.addEventListener("change", () => { state.filter = dom.filter.value; renderList(); });
dom.reviewer.addEventListener("input", () => { state.reviewerId = dom.reviewer.value.slice(0, 100); persist(); });
dom.importButton.addEventListener("click", () => dom.importFile.click());
dom.importFile.addEventListener("change", async () => {
  const file = dom.importFile.files[0]; if (!file) return;
  try {
    const value = JSON.parse(await file.text()); if (value.package_id !== packageId || !Array.isArray(value.annotations)) throw new Error("包 ID 不一致");
    const byId = new Map(bundle.cases.map((item) => [item.case_id, item])); const imported = {};
    for (const row of value.annotations) { const item = byId.get(row.case_id); if (!item || !validAnnotation(row, item)) throw new Error(`无效 case：${row.case_id}`); imported[row.case_id] = row; }
    state.annotations = imported; if (typeof value.reviewer_id === "string") state.reviewerId = value.reviewer_id; persist(); render(); toast("导入成功");
  } catch (error) { toast(`导入失败：${error.message}`, true); }
});
dom.exportButton.addEventListener("click", () => {
  const annotations = bundle.cases.map((item) => state.annotations[item.case_id]).filter(Boolean);
  if (annotations.length !== bundle.cases.length) { toast(`仍有 ${bundle.cases.length - annotations.length} 条未完成`, true); return; }
  if (!state.reviewerId.trim()) { toast("请填写标注者代号", true); return; }
  const payload = { schema_version: "terminology-span-human-review/v1", package_id: packageId, reviewer_id: state.reviewerId.trim(), annotations };
  const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], { type: "application/json" }); const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = `${packageId}.annotations.json`; link.click(); URL.revokeObjectURL(link.href);
});

loadDraft(); render();
