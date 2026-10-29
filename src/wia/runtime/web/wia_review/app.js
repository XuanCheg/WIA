/* Serve with the WIA review service, then open its root URL. No build step is required. */
"use strict";

const $ = (id) => document.getElementById(id);
const FAMILIES = ["AP", "NT", "GF", "SR", "GG", "TM"];
const DECISIONS = {approved: "Design / layout OK", needs_fix: "Needs changes", skipped: "Skipped", unreviewed: "Unreviewed"};
const state = {rows: [], total: 0, page: 0, pageSize: 100, scene: null,
  draft: {decision: "unreviewed", notes: ""}, saved: {decision: "unreviewed", notes: "", revision: 0},
  readiness: "all", detailBusy: false, saving: false, navigationBusy: false,
  catalogueRequest: 0, detailRequest: 0, imageId: null, imageGroup: "robot",
  compareEnabled: false, compareId: null, guardResolve: null, reviewTouched: false,
  rowsSignature: "", queueAnchor: null};

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}

function plain(value, fallback = "Not supplied") {
  if (value === null || value === undefined || value === "") return fallback;
  return typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
}

function pretty(value) { return plain(value, "Unknown").replaceAll("_", " "); }
function reviewOf(row) { return row && row.review && typeof row.review === "object" ? row.review : {}; }
function dirty() { return state.draft.decision !== state.saved.decision || state.draft.notes !== state.saved.notes || state.reviewTouched; }
function selectedVariant() { return state.scene && state.scene.selected_variant || {}; }
function selectedVariantMeta() { return state.scene && (state.scene.variants || []).find((variant) => variant.id === state.scene.selected_variant_id) || {}; }
function countLabel(value, singular, plural = singular + "s") {
  return Number.isInteger(value) && value >= 0 ? value + " " + (value === 1 ? singular : plural) : "Count unavailable";
}

function notice(id, message, kind = "") {
  const target = $(id);
  target.textContent = message;
  target.className = "notice" + (kind ? " " + kind : "") + (message ? "" : " hidden");
}

async function request(url, options = {}) {
  const response = await fetch(url, {cache: "no-store", credentials: "same-origin", ...options,
    headers: {Accept: "application/json", ...(options.headers || {})}});
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) {
    const error = new Error(plain(data.error || data.message, "Request failed (HTTP " + response.status + ")."));
    error.status = response.status;
    throw error;
  }
  return data;
}

function renderStats(payload) {
  const summary = payload.summary && typeof payload.summary === "object" ? payload.summary : {};
  const progress = summary.proposal_progress || {};
  const counts = [];
  const known = [["total", "Original proposals"], ["definitions_available", "Definition available"],
    ["with_images", "With images"], ["without_images", "Awaiting images"],
    ["awaiting_image_review", "Ready for image review"], ["with_layout_approval", "At least one layout approval"],
    ["reference_verified", "Reference verified"]];
  for (const [key, label] of known) {
    if (Number.isInteger(progress[key]) && progress[key] >= 0) counts.push([progress[key], label]);
  }
  $("stats").replaceChildren(...counts.map(([count, label]) => {
    const item = node("div", "stat");
    item.append(node("strong", "", String(count)), node("span", "", label));
    return item;
  }));
  $("progress-context").textContent = counts.length
    ? "Original proposals only. Layout approval means at least one current version is approved; it does not certify all versions or native success."
    : "Original proposal progress is unavailable in this snapshot.";
}

function statusOptions() {
  const select = $("status-filter");
  const value = select.value;
  const values = new Set([...Array.from(select.options).map((option) => option.value),
    ...state.rows.map((row) => row.status).filter((status) => typeof status === "string" && status)]);
  select.replaceChildren(node("option", "", "All statuses"));
  select.firstChild.value = "";
  for (const status of [...values].filter(Boolean).sort()) {
    const option = node("option", "", pretty(status));
    option.value = status;
    select.append(option);
  }
  select.value = value;
}

function badge(text) { return node("span", "badge", text); }

function orderedRows() {
  const other = [...new Set(state.rows.map((row) => row.primary_family || "Other"))].filter((family) => !FAMILIES.includes(family));
  return [...FAMILIES, ...other].flatMap((family) => state.rows.filter((row) => (row.primary_family || "Other") === family));
}

function queueVariant(row) {
  if (state.readiness === "awaiting_review") return row.next_review_variant_id || row.default_variant_id;
  if (state.readiness === "needs_fix") return row.next_needs_fix_variant_id || row.default_variant_id;
  if (state.readiness === "reference_verified") return row.reference_variant_id || row.default_variant_id;
  return undefined;
}

function renderRows() {
  const list = $("task-list");
  list.replaceChildren();
  const other = [...new Set(state.rows.map((row) => row.primary_family || "Other"))].filter((family) => !FAMILIES.includes(family));
  for (const family of [...FAMILIES, ...other]) {
    const rows = state.rows.filter((row) => (row.primary_family || "Other") === family);
    if (!rows.length) continue;
    const group = node("section", "family-group");
    const heading = node("h3", "family-heading", family);
    heading.append(node("span", "", rows.length + " on this page"));
    group.append(heading);
    for (const row of rows) {
      const button = node("button", "task-row" + (state.scene && row.id === state.scene.id ? " selected" : ""));
      button.type = "button";
      if (state.scene && row.id === state.scene.id) button.setAttribute("aria-current", "true");
      button.append(node("span", "row-alias", plain(row.document_alias, row.id)),
        node("strong", "", plain(row.title, "Untitled task")),
        node("span", "row-subclass", plain(row.primary_subclass, "Subclass not supplied")));
      const statuses = node("div", "row-status");
      statuses.append(badge(row.image_count > 0 ? "Images ready" : "Awaiting images"));
      if (row.image_count > 0) statuses.append(badge(countLabel(row.image_variant_count, "version with images", "versions with images")));
      if (row.status && !["definition_only", "unbuilt", "proposal"].includes(row.status)) statuses.append(badge("Research: " + pretty(row.status)));
      if (row.has_definition === false) statuses.append(badge("Definition pending"));
      if (row.unreviewed_image_variant_count > 0) statuses.append(badge(row.unreviewed_image_variant_count + " image version(s) to review"));
      if (row.needs_fix_variant_count > 0) statuses.append(badge(row.needs_fix_variant_count + " version(s) need changes"));
      const decision = row.decision || reviewOf(row).decision;
      const reviewLabel = "Default version: " + (DECISIONS[decision] || "Unreviewed") + (row.review_stale ? " · Earlier review is stale" : "");
      button.append(statuses, node("span", "row-review", reviewLabel));
      button.addEventListener("click", () => openScene(row.id, queueVariant(row)));
      group.append(button);
    }
    list.append(group);
  }
  if (!state.rows.length) list.append(node("p", "notice", "No tasks match these filters. Pending and unbuilt tasks are included when they match."));
  $("page-label").textContent = "Page " + (state.page + 1) + " / " + Math.max(1, Math.ceil(state.total / state.pageSize));
  $("page-prev").disabled = state.page === 0;
  $("page-next").disabled = (state.page + 1) * state.pageSize >= state.total;
  const index = orderedRows().findIndex((row) => state.scene && row.id === state.scene.id);
  if (index >= 0) state.queueAnchor = {index, signature: state.rowsSignature};
  updateQueueControls();
}

function catalogueQuery() {
  return new URLSearchParams({q: $("search").value.trim(), family: $("family-filter").value,
    status: $("status-filter").value, review: $("review-filter").value, scope: $("scope-filter").value,
    readiness: state.readiness,
    page: String(state.page), page_size: String(state.pageSize)});
}

async function loadCatalogue(automatic = false) {
  const sequence = ++state.catalogueRequest;
  const query = catalogueQuery();
  if (!automatic) $("list-status").textContent = "Refreshing catalogue…";
  try {
    const payload = await request("api/catalogue?" + query.toString());
    if (sequence !== state.catalogueRequest) return false;
    state.rows = Array.isArray(payload.scenes) ? payload.scenes : [];
    state.total = Number.isInteger(payload.total) ? payload.total : state.rows.length;
    state.page = Number.isInteger(payload.page) ? payload.page : state.page;
    state.pageSize = Number.isInteger(payload.page_size) && payload.page_size > 0 ? payload.page_size : 100;
    state.rowsSignature = query.toString();
    renderStats(payload);
    statusOptions();
    renderRows();
    $("list-status").textContent = state.total + " matches · " + ($("scope-filter").value === "original" ? "original proposals" : $("scope-filter").value === "extra" ? "additional proposals" : "all proposals");
    $("snapshot-date").textContent = payload.generated_at ? "Catalogue snapshot: " + plain(payload.generated_at) : "Catalogue snapshot time not supplied";
    notice("catalogue-error", "");
    return true;
  } catch (error) {
    if (sequence !== state.catalogueRequest) return;
    $("list-status").textContent = "List refresh unavailable";
    notice("catalogue-error", error.message + " The current detail and unsaved notes have been kept.", "error");
    return false;
  }
}

function renderProse(id, value, fallback) {
  const target = $(id);
  target.replaceChildren();
  if (Array.isArray(value)) {
    const list = node("ul");
    for (const item of value) list.append(node("li", "", plain(item)));
    target.append(value.length ? list : node("p", "", fallback || "None supplied"));
  } else target.append(node(typeof value === "object" && value !== null ? "pre" : "p", "", plain(value, fallback)));
}

async function copyText(text, button) {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const field = node("textarea");
      field.value = text;
      field.style.position = "fixed";
      field.style.left = "-10000px";
      field.readOnly = true;
      document.body.append(field);
      field.select();
      const copied = document.execCommand("copy");
      field.remove();
      if (!copied) throw new Error("Clipboard unavailable");
    }
    button.textContent = "Copied";
  } catch { button.textContent = "Select path to copy"; }
}

function renderPaths(id, entries) {
  const target = $(id);
  target.replaceChildren();
  for (const entry of entries) {
    if (!entry || typeof entry.path !== "string" || !entry.path) continue;
    const row = node("div", "path-entry");
    const heading = node("div", "path-heading");
    const button = node("button", "text-button", "Copy path");
    button.type = "button";
    button.addEventListener("click", () => copyText(entry.path, button));
    heading.append(node("span", "", plain(entry.label, "Evidence")), button);
    row.append(heading, node("code", "", entry.path));
    target.append(row);
  }
  if (!target.childElementCount) target.append(node("p", "caption", "No paths supplied for this version."));
}

const ROBOT_VIEWS = ["agentview_rgb", "head_rgb", "head", "wrist_rgb", "wrist", "right_wrist_rgb", "right_wrist",
  "chassis_front_left_rgb", "chassis_front_left", "chassis_front_right_rgb", "chassis_front_right",
  "chassis_left_rgb", "chassis_left", "chassis_right_rgb", "chassis_right", "chassis_rear_rgb", "chassis_rear"];

function viewName(image) {
  const label = String(image.label || "").split("/").pop().trim().toLowerCase().replace(/\.png$/, "");
  const base = label.replace(/^phase_[12]_/, "");
  if (ROBOT_VIEWS.includes(base) || /^(initial_overview|global_|object_|scene)/.test(label)) return label;
  const id = String(image.id || "").toLowerCase();
  const known = ["initial_overview_rgb", ...["phase_1", "phase_2"].flatMap((phase) => ROBOT_VIEWS.map((name) => phase + "_" + name)),
    ...ROBOT_VIEWS].find((name) => id === name || id.endsWith("_" + name));
  return known || label;
}

function isRobotImage(image) { return ROBOT_VIEWS.includes(viewName(image).replace(/^phase_[12]_/, "")); }
function isOverviewImage(image) { return viewName(image) === "initial_overview_rgb"; }
function isAgentImage(image) { return isRobotImage(image) || isOverviewImage(image); }
function imageWords(image) { return [image.role, viewName(image), image.label].filter(Boolean).join(" ").toLowerCase(); }

function imageLabel(image) {
  const view = viewName(image).replace(/^phase_[12]_/, "").replace(/_rgb$/, "");
  const names = {agentview: "Head", head: "Head", wrist: "Left wrist", right_wrist: "Right wrist",
    initial_overview: "Frozen initial overview", object_overview: "Object overview",
    object_top: "Objects from above", object_side: "Objects from the side",
    global_diagonal: "Room overview", global_overhead: "Room from above", scene: "Scene overview"};
  const label = names[view] || pretty(view).replace(/^./, (letter) => letter.toUpperCase());
  const words = imageWords(image);
  const phase = /phase[ _-]?2/.test(words) ? "Phase 2" : /phase[ _-]?1/.test(words) ? "Phase 1"
    : /released|release[ _-]?state/.test(words) ? "Released state"
    : /initial/.test(words) && view !== "initial_overview" ? "Initial" : "";
  return label + (phase ? " · " + phase : "");
}

function imageContext(image) {
  const words = imageWords(image);
  const contexts = [isOverviewImage(image) ? "Frozen initial overview · Public input · History-bearing"
    : isRobotImage(image) ? "Robot-mounted camera" : "Auxiliary inspection"];
  if (isOverviewImage(image)) return contexts.join(" · ");
  else if (/historical|history|frozen/.test(words)) contexts.push("Historical / history-bearing");
  if (/phase[ _-]?1/.test(words)) contexts.push("Phase 1");
  if (/phase[ _-]?2/.test(words)) contexts.push("Phase 2");
  if (/released|release[ _-]?state/.test(words)) contexts.push("Released state");
  if (/initial/.test(words) && !/initial_overview/.test(words)) contexts.push("Initial state");
  return contexts.join(" · ");
}

function images() { return state.scene && Array.isArray(state.scene.images) ? state.scene.images : []; }

function defaultImage(available) {
  const heads = available.filter((image) => ["head", "head_rgb", "agentview_rgb"].includes(viewName(image).replace(/^phase_[12]_/, "")));
  const rank = (image) => /phase[ _-]?2|released|release[ _-]?state/.test(imageWords(image)) ? 3
    : /initial/.test(imageWords(image)) ? 0 : /phase[ _-]?1/.test(imageWords(image)) ? 1 : 2;
  return heads.slice().sort((left, right) => rank(left) - rank(right))[0]
    || available.find(isRobotImage) || available[0] || null;
}

function defaultComparison(available) {
  for (const name of ["object_overview", "global_diagonal", "initial_overview_rgb"]) {
    const image = available.find((item) => viewName(item).replace(/^phase_[12]_/, "") === name);
    if (image) return image;
  }
  return defaultImage(available);
}

function imageURL(image) {
  if (typeof image.url !== "string" || !image.url) return null;
  try {
    const url = new URL(image.url, window.location.href);
    return url.origin === window.location.origin && ["http:", "https:"].includes(url.protocol) ? image.url : null;
  } catch { return null; }
}

function renderImage(image, comparison = false) {
  const hero = $(comparison ? "compare-hero" : "hero");
  const caption = $(comparison ? "compare-caption" : "image-caption");
  hero.replaceChildren();
  if (comparison) state.compareId = image ? image.id : null;
  else state.imageId = image ? image.id : null;
  const url = image && imageURL(image);
  if (!url) {
    const placeholder = node("div", "placeholder");
    placeholder.append(node("strong", "", image ? "Recorded image unavailable" : "No recorded image available"),
      node("span", "", "This version may be pending, unbuilt, or awaiting a capture. Its definition remains reviewable."));
    hero.append(placeholder);
    caption.textContent = "No available image in the selected view group.";
  } else {
    const picture = node("img");
    picture.alt = plain(image.label, "Recorded WIA view");
    picture.loading = "lazy";
    picture.decoding = "async";
    const loading = node("span", "image-loading", "Loading recorded image…");
    picture.addEventListener("load", () => loading.remove(), {once: true});
    picture.addEventListener("error", () => {
      if (!hero.contains(picture)) return;
      hero.replaceChildren(node("div", "placeholder", "The recorded image could not be loaded. Its evidence status is unchanged; try refreshing the detail."));
    }, {once: true});
    picture.addEventListener("click", () => {
      const enlarged = node("img");
      enlarged.alt = picture.alt;
      enlarged.src = url;
      $("enlarged-image").replaceChildren(enlarged);
      $("enlarged-caption").textContent = imageContext(image) + " · " + picture.alt;
      $("image-dialog").showModal();
    });
    picture.src = url;
    hero.append(picture, loading);
    caption.textContent = imageContext(image) + " · " + plain(image.label, "Recorded evidence") + ". Click to enlarge.";
  }
  if (comparison) return;
  $("view-select").value = state.imageId || "";
  for (const button of $("view-tabs").children) {
    const active = button.dataset.imageId === String(state.imageId);
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function renderComparison() {
  const choices = state.imageId ? images().filter((image) => image.id !== state.imageId) : [];
  if (!choices.length) state.compareEnabled = false;
  $("compare-toggle").checked = state.compareEnabled;
  $("compare-toggle").disabled = !choices.length;
  $("comparison-card").classList.toggle("hidden", !state.compareEnabled);
  $("compare-view-control").classList.toggle("hidden", !state.compareEnabled);
  $("image-pair").classList.toggle("comparing", state.compareEnabled);
  $("compare-view-select").replaceChildren();
  for (const image of choices) {
    const option = node("option", "", imageLabel(image) + " · " + imageContext(image));
    option.value = image.id;
    $("compare-view-select").append(option);
  }
  if (!state.compareEnabled) {
    $("compare-hero").replaceChildren();
    $("compare-caption").textContent = "";
    return;
  }
  const second = choices.find((image) => image.id === state.compareId) || defaultComparison(choices);
  renderImage(second, true);
  $("compare-view-select").value = state.compareId || "";
}

function renderImages(preserve = false) {
  const available = images();
  if (!preserve) {
    state.imageId = null;
    state.imageGroup = available.some(isAgentImage) || !available.length ? "robot" : "auxiliary";
    state.compareEnabled = false;
    state.compareId = null;
  }
  const visible = available.filter((image) => state.imageGroup === "all" || isAgentImage(image) === (state.imageGroup === "robot"));
  for (const button of document.querySelectorAll("[data-image-group]")) {
    const active = button.dataset.imageGroup === state.imageGroup;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  $("images-context").textContent = plain(selectedVariant().label, state.scene.selected_variant_id || "No version selected")
    + " · " + countLabel(available.length, "recorded image") + " · only chosen views are loaded";
  $("view-tabs").replaceChildren();
  $("view-select").replaceChildren();
  for (const image of visible) {
    const option = node("option", "", imageLabel(image));
    option.value = image.id;
    $("view-select").append(option);
    const button = node("button", "view-tab", imageLabel(image));
    button.title = plain(image.label, "Recorded view");
    button.type = "button";
    button.dataset.imageId = String(image.id);
    button.append(node("small", "", imageContext(image)));
    button.addEventListener("click", () => { renderImage(image); renderComparison(); });
    $("view-tabs").append(button);
  }
  if (!visible.length) {
    const option = node("option", "", "No images in this group");
    option.value = "";
    $("view-select").append(option);
  }
  $("view-select").disabled = !visible.length;
  renderImage(visible.find((image) => image.id === state.imageId) || defaultImage(visible));
  renderComparison();
}

function queuePlan(direction) {
  if (!state.total || state.rowsSignature !== catalogueQuery().toString()) return null;
  const rows = orderedRows();
  const current = rows.findIndex((row) => state.scene && row.id === state.scene.id);
  const anchor = state.queueAnchor && state.queueAnchor.signature === state.rowsSignature ? state.queueAnchor.index : null;
  const index = current >= 0 ? current + direction
    : anchor !== null ? anchor + (direction > 0 ? 0 : -1) : direction > 0 ? 0 : rows.length - 1;
  if (index >= 0 && index < rows.length) return {row: rows[index]};
  const page = state.page + direction;
  if (page >= 0 && page * state.pageSize < state.total) return {page};
  return null;
}

function updateQueueControls() {
  const busy = !state.scene || state.detailBusy || state.saving || state.navigationBusy || !!state.guardResolve;
  $("queue-prev").disabled = busy || !queuePlan(-1);
  $("queue-next").disabled = busy || !queuePlan(1);
  const index = orderedRows().findIndex((row) => state.scene && row.id === state.scene.id);
  $("queue-context").textContent = index < 0
    ? "Selected task is outside this filtered queue · " + state.total + " matches"
    : "Filtered queue · " + state.total + " tasks · " + (state.readiness === "awaiting_review" ? "opens the next unreviewed image version" : "previous / next task");
}

async function moveQueue(direction) {
  if (state.detailBusy || state.saving || state.navigationBusy) return;
  state.navigationBusy = true;
  updateReviewControls();
  try {
    if (!await guardDirty()) return;
    const plan = queuePlan(direction);
    if (!plan) {
      notice("detail-status", "No further task in this filtered queue. Change a filter to continue.");
      return;
    }
    let row = plan.row;
    if (!row) {
      const previousPage = state.page;
      state.page = plan.page;
      if (!await loadCatalogue()) { state.page = previousPage; return; }
      const rows = orderedRows();
      row = direction > 0 ? rows[0] : rows[rows.length - 1];
    }
    if (row) await openScene(row.id, queueVariant(row), false, {guarded: true});
  } finally { state.navigationBusy = false; updateReviewControls(); }
}

async function saveAndNext() {
  if ($("save-next").disabled || !state.scene || state.navigationBusy) return;
  const previous = {id: state.scene.id, variant: state.scene.selected_variant_id};
  if (!await saveReview()) return;
  const row = state.rows.find((item) => item.id === previous.id);
  if (state.readiness === "awaiting_review" && row && row.next_review_variant_id && row.next_review_variant_id !== previous.variant) {
    await openScene(row.id, row.next_review_variant_id);
  } else await moveQueue(1);
}

function updateReviewControls() {
  const busy = !state.scene || state.detailBusy || state.saving || state.navigationBusy;
  const disabled = busy || !state.scene.selected_variant_id;
  for (const button of document.querySelectorAll("[data-decision]")) {
    const active = button.dataset.decision === state.draft.decision;
    button.classList.toggle("selected", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = disabled;
  }
  $("notes").disabled = disabled;
  for (const button of document.querySelectorAll("[data-note-template]")) button.disabled = disabled;
  $("variant-select").disabled = busy || !(state.scene && state.scene.variants && state.scene.variants.length);
  $("refresh-detail").disabled = busy;
  $("save-review").disabled = disabled || !dirty();
  $("save-review").textContent = state.saving ? "Saving…" : "Save review";
  $("save-next").disabled = disabled || (!dirty() && state.draft.decision === "unreviewed");
  $("save-next").textContent = state.saving ? "Saving…" : "Save and next →";
  $("save-state").textContent = state.saving ? "Saving review…" : dirty() ? "Unsaved changes" : "No unsaved changes";
  $("save-state").classList.toggle("dirty", dirty());
  for (const id of ["guard-save", "guard-discard", "guard-cancel"]) $(id).disabled = state.saving;
  updateQueueControls();
}

function renderVariantOptions() {
  const variants = Array.isArray(state.scene.variants) ? state.scene.variants : [];
  $("variant-select").replaceChildren();
  for (const variant of variants) {
    const count = Number.isInteger(variant.image_count) ? variant.image_count
      : variant.id === state.scene.selected_variant_id ? images().length : null;
    const decision = variant.review_stale ? "Unreviewed (stale review)" : DECISIONS[variant.decision] || "Unreviewed";
    const option = node("option", "", plain(variant.label, variant.id) + " · " + countLabel(count, "image") + " · " + decision);
    option.value = variant.id;
    $("variant-select").append(option);
  }
  if (!variants.length) {
    const option = node("option", "", "No declared version available");
    option.value = "";
    $("variant-select").append(option);
  } else $("variant-select").value = state.scene.selected_variant_id;
  $("variant-context").textContent = variants.length ? countLabel(variants.length, "declared version") + " · images and feedback belong to each version"
    : "No declared version · feedback saving is unavailable";
}

function renderDetail(preserveImages = false) {
  const scene = state.scene;
  const variant = selectedVariant();
  $("welcome").classList.add("hidden");
  $("detail-content").classList.remove("hidden");
  $("detail-family").textContent = [scene.document_alias, scene.primary_family, scene.primary_subclass].filter(Boolean).join(" / ");
  $("detail-title").textContent = plain(scene.title, "Untitled task");
  $("detail-badges").replaceChildren(badge("Selected version: " + pretty(variant.status)),
    badge("Row implementation: " + pretty(scene.implementation_state)),
    badge(scene.proposal_membership === true ? "Original proposal" : scene.proposal_membership === false ? "Additional proposal" : "Proposal set not supplied"));
  renderVariantOptions();
  renderProse("objective", variant.objective, "No public objective supplied for this version.");
  renderProse("row-summary", scene.summary, "No row summary supplied yet.");
  const findings = scene.reassessment && typeof scene.reassessment === "object"
    ? Object.entries(scene.reassessment).filter(([, value]) => value !== null && value !== "")
      .map(([key, value]) => pretty(key) + ": " + plain(value)) : scene.reassessment;
  $("original-assessment").classList.toggle("hidden", !findings || Array.isArray(findings) && !findings.length);
  $("known-findings").classList.toggle("hidden", !findings || Array.isArray(findings) && !findings.length);
  renderProse("known-findings", findings, "");
  renderProse("success-criteria", variant.success_criteria, "No public success criteria supplied for this version.");
  renderProse("constraints", variant.constraints, "No public constraints supplied for this version.");
  renderProse("environment", variant.environment, "No environment description supplied for this version.");
  renderProse("evidence-summary", variant.evidence_summary, "No reference evidence summary supplied for this version.");
  $("evidence-status").replaceChildren();
  for (const [label, value] of [["Reference execution", variant.status === "reference_verified" ? "Verified for this selected version" : "Not verified for this selected version"],
    ["Native evidence", variant.native_evidence_status || variant.evidence_status || "Not supplied for this version"],
    ["Definition", selectedVariantMeta().has_definition === true ? "Available" : variant.missing_definition === false ? "Available" : "Proposal or unavailable"],
    ["Version status", variant.status]]) {
    $("evidence-status").append(node("dt", "", label), node("dd", "", pretty(value)));
  }
  renderPaths("evidence-paths", Array.isArray(variant.evidence) ? variant.evidence : []);
  renderPaths("definition-paths", [{label: "Task definition", path: variant.task_file}, {label: "Scene definition", path: variant.scene_file}]);
  $("notes").value = state.draft.notes;
  $("review-revision").textContent = "Review revision: " + plain(state.saved.revision, "not supplied");
  $("stale-review").classList.toggle("hidden", !state.saved.stale);
  $("stale-review").textContent = "Earlier review: " + (DECISIONS[state.saved.previousDecision] || "Unreviewed") + ". Its fingerprint is stale. Notes are retained; choose and save a current decision after inspecting this version.";
  if (!scene.selected_variant_id) notice("review-feedback", "This pending row has no declared reviewable version yet. Its design and evidence remain available for inspection.");
  renderImages(preserveImages);
  updateReviewControls();
}

async function saveReview() {
  if (!state.scene || !state.scene.selected_variant_id || state.saving || state.detailBusy) return false;
  if (!dirty()) return true;
  state.saving = true;
  updateReviewControls();
  notice("review-feedback", "Saving this version's review…");
  const payload = {decision: state.draft.decision, notes: state.draft.notes,
    revision: state.saved.revision, fingerprint: state.scene.fingerprint,
    variant_id: state.scene.selected_variant_id};
  try {
    const response = await request("api/reviews/" + encodeURIComponent(state.scene.id), {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    state.saved = {...state.saved, ...response.review, decision: response.review && response.review.decision || payload.decision,
      notes: response.review && typeof response.review.notes === "string" ? response.review.notes : payload.notes, stale: false};
    state.draft = {decision: state.saved.decision, notes: state.saved.notes};
    state.reviewTouched = false;
    state.scene.review = state.saved;
    state.scene.review_stale = false;
    state.scene.decision = state.saved.decision;
    Object.assign(selectedVariantMeta(), {decision: state.saved.decision, review_stale: false});
    renderVariantOptions();
    $("review-revision").textContent = "Review revision: " + plain(state.saved.revision);
    $("stale-review").classList.add("hidden");
    notice("review-feedback", "Review saved. This records human design feedback only.");
    await loadCatalogue(true);
    return true;
  } catch (error) {
    const message = error.status === 409
      ? "The saved review or evidence changed (409 conflict). Your unsaved decision and notes are preserved. Copy any notes you want to keep, then refresh the detail to load the current revision."
      : "Review was not saved. " + error.message + " Your unsaved decision and notes are preserved.";
    notice("review-feedback", message, "error");
    $("guard-feedback").textContent = message;
    return false;
  } finally {
    state.saving = false;
    updateReviewControls();
  }
}

function finishGuard(allowed) {
  const resolve = state.guardResolve;
  state.guardResolve = null;
  $("unsaved-dialog").close();
  updateReviewControls();
  if (resolve) resolve(allowed);
}

function guardDirty() {
  if (!dirty()) return Promise.resolve(true);
  if (state.guardResolve || state.saving) return Promise.resolve(false);
  $("guard-feedback").textContent = "";
  return new Promise((resolve) => {
    state.guardResolve = resolve;
    $("unsaved-dialog").showModal();
    updateQueueControls();
  });
}

function sceneLink() {
  const params = new URLSearchParams(window.location.hash.slice(1));
  return params.get("task") ? {id: params.get("task"), variant: params.get("variant") || undefined} : null;
}

function writeSceneLink() {
  if (!state.scene) return;
  const params = new URLSearchParams({task: state.scene.id});
  if (state.scene.selected_variant_id) params.set("variant", state.scene.selected_variant_id);
  // Replace the shareable location without adding navigation entries that bypass the draft guard.
  window.history.replaceState(null, "", window.location.pathname + window.location.search + "#" + params.toString());
}

async function openScene(id, variantId, refresh = false, options = {}) {
  if (state.detailBusy || state.saving) return false;
  if (!refresh && state.scene && state.scene.id === id && (!variantId || variantId === state.scene.selected_variant_id)) return true;
  if (!options.guarded && !await guardDirty()) return false;
  const previous = state.scene && {id: state.scene.id, variant: state.scene.selected_variant_id};
  const sequence = ++state.detailRequest;
  state.detailBusy = true;
  updateReviewControls();
  notice("detail-status", "Loading task and recorded evidence…");
  const query = new URLSearchParams({id});
  if (variantId) query.set("variant_id", variantId);
  try {
    const payload = await request("api/scene?" + query.toString());
    if (sequence !== state.detailRequest) return false;
    if (!payload.scene || typeof payload.scene !== "object") throw new Error("The task detail response is unavailable.");
    state.scene = payload.scene;
    const review = reviewOf(state.scene);
    const stale = !!(state.scene.review_stale || review.stale);
    state.saved = {...review, decision: stale ? "unreviewed" : review.decision || "unreviewed",
      previousDecision: review.decision, stale, notes: typeof review.notes === "string" ? review.notes : "", revision: review.revision ?? 0};
    state.draft = {decision: state.saved.decision, notes: state.saved.notes};
    state.reviewTouched = false;
    notice("detail-status", "");
    notice("review-feedback", "");
    const sameVersion = previous && previous.id === state.scene.id && previous.variant === state.scene.selected_variant_id;
    if ($("image-dialog").open) $("image-dialog").close();
    renderDetail(refresh && sameVersion);
    renderRows();
    writeSceneLink();
    if (!refresh) { $("detail-title").tabIndex = -1; $("detail-title").focus({preventScroll: true}); }
    return true;
  } catch (error) {
    notice("detail-status", "Could not refresh the detail. " + error.message + " Existing notes and evidence are kept.", "error");
    return false;
  }
  finally { state.detailBusy = false; updateReviewControls(); }
}

let searchTimer;
$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.page = 0; loadCatalogue(); }, 300);
});
for (const id of ["family-filter", "status-filter", "review-filter", "scope-filter"]) $(id).addEventListener("change", () => { state.page = 0; loadCatalogue(); updateQueueControls(); });
for (const button of document.querySelectorAll("[data-readiness]")) button.addEventListener("click", () => {
  state.readiness = button.dataset.readiness;
  for (const item of document.querySelectorAll("[data-readiness]")) {
    const active = item === button;
    item.classList.toggle("active", active);
    item.setAttribute("aria-pressed", String(active));
  }
  state.page = 0;
  loadCatalogue();
  updateQueueControls();
});
$("refresh-list").addEventListener("click", () => loadCatalogue());
$("page-prev").addEventListener("click", () => { if (state.page > 0) { state.page -= 1; loadCatalogue(); } });
$("page-next").addEventListener("click", () => { if ((state.page + 1) * state.pageSize < state.total) { state.page += 1; loadCatalogue(); } });
$("refresh-detail").addEventListener("click", () => { if (state.scene) openScene(state.scene.id, state.scene.selected_variant_id, true); });
$("variant-select").addEventListener("change", (event) => {
  if (!state.scene) return;
  const requested = event.target.value;
  event.target.value = state.scene.selected_variant_id || "";
  openScene(state.scene.id, requested);
});
for (const button of document.querySelectorAll("[data-decision]")) button.addEventListener("click", () => {
  if (!state.scene || state.detailBusy || state.saving) return;
  state.draft.decision = button.dataset.decision;
  if (state.saved.stale) state.reviewTouched = true;
  updateReviewControls();
});
$("notes").addEventListener("input", () => { state.draft.notes = $("notes").value; updateReviewControls(); });
$("save-review").addEventListener("click", saveReview);
$("save-next").addEventListener("click", saveAndNext);
$("queue-prev").addEventListener("click", () => moveQueue(-1));
$("queue-next").addEventListener("click", () => moveQueue(1));
for (const button of document.querySelectorAll("[data-note-template]")) button.addEventListener("click", () => {
  if ($("notes").disabled) return;
  const templates = {
    Visibility: "Visibility: Check that every required object is clearly visible in the initial robot views. Details: ",
    Occlusion: "Occlusion: Identify what blocks the required object or interaction. Suggested change: ",
    "Task wording": "Task wording: Clarify the objective, labels, or success criteria. Suggested wording: ",
    "Layout/reach": "Layout/reach: Describe the placement or access concern and the evidence needed. Details: ",
    "Scoring concern": "Scoring concern: Describe the criterion or measurement that needs checking. Details: ",
    "Mechanism concern": "Mechanism concern: Describe whether the intended mechanism is observable and necessary. Details: "
  };
  state.draft.notes += (state.draft.notes ? "\n\n" : "") + templates[button.dataset.noteTemplate];
  $("notes").value = state.draft.notes;
  $("notes").focus();
  $("notes").setSelectionRange(state.draft.notes.length, state.draft.notes.length);
  updateReviewControls();
});
for (const button of document.querySelectorAll("[data-image-group]")) button.addEventListener("click", () => {
  state.imageGroup = button.dataset.imageGroup;
  renderImages(true);
});
$("view-select").addEventListener("change", () => {
  renderImage(images().find((image) => image.id === $("view-select").value) || null);
  renderComparison();
});
$("compare-toggle").addEventListener("change", () => { state.compareEnabled = $("compare-toggle").checked; renderComparison(); });
$("compare-view-select").addEventListener("change", () => {
  renderImage(images().find((image) => image.id === $("compare-view-select").value) || null, true);
});
$("guard-cancel").addEventListener("click", () => finishGuard(false));
$("guard-discard").addEventListener("click", () => finishGuard(true));
$("guard-save").addEventListener("click", async () => { if (await saveReview()) finishGuard(true); });
$("unsaved-dialog").addEventListener("cancel", (event) => { event.preventDefault(); finishGuard(false); });
$("close-image").addEventListener("click", () => $("image-dialog").close());
$("image-dialog").addEventListener("close", () => $("enlarged-image").replaceChildren());
window.addEventListener("beforeunload", (event) => { if (dirty() || state.saving) { event.preventDefault(); event.returnValue = ""; } });
window.addEventListener("hashchange", async () => {
  const link = sceneLink();
  if (!link || !await openScene(link.id, link.variant)) writeSceneLink();
});
document.addEventListener("keydown", (event) => {
  if (event.defaultPrevented || event.isComposing || document.querySelector("dialog[open]")) return;
  if ((event.ctrlKey || event.metaKey) && !event.altKey && event.key === "Enter") {
    if (!$("save-next").disabled) { event.preventDefault(); saveAndNext(); }
    return;
  }
  const target = event.target;
  if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || target.isContentEditable
      || target.closest("input, textarea, select, button, a, summary, [role=button], [role=textbox], [contenteditable]")) return;
  if (event.key === "j" || event.key === "k") {
    const direction = event.key === "j" ? 1 : -1;
    if (!$(direction > 0 ? "queue-next" : "queue-prev").disabled) { event.preventDefault(); moveQueue(direction); }
  }
});
updateReviewControls();
loadCatalogue().then(() => { const link = sceneLink(); if (link) openScene(link.id, link.variant); });
setInterval(() => { if (!document.hidden) loadCatalogue(true); }, 15000);
