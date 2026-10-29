"use strict";
const $ = (id) => document.getElementById(id);
const labels = {unreviewed: "Unreviewed", approved: "Approved", needs_fix: "Needs repair", skipped: "Skipped"};
let scenes = [], selected = null, detail = null, view = null, mode = "overview", dirty = false, generation = 0;
let saveTimer = null, saving = null, loading = false, catalogueController = null, detailController = null;
let catalogueGeneration = 0, detailGeneration = 0;
let selectedVariant = null, coreGrouped = false, coreGroupCount = 0;
let cameraGroup = "global", manualCamera = false;
let categories = [], page = 0, matched = 0, pageSize = 48;
let scope = ['diagnostics','completed','additional'].includes(new URL(location.href).searchParams.get('scope')) ? new URL(location.href).searchParams.get('scope') : 'all';
let categoryBig = new URL(location.href).searchParams.get('big_task') || '';
let categorySmall = new URL(location.href).searchParams.get('small_task') || '';
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pretty = (value) => String(value || "Unknown").replaceAll("_", " ");
const objectNames = (row) => (row.targets?.length ? row.targets : row.target ? [row.target] : []).map((item) => pretty(item.category)).join(" + ");
const taskName = (row) => row.name || [row.scene, row.sample].filter(Boolean).join(' / ') || row.id;
const taskObjective = (row) => row.objective || row.robot_task_public?.objective || 'No public operation objective is available.';
const originalAnswerText = (answer) => answer == null ? 'Not available in the original ESI source.' : typeof answer === 'string' ? answer : JSON.stringify(answer);
const placeholder = () => '<div class="placeholder"><span class="placeholder-icon">▧</span>No native screenshot available<br><small>Inspect diagnostics and leave repair guidance or notes.</small></div>';
const preferredImage = (images) => images.find(i=>(i.original_view||i.view)==='global_overhead') || images.find(i=>(i.original_view||i.view)==='global_diagonal') || images[0];

// Composite public metadata renderers.
function categoryLabels(row) {
  return [...new Set([row.subcategory,...(row.category_tags||[]).map(tag=>tag.small_task)].filter(Boolean))].join(' · ');
}
function compositeBadge(row) {
  return row.composite?.member_count>1?`<span class="badge composite-badge">${esc(row.composite.member_count)} goals</span>`:'';
}
function versionRole(version, group) {
  return version?.id===group?.original_variant_id||['original','original_baseline'].includes(version?.role)?'Original':'Draft';
}
function versionsBadge(row) {
  const group=row.core_group;
  if(!group?.versions?.length) return '';
  const displayed=group.versions.find(version=>version.id===group.default_variant_id);
  return `<span class="badge version-badge">${group.versions.length} ${group.versions.length===1?'version':'versions'} · default ${versionRole(displayed,group).toLowerCase()}</span>`;
}
function renderVersionSelector(row) {
  const group=row.core_group,versions=group?.versions||[];
  $('version-panel').classList.toggle('hidden',!versions.length);
  $('task-version').innerHTML=versions.map(version=>`<option value="${esc(version.id)}">${esc(versionRole(version,group))} · ${esc(version.label||version.id)} · ${esc(pretty(version.capture_state||'unavailable'))}</option>`).join('');
  $('task-version').value=group?.active_variant_id||row.id;
  $('version-context').textContent=group?'Reviewing this version only. Images and notes stay with this version; the default display is provisional.':'';
}
function updateSummaryReview(row) {
  const parentId=row.core_group?.id||row.id,summary=scenes.find(item=>item.id===parentId);
  if(!summary) return;
  if(summary.core_group) {
    summary.core_group.versions=(summary.core_group.versions||[]).map(version=>version.id===row.id
      ?{...version,decision:row.decision,review:{...row.review}}:version);
    // A review of the original must never become the default draft's review.
    if(summary.core_group.default_variant_id!==row.id) return;
  }
  summary.decision=row.decision;summary.review={...row.review};summary.review_stale=row.review_stale;
}
function previousReviewBadge(row) {
  const summary=row.source_review_summary;
  if(!summary?.reviewed_goals) return '';
  return `<span class="badge prior-review-badge">${summary.is_merged?'Merged':'Revised'} · ${esc(summary.reviewed_goals)} previously reviewed</span>`
    +(summary.needs_fix?`<span class="badge warning">Prior repair feedback · ${esc(summary.needs_fix)}</span>`:'');
}
function previousReviewMarkup(goal) {
  const review=goal.previous_review;
  if(!review||!review.decision||review.decision==='unreviewed') return '';
  const decision={approved:'Approved',needs_fix:'Needs repair',skipped:'Skipped'}[review.decision]||review.decision;
  return `<aside class="previous-review ${review.decision==='needs_fix'?'prior-needs-fix':''}"><strong>Previously reviewed source · ${esc(decision)}</strong>${review.notes?`<p>${esc(review.notes)}</p>`:''}<small>${esc(review.candidate_id)} · revision ${esc(review.revision??0)}</small><p>Historical result applies to the original task. Review this version separately.</p></aside>`;
}
function executionProgressMarkup(row) {
  const progress=row.progress;
  if(progress?.status!=='measured') return '<strong>Not executed</strong><span>Screenshots record the layout; they do not score task completion.</span>';
  const score=progress.completion_score;
  const fraction=Number.isInteger(progress.completed_goals)&&Number.isInteger(progress.total_goals)
    ?`${progress.completed_goals} / ${progress.total_goals} goals completed`:'';
  return `<small>${esc(progress.run_mode||'')} · ${esc(progress.evaluation_validity||'complete')} · ${esc(progress.scoring_version||'legacy')}</small><strong>${typeof score==='number'&&Number.isFinite(score)?`${(score*100).toFixed(1)}% completion`:'Measured progress unavailable'}</strong><span>${esc(fraction)}</span>${typeof progress.diagnostic_lower_bound==='number'&&score===null?`<span>Diagnostic lower bound: ${(100*progress.diagnostic_lower_bound).toFixed(1)}% (not a complete comparable score)</span>`:''}`;
}
function previousCaptureMarkup(goal) {
  const previous=goal.previous_capture;
  if(!previous) return '';
  return `<aside class="previous-capture"><strong>Previous capture: ${esc([previous.state,previous.failure_stage].filter(Boolean).join(' · ')||'Recorded diagnostic')}</strong>${previous.error?`<p>${esc(previous.error)}</p>`:''}<small>${esc([previous.candidate_id,previous.job_id].filter(Boolean).join(' · '))}</small></aside>`;
}
function repairMarkup(row) {
  const repair=row.task_repair, conflict=row.source_conflict_review;
  return (repair?`<article class="composite-goal"><strong>Task repair · v${esc(repair.previous_version)} → v${esc(repair.current_version)}</strong><p>Validation: ${esc(repair.native_validation)}. Existing images: historical layout evidence only.</p><details><summary>Previous public goal</summary><pre>${esc(JSON.stringify(repair.old_public,null,2))}</pre></details><details><summary>Current public goal</summary><pre>${esc(JSON.stringify(repair.new_public,null,2))}</pre></details></article>`:'')
    +(row.background_movements?.length?`<article class="composite-goal"><strong>Measured background movements</strong><pre>${esc(JSON.stringify(row.background_movements,null,2))}</pre></article>`:'')
    +(row.evaluation_failure?.failure_reason?`<article class="composite-goal"><strong>Evaluation unavailable</strong><pre>${esc(JSON.stringify(row.evaluation_failure,null,2))}</pre></article>`:'')
    +(conflict?`<article class="composite-goal"><strong>Source conflict requires review</strong><pre>${esc(JSON.stringify(conflict,null,2))}</pre></article>`:'');
}
function compositeGoalsMarkup(row) {
  return repairMarkup(row) + (row.composite?.public_goals||[]).map((goal,index)=>`<article class="composite-goal"><p class="eyebrow">GOAL ${index+1} · ${esc([goal.big_task,goal.small_task].filter(Boolean).join(' / '))}</p><p class="goal-source">${esc(goal.source)}</p>${previousReviewMarkup(goal)}<p>${esc(goal.objective)}</p><p><strong>Success criteria</strong><br>${esc(goal.success_criteria)}</p>${goal.constraints?.length?`<ul>${goal.constraints.map(item=>`<li>${esc(item)}</li>`).join('')}</ul>`:''}${goal.original_question?`<p><strong>Original question</strong><br>${esc(goal.original_question)}</p>`:''}${Object.hasOwn(goal,'original_esi_answer')?`<p><strong>Original ESI answer</strong><br>${esc(originalAnswerText(goal.original_esi_answer))}</p>`:''}${goal.original_options?.length?`<p class="goal-options">Options: ${goal.original_options.map(option=>esc(option)).join(' · ')}</p>`:''}${previousCaptureMarkup(goal)}</article>`).join('');
}
// End composite public metadata renderers.

function globalState(row) {
  if(row.has_global_preview) return {label:"Global ready",message:"Global views are available."};
  return {label:"Global unavailable",message:row.capture?.error||'No global view is available for this task. Inspect the recorded diagnostics.'};
}

function filtered() {
  return scenes.filter((r) => (!$("flag-filter").checked || r.aperture_warning || r.warning_codes?.length));
}
function badges(row) {
  const complete=row.native_capture_complete??row.has_final_preview;
  return `<span class="badge ${complete?'ready':'warning'}">${complete?'Capture complete':row.instance_status==='capture_pending'?'Capture pending':'Diagnostics'}</span>`
    + (row.core_selection_status ? `<span class="badge">${esc(pretty(row.core_selection_status))}</span>` : '')
    + compositeBadge(row)
    + versionsBadge(row)
    + previousReviewBadge(row)
    + (row.is_category_sample ? '<span class="badge ready">Category sample</span>' : '')
    + (row.placement_state==='needs_repair' ? '<span class="badge warning">Placement needs repair</span>' : '')
    + (row.aperture_warning ? '<span class="badge warning">Size flag</span>' : "")
    + (row.review_stale ? '<span class="badge stale">Layout changed</span>' : "")
    + (scope==='diagnostics'&&row.instance_status ? `<span class="badge stale">${esc(row.instance_status.replaceAll('_',' '))}</span>` : '')
    + (row.warning_codes?.length ? `<span class="badge warning">${row.warning_codes.includes('assumption_review_required')?'Assumption needs review':`${row.warning_codes.length} warnings`}</span>` : '');
}
function renderCategories() {
  const bigTasks = [...new Set(categories.map(c=>c.big_task))];
  $('category-tree').innerHTML = bigTasks.map(big=>{
    const groups=categories.filter(c=>c.big_task===big);
    return `<details ${!categoryBig||categoryBig===big?'open':''}><summary>${esc(big)}</summary>${groups.map(c=>{
      const n=c.counts||{},active=c.big_task===categoryBig&&c.small_task===categorySmall;
      const count=(scope==='all')?(n.captured||0)+(n.failed||0):scope==='completed'?n.captured:n.failed;
      return `<button class="category-item ${active?'active':''}" data-big="${esc(c.big_task)}" data-small="${esc(c.small_task)}" ${active?'aria-current="true"':''}><strong>${esc(c.small_task)}</strong><span>${count||0} ${coreGrouped?'Core source entries':scope==='all'?'candidates':scope==='additional'?'additional prototypes':scope==='completed'?'completed':'pending / diagnostic'} · ${c.source_count||0} source tasks</span><span>${n.reviewed||0} reviewed · ${n.approved||0} approved</span></button>`;
    }).join('')}</details>`;
  }).join('');
  const group=categories.find(c=>c.big_task===categoryBig&&c.small_task===categorySmall);
  $('category-heading').innerHTML=group?`<h2>${esc(group.big_task)} / ${esc(group.small_task)}</h2><p>${matched} ${scope==='completed'?'completed tasks':'failed tasks'} · ${group.source_count||0} source tasks</p>`:`<h2>${scope==='completed'?'Completed tasks':'Diagnostics'}</h2><p>${scope==='completed'?'Current-version captures, grouped by ESI Big Task. Select a task to open its images and review.':'Failed tasks are retained here for inspection and repair guidance.'}</p>`;
  if(scope==='all') $('category-heading').innerHTML=`<h2>${group?esc(group.big_task+' / '+group.small_task):'Provisional Core candidates'}</h2><p>${matched} candidates · final selection and freezing are yours.</p>`;
  if(coreGrouped) $('category-heading').innerHTML=`<h2>${group?esc(group.big_task+' / '+group.small_task):'Core tasks'}</h2><p>${matched} Core source entries · expand a task to review its original and draft versions.</p>`;
  if(scope==='additional') $('category-heading').innerHTML=`<h2>${group?esc(group.big_task+' / '+group.small_task):'Additional prototypes'}</h2><p>${matched} additional prototypes · outside the original Core membership.</p>`;
  $('page-label').textContent=`Page ${page+1} / ${Math.max(1,Math.ceil(matched/pageSize))} · ${matched} matches`;
  $('page-prev').disabled=page===0;
  $('page-next').disabled=(page+1)*pageSize>=matched;
}
async function chooseCategory(big,small) {
  try {await persist();} catch {return;}
  categoryBig=big;categorySmall=small;page=0;clearDetail();
  for(const id of ['version-filter','failure-filter','scene-filter']) $(id).value='all';
  const url=new URL(location.href);
  for(const [key,value] of [['big_task',big],['small_task',small]]) value?url.searchParams.set(key,value):url.searchParams.delete(key);
  history.replaceState({},'',url);await reload(true);
}
function renderStats() {
  const reviewed = scenes.filter((r) => r.decision !== "unreviewed").length, approved = scenes.filter((r) => r.decision === "approved").length;
  $("stats").innerHTML = [[matched,coreGrouped?"Core tasks":scope==='additional'?"Additional prototypes":scope==='all'?"Candidates":scope==='completed'?"Completed":"Pending / diagnostic"],[scenes.length,"On this page"],[reviewed,"Reviewed on page"],[approved,"Approved on page"]]
    .map(([number,label],i)=>`<div class="stat ${i===3?'green':''}"><strong>${number}</strong><span>${label}</span></div>`).join("");
}
function renderCollection() {
  const rows = filtered();
  $("result-count").textContent = `${rows.length} ${coreGrouped?'Core source entries':scope==='additional'?'additional prototypes':rows.length===1?'task':'tasks'} on this page · ${rows.filter(r=>r.decision==="unreviewed").length} unreviewed`;
  $("list-count").textContent = rows.length;
  $("empty").classList.toggle("hidden", rows.length > 0 || mode==='detail');
  const groups=[...new Set(rows.map(r=>r.category||'Other tasks'))];
  $("cards").innerHTML = groups.map((big,index)=>`<section class="task-group" aria-labelledby="task-group-${index}"><h3 id="task-group-${index}">${esc(big)}</h3><ul class="task-rows">${rows.filter(r=>(r.category||'Other tasks')===big).map(r=>`<li><button class="task-row" data-id="${esc(r.id)}"><span class="task-row-top"><span class="task-small">${esc(categoryLabels(r))}</span>${compositeBadge(r)}${versionsBadge(r)}${previousReviewBadge(r)}${r.placement_state==='needs_repair'?'<span class="badge warning">Placement needs repair</span>':''}<span class="task-review"><i class="status-mark ${esc(r.decision)}"></i>${esc(labels[r.decision]||r.decision)}</span></span><strong>${esc(taskName(r))}</strong><span class="task-context">${esc([r.scene,pretty(r.room),r.sample].filter(Boolean).join(' · '))}</span><span class="task-objective">${esc(taskObjective(r))}</span><span class="task-open" aria-hidden="true">Inspect →</span></button></li>`).join('')}</ul></section>`).join('');
  $("scene-list").innerHTML = groups.map(big=>`<section class="list-group"><h3>${esc(big)}</h3><ul>${rows.filter(r=>(r.category||'Other tasks')===big).map(r=>`<li><button class="list-item ${r.id===selected?'selected':''}" data-id="${esc(r.id)}" ${r.id===selected?'aria-current="true"':''}><span><small>${esc(categoryLabels(r))} ${compositeBadge(r)}${versionsBadge(r)}${previousReviewBadge(r)}</small><strong>${esc(taskName(r))}</strong><small>${esc([r.scene,r.sample].filter(Boolean).join(' · '))}</small><span class="list-objective">${esc(taskObjective(r))}</span><small><i class="status-mark ${esc(r.decision)}"></i>${esc(labels[r.decision]||r.decision)}</small></span></button></li>`).join('')}</ul></section>`).join('');
  renderStats();
  updatePosition();
}
function current() { return detail&&(detail.id===selected||detail.core_group?.id===selected)?detail:null; }
function neighbors() {
  const rows=filtered(),index=rows.findIndex((r)=>r.id===selected);
  if(index>=0) return {previous:rows[index-1],next:rows[index+1],index,rows};
  return {previous:null,next:null,index,rows};
}
function updatePosition() {
  const {previous,next,index,rows}=neighbors();
  $("position").textContent=index<0?`${rows.length} remaining`:`${index+1} / ${rows.length}`;
  $("previous").disabled=!previous;
  $("next").disabled=!next;
}
function groupImages(row, group) {
  if(group==='diagnostics') return row.images.filter(i=>i.kind!=='native'||i.view.startsWith('diagnostic_'));
  return row.images.filter((i)=>{
    const v=i.original_view||i.view;
    return group==="global"?(v.startsWith("global_")&&!v.startsWith("global_target"))||/^phase_[12]_global_/.test(v)
    :group==="object"?v.startsWith("global_target")||v.startsWith("object_")||/^phase_[12]_object_/.test(v):group==="robot"?["head","wrist","right_wrist"].includes(v)||v.startsWith("chassis_"):v==="scene";
  });
}
function diagnosticsMarkup(row) {
  const evidence=row.images.filter(i=>i.kind!=='native'||i.view.startsWith('diagnostic_'));
  const markup = `<div class="diagnostics-panel"><h3>Diagnostics</h3><p><strong>${esc(row.instance_status||row.conversion_status)}</strong> · ${esc(row.failure_stage||'No recorded failure stage')}</p>${row.no_native_image?'<p class="diagnostic-warning">No native screenshot available. Reference and partial images below are labelled evidence.</p>':''}<pre>${esc(row.failure||row.exception_summary||'No recorded exception.')}</pre><h4>Warnings and reversible assumptions</h4><ul>${(row.warning_codes||[]).map(w=>`<li>${esc(w)}</li>`).join('')}</ul>${(row.assumptions||[]).map(a=>`<pre>${esc(typeof a==='string'?a:JSON.stringify(a,null,2))}</pre>`).join('')}<h4>Candidate changes / original comparison</h4><pre>${esc(JSON.stringify(row.change_summary||[],null,2))}</pre><p>${esc((row.known_limitations||[]).join(' '))}</p><h4>Attempted repairs / next step</h4><pre>${esc(JSON.stringify(row.attempted_repairs||[],null,2))}</pre><p>${esc(row.next_step||'Leave guidance for the category converter.')}</p><h4>Evidence</h4><div class="diagnostic-evidence">${evidence.map(i=>`<a href="${esc(i.url)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(i.url)}" alt="${esc(i.label)}"><span>${esc(i.label)}</span></a>`).join('')}</div>${(row.diagnostic_logs||[]).map((path,index)=>`<a class="diagnostic-log" target="_blank" rel="noopener" href="api/diagnostic-log?id=${encodeURIComponent(row.id)}&index=${index}">${esc(path)} ↗</a>`).join('')}</div>`;
  const prior = row.prior_conversion_failures?.length ? `<h4>Previous task diagnostics</h4><p>Historical CPU conversion and native capture or placement errors are retained separately from the current task state.</p><pre>${esc(JSON.stringify(row.prior_conversion_failures,null,2))}</pre>` : "";
  return markup.replace("<h4>Evidence</h4>",prior+"<h4>Evidence</h4>");
}
function renderVisual(row) {
  if(!manualCamera) cameraGroup=row.has_global_preview?"global":row.images.some(i=>i.view==="head")?"robot":"diagnostics";
  const images=groupImages(row,cameraGroup);
  const image=manualCamera?images.find((i)=>i.view===view)||preferredImage(images):preferredImage(images);
  view=image?.view || null;
  $("camera-groups").innerHTML=[["global","Global"],["object","Object"],["robot","Robot"],["esi","ESI"],['diagnostics','Diagnostics']].map(([group,label])=>{
    const count=groupImages(row,group).length;
    return `<button class="camera-group ${group===cameraGroup?'active':''}" data-group="${group}">${label}<span>${count||(group==="global"?esc(globalState(row).label.replace("Global ","")):"unavailable")}</span></button>`;
  }).join("");
  const missingGlobal=cameraGroup==="global";
  $("hero").innerHTML=image?`<img src="${esc(image.url)}" alt="${esc(image.label)}"><span class="hero-label">${esc(image.label)} · click to enlarge</span>`:missingGlobal?`<div class="placeholder"><span class="placeholder-icon">▧</span>${esc(globalState(row).label)}<br><small>${esc(globalState(row).message)}${row.images.some(i=>i.view==="head")?" Head and wrist images are available in Robot views.":""}</small></div>`:placeholder();
  $("view-tabs").innerHTML=images.map((i)=>`<button class="view-tab ${i.view===view?'active':''}" data-view="${esc(i.view)}">${esc(i.label)}</button>`).join("");
  $("global-grid").classList.toggle("hidden",cameraGroup!=="global"||!images.length);
  $("global-grid").innerHTML=cameraGroup==="global"?images.map(i=>`<button class="global-tile ${i.view===view?'active':''}" data-global-view="${esc(i.view)}"><img src="${esc(i.url)}" alt="${esc(i.label)}" loading="lazy"><span>${esc(i.label)}</span></button>`).join(""):"";
  const cameraEvidence=row.human_review_cameras;
  const phaseEvidence=view?.startsWith('phase_2_')?cameraEvidence?.phase_2:cameraEvidence?.phase_1||cameraEvidence;
  const cameraView=phaseEvidence?.views?.[view?.replace(/^phase_[12]_/, '')];
  const visibilityWarning=cameraView?.objects_below_64_pixels?.length?` Limited object visibility: ${cameraView.objects_below_64_pixels.join(', ')}.`:'';
  const cutaway=cameraView?.human_cutaway_above_framed_bounds||cameraView?.human_cutaway_before_framed_bounds?' Human cutaway before the framed task objects; mounted robot views are unchanged.':'';
  $("preview-caption").textContent=(row.preview_provenance||'')+(!row.has_global_preview?` ${globalState(row).message}`:"")+visibilityWarning+cutaway;
  if(cameraGroup==='diagnostics') {
    $('hero').innerHTML=diagnosticsMarkup(row);$('hero').classList.add('diagnostic-mode');
    $('view-tabs').innerHTML='';$('global-grid').classList.add('hidden');
  } else $('hero').classList.remove('diagnostic-mode');
}
function renderDecision(row) {
  for (const button of document.querySelectorAll("[data-decision]")) {
    button.classList.toggle("selected",button.dataset.decision===row.decision);
    button.disabled=button.dataset.decision==="approved"&&!canApproveLayout(row);
    button.title=button.dataset.decision==='approved'?"Record your layout approval; capture and execution status are tracked separately.":"";
  }
}
function renderDetail() {
  const row=current();
  if (!row) return;
  $('scene-id').textContent=row.id;
  const editorURL=new URL(location.href);
  editorURL.port='8615';editorURL.pathname='/';editorURL.search='';editorURL.hash='';
  editorURL.searchParams.set('scene_id',row.id);
  $('open-native-editor').href=editorURL.href;
  document.querySelector('.live-panel')?.classList.toggle('hidden',!!(row.core_group||row.task_edit2_capture_campaign||row.original_baseline));
  $("detail-title").textContent=taskName(row);
  $("detail-room").textContent=[row.category,row.subcategory,row.scene,pretty(row.room)].filter(Boolean).join(' / ');
  $("detail-badges").innerHTML=badges(row);
  renderVersionSelector(row); renderVisual(row); renderDecision(row); updatePosition();
  $("question").textContent=row.question;
  $('original-esi-answer').textContent=originalAnswerText(row.original_esi_answer);
  const operation=row.robot_task_public;
  $('robot-operation-panel').classList.remove('hidden');
  $('robot-objective').textContent=taskObjective(row);
  $('robot-success').textContent=operation?.success_criteria||'Review the recorded failure or guidance request.';
  $('composite-goals-panel').classList.toggle('hidden',!row.composite&&!row.task_repair&&!row.source_conflict_review);
  $('composite-goals').innerHTML=compositeGoalsMarkup(row);
  $('execution-progress-panel').classList.toggle('hidden',!row.composite&&!row.progress);
  $('execution-progress').innerHTML=executionProgressMarkup(row);
  $('review-checks-panel').classList.toggle('hidden',!row.review_goals?.length);
  $('review-checks').innerHTML=(row.review_goals||[]).map((goal,index)=>`<label class="review-check"><input type="checkbox" data-review-check="${index}" ${row.review_checks?.[String(index)]===true?'checked':''}><span>${esc(goal)}</span></label>`).join('');
  const targets=row.targets?.length?row.targets:row.candidate_id&&!row.physical_task_id?[]:[row.target];
  $("target").innerHTML=targets.map((target,index)=>{
    const location=row.target_locations?.[index]||(index===0?row.target_location:null);
    return `<div class="object-entry"><strong>${esc(pretty(target.category))}</strong><span class="object-sub">${esc(target.name||target.binding)} · ${esc(target.model)}</span><span class="object-sub">Original scale ${esc(JSON.stringify(target.original_scale||target.scale))}</span>`
      +(location?`<span class="object-sub">Support: ${esc(location.support)}</span><span class="object-sub">Position (m): ${esc(location.position_m.map(v=>v.toFixed(2)).join(", "))}</span>`:"")+"</div>";
  }).join("")||(operation?'Original task objects are retained; inspect the operation objective.':row.candidate_id?'This candidate is retained for scene diagnosis.':'');
  $("containers").innerHTML=row.containers.map(c=>`<div class="object-entry"><strong>${esc(pretty(c.category))}</strong><span class="object-sub">${esc(c.name)} · ${esc(c.model)}</span></div>`).join("");
  $("notes").value=row.review.notes || "";
  $("failure").classList.toggle("hidden",!row.failure);
  $("failure").textContent=row.failure;
  $("copy-command").disabled=row.conversion_status!=="ready"||!row.viewer_command;
  const capture=row.capture||{},gpu=capture.gpu||{};
  const entries=[["Category",`${row.category||""} / ${row.subcategory||""}`],["Source",row.source],["Candidate",row.candidate_id],["Scene file",row.scene_file],["Task file",row.task_file],["Reuse",pretty(row.reuse_level)],["Support",row.support],["Verification",row.verification_level],["Worker",capture.worker],["GPU",gpu.uuid],["Capture duration",capture.duration_s==null?null:`${capture.duration_s.toFixed(1)} s`],["Capture attempts",capture.attempts],["Fingerprint",row.fingerprint],["Diagnostic evidence",row.diagnostic_dir]];
  $("provenance").innerHTML=`<dl>${entries.map(([k,v])=>`<dt>${k}</dt><dd>${esc(v??"—")}</dd>`).join("")}</dl>`
    +(row.review_history?.length?`<p>Previous feedback</p>${row.review_history.map(r=>`<p>${esc(labels[r.decision]||r.decision)} · ${esc(r.notes)}</p>`).join("")}`:"");
}
function setMode(next) {
  mode=next;
  $("overview").classList.toggle("hidden",mode!=="overview");
  $("inspection").classList.toggle("hidden",mode!=="detail"||!selected);
  $("grid-mode").classList.toggle("active",mode==="overview");
  $("detail-mode").classList.toggle("active",mode==="detail");
  $("detail-mode").disabled=!selected;
  $("empty").classList.toggle("hidden",filtered().length>0||mode==='detail');
}
function clearDetail() {
  detailGeneration++;detailController?.abort();detailController=null;
  selected=null;selectedVariant=null;detail=null;view=null;
  $('hero').innerHTML='';$('global-grid').innerHTML='';$('view-tabs').innerHTML='';$('camera-groups').innerHTML='';
  $('lightbox').close();$('lightbox').querySelector('img').removeAttribute('src');
  $('detail-content').classList.add('hidden');$('detail-status').classList.add('hidden');
  const url=new URL(location.href);url.searchParams.delete('scene');url.searchParams.delete('variant_id');history.replaceState({},'',url);
  setMode('overview');
}
function error(message) { $("error").textContent=message; $("error").classList.toggle("hidden",!message); }
function saveState(text,failed=false) { $("save-state").textContent=text; $("save-state").classList.toggle("failed",failed); }

async function fetchJSON(url,options={}) {
  const {timeout=15000,controller=new AbortController(),...request}=options;
  let timedOut=false;
  const timer=setTimeout(()=>{timedOut=true;controller.abort();},timeout);
  try {
    const response=await fetch(url,{cache:'no-store',...request,signal:controller.signal});
    const value=await response.json();
    if(!response.ok) throw new Error(value.error||`Request failed (${response.status}).`);
    return value;
  } catch(e) {
    if(timedOut) throw new Error('The request timed out. Please try again.');
    throw e;
  } finally {clearTimeout(timer);}
}

async function persist() {
  clearTimeout(saveTimer);
  if (saving) { await saving; if(dirty) return persist(); return; }
  if (!dirty) return;
  const row=current(),version=generation;
  if(!row) return;
  const payload={decision:row.decision,notes:row.review.notes||"",fingerprint:row.fingerprint,revision:row.review.revision||0,review_checks:row.review_checks||{}};
  saveState("Saving…");
  saving=(async()=>{
    const value=await fetchJSON(`api/reviews/${encodeURIComponent(row.id)}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    row.review.revision=value.review.revision;row.review.updated_at=value.review.updated_at;row.review.fingerprint=value.review.fingerprint;
    row.review_stale=false;
    updateSummaryReview(row);renderCollection();
    if(generation===version) dirty=false;
    saveState(dirty?"Unsaved changes":"Saved to disk"); error("");
  })();
  try { await saving; }
  catch(e) {saveState("Save failed",true);error(e.message+" Your edits remain on this page. Refresh only after preserving them.");throw e;}
  finally {saving=null;}
  if(dirty) return persist();
}
function scheduleSave() {dirty=true;generation++;saveState("Unsaved changes");clearTimeout(saveTimer);saveTimer=setTimeout(()=>persist().catch(()=>{}),650);}
function sceneURL(id,variantId) {
  const url=new URL(location.href);url.searchParams.set('scene',id);
  variantId?url.searchParams.set('variant_id',variantId):url.searchParams.delete('variant_id');
  history.replaceState({},'',url);
}
async function selectScene(id,variantId=null) {
  if(!id) return;
  try {await persist();} catch {return;}
  const requestedVariant=variantId||(id===selected?selectedVariant:null)
    ||scenes.find(row=>row.id===id)?.core_group?.default_variant_id||null;
  if(id===selected&&current()&&(!requestedVariant||requestedVariant===current().id)) {setMode('detail');return;}
  const version=++detailGeneration;
  detailController?.abort();const controller=detailController=new AbortController();
  selected=id;selectedVariant=requestedVariant;
  detail=null;
  manualCamera=false;cameraGroup="global";view=null;
  sceneURL(id,requestedVariant);
  $('hero').innerHTML='';$('global-grid').innerHTML='';
  $('lightbox').close();$('lightbox').querySelector('img').removeAttribute('src');
  $('detail-content').classList.add('hidden');$('detail-status').classList.remove('hidden');
  $('detail-status').textContent='Opening task…';
  renderCollection();setMode("detail");
  try {
    const params=new URLSearchParams({id});if(requestedVariant) params.set('variant_id',requestedVariant);
    const data=await fetchJSON(`api/scene?${params}`,{timeout:45000,controller});
    if(version!==detailGeneration) return;
    if(!data.scene) throw new Error('This task is unavailable.');
    detail={...scenes.find(r=>r.id===id),...data.scene};detail.images=detail.images||[];detail.containers=detail.containers||[];
    detail.review=detail.review||{notes:'',revision:0};
    if(detail.core_group) {
      selected=detail.core_group.id;selectedVariant=detail.core_group.active_variant_id||detail.id;
      sceneURL(selected,selectedVariant);
    } else {
      selected=detail.id;selectedVariant=null;sceneURL(selected,null);
    }
    renderCollection();
    renderDetail();$('detail-status').classList.add('hidden');$('detail-content').classList.remove('hidden');
  } catch(e) {
    if(version!==detailGeneration||e.name==='AbortError') return;
    $('detail-status').innerHTML=`<span>${esc(e.message)}</span><button class="button quiet" id="retry-detail" type="button">Try again</button>`;
  } finally {if(version===detailGeneration) detailController=null;}
}
async function step(direction) {
  const choices=neighbors(),next=direction>0?choices.next:choices.previous;
  if(next) await selectScene(next.id);
}
function canApproveLayout(row) {
  return !!row;
}
function choose(decision) {
  const row=current();if(!row) return;
  if(decision==="approved"&&!canApproveLayout(row)) return;
  row.decision=decision;row.review.decision=decision;scheduleSave();renderDecision(row);
  updateSummaryReview(row);
  renderCollection();
}
async function reload(force=false) {
  if((loading&&!force)||dirty||saving) return;
  const version=++catalogueGeneration;
  catalogueController?.abort();const controller=catalogueController=new AbortController();
  loading=true;
  if(force) {scenes=[];matched=0;renderCollection();saveState('Loading task list…');}
  try {
    const params=new URLSearchParams({scope,page:String(page),page_size:String(pageSize),big_task:categoryBig,small_task:categorySmall,
      converter_version:$('version-filter').value,reuse_level:$('reuse-filter').value,review_status:$('decision-filter').value,
      capture_status:$('conversion-filter').value,failure_stage:$('failure-filter').value,warning:$('warning-filter').value,
      scene_model:$('scene-filter').value,sort:$('sort-filter').value,search:$('search').value.trim()});
    const data=await fetchJSON(`api/catalogue?${params}`,{controller});
    if(version!==catalogueGeneration||dirty||saving) return;
    scenes=data.scenes||[];
    categories=scope==='additional'?[]:(data.categories||[]);matched=data.matched??scenes.length;
    coreGrouped=data.core_grouped===true&&scope!=='additional';coreGroupCount=data.core_group_count??matched;
    updateScope();
    const filterFields=[['version-filter','converter_version'],['failure-filter','failure_stage'],['scene-filter','scene']];
    for(const [id,field] of filterFields) {
      const control=$(id),old=control.value;
      const values=data.filter_options?.[field]||[...new Set(scenes.map(r=>r[field]).filter(Boolean))].sort();
      control.innerHTML=`<option value="all">${esc(control.options[0].textContent)}</option>`+[...new Set([...values,...(old!=='all'?[old]:[])])].map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('');control.value=old;
    }
    const warning=$('warning-filter'),oldWarning=warning.value;
    const warningValues=data.filter_options?.warning_codes||[...new Set(scenes.flatMap(r=>r.warning_codes||[]))].sort();
    warning.innerHTML='<option value="all">All warnings</option><option value="any">Has warning</option>'+[...new Set([...warningValues,...(!['all','any'].includes(oldWarning)?[oldWarning]:[])])].map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('');warning.value=oldWarning;
    renderCategories();
    renderCollection();
    $("review-location").textContent='Reviews save automatically.';
    const refreshing=data.indexing?.status==='refreshing';
    const samples=data.category_samples;
    const status=refreshing?'Updating task list':coreGrouped?`${coreGroupCount} Core source entries · original and draft versions reviewed separately`:scope==='additional'?`${matched} additional prototypes · outside the original Core membership`:samples?.active?`Category samples: ${samples.completed}/${samples.requested} captured · remaining batch on hold`:scope==='all'?`${matched} provisional Core candidates · ${data.capture_queue?.completed??0} rendered`:scope==='completed'?`${matched} completed tasks available`:`${matched} pending captures or diagnostics available`;
    $("capture-progress").innerHTML=`<span class="dot"></span><span>${esc(status)}</span>`;
    $("capture-progress").classList.toggle("complete",!refreshing);
    if(data.indexing?.status==='error') error(data.indexing.error||'The task list could not be updated. Showing the available tasks.');
    else error('');
    saveState("Saved to disk");
  } catch(e) {
    if(version!==catalogueGeneration||e.name==='AbortError') return;
    error(`Could not load the task list. ${e.message}`);saveState("Task list unavailable",true);
    $('capture-progress').textContent='Task list unavailable. Retry or wait for the next update.';
  } finally {if(version===catalogueGeneration){loading=false;catalogueController=null;}}
}

let filterTimer=null;
for (const id of ['search','conversion-filter','decision-filter','version-filter','reuse-filter','failure-filter','warning-filter','scene-filter','sort-filter']) $(id).addEventListener(id==='search'?'input':'change',()=>{
  clearTimeout(filterTimer);filterTimer=setTimeout(async()=>{try{await persist();page=0;clearDetail();await reload(true);}catch{}},id==='search'?250:0);
});
$('flag-filter').addEventListener('change',async()=>{try{await persist();clearDetail();renderCollection();}catch{}});
$('review-scope').addEventListener('change',async()=>{
  const next=$('review-scope').value;
  try{await persist();}catch{$('review-scope').value=scope;return;}
  scope=next;page=0;clearDetail();
  if(scope==='additional') {categoryBig='';categorySmall='';}
  const url=new URL(location.href);scope==='all'?url.searchParams.delete('scope'):url.searchParams.set('scope',scope);history.replaceState({},'',url);
  if(scope==='additional') {url.searchParams.delete('big_task');url.searchParams.delete('small_task');history.replaceState({},'',url);}
  updateScope();await reload(true);
});
$('category-tree').addEventListener('click',e=>{const item=e.target.closest('[data-small]');if(item) chooseCategory(item.dataset.big,item.dataset.small);});
$('all-categories').addEventListener('click',()=>chooseCategory('',''));
for(const [id,delta] of [['page-prev',-1],['page-next',1]]) $(id).addEventListener('click',async()=>{try{await persist();page=Math.max(0,page+delta);clearDetail();await reload(true);}catch{}});
for (const id of ["cards","scene-list"]) $(id).addEventListener("click",e=>{const item=e.target.closest("[data-id]");if(item) selectScene(item.dataset.id);});
$("grid-mode").addEventListener("click",async()=>{try{await persist();clearDetail();renderCollection();}catch{}});
$("detail-mode").addEventListener("click",()=>{if(selected) setMode('detail');});
$('detail-status').addEventListener('click',e=>{if(e.target.closest('#retry-detail')) selectScene(selected,selectedVariant);});
$('task-version').addEventListener('change',async()=>{
  const row=current();if(!row) return;
  const parent=row.core_group?.id||selected,variantId=$('task-version').value;
  await selectScene(parent,variantId);
  if(current()) $('task-version').value=current().id;
});
$('retry-catalogue').addEventListener('click',()=>reload(true));
$("previous").addEventListener("click",()=>step(-1));$("next").addEventListener("click",()=>step(1));
for(const button of document.querySelectorAll("[data-decision]")) button.addEventListener("click",()=>choose(button.dataset.decision));
$("notes").addEventListener("input",()=>{const row=current();if(row){row.review.notes=$("notes").value;scheduleSave();}});
$('review-checks').addEventListener('change',event=>{const input=event.target.closest('[data-review-check]'),row=current();if(input&&row){row.review_checks={...(row.review_checks||{}),[input.dataset.reviewCheck]:input.checked};scheduleSave();}});
$("camera-groups").addEventListener("click",e=>{const button=e.target.closest("[data-group]");if(button){manualCamera=true;cameraGroup=button.dataset.group;view=null;renderVisual(current());}});
$("view-tabs").addEventListener("click",e=>{const button=e.target.closest("[data-view]");if(button){manualCamera=true;view=button.dataset.view;renderVisual(current());}});
$("global-grid").addEventListener("click",e=>{const button=e.target.closest("[data-global-view]");if(button){manualCamera=true;view=button.dataset.globalView;renderVisual(current());}});
$("hero").addEventListener("click",()=>{if(cameraGroup==='diagnostics') return;const image=current()?.images.find(i=>i.view===view);if(!image) return;$("lightbox").querySelector("img").src=image.url;$("lightbox").querySelector("p").textContent=`${current().scene} · ${image.label}`;$("lightbox").showModal();});
$("close-lightbox").addEventListener("click",()=>$("lightbox").close());
$("copy-command").addEventListener("click",async()=>{
  try{await navigator.clipboard.writeText(current().viewer_command);$("copy-command").textContent="Copied";setTimeout(()=>$("copy-command").textContent="Copy launch command",1800);}
  catch{error("Clipboard is unavailable. The launch command is available in the scene API; open this page through localhost forwarding to enable copy.");}
});
$('copy-scene-id').addEventListener('click',async()=>{
  const row=current();if(!row) return;
  try {await navigator.clipboard.writeText(row.id);$('copy-scene-id').textContent='Copied';setTimeout(()=>$('copy-scene-id').textContent='Copy ID',1600);}
  catch {error('Copy the Scene ID shown above; clipboard access requires a localhost or secure connection.');}
});
if(!location.pathname.startsWith("/review")) $("gumi-link").href=`${location.protocol}//${location.hostname}:8615/gumi/`;
document.addEventListener("keydown",e=>{
  if(e.target.matches("input,textarea,select")||e.ctrlKey||e.metaKey||e.altKey||$("lightbox").open||mode!=="detail") return;
  if(e.key==="ArrowLeft"){e.preventDefault();step(-1);}if(e.key==="ArrowRight"){e.preventDefault();step(1);}
  const decision={"1":"approved","2":"needs_fix","3":"skipped"}[e.key];if(decision){e.preventDefault();choose(decision);}
});
window.addEventListener("beforeunload",e=>{if(dirty||saving){e.preventDefault();e.returnValue="";}});
function updateScope() {
  $('review-scope').value=scope;
  const options=$('review-scope').options;
  [...options].find(option=>option.value==='all').textContent=coreGrouped?'Core source entries':'All candidates';
  [...options].find(option=>option.value==='additional').hidden=!coreGrouped&&scope!=='additional';
  $('category-nav-title').textContent=coreGrouped?'Core tasks':'ESI tasks';
  $('intro-title').textContent=scope==='completed'?'Review completed tasks.':'Inspect failed tasks.';
  $('intro-description').textContent=scope==='completed'?'Browse task names and robot objectives. Select a task to open its images.':'Browse retained failures. Select a task to inspect its diagnostics and leave repair guidance.';
  if(scope==='all') {
    $('intro-title').textContent='Review the repaired Core candidates.';
    $('intro-description').textContent='Preselected Core samples and the protected Rigid baseline. Inspect the screenshots and decide the final Core selection.';
  }
  if(coreGrouped) {
    $('intro-title').textContent='Review Core tasks and their versions.';
    $('intro-description').textContent='Each entry represents one original Core source. Inspect its original and draft versions separately; default display does not make a final selection.';
  }
  if(scope==='additional') {
    $('intro-title').textContent='Review additional prototypes.';
    $('intro-description').textContent='These prototypes are outside the original Core membership and remain separate from Core source entries.';
  }

  $('conversion-filter').classList.toggle('hidden',scope==='completed');
  $('failure-filter').classList.toggle('hidden',scope==='completed');
  $('sort-filter').options[0].textContent=scope==='completed'?'Unreviewed first':'Unreviewed / failed / warned first';
  if(scope==='completed') {$('conversion-filter').value='all';$('failure-filter').value='all';}
}
updateScope();setMode('overview');
reload().then(()=>{const params=new URL(location.href).searchParams,id=params.get('scene');if(id) selectScene(id,params.get('variant_id'));});
setInterval(reload,12000);
