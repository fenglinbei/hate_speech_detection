'use strict';
// Display-only reader. Scientific inputs, scores and source aggregates are immutable.
const $ = id => document.getElementById(id);
const META_LIMIT = 4, LAYER_LIMIT = 2, cache = new Map(), catalogs = new Map();
const conditionNames = {C0:'无参考材料',D:'仅示例',L:'仅词典释义',LC:'词典释义＋类别',LD:'词典释义＋示例',LDC:'词典释义＋类别＋示例'};
const groupNames = {baseline:'基线',answer:'答案替换',definition:'释义替换',body:'正文 × 答案',order:'同标签换序',dictionary:'释义组合'};
let DATA, directory, mode = 'layers', selectedSpan = null, revision = 0, controller, lastExport;
const average = xs => xs.reduce((a,b) => a+b,0)/xs.length;
const percent = v => (v*100).toFixed(5)+'%';
const signed = v => (v >= 0 ? '+' : '')+v.toFixed(6);
function option(select,value,label) { const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o); }
function label(r) {
  const h=r.hosted;
  if(h.display_label)return h.display_label;
  return `${r.condition} · ${conditionNames[r.condition]} · ${h.item_id ? h.item_id+' '+h.title.replace(/^#\d+ /,'') : '基线'}`+
    (h.module==='body' ? (h.variant==='same'?'（原答案）':'（翻转答案）') : '');
}
function request(id=$('condition').value) { return DATA?.requests.find(r=>r.request_id===id); }
function comparison() { return $('compare').value || ($('compareRole').value ? $('condition').value : ''); }
function touch(key,p) { cache.delete(key);cache.set(key,p);while(cache.size>META_LIMIT)cache.delete(cache.keys().next().value);return p; }
function state() {
  return {request:$('condition').value,round:$('round').value,mode,role:$('role').value,metric:$('metric').value,
    head:+$('head').value,layer:+$('layer').value,compare:comparison(),compareRole:$('compareRole').value||$('role').value,
    parts:$('parts').value,scope:$('scope').value,alignment:$('alignment').value,ceiling:+$('ceiling').value,sort:$('sort').value};
}
async function json(url,signal) {
  const response=await fetch(url,{signal});if(!response.ok)throw Error(`读取失败（HTTP ${response.status}），请重试。`);
  return response.json();
}
async function metadata(r,signal) {
  if(cache.has(r.request_id))return touch(r.request_id,cache.get(r.request_id));
  const p=await json(r.hosted.meta,signal);
  if(p.request_id!==r.request_id || p.prompt_sha256!==r.prompt_sha256 || p.shape.join(',')!==[36,32,6,r.prompt_tokens].join(',') || p.layer_files.length!==36)
    throw Error('结果与输入目录不一致，请刷新页面。');
  p.bySpan=new Map(p.aggregates.map(a=>[a.role+'|'+a.span_id,a]));
  p.meanBySpan=new Map(p.mean_aggregates.map(a=>[a.role+'|'+a.span_id,a]));
  p.layers=new Map();p.mean=null;
  return touch(r.request_id,p);
}
async function binary(r,p,s,signal) {
  if(!r.roles[s.role].length)return;
  const isMean=s.mode==='mean',record=isMean?p.mean_file:p.layer_files[s.layer];
  if(isMean?p.mean:p.layers.has(s.layer))return;
  const url=new URL(record.file,new URL(r.hosted.meta,location.href));
  const response=await fetch(url,{signal});if(!response.ok)throw Error(`注意力文件读取失败（HTTP ${response.status}），请重试。`);
  const buffer=await response.arrayBuffer();
  if(buffer.byteLength!==record.bytes)throw Error('注意力文件大小校验失败。');
  // HTTPS and localhost provide SubtleCrypto; validate transport before using it.
  if(crypto.subtle) {
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),b=>b.toString(16).padStart(2,'0')).join('');
    if(hash!==record.sha256)throw Error('注意力文件摘要校验失败。');
  }
  if(isMean)p.mean=new Float64Array(buffer);
  else {p.layers.set(s.layer,new Float32Array(buffer));while(p.layers.size>LAYER_LIMIT)p.layers.delete(p.layers.keys().next().value);}
}
function status(text,kind='') { $('message').textContent=text;$('message').parentElement.className='loadline '+kind;$('retry').hidden=kind!=='error'; }
function populateConditions(preferred='') {
  const rows=DATA.requests.filter(r=>r.query_id===$('query').value),group=$('group').value;
  const visible=rows.filter(r=>!group||r.hosted.module===group);
  $('condition').replaceChildren();$('compare').replaceChildren();option($('compare'),'','不比较条件');
  visible.forEach(r=>option($('condition'),r.request_id,label(r)));rows.forEach(r=>option($('compare'),r.request_id,label(r)));
  $('condition').value=visible.find(r=>r.request_id===preferred)?.request_id || visible.find(r=>r.condition==='LD'&&r.hosted.module==='baseline')?.request_id || visible[0]?.request_id || '';
  $('compareRole').value='';selectedSpan=null;
}
function populateGroups() {
  $('group').replaceChildren();option($('group'),'','全部类型');
  for(const k of Object.keys(groupNames))if(DATA.requests.some(r=>r.query_id===$('query').value&&r.hosted.module===k))option($('group'),k,groupNames[k]);
  $('group').disabled=$('round').value==='original';
}
function aligned(r,cr,s,snapshot) {
  if(snapshot.alignment!=='slot'||!s.kind.startsWith('demo'))return s.id;
  const slot=s.label.match(/^示例(\d+) /);if(!slot)return s.id;
  return cr.spans.find(x=>x.kind===s.kind&&x.label.startsWith('示例'+slot[1]+' '))?.id || '__missing__';
}
function value(p,span,layer,head,role,metric,isMean) {
  const a=(isMean?p?.meanBySpan:p?.bySpan)?.get(role+'|'+span);
  if(!a?.visible)return null;
  const values=isMean?a[metric]:a[metric][layer];return head<0?average(values):values[head];
}
function difference(r,p,cr,cp,span,layer,head,s) {
  const v=value(p,span.id,layer,head,s.role,s.metric,s.mode==='mean');
  if(!cr)return v;
  const b=value(cp,aligned(r,cr,span,s),layer,head,s.compareRole,s.metric,s.mode==='mean');
  return v===null||b===null?null:v-b;
}
function color(v,max,difference=false) {
  if(v===null)return '#e8edf0';
  const t=Math.min(1,Math.abs(v)/Math.max(max,1e-12)),base=difference&&v<0?[192,84,57]:[8,126,136];
  return `rgb(${base.map(x=>Math.round(246+(x-246)*t)).join(',')})`;
}
function matrix(target,labels,rows,s,onpick,ids=[]) {
  target.replaceChildren();const table=document.createElement('table');table.className='matrix';
  const thead=document.createElement('thead'),tr=document.createElement('tr');tr.append(document.createElement('th'));
  for(const l of labels){const th=document.createElement('th');th.textContent=l;tr.append(th)}thead.append(tr);table.append(thead);
  const body=document.createElement('tbody');
  rows.forEach((values,i)=>{const tr=document.createElement('tr'),th=document.createElement('th');th.textContent='层 '+i;tr.append(th);
    values.forEach((v,j)=>{const td=document.createElement('td');td.style.background=color(v,s.ceiling,!!s.compare);td.dataset.value=v===null?'':v;
      td.dataset.span=ids[j]||'';td.tabIndex=0;td.setAttribute('role','button');
      td.title=`层 ${i} · ${labels[j]}：${v===null?'NA · 未提供 / 不可见 / 未加载':percent(v)+'（原值 '+v.toPrecision(9)+'）'}`;
      td.setAttribute('aria-label',td.title);td.onclick=()=>onpick(i,j);td.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onpick(i,j)}};tr.append(td)});body.append(tr)});
  table.append(body);target.append(table);
}
function bars(target,items,s,onpick) {
  target.replaceChildren();
  for(const x of items){const row=document.createElement('div');row.className='barRow'+(x.value===null?' missing':'')+(x.id===selectedSpan?' selected':'');
    row.dataset.span=x.id;row.dataset.value=x.value===null?'':x.value;row.tabIndex=0;row.setAttribute('role','button');
    const title=document.createElement('span');title.className='barLabel';title.textContent=x.label;
    const track=document.createElement('span');track.className='barTrack';const fill=document.createElement('span');fill.className='barFill';
    if(s.compare){const zero=document.createElement('span');zero.className='barZero';zero.style.left='50%';track.append(zero)}
    if(x.value!==null){const w=Math.min(1,Math.abs(x.value)/s.ceiling)*(s.compare?50:100);fill.style.width=w+'%';
      fill.style.left=(s.compare?(x.value<0?50-w:50):0)+'%';fill.style.backgroundColor=x.value<0&&s.compare?'#c05439':'#087e88';track.append(fill)}
    const num=document.createElement('span');num.className='barValue';num.textContent=x.value===null?'NA':percent(x.value);
    row.title=x.label+'：'+(x.value===null?'未提供 / 不可见 / 未加载':x.value.toPrecision(12))+(Math.abs(x.value)>s.ceiling?'；超过显示范围，条形已截断':'');
    row.setAttribute('aria-label',row.title);row.append(title,track,num);row.onclick=()=>onpick(x);row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onpick(x)}};target.append(row);
  }
}
function prompt(r,p,s) {
  const data=s.mode==='mean'?p?.mean:p?.layers.get(s.layer),T=r.prompt_tokens,ri=DATA.role_order.indexOf(s.role),chars=Array.from(r.prompt_text);
  const weights=new Float64Array(T),cw=new Float64Array(chars.length),count=new Uint32Array(chars.length),maxPos=r.roles[s.role].length?Math.max(...r.roles[s.role]):-1;
  const visibleEnd=maxPos>=0?r.token_offsets[maxPos][1]:0;
  if(data){for(let j=0;j<T;j++){if(s.head<0){for(let h=0;h<32;h++)weights[j]+=data[(h*6+ri)*T+j]/32}else weights[j]=data[(s.head*6+ri)*T+j]}}
  r.token_offsets.forEach(([a,b],j)=>{for(let i=a;i<b;i++){cw[i]+=weights[j];count[i]++}});
  const frag=document.createDocumentFragment();chars.forEach((ch,i)=>{const node=document.createElement('span');node.textContent=ch;
    if(i>=visibleEnd){node.className='unseen';node.title='此读取位置尚不可见'}
    else if(data){const v=count[i]?cw[i]/count[i]:0;node.style.backgroundColor=color(v,s.ceiling);node.title='当前条件所覆盖 token 的权重均值：'+v.toPrecision(8)}
    else node.title='注意力尚未加载';frag.append(node)});$('prompt').replaceChildren(frag);
  $('promptTitle').textContent='完整 prompt · '+(s.mode==='mean'?'36 层平均':'层 '+s.layer)+' · '+(s.head<0?'32 头平均':'头 '+s.head);
}
function render(s=state(),loaded=false) {
  const r=request(s.request);if(!r)return;const p=cache.get(r.request_id),cr=s.compare?request(s.compare):null,cp=cr?cache.get(cr.request_id):null;
  const score=DATA.scores[r.request_id],other=cr?DATA.scores[cr.request_id]:null;
  $('conditionTitle').textContent=label(r);$('score').replaceChildren();
  const prediction=document.createElement('b');prediction.textContent=score.raw_prediction;
  $('score').append('模型：',prediction,document.createElement('br'),`参考：${score.reference} · m = ${signed(score.m)}`);
  $('score').title='m = z(无) − z(有)；正值偏无，负值偏有。';
  $('delta').textContent=other?`Δm = ${signed(score.m-other.m)}（当前 − 对比条件）；数值界限 ≤ ${(score.margin_error_bound+other.margin_error_bound).toPrecision(5)}`:'m 为“无”与“有”的 logit 差，正值偏向“无”。';
  const positions=r.roles[s.role];
  $('geometry').textContent=`${r.prompt_tokens} tokens；读取位置（从 0 起）：${positions.length>8?positions[0]+'…'+positions.at(-1)+'，共 '+positions.length+' 个':positions.join(', ')||'该材料未提供'}`;
  $('scopeNote').textContent=DATA.scope_note||($('round').value==='replacement'?'本轮独立评分，含 88 个输入和 258 个配对读数。':'首轮 12 条件；完整保留原始分数和注意力数据。');
  $('scienceScope').textContent=DATA.science_scope||'仅两个已暴露案例，条件共享端点，不能当成独立确认。';
  $('report').href=DATA.report_url||'reports/';
  $('mechanism').hidden=!DATA.mechanism_url;$('mechanism').href=DATA.mechanism_url||'#';
  $('roundNotes').replaceChildren();if(['hehe','presentation'].includes($('round').value))for(const text of DATA.notes||[]){const p=document.createElement('p');p.textContent=text;$('roundNotes').append(p)}
  $('baseline').disabled=!(r.hosted.baseline&&r.hosted.baseline!==r.request_id);
  $('resetCompare').disabled=!s.compare;$('compareBadge').textContent=s.compare?'已启用差分':'未比较';
  $('layerLabel').hidden=s.mode==='mean';$('sort').disabled=s.mode!=='mean';
  $('meanNote').textContent=s.mode==='mean'?'每条是该片段在 36 层的等权平均。它概括总体分布，可能掩盖某一层的峰值；可切回逐层热图。':'每行是一层，列是原文中的参考片段；点击单元格可查看该片段的头分布及该层的原文着色。';
  $('chartTitle').textContent=s.mode==='mean'?'36 层平均 · 注意力分布':'层 × 参考片段';
  const roleLabel=DATA.roles[s.role];
  $('readout').textContent=`${roleLabel}；${s.head<0?'32 头平均':'头 '+s.head}；${s.metric==='mass'?'片段总质量':'每可见 token 平均'}`+
    (cr?`；当前 − ${label(cr)} / ${DATA.roles[s.compareRole]}`:'')+'。';
  $('scale').textContent=(cr?`差值范围 −${percent(s.ceiling)}…+${percent(s.ceiling)}，橙负、青正`:`显示范围 0…${percent(s.ceiling)}`)+'；超过范围的图形饱和，数值保留。';
  $('chartNote').textContent=(s.scope==='references'?'只显示示例和词典；其余上下文仍在分母内。':'显示全部上下文，跨边界 token 单列。')+
    (s.metric==='density'?'密度分母为各读取位置可见的片段 token 数均值，不能相加。':'同一层级的质量可比较；父片段与子片段重叠，不能跨层级相加。')+
    (cr&&s.compareRole!==s.role?'读取位置对比同时改变目标位置和可见前缀，仅作描述。':'')+
    (cr?'当前独有或对比独有的片段为 NA；列表按当前输入展开。':'');
  const wanted=s.parts==='whole'?['demo','lexicon','system','query','structure','boundary','placeholder']:['demo_text','demo_answer','lexicon_term','lexicon_definition','lexicon_definition_component','lexicon_category','system','query','structure','boundary','placeholder'];
  const spans=r.spans.filter(x=>wanted.includes(x.kind)&&(s.scope==='all'||x.kind.startsWith('demo')||x.kind.startsWith('lexicon')));
  if(!selectedSpan||!spans.some(x=>x.id===selectedSpan))selectedSpan=spans[0]?.id||null;
  if(!spans.length){$('chart').innerHTML='<p class="empty">当前条件没有示例或词典。可在“显示范围”中选择全部上下文。</p>';lastExport=null}
  else if(s.mode==='mean'){
    const items=spans.map(x=>({id:x.id,label:x.label,value:difference(r,p,cr,cp,x,0,s.head,s)}));
    if(s.sort==='attention')items.sort((a,b)=>(b.value===null?-Infinity:Math.abs(b.value))-(a.value===null?-Infinity:Math.abs(a.value)));
    bars($('chart'),items,s,x=>{selectedSpan=x.id;refresh()});lastExport={mode:'mean',items,s};
  }else{
    const rows=Array.from({length:36},(_,l)=>spans.map(x=>difference(r,p,cr,cp,x,l,s.head,s)));
    matrix($('chart'),spans.map(x=>x.label),rows,s,(l,j)=>{selectedSpan=spans[j].id;$('layer').value=l;refresh()},spans.map(x=>x.id));
    lastExport={mode:'layers',labels:spans.map(x=>x.label),rows,s};
  }
  const selected=spans.find(x=>x.id===selectedSpan);
  $('headchart').replaceChildren();$('headtitle').textContent=selected?(s.mode==='mean'?'36 层平均 × 32 头 · ':'层 × 32 头 · ')+selected.label:'片段的注意力头分布';
  if(selected&&s.mode==='mean')bars($('headchart'),Array.from({length:32},(_,h)=>({id:String(h),label:'头 '+h,value:difference(r,p,cr,cp,selected,0,h,s)})),s,x=>{$('head').value=x.id;refresh()});
  else if(selected)matrix($('headchart'),Array.from({length:32},(_,h)=>'头 '+h),Array.from({length:36},(_,l)=>Array.from({length:32},(_,h)=>difference(r,p,cr,cp,selected,l,h,s))),s,(l,h)=>{$('head').value=h;$('layer').value=l;refresh()});
  const chars=Array.from(r.prompt_text);$('materials').replaceChildren();
  for(const x of r.spans.filter(x=>['demo','lexicon','query'].includes(x.kind))){const title=document.createElement('h3');title.textContent=x.label;const text=document.createElement('div');text.textContent=chars.slice(x.char_start,x.char_end).join('');$('materials').append(title,text)}
  prompt(r,p,s);$('export').disabled=!loaded||!lastExport;
  window.viewerState={...s,loaded,selectedSpan,cache:{conditions:cache.size,layerFiles:[...cache.values()].reduce((n,p)=>n+p.layers.size,0),meanFiles:[...cache.values()].filter(p=>p.mean).length}};
}
function saveURL(s) {
  const params=new URLSearchParams();for(const k of ['round','request','mode','role','metric','head','layer','parts','scope','sort','alignment','ceiling'])params.set(k,String(s[k]));
  if(s.compare)params.set('compare',s.compare);if(s.compareRole!==s.role)params.set('compareRole',s.compareRole);
  history.replaceState(null,'','#'+params.toString());
}
async function refresh() {
  if(!DATA)return;const n=++revision;controller?.abort();controller=new AbortController();const signal=controller.signal,s=state(),r=request(s.request);
  if(!r)return;render(s,false);status('正在加载所选条件…','busy');
  try {
    const cr=s.compare&&s.compare!==r.request_id?request(s.compare):null;
    const settled=await Promise.allSettled([metadata(r,signal),...(cr?[metadata(cr,signal)]:[])]);
    for(const result of settled)if(result.status==='rejected')throw result.reason;
    if(n!==revision)return;const p=settled[0].value;render(s,false);await binary(r,p,s,signal);
    if(n!==revision)return;render(s,true);saveURL(s);status('结果已加载 · '+(s.mode==='mean'?'36 层平均':'逐层查看')+' · 本页仅作 CPU 展示');
  }catch(e){if(n!==revision||e.name==='AbortError')return;status(e.message,'error');render(s,false)}
}
async function selectRound(preferred='',params=null) {
  const key=$('round').value,n=++revision;controller?.abort();controller=new AbortController();status('正在读取实验目录…','busy');DATA=null;
  for(const node of document.querySelectorAll('.controls select,.controls button'))if(node.id!=='round')node.disabled=true;
  $('export').disabled=true;window.viewerState={loaded:false,round:key};
  try {
    if(!catalogs.has(key))catalogs.set(key,await json(directory.rounds.find(x=>x.key===key).catalog,controller.signal));
    if(n!==revision)return;DATA=catalogs.get(key);
    const previousQuery=$('query').value,preferredRequest=DATA.requests.find(r=>r.request_id===preferred),queries=[...new Set(DATA.requests.map(r=>r.query_id))];
    $('query').replaceChildren();for(const q of queries)option($('query'),q,DATA.query_labels?.[q]||'#'+q);
    $('query').value=preferredRequest?.query_id||(queries.includes(previousQuery)?previousQuery:queries[0]);
    populateGroups();populateConditions(preferred);
    $('role').replaceChildren();$('compareRole').replaceChildren();option($('compareRole'),'','使用当前读取位置');
    for(const k of DATA.role_order){option($('role'),k,DATA.roles[k]);option($('compareRole'),k,DATA.roles[k])}
    for(const node of document.querySelectorAll('.controls select,.controls button'))node.disabled=false;
    $('group').disabled=key==='original';
    if(params){for(const id of ['role','metric','head','layer','parts','scope','sort','alignment','ceiling','compare','compareRole'])if(params.has(id)&&[...$(id).options].some(o=>o.value===params.get(id)))$(id).value=params.get(id);
      setMode(params.get('mode')==='mean'?'mean':'layers');if(comparison())$('comparisonOptions').open=true;}
    await refresh();
  }catch(e){if(n===revision&&e.name!=='AbortError')status(e.message,'error')}
}
function setMode(next) {
  mode=next;for(const [id,key] of [['tabLayers','layers'],['tabMean','mean']]){$(id).classList.toggle('active',key===mode);$(id).setAttribute('aria-pressed',String(key===mode))}
}
function exportSVG() {
  if(!window.viewerState?.loaded||!lastExport)return;
  const e=lastExport,s=e.s,esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const width=e.mode==='mean'?1000:Math.max(900,170+e.labels.length*48),height=e.mode==='mean'?175+e.items.length*30:900;
  let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="12" fill="#193747">`;
  const heading=[DATA.title+' / '+s.request+' / '+(e.mode==='mean'?'36 层等权平均':'逐层热图'),$('readout').textContent,$('scale').textContent,
    `片段：${s.parts}；范围：${s.scope}；对齐：${s.alignment}；质量不按筛选范围重新归一化`];
  heading.forEach((t,i)=>svg+=`<text x="16" y="${24+i*23}" font-size="${i?10:13}">${esc(t)}</text>`);
  if(e.mode==='mean')e.items.forEach((x,i)=>{const y=125+i*30,start=s.compare?590:330,w=x.value===null?0:Math.min(1,Math.abs(x.value)/s.ceiling)*(s.compare?260:520);
    svg+=`<text x="16" y="${y+14}">${esc(x.label)}</text><rect x="330" y="${y}" width="520" height="20" fill="#f2f6f7"/>`;
    if(s.compare)svg+=`<path d="M590 ${y}v20" stroke="#93a8b3"/>`;
    svg+=`<rect x="${x.value<0&&s.compare?start-w:start}" y="${y}" width="${w}" height="20" fill="${x.value<0&&s.compare?'#c05439':'#087e88'}"><title>${x.value===null?'NA':x.value}</title></rect><text x="980" text-anchor="end" y="${y+14}">${x.value===null?'NA':esc(percent(x.value))}</text>`;
  });
  else{
    e.labels.forEach((label,j)=>svg+=`<text transform="translate(${138+j*48},220) rotate(-62)" font-size="10">${esc(label)}</text>`);
    e.rows.forEach((row,i)=>{svg+=`<text x="75" y="${249+i*18}">层 ${i}</text>`;row.forEach((v,j)=>svg+=`<rect x="${130+j*48}" y="${236+i*18}" width="46" height="16" fill="${color(v,s.ceiling,!!s.compare)}"><title>${v===null?'NA':v}</title></rect>`)});
  }
  svg+='</g></svg>';const a=document.createElement('a'),url=URL.createObjectURL(new Blob([svg],{type:'image/svg+xml'}));a.href=url;a.download=s.request+'-'+s.mode+'-'+s.role+'.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
for(const id of ['condition','role','metric','head','layer','parts','scope','compare','compareRole','alignment','ceiling','sort'])$(id).onchange=refresh;
$('query').onchange=()=>{populateGroups();populateConditions();refresh()};
$('group').onchange=()=>{populateConditions($('condition').value);refresh()};
$('round').onchange=()=>selectRound();
$('tabLayers').onclick=()=>{setMode('layers');refresh()};$('tabMean').onclick=()=>{setMode('mean');refresh()};
$('baseline').onclick=()=>{$('compare').value=request().hosted.baseline;$('compareRole').value='';refresh()};
$('resetCompare').onclick=()=>{$('compare').value='';$('compareRole').value='';refresh()};
$('retry').onclick=()=>DATA?refresh():initialize();$('export').onclick=exportSVG;
$('copyPrompt').onclick=async()=>{try{await navigator.clipboard.writeText(request().prompt_text);status('完整 prompt 已复制。')}catch{status('无法访问剪贴板；可在下方原文中选择并复制。')}};
option($('head'),'-1','全部头平均');for(let h=0;h<32;h++)option($('head'),h,'头 '+h);
for(let l=0;l<36;l++)option($('layer'),l,'层 '+l);$('layer').value='29';
async function initialize() {
  try {
    const params=new URLSearchParams(location.hash.slice(1));directory=await json('catalog.json');$('round').replaceChildren();
    for(const r of directory.rounds)option($('round'),r.key,r.label);$('round').disabled=false;
    $('round').value=directory.rounds.some(r=>r.key===params.get('round'))?params.get('round'):directory.default_round;
    const preferred=params.get('request')||'';
    await selectRound(preferred,params);
  }catch(e){status(e.message,'error')}
}
initialize();
