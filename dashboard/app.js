let events=[],visitors=[],latestStats=null;
const $=id=>document.getElementById(id);

function escapeHTML(v){return String(v??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;")}
function shortTime(v){if(!v)return"—";const m=String(v).match(/(\d{2}:\d{2}:\d{2})/);return m?m[1]:String(v)}
function kind(type){type=String(type||"").toUpperCase();if(type.includes("EXIT"))return"exit";if(type==="ENTRY"||type==="VISITOR_REGISTERED")return"entry";return"other"}
function symbol(type){const k=kind(type);return k==="exit"?"↓":k==="entry"?"↑":"•"}

async function fetchJSON(url){const r=await fetch(url,{cache:"no-store"});if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);return r.json()}

function renderEvents(){
  const list=$("eventList"),table=$("eventTable");if(!list||!table)return;
  if(!events.length){list.innerHTML=`<div class="empty">No events recorded in this session.</div>`;table.innerHTML=`<tr><td colspan="5" class="empty">No events recorded.</td></tr>`;return}
  list.innerHTML=events.slice(0,8).map(e=>{
    const k=kind(e.event_type),v=e.visitor_id!=null?`Visitor ${escapeHTML(e.visitor_id)}`:"Pipeline";
    return `<div class="event-row"><div class="event-icon ${k}">${symbol(e.event_type)}</div><div class="event-info"><strong>${escapeHTML(e.event_type||"EVENT")}</strong><small>${v}</small></div><span class="event-time">${escapeHTML(shortTime(e.timestamp))}</span></div>`
  }).join("");
  table.innerHTML=events.slice(0,100).map(e=>{
    const k=kind(e.event_type),v=e.visitor_id!=null?`Visitor ${escapeHTML(e.visitor_id)}`:"—";
    return `<tr><td><span class="tag ${k}">${escapeHTML(e.event_type||"EVENT")}</span></td><td>${v}</td><td>${escapeHTML(e.timestamp||"—")}</td><td>${e.image_path?"Image saved":"—"}</td><td>${k==="exit"?"Closed":"Recorded"}</td></tr>`
  }).join("");
  $("eventCountLabel").textContent=`${events.length} record${events.length===1?"":"s"}`;
}

function renderVisitors(filter=""){
  const grid=$("visitorGrid");if(!grid)return;
  const arr=visitors.filter(v=>String(v.id).includes(filter));
  if(!arr.length){grid.innerHTML=`<div class="card empty">No visitors found.</div>`;return}
  grid.innerHTML=arr.map(v=>`<article class="visitor-card" data-id="${escapeHTML(v.id)}"><div class="avatar">V${String(v.id).padStart(2,"0")}</div><strong>Visitor ${escapeHTML(v.id)}</strong><small>Registered identity</small><div class="seen">● Last seen ${escapeHTML(v.last_seen||"—")}</div></article>`).join("");
  document.querySelectorAll(".visitor-card").forEach(c=>c.onclick=()=>openVisitor(c.dataset.id));
}

function openVisitor(id){
  const v=visitors.find(x=>String(x.id)===String(id));if(!v)return;
  $("modalTitle").textContent=`Visitor ${v.id}`;$("modalAvatar").textContent=String(v.id).padStart(2,"0");
  document.querySelector("#visitorModal .detail-grid").innerHTML=`<div><span>First seen</span><strong>${escapeHTML(v.first_seen||"—")}</strong></div><div><span>Last seen</span><strong>${escapeHTML(v.last_seen||"—")}</strong></div><div><span>Visitor ID</span><strong>${escapeHTML(v.id)}</strong></div><div><span>Recognition</span><strong>InsightFace / ArcFace</strong></div>`;
  $("visitorModal").classList.add("show");
}

function renderSummary(s){
  const entries=Number(s.entries||0),exits=Number(s.exits||0),active=Number(s.active_visitors||0);
  $("netOccupancy").textContent=Math.max(0,entries-exits);
  $("exitRate").textContent=`${entries?Math.round(exits/entries*100):0}%`;
  $("registeredCount").textContent=visitors.length;
  const vals=[entries,active,exits,Number(s.unique_visitors||0),Math.max(0,entries-exits),entries+exits],max=Math.max(...vals,1);
  $("activityBars").innerHTML=vals.map((v,i)=>`<div class="bar" style="height:${Math.max(7,v/max*58)}%"><span>${i+1}</span></div>`).join("");
}

async function refreshDashboard(){
  try{
    const [stats,ev,vs]=await Promise.all([fetchJSON("/api/stats"),fetchJSON("/api/events"),fetchJSON("/api/visitors")]);
    latestStats=stats;events=ev||[];visitors=vs||[];
    $("uniqueCount").textContent=stats.unique_visitors??0;$("activeCount").textContent=stats.active_visitors??0;$("entryCount").textContent=stats.entries??0;$("exitCount").textContent=stats.exits??0;
    $("trackSummary").textContent=`${stats.active_visitors??0} active track${Number(stats.active_visitors||0)===1?"":"s"}`;
    renderEvents();renderVisitors($("visitorSearch")?.value.trim()||"");renderSummary(stats);
    $("engineState").textContent="Connected";
  }catch(e){console.error("Dashboard API error:",e);$("engineState").textContent="Backend unavailable"}
}

function switchView(view){
  document.querySelectorAll(".view").forEach(v=>v.classList.remove("active-view"));$(view)?.classList.add("active-view");
  document.querySelectorAll(".nav-item").forEach(n=>n.classList.toggle("active",n.dataset.view===view));
  const titles={overview:"Visitor monitoring",visitors:"Visitor directory",events:"Event log",system:"System status"};
  $("pageTitle").textContent=titles[view]||titles.overview;document.querySelector(".crumb").textContent=`Operations / ${titles[view]||titles.overview}`;
}

document.querySelectorAll(".nav-item").forEach(n=>n.onclick=()=>switchView(n.dataset.view));
document.querySelectorAll("[data-view-target]").forEach(b=>b.onclick=()=>switchView(b.dataset.viewTarget));
$("visitorSearch")?.addEventListener("input",e=>renderVisitors(e.target.value.trim()));
$("modalClose")?.addEventListener("click",()=>$("visitorModal").classList.remove("show"));
$("visitorModal")?.addEventListener("click",e=>{if(e.target.id==="visitorModal")e.currentTarget.classList.remove("show")});

$("liveFeed").onload=()=>{$("feedMessage").style.display="none"};
$("liveFeed").onerror=()=>{$("feedMessage").style.display="grid";$("feedMessage").textContent="Camera stream unavailable"};
function updateClock(){$("clock").textContent=new Date().toLocaleTimeString([],{hour12:false})}

updateClock();refreshDashboard();setInterval(updateClock,1000);setInterval(refreshDashboard,2000);
