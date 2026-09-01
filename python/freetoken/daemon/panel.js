"use strict";
/* FreeToken 控制面板逻辑 */

/* ═══════════ 工具 ═══════════ */
var $ = function(id){ return document.getElementById(id); };
function el(tag, cls, text){
  var e = document.createElement(tag);
  if(cls) e.className = cls;
  if(text !== undefined) e.textContent = text;
  return e;
}
async function api(method, path, body){
  try {
  var opt = { method: method, headers: { "Content-Type": "application/json" } };
  if(body !== undefined) opt.body = JSON.stringify(body);
  var res = await fetch(path, opt);
  var data = null;
  try { data = await res.json(); } catch(e) {}
  if(!res.ok) throw new Error((data && (data.detail || data.error)) || ("HTTP " + res.status));
  return data;
  } catch(e){ throw new Error("无法连接到服务端 (" + e.message + ")"); }
}
function fmtGB(bytes){
  if(bytes === undefined || bytes === null || isNaN(bytes)) return "?";
  if(bytes === 0) return "0.0";
  return (bytes / 1073741824).toFixed(1);
}

function getTagClass(tag){
  if(tag === "MoE" || tag === "稀疏") return "tag-moe";
  if(tag === "稠密") return "tag-dense";
  if(tag === "NVFP4量化") return "pill-nv";
  if(tag === "FP8量化" || tag === "MXFP4量化") return "pill-fp";
  return "tagpill";
}
function toast(msg, ok){
  var t = el("div", "toast", msg);
  t.style.cssText = "position:fixed;top:18px;right:18px;padding:10px 18px;border-radius:8px;font-size:13px;z-index:9999;background:" + (ok === false ? "#b62324" : "#238636") + ";color:#fff;box-shadow:0 4px 16px rgba(0,0,0,.4)";
  document.body.appendChild(t);
  setTimeout(function(){ t.remove(); }, 2600);
}

/* ═══════════ 设置（本地持久化） ═══════════ */
var SET_KEY = "ft.settings.v1";
var settings = {
  sHost:"127.0.0.1", sPort:1919, sDaemonPort:1900, sAutostart:false, sCors:"*",
  sConcurrency:32, sCpuThreads:8, sVramBudget:0, sMoeBackend:"offload",
  sDenseEngine:"igpu", sIgpuService:"", sIgpuNoFallback:false,
              sKvDevice:"cpu", sKvQuant:"bf16", sCtFp8:"native", sMtp:false, sMtpK:2, sMtpIgpuFc:false, sMtpIgpuVerifyGraph:false,
  sZoom:"1", sTheme:"light", sLang:"zh", sModelDir:"", sMirror:"", sHfToken:"",
  /* 高级引擎参数（「高级引擎参数」卡片）：空/默认值不 push，用引擎默认 */
  sMoeCacheSize:256, sMoeCpuThreads:0, sNumTokens:"", sMemoryRatio:0.95,
  sMaxRunningReq:4, sKvReserveTokens:1024, sDisableMoePrefillOverlap:true,
};
function loadSettings(){
  /* 跨 session 读取：优先 pywebview.js_api (落 JSON 文件), fallback localStorage。
     注意：pywebview 的 js_api 方法返回 Promise，必须 await，否则拿到的是
     Promise 对象（typeof "object" 但不含任何设置键），设置会回退到默认值。 */
  var loaded = _loadPersistedSettings();
  if(loaded && loaded.then) return loaded.then(function(data){
    if(data) try { Object.assign(settings, data); } catch(e){}
  });
  if(loaded) try { Object.assign(settings, loaded); } catch(e){}
  return Promise.resolve();
}
function _persistSettings(obj){
  /* 跨 session 稳定持久：优先 pywebview.js_api 落 JSON 文件，fallback localStorage。
     若 bridge 尚未注入，先等最多 3s（与读取路径对称），再决定落盘目标。 */
  var payload = JSON.stringify(obj);
  var writeLocal = function(){ try { localStorage.setItem(SET_KEY, payload); return true; } catch(_e2){ return false; } };
  return _waitForPywebview(3000).then(function(ready){
    if(!ready) return writeLocal();
    try {
      if(window.pywebview && window.pywebview.api && window.pywebview.api.set_settings){
        return window.pywebview.api.set_settings(JSON.parse(payload))
          .then(function(ok){ if(ok === false) return writeLocal(); return true; })
          .catch(function(){ return writeLocal(); });
      }
    } catch(_e){}
    return writeLocal();
  });
}
function _waitForPywebview(timeoutMs){
  return new Promise(function(resolve){
    var t0 = Date.now();
    (function poll(){
      try {
        if(window.pywebview && window.pywebview.api && window.pywebview.api.get_settings) return resolve(true);
      } catch(_e){}
      if(Date.now() - t0 >= (timeoutMs || 3000)) return resolve(false);
      setTimeout(poll, 50);
    })();
  });
}
function _loadPersistedSettings(){
  /* pywebview js_api 在页面加载后异步注入；脚本启动时往往尚不存在。
     先等待注入（最多 3s），再走 js_api 读取；超时才退回 localStorage。 */
  return _waitForPywebview(3000).then(function(ready){
    if(ready){
      try { return Promise.resolve(window.pywebview.api.get_settings()); } catch(_e){}
    }
    try { return JSON.parse(localStorage.getItem(SET_KEY) || "{}"); } catch(_e){ return {}; }
  });
}
function saveSettings(){
  var ids = Object.keys(settings);
  for(var i=0;i<ids.length;i++){
    var node = $(ids[i]);
    if(!node) continue;
    settings[ids[i]] = node.type === "checkbox" ? node.checked : node.value;
  }
  /* 优先写到 pywebview.js_api（持久到磁盘 JSON）；fallback 到 localStorage。
     js_api 返回 Promise（恒 truthy），走 _persistSettings 统一处理：
     Promise resolve 为 false 或 reject 时兜底写 localStorage。 */
  var written = false;
  try {
    if(window.pywebview && window.pywebview.api && window.pywebview.api.set_settings){
      _persistSettings(settings); written = true;
    }
  } catch(e){}
  if(!written){
    try { _persistSettings(settings); written = true; } catch(e){}
  }
  applyZoom();
  /* 模型存储目录同步给 daemon（下载与扫描共用） */
  var mdNode = $("sModelDir");
  if(mdNode){
    var mNode = $("sMirror");
    api("POST", "/settings/model-dir", { path: mdNode.value, mirror: mNode ? mNode.value : "" })
      .then(function(r){ $("sModelDir").value = r.path || ""; })
      .catch(function(){ /* 目录同步失败不打断设置流程 */ });
  }
}
function loadModelDirFromDaemon(){
  api("GET", "/settings/model-dir")
    .then(function(r){
      var node = $("sModelDir");
      if(node && r.path) node.value = r.path;
      /* 镜像偏好以服务端为准回填，并写回本地缓存（用户需求②） */
      var mNode = $("sMirror");
      if(mNode && typeof r.mirror === "string" && r.mirror !== ""){
        mNode.value = r.mirror;
        settings.sMirror = r.mirror;
        try{ _persistSettings(settings); }catch(_e){}
      }
    })
    .catch(function(){});
}
function fillSettingsForm(){
  Object.keys(settings).forEach(function(id){
    var node = $(id); if(!node) return;
    if(node.type === "checkbox") node.checked = !!settings[id];
    else node.value = settings[id];
  });
  applyZoom();
  /* ── 自动持久化 ──：所有 #page-settings 下的表单控件在 change/input 时自动保存到
     localStorage（debounced 300ms），不再依赖「保存设置」按钮的点击。新增设置字段
     只要在 settings 对象里加了 key 并在 panel.html 里放了同名 id 的 input/select/
     checkbox，就会自动纳入持久化，无需在此处再写一行。 */
  installAutoSave();
}
var _autoSaveTimer = null;
var _autoSaveInstalling = false;
function installAutoSave(){
  if(_autoSaveInstalling) return;
  _autoSaveInstalling = true;
  var root = $("page-settings");
  if(!root){ _autoSaveInstalling = false; return; }
  var nodes = root.querySelectorAll("input,select,textarea");
  function flushSave(){
    if(_autoSaveTimer){ clearTimeout(_autoSaveTimer); _autoSaveTimer = null; }
    saveSettings();
    var msg = $("saveMsg");
    if(msg){
      msg.textContent = "✓ 已自动保存 " + new Date().toLocaleTimeString();
      setTimeout(function(){ msg.textContent = ""; }, 1500);
    }
  }
  for(var i=0;i<nodes.length;i++){
    var node = nodes[i];
    /* slider/range 实时刷：input + change 都保存；其它控件仅 change 触发，避免输入时反复写盘 */
    var evt = (node.type === "range" || node.type === "number") ? "input" : "change";
    node.addEventListener(evt, function(){
      if(_autoSaveTimer) clearTimeout(_autoSaveTimer);
      _autoSaveTimer = setTimeout(flushSave, 300);
    });
  }
  /* 复选框点击瞬间也保存（用户已确认意图，不必 debounce） */
  root.querySelectorAll("input[type=checkbox]").forEach(function(cb){
    cb.addEventListener("click", flushSave);
  });
}
function applyZoom(){
  var z = parseFloat(settings.sZoom) || 1;
  document.body.style.zoom = z;
}
function applyTheme(){
  var light = settings.sTheme === "light";
  document.body.classList.toggle("light", light);
  var tLabel = $("themeLabel");
  if(tLabel) tLabel.textContent = light ? "深色模式" : "日间模式";
}
/* 启动引擎时把引擎类设置映射成 CLI 参数 */
function buildEngineArgs(id){
  var a = [];
  var moeBe = settings.sMoeBackend || "";
  if(moeBe && moeBe !== "auto") a.push("--moe-backend", moeBe);
  /* ── 高级引擎参数：显式设置才 push；空/默认值不 push，交给引擎自己的默认值 ── */
  /* MoE 显存槽缓存：仅 offload/hybrid 有意义（cpu 模式专家全驻内存，无显存槽） */
  var _mcs = parseInt(settings.sMoeCacheSize, 10);
  if((moeBe === "offload" || moeBe === "hybrid") && _mcs > 0 && _mcs !== 256)
    a.push("--moe-cache-size", String(_mcs));
  /* MoE CPU 解码线程：仅 cpu/hybrid 模式；0=自动（物理核数） */
  var _mct = parseInt(settings.sMoeCpuThreads, 10);
  if((moeBe === "cpu" || moeBe === "hybrid") && _mct > 0)
    a.push("--moe-cpu-threads", String(_mct));
  if(settings.sDenseEngine) a.push("--dense-ffn-engine", settings.sDenseEngine);
  if(settings.sIgpuService) a.push("--igpu-service", settings.sIgpuService);
  if(settings.sIgpuNoFallback) a.push("--igpu-no-fallback");
  if(settings.sKvDevice && settings.sKvDevice !== "cuda") a.push("--kv-device", settings.sKvDevice);
  if(settings.sKvQuant && settings.sKvQuant !== "bf16") a.push("--kv-quant", settings.sKvQuant);
  /* KV 预算 tokens：留空=引擎按剩余显存自动分配 */
  var _nt = parseInt(settings.sNumTokens, 10);
  if(_nt > 0) a.push("--num-tokens", String(_nt));
  if(settings.sMtp) a.push("--mtp");
  if(settings.sMtp && settings.sMtpK !== 3) a.push("--mtp-k", String(+settings.sMtpK));
  if(settings.sMtp && settings.sMtpIgpuFc) a.push("--mtp-igpu-fc");
  /* P0 verify graph: 24-layer Qwen3_5Model forward captured into a CUDA graph on MTP verify batches.
     ~265 launches collapse into 1 dispatch -- the largest single perf win for K>=2.
     Default OFF so the first user click is a no-op (preserves previous behavior); toggling on
     triggers engine restart via the standard reload path. */
  if(settings.sMtp && settings.sMtpIgpuVerifyGraph) a.push("--mtp-igpu-verify-graph");
  if(settings.sCtFp8 && settings.sCtFp8 !== "native") a.push("--ct-fp8", settings.sCtFp8);
  /* 显存占比：滑杆始终显式给出（默认 0.95），钳在 0.5~0.98。
     原 sVramBudget/sConcurrency/sCpuThreads 的换算推送已由「高级引擎参数」卡片接管 */
  var _mr = parseFloat(settings.sMemoryRatio);
  if(!isFinite(_mr)) _mr = 0.95;
  _mr = Math.min(0.98, Math.max(0.5, _mr));
  a.push("--memory-ratio", _mr.toFixed(2));
  /* 单请求最大并发：默认 4 不 push（用引擎默认） */
  var _mrr = parseInt(settings.sMaxRunningReq, 10);
  if(_mrr > 0 && _mrr !== 4) a.push("--max-running-requests", String(_mrr));
  /* 预留 KV tokens：默认 1024 不 push */
  var _krt = parseInt(settings.sKvReserveTokens, 10);
  if(_krt > 0 && _krt !== 1024) a.push("--kv-reserve-tokens", String(_krt));
  /* 禁用 MoE prefill 重叠：布尔 flag，勾选即 push（默认勾选） */
  if(settings.sDisableMoePrefillOverlap) a.push("--disable-moe-prefill-overlap");
  /* MoE 大模型：预算最小方案塞不下会 assert 自杀——压低 KV 底座。
     显存占比已由「显存占比」滑杆显式给出（默认 0.95），不再强制覆盖 */
  var mid = (typeof id==="string"?id:"") || window._ftPending || "";
  if(/(A3B|MoE|Moe|moe|NVFP4|FP4)/.test(mid)){
    if(a.indexOf("--kv-reserve-tokens")<0) a.push("--kv-reserve-tokens","2048");
        /* 256K-capable layout: host KV pool + CPU SDPA attention fallback.
       16 full-attn layers x 64 KB/token = 16 GB at 262k -- VRAM can never hold
       it, and PCIe spill reads lose to host RAM bandwidth at long ctx. */
    if(a.indexOf("--kv-device")<0) a.push("--kv-device","cpu");
  }
  return a;
}

/* ═══════════ 页面切换 ═══════════ */
document.querySelectorAll(".nav-item").forEach(function(btn){
  btn.addEventListener("click", function(){
    document.querySelectorAll(".nav-item").forEach(function(b){ b.classList.remove("active"); });
    btn.classList.add("active");
    var name = btn.dataset.page;
    document.querySelectorAll(".page").forEach(function(p){ p.classList.remove("active"); });
    $("page-" + name).classList.add("active");
    if(name === "models") refreshLocalModels();
    if(name === "console"){ refreshInstalled(); refreshEngineState(); }
    if(name === "chat"){ refreshChatModels(); renderHistory(); }
    if(name === "logs") pollLogs();
  });
});
function setSidebarCollapsed(collapsed){
  var s = $("sidebar");
  if(!s) return;
  var isC = s.classList.contains("collapsed");
  if(isC === collapsed) {} else if(collapsed){ s.classList.add("collapsed"); } else { s.classList.remove("collapsed"); }
  $("collapseBtn").innerHTML = collapsed ? "▶" : "◀ 收起";
  $("collapseBtn").title = collapsed ? "展开侧边栏" : "收起侧边栏";
}
$("collapseBtn").addEventListener("click", function(){ setSidebarCollapsed(!$("sidebar").classList.contains("collapsed")); });
var _autoFolded = false;
window.addEventListener("resize", function(){
  var narrow = window.innerWidth <= 900;
  if(narrow && !_autoFolded){ _autoFolded = true; setSidebarCollapsed(true); }
  else if(!narrow && _autoFolded){ _autoFolded = false; setSidebarCollapsed(false); }
});
$("themeBtn").addEventListener("click", function(){
  settings.sTheme = settings.sTheme === "light" ? "dark" : "light";
  _persistSettings(settings);
  applyTheme();

});

/* 通用 tab 组切换（模型库 / 日志） */
function bindTabs(sel, panePrefix){
  document.querySelectorAll(sel + " .tab").forEach(function(t){
    t.addEventListener("click", function(){
      document.querySelectorAll(sel + " .tab").forEach(function(x){ x.classList.remove("active"); });
      t.classList.add("active");
      var key = t.dataset.tab || t.dataset.logtab;
      document.querySelectorAll("[id^=" + panePrefix + "]").forEach(function(p){ p.hidden = true; });
      var pane = $(panePrefix + "-" + key);
      if(pane) pane.hidden = false;
      if(key === "library") renderBuiltin();
    });
  });
}
bindTabs("#page-models", "tabpane");
bindTabs("#page-logs", "logpane");
document.querySelectorAll("[data-logtab]").forEach(function(t){
  t.addEventListener("click", function(){ pollLogs(); });
});

/* ═══════════ 系统监控 ═══════════ */
var lastSys = {};
function setRing(nodeId, pctId, pct){
  var p = Number(pct);
  if(!isFinite(p)) p = 0;
  p = Math.max(0, Math.min(100, p));
  var ring = $(nodeId);
  if(ring) ring.setAttribute("stroke-dasharray", p + " 100");
  var label = $(pctId);
  if(label) label.textContent = Math.round(p) + "%";
}
async function pollSys(){
  try {
    var d = await api("GET", "/sysinfo");
    if(!d) return;
    lastSys = d;
    var vramT = Number(d.vramTotal) || 0, vramU = Number(d.vramUsed) || 0;
    var memT  = Number(d.memTotal)  || 0, memU  = Number(d.memUsed)  || 0;
    var vp = vramT > 0 ? vramU / vramT * 100 : 0;
    var mp = memT  > 0 ? memU  / memT  * 100 : 0;
    $("monVramBar").style.width = vp + "%";
    $("monMemBar").style.width  = mp + "%";
    $("monVramText").textContent = fmtGB(vramU) + "/" + fmtGB(vramT) + "G";
    $("monMemText").textContent  = fmtGB(memU)  + "/" + fmtGB(memT)  + "G";
    setRing("ringGpu", "ringGpuPct", Number(d.gpuUtil) || 0);
    setRing("ringCpu", "ringCpuPct", Number(d.cpuUtil) || 0);
    setRing("ringVram", "ringVramPct", vp);
    setRing("ringMem",  "ringMemPct",  mp);
  } catch(e){
    /* /sysinfo 偶发失败（daemon 重启、轮询超时）静默：UI 保留上一次值即可 */
  }
}
setInterval(pollSys, 3000);
pollSys();

/* ═══════════ 模型库 ═══════════ */
var BUILTIN = [
  { id:"Qwen/Qwen2.5-0.5B-Instruct", size:"0.5B", desc:"超轻量对话，1GB 显存即可" },
  { id:"Qwen/Qwen2.5-1.5B-Instruct", size:"1.5B", desc:"轻量对话，适合 4GB 显存" },
  { id:"Qwen/Qwen2.5-7B-Instruct", size:"7B", desc:"均衡对话主力，8GB+ 显存" },
  { id:"Qwen/Qwen2.5-14B-Instruct", size:"14B", desc:"更强的推理与写作，12GB+" },
  { id:"Qwen/Qwen2.5-32B-Instruct", size:"32B", desc:"高性能旗舰，24GB+ 或 iGPU 分层" },
  { id:"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", size:"7B", desc:"推理增强蒸馏版" },
  { id:"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", size:"32B", desc:"深度推理旗舰（MoE 友好）" },
  { id:"meta-llama/Llama-3.2-3B-Instruct", size:"3B", desc:"Meta 轻量多语对话" },
  { id:"mistralai/Mistral-7B-Instruct-v0.3", size:"7B", desc:"Mistral 经典指令模型" },
];
function modelRow(opt){
  var row = el("div", "model-row" + (opt.selected ? " selected" : ""));
  row.dataset.modelId = opt.id;
  row.dataset.sizeBytes = opt.sizeBytes || 0;
  row.dataset.source = opt.source || "";
  row.dataset.path = opt.path || "";
  var info = el("div", "model-info");
  info.appendChild(el("div", "model-name", opt.name));
  info.appendChild(el("div", "model-path", opt.path));
  info.appendChild(el("div", "model-size", opt.sizeBytes ? fmtGB(opt.sizeBytes) + " GB" : "—"));
  row.appendChild(info);
  var tags = el("div", "model-tags");
  (opt.tags || []).forEach(function(t){ tags.appendChild(el("span", getTagClass(t), t)); });
  (opt.pills || []).forEach(function(t){
    var pill = el("span", t === "NVFP4量化" ? "tagpill pill-nv" : "tagpill pill-hl", t);
    tags.appendChild(pill);
  });
  row.appendChild(tags);
  if(opt.fit !== false){
    row.appendChild(makeFitBadge(opt.id, opt.sizeBytes));
    /* 本地模型：只显示本地量化；非本地不显示量化下拉 */
    if(opt.source === "local-dir" || opt.source === "hf-cache"){
      var qs = el("select", "quant-sel");
      qs.title = "本地量化版本";
      qs.dataset.loading = "1";
      qs.appendChild(el("option", "", "检测中…"));
      qs.addEventListener("click", function(ev){ ev.stopPropagation(); });
      qs.addEventListener("change", function(){ paintBadges(curBackend()); updateModelSize(row, qs, opt); });
      row.appendChild(qs);
      /* 异步加载本地量化列表 */
      (async function(qsEl, mid){
        try {
          var r = await api("POST", "/models/local-quants", { path: mid });
          var qsList = (r.quants || []);
          qsEl.textContent = "";
          if(qsList.length){
            qsList.forEach(function(q){
              var op = el("option", "", q.label);
              op.value = q.label.toLowerCase();
              op.dataset.ratio = q.ratio;
              qsEl.appendChild(op);
            });
            qsEl.value = qsList[0].label.toLowerCase();
            /* 记录当前量化的 ratio */
            var rt = qsList[0].ratio;
            if(rt && qsEl.closest) var rw = qsEl.closest(".model-row"); if(rw) rw.dataset.curRatio = String(rt);
          } else {
            qsEl.appendChild(el("option", "", "默认"));
          }
          qsEl.dataset.loading = "";
          paintBadges(curBackend());
        } catch(e){};
      })(qs, opt.path || opt.id);
    }
  }
  var acts = el("div", "model-actions");
  (opt.buttons || []).forEach(function(b){
    var btn = el("button", "btn btn-sm" + (b.cls ? " " + b.cls : ""), b.text);
    btn.type = "button";
    btn.addEventListener("click", function(ev){ ev.stopPropagation(); b.onClick(opt.id, row); });
    acts.appendChild(btn);
  });
  row.appendChild(acts);
  if(opt.onSelect) row.addEventListener("click", function(){ opt.onSelect(opt.id); });
  return row;
}
/* ---------- 行内适配判断 ---------- */
var fitCache = {};
var SIZE_HINT = { "0.5B": 1.1e9, "1.5B": 3.2e9, "3B": 6.4e9, "7B": 15e9, "14B": 30e9, "32B": 68e9 };
function bytesForLabel(label, id){
  var M = { "0.5B":1.1e9, "1.5B":3.2e9, "3B":6.4e9, "7B":15e9, "14B":30e9, "20B-A3.6B":42.8e9, "22B":47e9,
            "16B-A2.4B":34.2e9, "30B-A3B":64e9, "32B":68e9, "106B-A12B":227e9 };
  if(M[label]) return M[label];
  var b = sizeHintFor(id);
  if(b) return b;
  var m = (label || "").match(/(\d+(?:\.\d+)?)B/) || (id || "").match(/(\d+(?:\.\d+)?)B/);
  return m ? parseFloat(m[1]) * 2.14e9 : 0;
}
function curBackend(){
  var s = $("estBackend");
  return s ? s.value : "igpu";
}
function sizeHintFor(id){
  var hit = id.match(/-([0-9.]+)(B|M)(?=[-_]|$|I)/);
  if(hit) return parseFloat(hit[1]) * (hit[2] === "B" ? 2.1e9 : 2.1e6);
  return 8e9;
}
function capacityFor(backend){
  var vram = lastSys.vramTotal || 0;
  var dram = lastSys.memTotal || 0;
  if(backend === "gpu") return vram * 0.88;
  return vram + Math.max(0, dram - 6e9) * 0.45;
}
function fitOf(id, bytes){
  var be = curBackend();
  var key = id + "|" + be;
  if(fitCache[key]) return fitCache[key];
  return { ok: (bytes || sizeHintFor(id)) < capacityFor(be) };
}
function makeFitBadge(id, bytes){
  var cached = fitCache[id + "|" + curBackend()];
  var cls = "fit-badge ", txt = "评估中…";
  if(cached){
    var f0 = fitOf(id, bytes);
    cls += cached.ok ? (f0.mode === "layered" ? "warn" : "ok") : "bad";
    txt = cached.ok ? "可运行" : "跑不动";
    /* 推荐量化版本直接亮在徽章上（含搜索结果行） */
    var _best = null;
    (cached.variants||[]).forEach(function(v){ if(v.fit && (!_best || v.totalBytes < _best.totalBytes)) _best = v; });
    if(_best) txt += " · 推荐 " + (_best.label || (_best.key||"").toUpperCase());
  } else { cls += "pending"; }
  var s = el("span", cls, txt);
  s.dataset.fitId = id;
  s.dataset.fitBe = curBackend();
  return s;
}
/* 同步启发式：按 id 中的参数量与各后端容量立即给结论，异步评估随后精化 */
function ftParseParamsB(id){
  var m = String(id||"").match(/(\d+(?:\.\d+)?)\s*B/i);
  return m ? parseFloat(m[1]) : 0;
}
function ftSyncBadge(id){
  try{
    document.querySelectorAll('.fit-badge[data-fit-id]').forEach(function(bd){
      if(bd.dataset.fitId !== id) return;
      if(bd.dataset.ftSynced) return;
      bd.dataset.ftSynced = "1";
      var pb = ftParseParamsB(id);
      if(!pb){ bd.className="fit-badge pending"; bd.textContent="评估中…"; return; }
      var bf16GB = pb * 2, q4GB = pb * 0.62;
      var vram = 8.0, dram = 32.0; /* 兜底容量；精化评估会覆盖 */
      try{ if(typeof capacityFor==="function"){ /* 精确值由后续评估给出 */ } }catch(_e){}
      var txt, cls;
      if(bf16GB <= vram){ cls="fit-badge ok"; txt="可运行 · 推荐 BF16"; }
      else if(q4GB <= vram){ cls="fit-badge ok"; txt="可运行 · 推荐 NVFP4"; }
      else if(bf16GB <= vram + dram){ cls="fit-badge warn"; txt="可运行 · 推荐 FP8（混合）"; }
      else { cls="fit-badge warn"; txt="超大模型 · 建议 Q4＋CPU 卸载"; }
      bd.className = cls; bd.textContent = txt;
    });
  }catch(_e){}
}
/* 量化下拉：把评估出的各档真实大小写进 option，并同步行尺寸显示 */
function ftFillQuantSizes(id){
  document.querySelectorAll('.model-row[data-model-id]').forEach(function(row){
    if(row.dataset.modelId !== id) return;
    var qs = row.querySelector(".quant-sel"); if(!qs) return;
    var be = curBackend();
    var f = fitCache[id + "|" + be];
    if(!f || !f.variants || !f.variants.length) return;
    for(var i=0;i<qs.options.length;i++){
      var op = qs.options[i]; var k = op.value || "bf16";
      var v = f.variants.find(function(x){ return x.key === k; });
      if(v){
        op.dataset.bytes = v.totalBytes || 0;
        op.textContent = (op.value===""?"自动(推荐)":(v.label||k.toUpperCase())) + " · " + fmtGB(v.totalBytes||0);
      }
    }
    row.dataset.variants = JSON.stringify(f.variants.map(function(v){ return { key:v.key, bytes:v.totalBytes||0 }; }));
    row.dataset.bestKey = f.bestKey || "";
    ftApplyQuantSize(row);
  });
}
function ftApplyQuantSize(row){
  try{
    var qs = row.querySelector(".quant-sel"); if(!qs) return;
    var sizeEl = row.querySelector(".model-size");
    var op = qs.options[qs.selectedIndex];
    var bytes = op && parseInt(op.dataset.bytes || "0");
    if(sizeEl && bytes){ sizeEl.textContent = fmtGB(bytes) + " GB"; }
    else if(sizeEl && op && op.value === "" && row.dataset.bestKey){
      var bk = JSON.parse(row.dataset.variants || "[]").find(function(v){ return v.key === row.dataset.bestKey; });
      if(bk && bk.bytes) sizeEl.textContent = fmtGB(bk.bytes) + " GB";
    }
  }catch(_e){}
}
var estQueue = [], estRunning = 0;
function refineFit(id, be){
  if(fitCache[id + "|" + be]){ paintBadges(be); return; }
  estQueue.push([id, be]);
  pumpEst();
}
function pumpEst(){
  while(estRunning < 2 && estQueue.length){
    var job = estQueue.shift();
    estRunning++;
    var estCall = api("POST", "/engine/estimate-quants", { model: job[0], backend: "auto", args: [] });
    var estGuard = new Promise(function(res){ setTimeout(function(){ res({ __timeout:true }); }, 15000); });
    Promise.race([estCall, estGuard])
      .then(function(r){
        if(r && r.__timeout){ r = { variants:[], anyFit:false, bestKey:"", bestLabel:"", timeout:true }; }
        var best = (r.variants || []).find(function(v){ return v.key === r.bestKey; });
        fitCache[job[0] + "|" + job[1]] = {
          ok: !!r.anyFit,
          note: best ? best.note : "",
          variants: r.variants || [],
          bestKey: r.bestKey, bestLabel: r.bestLabel, nvidiaFit: !!r.nvidiaFit,
          bestBackends: best ? (best.backends || []) : [],
        };
      })
      .catch(function(){})
      .then(function(){ estRunning--; paintBadges(job[1]); try{ ftFillQuantSizes(job[0]); }catch(_e){} pumpEst(); });
  }
}
function paintBadges(be){
  document.querySelectorAll(".fit-badge").forEach(function(bd){
    if(bd.dataset.fitBe !== be) return;
    var f = fitCache[bd.dataset.fitId + "|" + be];
    if(!f) return;
    /* 行内选中的量化档优先；无缓存明细则退回整体结论 */
    var row = bd.closest(".model-row");
    var sel = row ? row.querySelector(".quant-sel") : null;
    var v = null;
    if(sel && f.variants && f.variants.length){
      v = f.variants.find(function(x){ return x.key === sel.value; });
    }
    if(f.err || !(f.variants || []).length){
      /* 元数据不可达：按本机容量做多模式粗估，绝不武断显示跑不动 */
      var bytes = parseInt(bd.closest(".model-row").dataset.sizeBytes || "0") || sizeHintFor(bd.dataset.fitId) || 0;
      var BE_CN2 = { gpu:"GPU", hybrid:"混合", igpu:"iGPU", cpu:"CPU" };
      var fits2 = ["gpu","hybrid","igpu","cpu"].filter(function(be){
        var cap = capacityFor(be);
        return bytes > 0 && (bytes * 0.55 < cap || bytes < cap);
      });
      if(bytes > 0 && fits2.length){
        bd.className = "fit-badge warn";
        bd.textContent = "预估可跑" + (fits2.length < 4 ? " · " + fits2.map(function(x){return BE_CN2[x];}).join("/") : "");
        bd.title = "模型信息暂不可达 · 已按本机容量预估";
      } else if(bytes > 0){
        bd.className = "fit-badge pending"; bd.textContent = "待精确评估";
        bd.title = "网络不可达，无法获取模型元数据";
      } else {
        bd.className = "fit-badge pending"; bd.textContent = "待评估";
      }
      return;
    }
    var fitNow = v ? v.fit : f.ok;
    var mode = v ? v.mode : null;
    var label = v ? v.label : f.bestLabel;
    var nv = v ? v.nvidia : false;
    var BE_CN = { gpu:"GPU", hybrid:"混合", igpu:"iGPU", cpu:"CPU" };
    var bes = (v && v.backends && v.backends.length) ? v.backends : (fitNow ? (f.bestBackends || []) : []);
    var besTxt = bes.map(function(b){ return BE_CN[b] || b; }).join("/");
    var cls = fitNow ? (mode === "layered" ? "warn" : "ok") : "bad";
    bd.className = "fit-badge " + cls;
    var suffix = "";
    if(label && label !== "BF16") suffix += " · " + label;
    if(fitNow && besTxt) suffix += " · " + besTxt;
    if(fitNow){
      bd.textContent = mode === "layered" ? "分层可跑" + suffix : "可运行" + suffix;
    } else {
      bd.textContent = f.ok && f.bestLabel ? "可跑 · 需" + f.bestLabel : "跑不动";
    }
    if(fitNow && nv) bd.dataset.nv = "1"; else delete bd.dataset.nv;
    bd.title = v ? v.note : (f.note || bd.title);
  });
}
var FT_BUILD = "2k25-0825-qsel";
try{ var _fb=document.getElementById("buildTag"); if(_fb) _fb.textContent = "build " + FT_BUILD; }catch(_e){}
var selectedModel = "";

function updateModelSize(row, qs, opt){
  var sizeNode = row ? row.querySelector(".model-size") : null;
  if(!sizeNode) return;
  var base = opt.sizeBytes || 0;
  if(!base){ sizeNode.textContent = "未知大小"; return; }
  var curRatio = parseFloat(row.dataset.curRatio || "1.0");
  if(qs && qs.selectedOptions && qs.selectedOptions[0]){
    var selRatio = qs.selectedOptions[0].dataset.ratio;
    if(selRatio){
      var ratio = parseFloat(selRatio);
      if(ratio > 0 && curRatio > 0) sizeNode.textContent = fmtGB(base * ratio / curRatio) + " GB";
    }
  }
}
function bestLocalModel(list){
  /* 真正的本地模型：local-dir 优先，hf-cache 需有完整权重 */
  var real = (list || []).filter(function(m){
    return m.source === "local-dir" || (m.source === "hf-cache" && (m.sizeBytes || 0) > 300 * 1024 * 1024);
  });
  if(!real.length) return null;
  real.sort(function(a, b){ return (b.sizeBytes || 0) - (a.sizeBytes || 0); });
  return real[0];
}
function pickModel(id){
  selectedModel = id;
  document.querySelectorAll(".model-row").forEach(function(r){ r.classList.remove("selected"); });
  /* 简单高亮：按文本匹配 */
  document.querySelectorAll(".model-row").forEach(function(r){
    if(r.textContent.indexOf(id.split("/").pop()) >= 0) r.classList.add("selected");
  });
}
async function refreshLocalModels(){
  var box = $("localModelList");
  var list;
  try { list = await api("GET", "/models"); }
  catch(e){ return; /* 取数失败：保留旧列表 */
  }
  box.textContent = "";
  try {
    if(!list.length){ box.appendChild(el("div", "list-empty", "暂无已下载模型 —— 到「内置模型库」一键下载")); return; }
    list.forEach(function(m){
      try{
        box.appendChild(modelRow({
          id: m.id, name: m.name, path: m.path, sizeBytes: m.sizeBytes, source: m.source,
          tags: [fmtGB(m.sizeBytes || 0) + " GB", m.source || "local"],
          buttons: [
            { text: "启动", cls: "btn-success", onClick: startWithModel },
            { text: "转 NVFP4", cls: "", onClick: convertToNvfp4, title: "把该模型量化转换为 NVFP4（引擎原生 W4A16，体积约 1/2）" },
          ],
          onSelect: pickModel,
        }));
      }catch(_e){}
      refineFit(m.id, curBackend());
    });
    /* 默认选中本地类型：真实本地模型优先（未选过或原选择已失效时） */
    if(!selectedModel || !list.some(function(x){ return x.id === selectedModel; })){
      var bl = bestLocalModel(list);
      if(bl) pickModel(bl.id);
    }
  } catch(e){ /* 渲染失败保留现状，不闪空 */ }
  refreshDownloadJobs();
}
function renderBuiltin(){
  var box = $("builtinList");
  box.textContent = "";
  BUILTIN.forEach(function(m){
    box.appendChild(modelRow({
      id: m.id, name: m.id.split("/")[1], path: m.desc, sizeBytes: SIZE_HINT[m.size],
      tags: [m.size],
      buttons: [{ text: "下载", cls: "btn-primary", onClick: downloadModel }],
    }));

  });
}
async function searchModels(){
  var q = $("searchQ").value.trim();
  if(!q) return;
  var src = $("searchSource").value;
  var box = $("searchResults");
  box.textContent = "";
  box.appendChild(el("div", "list-empty", "搜索中…"));
  try {
    var r = await Promise.race([
      api("GET", "/models/search?q=" + encodeURIComponent(q) + "&source=" + src + "&limit=12"),
      new Promise(function(_res){ setTimeout(function(){ _res({ results:[{ error:"搜索超时——网络/镜像暂时不可达，请稍后重试" }] }); }, 45000); })
    ]);
    box.textContent = "";
    var items = (r && r.results) || [];
    if(items.length && items[0].error){ box.appendChild(el("div", "list-empty", items[0].error)); return; }
    if(!items.length){ box.appendChild(el("div", "list-empty", "未找到结果")); return; }
    items.forEach(function(m){
      var _row = modelRow({
        id: m.id, name: m.title || m.id, path: m.id, sizeBytes: sizeHintFor(m.id),
        tags: ["⬇ " + (m.downloads || 0), "♥ " + (m.likes || 0), m.source || src],
        buttons: [{ text: "下载", cls: "btn-primary", onClick: downloadModel }],
      });
      /* 量化版本下拉卡片（评估完成后自动填真实大小） */
      var _qs = document.createElement("select");
      _qs.className = "quant-sel";
      [["","自动(推荐)"],["bf16","BF16"],["fp8","FP8"],["q4","Q4"],["nvfp4","NVFP4"]].forEach(function(kv){
        var op = document.createElement("option"); op.value = kv[0]; op.textContent = kv[1];
        _qs.appendChild(op);
      });
      _qs.addEventListener("change", function(){ ftApplyQuantSize(_row); });
      var _tg = _row.querySelector(".model-tags");
      if(_tg) _tg.appendChild(_qs);
      box.appendChild(_row);
      ftSyncBadge(m.id);
      refineFit(m.id, curBackend());
    });
  } catch(e){
    box.textContent = "";
    box.appendChild(el("div", "list-empty", "搜索失败：" + e.message));
  }
}
$("searchBtn").addEventListener("click", searchModels);
$("searchQ").addEventListener("keydown", function(e){ if(e.key === "Enter") searchModels(); });
async function downloadModel(id, rowEl){
  /* 量化选择：非 bf16/自动 时先解析官方预量化仓库
     Fix: 解析失败不再静默回退 BF16（用户观感即"只能选 BF16"）。
     行为：成功→用预量化仓库；失败→弹窗告知命中的候选（如果有），让用户二选一。 */
  var key = "", finalId = id, source = ($("searchSource") && $("searchSource").value) || "hf";
  try{
    var qs = rowEl ? rowEl.querySelector(".quant-sel") : null;
    if(qs) key = qs.value || "";
  }catch(_e){}
  var quantResolved = false;
  if(key && key !== "bf16"){
    try{
      toast("正在解析官方 " + key.toUpperCase() + " 版本…", true);
      var rr = await api("POST", "/models/resolve-quant", { model: id, key: key, source: source });
      if(rr && rr.found && rr.id){
        finalId = rr.id; quantResolved = true;
        /* 修：后端 pure===false 时说明探测到的仓库是混合精度（如 NVFP4 名称下含 FP8 层），
           dev tree 引擎加载时会因 Float8 promotion 不支持而失败——必须警告并让用户决定。 */
        if(rr.pure === false){
          var altTxt = (rr.alternatives || []).map(function(a){ return a.id + (a.source ? (" 〔" + a.source + "〕") : ""); }).join("\n  · ");
          var warnMsg = "⚠ " + key.toUpperCase() + " 探测命中 " + finalId + "，但后端分析发现该仓库是混合精度（" +
                        "同时含 FP8 / NVFP4 等多种层量化），dev tree 引擎加载时可能报 'Float8 promotion not supported'。\n\n" +
                        (altTxt ? ("社区提供的纯 " + key.toUpperCase() + " 版本：\n  · " + altTxt + "\n\n") : "") +
                        "点击 [确定] 仍按 " + finalId + " 下载；\n" +
                        "点击 [取消] 重新选择其他精度。";
          if(!confirm(warnMsg)){
            toast("已取消下载", false);
            return;
          }
        }
        toast("已匹配 " + key.toUpperCase() + "：" + rr.id + (rr.source?(" 〔"+rr.source+"〕") : ""), true);
      } else {
        var tried = (rr && rr.tried) ? rr.tried.join(", ") : "";
        var promptMsg = "未找到 " + id + " 的官方 " + key.toUpperCase() + " 预量化仓库。\n\n" +
                        "尝试过：" + (tried || "（无）") + "\n\n" +
                        "点击 [确定] 仍按原版 (" + id + ") 下载；\n" +
                        "点击 [取消] 中止下载以换其他精度。";
        if(!confirm(promptMsg)){
          toast("已取消下载，请重选其他精度", false);
          return;
        }
        toast("未找到 " + key.toUpperCase() + " 版本，按原版权重下载", false);
      }
    }catch(e){
      if(!confirm("解析 " + key.toUpperCase() + " 仓库失败（" + e.message + "）。\n\n点击 [确定] 按原版下载；\n点击 [取消] 中止。")){
        toast("已取消下载", false);
        return;
      }
    }
  }
  var sizeTxt = "";
  try{
    var sizeEl = rowEl ? rowEl.querySelector(".model-size") : null;
    if(sizeEl) sizeTxt = sizeEl.textContent.trim();
  }catch(_e){}
  if(!confirm("下载模型：\n" + finalId + (sizeTxt?("\n大小约：" + sizeTxt):"") + "\n\n将占用数 GB 磁盘空间，确认继续？")) return;
  var src = ($("searchSource") && $("searchSource").value === "modelscope") ? "modelscope" : "hf";
  try {
    await api("POST", "/models/download", { id: finalId, source: src });
    toast("下载已开始：" + finalId, true);
    refreshDownloadJobs(true);
  } catch(e){ toast("下载失败：" + e.message, false); }
}
var dlPolling = false;
function refreshDownloadJobs(force){
  if(dlPolling && !force) return;
  dlPolling = true;
  var tick = async function(){
    var jobs = [];
    try {
      var r = await api("POST", "/models/download/status", {});
      jobs = r.jobs || r.active || (Array.isArray(r) ? r : []);
      if(!Array.isArray(jobs)) jobs = Object.values(jobs);
    } catch(e){}
    var box = $("downloadJobs");
    box.textContent = "";
    box.hidden = jobs.length === 0;
    var hadBusy = false;
    jobs.forEach(function(j){
      var st = j.status || "";
      if(st === "queued" || st === "downloading") hadBusy = true;
      var row = el("div", "dl-job" + (st === "done" ? " ok" : st === "failed" ? " fail" : ""));
      row.appendChild(el("span", "spinner"));
      var label = (j.id || "?");
      var pct = 0;
      if(j.bytesTotal > 0 && st !== "done"){
        pct = Math.round(j.bytesDone / j.bytesTotal * 100);
        label += " · " + pct + "% · " + fmtGB(j.bytesDone) + "/" + fmtGB(j.bytesTotal) + " GB";
      } else if(st === "done"){
        pct = 100;
        label += " · 完成";
      } else {
        label += " · " + (j.message || st || "下载中");
      }
      row.appendChild(el("span", "", label));
      if(st === "downloading" || st === "queued" || st === "done"){
        var track = el("div", "dl-progress");
        var bar = el("div", "dl-progress-bar");
        bar.style.width = pct + "%";
        track.appendChild(bar);
        row.appendChild(track);
      }
      box.appendChild(row);
    });
    dlPolling = false;
    if(hadBusy){
      dlPolling = true;
      setTimeout(tick, 2000);
    } else if(jobs.length){
      /* 全部结束后 4 秒清空并刷新本地列表 */
      setTimeout(function(){ $("downloadJobs").hidden = true; refreshLocalModels(); }, 4000);
    }
  };
  tick();
}

/* ═══════════ 推荐模型：联网 Coding-Agent 优先，MoE+稠密混排 ═══════════ */
async function renderRecommendations(){
  var box = $("recList");
  box.textContent = "";
  box.appendChild(el("div", "list-empty", "正在拉取在线推荐…"));
  var items = null;
  try {
    var r = await api("GET", "/models/recommend?limit=6");
    items = r.items || [];
  } catch(e){}
  if(!items || !items.length){
    /* 离线兜底：本地内置池按承载度排序取三 */
    var cap = capacityFor("igpu");
    items = BUILTIN.map(function(m){
      var bytes = SIZE_HINT[m.size];
      return { id: m.id, name: m.id.split("/")[1], sizeLabel: m.size, desc: m.desc,
               tags: [], bytes: bytes || bytesForLabel(m.size, m.id), score: bytes < cap ? -Math.abs(bytes - cap * 0.7) : -(1e15 + bytes) };
    }).sort(function(a, b){ return b.score - a.score; }).slice(0, 3)
      .map(function(x){ return { id: x.id, name: x.name, sizeLabel: x.sizeLabel, desc: x.desc, tags: [], bytes: x.bytes || bytesForLabel(x.sizeLabel, x.id) }; });
  }
  var localIds = {};
  try {
    var locals = await api("GET", "/models");
    (locals || []).forEach(function(x){ localIds[x.id] = true; });
  } catch(e){}
  $("recMeta").textContent = "Coding Agent 优先 · MoE 与稠密混排" + (lastSys.gpuName ? " · " + lastSys.gpuName : "");
  box.textContent = "";
  /* 保证展示中同时含 MoE 与稠密：若前几名全同型，从余下补位 */
  var shown = items.slice(0, 4);
  if(shown.length >= 2){
    var hasMoe = shown.some(function(x){ return (x.tags || []).indexOf("MoE") >= 0; });
    var hasDense = shown.some(function(x){ return (x.tags || []).indexOf("稠密") >= 0; });
    if(!hasMoe || !hasDense){
      var want = !hasMoe ? "MoE" : "稠密";
      var pick = items.find(function(x, i){ return i >= shown.length && (x.tags || []).indexOf(want) >= 0; });
      if(pick) shown[shown.length - 1] = pick;
    }
  }
  shown.forEach(function(item, i){
    var hasLocal = !!localIds[item.id];
    var tags = [item.sizeLabel || ""].filter(Boolean).concat(hasLocal ? ["已下载"] : []);
    box.appendChild(modelRow({
      id: item.id, name: "No." + (i + 1) + " " + (item.name || item.id.split("/")[1]),
      path: item.desc || item.id, sizeBytes: item.bytes || bytesForLabel(item.sizeLabel, item.id),
      pills: (item.nvidia ? ["NVFP4量化"] : []).concat(item.tags || []),
      tags: tags,
      buttons: [hasLocal
        ? { text: "启动", cls: "btn-success", onClick: startWithModel }
        : { text: "下载", cls: "btn-primary", onClick: downloadModel }],
    }));
    refineFit(item.id, curBackend());
  });
}
/* ═══════════ 控制台 ═══════════ */
async function refreshInstalled(){
  var box = $("installedModels");
  var list;
  try { list = await api("GET", "/models"); }
  catch(e){ return; /* 取数失败：保留旧列表，避免闪空 */
  }
  try {
    if(!list.length){ box.appendChild(el("div", "list-empty", "暂无模型")); return; }
    list.forEach(function(m){
      box.appendChild(modelRow({
        id: m.id, name: m.name, path: m.path,
        tags: [fmtGB(m.sizeBytes || 0) + " GB"],
        buttons: [
          { text: "启动", cls: "btn-success", onClick: startWithModel },
          { text: "切换", onClick: switchToModel, act:"switch" },
        ],
      }));
    });
  } catch(e){ }
}
async function refreshEngineState(){ /* 已并入 FT_ENGINE_STATE 单拍 */ }

async function quantTarget(id, rowEl){
  /* 返回 {useId, hint}：选中 FP8/Q4 时先解析官方预量化仓库；无则回退原版 */
  var row = rowEl || (typeof id === "string" ? document.querySelector('.model-row[data-model-id="' + CSS.escape(id) + '"]') : null);
  var sel = row ? row.querySelector(".quant-sel") : null;
  var out = { useId: id, hint: "" };
  /* 本地模型：目录本身即含量化，直接启动 */
  if(row && (row.dataset.source === "local-dir" || row.dataset.source === "hf-cache")){
    var lq = sel && sel.selectedOptions && sel.selectedOptions[0] ? sel.selectedOptions[0].textContent : "";
    if(lq && lq !== "默认") out.hint = "本地 " + lq + " 权重";
    return out;
  }
  if(!sel || sel.value === "bf16") return out;
  try {
    var src = ($("searchSource") && $("searchSource").value) || "hf";
    var r = await api("POST", "/models/resolve-quant", { model: id, key: sel.value, source: src });
    if(r && r.found && r.id){
      out.useId = r.id;
      out.hint = "已用 " + sel.value.toUpperCase() + " 预量化仓库 " + r.id + (r.source?(" 〔"+r.source+"〕"):"");
    }
    else {
      var tried = (r && r.tried) ? r.tried.join(", ") : "";
      out.hint = "未找到 " + sel.value.toUpperCase() + " 预量化仓库（尝试过：" + (tried || "（无）") + "），按原版权重启动";
    }
  } catch(e){ out.hint = "量化仓库解析失败（" + (e.message||e) + "），按原版启动"; }
  return out;
}
function incompleteCache(rowEl){
  var row = typeof rowEl === "string"
    ? document.querySelector('.model-row[data-model-id="' + CSS.escape(rowEl) + '"]')
    : rowEl;
  if(!row) return false;
  var src = row.dataset.source || "";
  var sz = +row.dataset.sizeBytes || 0;
  return src === "hf-cache" && sz > 0 && sz < 500 * 1024 * 1024;
}
async function startWithModel(id, rowEl){
  id = id || selectedModel;
  if(!id){ toast("请先选择模型", false); return; }
  if(_ftBusy.active){ toast("正在加载 " + _ftShortModel(_ftBusy.model) + "，请等待完成后再启动其他模型", false); return; }
  _ftBusy.active = true; _ftBusy.model = id; try{ ftApplyLock(); }catch(_){}   // 乐观锁，轮询随后校准
  toast("开始加载 " + _ftShortModel(id) + " · 首次冷启动约需 20~40 秒", true);
  if(incompleteCache(rowEl || id)){
    toast("该模型只有配置缓存（权重未下载），请先下载完整模型", false);
    _ftBusy.active = false; _ftBusy.model = null; try{ ftApplyLock(); }catch(_){}
    return;
  }
  /* quantTarget 是 async/await，失败会冒泡——统一加保护，避免 busy 锁残留 */
  var q;
  try { q = await quantTarget(id, rowEl); }
  catch(qe){
    _ftBusy.active = false; _ftBusy.model = null; try{ ftApplyLock(); }catch(_){}
    toast("启动参数解析失败：" + qe.message, false);
    return;
  }
  /* 禁用启动按钮，显示加载动画 */
  var btn = rowEl ? rowEl.querySelector(".btn-success") || rowEl.querySelector(".btn-primary") : null;
  var origText = btn ? btn.textContent : "";
  if(btn){ btn.disabled = true; btn.innerHTML = "<span class=\'spinner\'></span> 加载中…"; }
  try {
    var extraArgs = buildEngineArgs(id);
    /* 确保包含混合推理参数 */
    if(extraArgs.indexOf("--moe-backend") < 0) extraArgs.push("--moe-backend", "hybrid");
    if(extraArgs.indexOf("--disable-pynccl") < 0) extraArgs.push("--disable-pynccl");
    if(extraArgs.indexOf("--memory-ratio") < 0) extraArgs.push("--memory-ratio", "0.9");
    await api("POST", "/engine/start", { model: q.useId, port: null, args: extraArgs });
  } catch(e){
    if(/serve_conflict|switch to replace|already running/i.test(e.message)){
      try {
        var extraArgs2 = buildEngineArgs(id);
        if(extraArgs2.indexOf("--moe-backend") < 0) extraArgs2.push("--moe-backend", "hybrid");
        if(extraArgs2.indexOf("--disable-pynccl") < 0) extraArgs2.push("--disable-pynccl");
        if(extraArgs2.indexOf("--memory-ratio") < 0) extraArgs2.push("--memory-ratio", "0.9");
        await api("POST", "/engine/switch", { model: q.useId, port: null, args: extraArgs2, force: false });
      } catch(e2){
        if(btn){ btn.disabled = false; btn.textContent = origText; }
        toast("切换失败：" + e2.message, false);
        return;
      }
    } else {
      if(btn){ btn.disabled = false; btn.textContent = origText; }
      toast("启动失败：" + e.message, false);
      return;
    }
  }
  /* 轮询引擎就绪（最多 5 分钟） */
  var ready = false;
  for(var i=0;i<150;i++){
    await new Promise(function(r){ setTimeout(r, 2000); });
    try {
      var r = await api("GET", "/engine/ready");
      if(r.ready){ ready = true; break; }
    } catch(e){}
    if(i%15===14 && btn) btn.innerHTML = "<span class=\'spinner\'></span> 加载中 " + Math.round((i+1)*2/60) + " 分钟…";
  }
  if(btn){ btn.disabled = false; btn.textContent = ready ? "已启动" : origText; }
  if(ready){
    toast((q.hint ? q.hint + " · " : "") + "启动成功：" + q.useId, true);
  } else {
    toast("启动超时，请检查引擎日志", false);
    _ftBusy.active = false; _ftBusy.model = null; try{ ftApplyLock(); }catch(_){}
  }
  refreshEngineState();
}
async function switchToModel(id){
  if(_ftBusy.active){ toast("正在加载 " + _ftShortModel(_ftBusy.model) + "，请等待完成后再切换", false); return; }
  _ftBusy.active = true; _ftBusy.model = id; try{ ftApplyLock(); }catch(_){}
  toast("切换到 " + id + " …");
  var q;
  try { q = await quantTarget(id, null); }
  catch(qe){
    _ftBusy.active = false; _ftBusy.model = null; try{ ftApplyLock(); }catch(_){}
    toast("切换参数解析失败：" + qe.message, false);
    return;
  }
  try {
    await api("POST", "/engine/switch", { model: q.useId, port: null, args: buildEngineArgs(q.useId), force: false });
    toast(q.hint ? q.hint + " · 切换成功" : "切换成功", true);
    refreshEngineState();
  } catch(e){
    _ftBusy.active = false; _ftBusy.model = null; try{ ftApplyLock(); }catch(_){}
    toast("切换失败：" + e.message, false);
  }
}
$("startBtn").addEventListener("click", function(){ startWithModel(selectedModel); });
$("stopBtn").addEventListener("click", async function(){
  var xb=$("stopBtn"); if(xb) xb.disabled = true;
  var t=$("engineStateText"); var old=t?t.textContent:""; if(t) t.textContent="正在停止…";
  try {
    try { await api("POST", "/engine/stop", { force: false }); }
    catch(_e1){ await new Promise(function(r){ setTimeout(r, 1500); });
               await api("POST", "/engine/stop", { force: true }); }
    _ftBusy.active = false; _ftBusy.model = null; try{ ftApplyLock(); }catch(_){}
    toast("已停止", true);
  }
  catch(e){ toast("停止失败：" + e.message, false); if(t) t.textContent=old; }
  refreshEngineState();
});
async function doEstimate(){
  var m = $("estModel").value.trim();
  if(!m){ toast("请输入模型路径或 ID", false); return; }
  var box = $("fitResult");
  box.textContent = "";
  box.appendChild(el("span", "spinner", ""));
  box.appendChild(document.createTextNode(" 计算中…"));
  try {
    var r = await api("POST", "/engine/estimate", { model: m, backend: $("estBackend").value, args: [] });
    box.textContent = "";
    var head = el("div", r.fit ? "fit-ok" : "fit-bad", r.fit ? "✅ 可以运行" : "❌ 无法运行");
    box.appendChild(head);
    if(r.note) box.appendChild(el("div", "fit-detail", r.note));
    if(r.aBytes !== undefined){
      box.appendChild(el("div", "fit-detail",
        "A 组(dGPU 注意力+KV): " + fmtGB(r.aBytes||0) + " GB   B 组(" + ($("estBackend").value === "igpu" ? "iGPU" : "DRAM/CPU") + " FFN): " + fmtGB(r.bBytes||0) + " GB"));
    }
  } catch(e){
    box.textContent = "";
    box.appendChild(el("div", "fit-bad", "估算失败：" + e.message));
  }
}
$("estimateBtn").addEventListener("click", doEstimate);
$("estModel").addEventListener("keydown", function(e){ if(e.key === "Enter") doEstimate(); });

/* ═══════════ 对话 ═══════════ */
var CHATS_KEY = "ft.chats.v1";
var chats = [];       /* [{id,title,messages:[{role,content}]}] */
var curChat = null;
function loadChats(){
  try { chats = JSON.parse(localStorage.getItem(CHATS_KEY) || "[]"); } catch(e){ chats = []; }
  if(!chats.length) newChat(true);
  else curChat = chats[0];
}
function storeChats(){ localStorage.setItem(CHATS_KEY, JSON.stringify(chats.slice(0, 40))); }
function newChat(silent){
  curChat = { id: Date.now(), title: "新对话", messages: [] };
  chats.unshift(curChat);
  storeChats();
  renderHistory(); renderMessages();
  if(!silent) $("chatInput").focus();
}
function renderHistory(){
  var box = $("historyList");
  box.textContent = "";
  chats.forEach(function(c){
    var item = el("div", "history-item" + (c === curChat ? " active" : ""));
    var t = el("span", "hi-title", c.title || "新对话");
    var del = el("button", "history-del", "×");
    del.type = "button"; del.title = "删除此对话";
    del.addEventListener("click", function(ev){
      ev.stopPropagation();
      chats.splice(chats.indexOf(c), 1);
      if(curChat === c){ curChat = chats[0]; if(!curChat){ newChat(true); return; } }
      storeChats(); renderHistory(); renderMessages();
      toast("已删除对话", true);
    });
    item.appendChild(t); item.appendChild(del);
    item.addEventListener("click", function(){ curChat = c; renderHistory(); renderMessages(); });
    box.appendChild(item);
  });
}
function renderMessages(){
  var box = $("messageList");
  box.textContent = "";
  if(!curChat.messages.length){
    box.appendChild(el("div", "empty-hint", "启动引擎后即可开始对话"));
    return;
  }
  curChat.messages.forEach(function(m){ box.appendChild(msgBubble(m.role, m.content, m)); });
  ftMaybeScroll(box);//ft
}
function makeReasonBlock(reasonText){
  var wrap = el("div", "reason-block");
  var head = el("div", "reason-toggle", "思考过程 ▸");
  var body = el("div", "reason-body", reasonText || "");
  head.addEventListener("click", function(){
    body.classList.toggle("open");
    head.textContent = body.classList.contains("open") ? "思考过程 ▾" : "思考过程 ▸";
  });
  wrap.appendChild(head); wrap.appendChild(body);
  return { wrap: wrap, head: head, body: body };
}
function msgBubble(role, content, meta){
  var bubble = el("div", "msg " + role);
  var hasReason = meta && meta.reasoning;
  if(hasReason){ bubble.appendChild(makeReasonBlock(meta.reasoning).wrap); }
  bubble.appendChild(el("div", "msg-ans", content || ""));
  if(meta && meta.timing){
    var t = meta.timing, parts = [];
    if(t.firstTok) parts.push("首字 " + t.firstTok + "s");
    if(t.decodeTps && t.decodeTps !== "-") parts.push("解码 " + t.decodeTps + " t/s");
    if(t.totalTok) parts.push(t.totalTok + " tokens");
    if(t.elapsed) parts.push("共 " + t.elapsed + "s");
    if(parts.length){ bubble.appendChild(el("div", "msg-timing", "⏱ " + parts.join(" · "))); }
  }
  return bubble;
}

async function refreshChatModels(){
  var sel = $("chatModelSelect");
  try {
    var list = await api("GET", "/models");
    sel.textContent = "";
    list.forEach(function(m){
      var o = el("option", "", m.name);
      o.value = m.id;
      sel.appendChild(o);
    });
    if(window._ftCurModelId && list.some(function(x){ return x.id === window._ftCurModelId; })){
      /* 引擎在跑：以实际加载的模型为准（用户需求①） */
      sel.value = window._ftCurModelId;
    } else if(selectedModel && list.some(function(x){ return x.id === selectedModel; })){
      sel.value = selectedModel;
    } else {
      var bl = bestLocalModel(list);
      if(bl){ selectedModel = bl.id; sel.value = bl.id; }
    }
  } catch(e){ sel.textContent = ""; var o = el("option","","加载失败"); sel.appendChild(o); }
  updateCtxHint();
}
async function updateCtxHint(){
  /* 根据运行中引擎动态获取上下文上限 */
  var hint = $("ctxHint");
  hint.textContent = "";
  try {
    var st = await api("GET", "/engine/status");
    if(!st.running) return;
    var port = st.port || 1919;
    var r = await fetch("/v1/models");
    var d = await r.json();
    var info = d.data && d.data[0] || {};
    var ctx = info.context_length || info.max_model_len || info.ctx_len;
    if(ctx){
      hint.textContent = "模型上限 " + ctx;
      var mt = $("pMaxTokens");
      mt.max = ctx;
      if(+mt.value > ctx) mt.value = ctx;
      paintRange(mt); $("valMaxTokens").textContent = mt.value;
    }
  } catch(e){}
}
function chatParams(){
  var stopEl = $("pStop");
  var stop = stopEl ? String(stopEl.value||"").split(",").map(function(s){ return s.trim(); }).filter(Boolean) : [];
  /* 数值参数加兜底：input.value 为空或非数字时退回安全默认，避免 NaN 打到后端 */
  var _num = function(id, def){
    var el = $(id); if(!el) return def;
    var v = +el.value;
    return isFinite(v) ? v : def;
  };
  var p = {
    temperature: _num("pTemp", 0.7),
    top_p:      _num("pTopP", 0.8),
    top_k:      _num("pTopK", 20),
    max_tokens: _num("pMaxTokens", 1024),
  };
  if(stop.length) p.stop = stop;
  return p;
}
function paintRange(input){
  var min = +input.min || 0, max = +input.max || 100;
  input.style.setProperty("--fill", ((+input.value - min) / (max - min) * 100) + "%");
}
function bindRange(id, valId, fmt){
  var node = $(id), out = $(valId);
  if(!node) return;
  var upd = function(){
    paintRange(node);
    if(out) out.textContent = fmt ? fmt(+node.value) : node.value;
  };
  node.addEventListener("input", upd);
  upd();
}
bindRange("pTemp", "valTemp", function(v){ return v.toFixed(2); });
bindRange("pTopP", "valTopP", function(v){ return v.toFixed(2); });
bindRange("pTopK", "valTopK");
bindRange("pMaxTokens", "valMaxTokens");
bindRange("sConcurrency", "valConcurrency");
bindRange("sCpuThreads", "valCpuThreads");
var _vramFmt = function(v){ return v > 0 ? v.toFixed(1) + "G" : "自动"; };
bindRange("sVramBudget", "valVramBudget", _vramFmt);
bindRange("sMemoryRatio", "valMemoryRatio", function(v){ return v.toFixed(2); });
/* 高级引擎参数：随 MoE 后端联动
   --moe-cache-size 仅 offload/hybrid 显示；--moe-cpu-threads 仅 cpu/hybrid 可用（置灰） */
function syncAdvEngineRows(){
  var beNode = $("sMoeBackend");
  var be = beNode ? beNode.value : String(settings.sMoeBackend || "");
  var gpuMode = (be === "offload" || be === "hybrid");
  var cpuMode = (be === "cpu" || be === "hybrid");
  var cs = $("sMoeCacheSize");
  if(cs){
    var csRow = cs.closest(".set-row");
    if(csRow) csRow.style.display = gpuMode ? "" : "none";
    cs.disabled = !gpuMode;
  }
  var ct = $("sMoeCpuThreads");
  if(ct){
    ct.disabled = !cpuMode;
    var ctRow = ct.closest(".set-row");
    if(ctRow) ctRow.style.opacity = cpuMode ? "" : ".5";
  }
}
$("sMoeBackend").addEventListener("change", syncAdvEngineRows);
$("toggleParamsBtn").addEventListener("click", function(){ $("paramDrawer").classList.toggle("open"); });
$("closeParamsBtn").addEventListener("click", function(){
  var d = $("paramDrawer"); if(d) d.classList.remove("open");
  try{ if(history.state && history.state.ftDrawer) history.back(); }catch(_){}
});
$("newChatBtn").addEventListener("click", function(){ newChat(false); });
$("clearChatBtn").addEventListener("click", function(){
  if(curChat){ curChat.messages = []; storeChats(); renderMessages(); }
});
$("exportChatBtn").addEventListener("click", function(){
  if(!curChat || !curChat.messages.length){ toast("当前对话为空", false); return; }
  var text = curChat.messages.map(function(m){ return (m.role === "user" ? "[用户] " : "[助手] ") + m.content; }).join("\n\n");
  var blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "mytoken-chat-" + curChat.id + ".txt";
  a.click();
});
var sending = false;
/* 贴底检测：用户上翻后输出不再拽底 */
function ftNearBottom(box){ return box.scrollHeight - box.scrollTop - box.clientHeight < 90; }
function ftMaybeScroll(box){ if(ftNearBottom(box)) box.scrollTop = box.scrollHeight; }
async function sendMessage(){
  if(sending) return;
  var sendStartTime = Date.now();
  var input = $("chatInput");
  var text = input.value.trim();
  if(!text) return;
  input.value = ""; input.style.height = "auto";
  curChat.messages.push({ role: "user", content: text });
  if(curChat.title === "新对话") { curChat.title = text.slice(0, 18); renderHistory(); }
  renderMessages();
  sending = true;
  var box = $("messageList");
  var think = el("div", "msg assistant thinking");
  var tdots = el("span", "ft-dots"); tdots.innerHTML = "<i></i><i></i><i></i>";
  var tstat = el("span", "ft-tstat", " 思考中…");
  think.appendChild(tdots); think.appendChild(tstat);
  box.appendChild(think); ftMaybeScroll(box);
  try {
    /* 引擎就绪检查（含等待） */
    var st = await api("GET", "/engine/status");
    if(!st.running){
      tstat.textContent = "模型正在加载，请稍候…";
      var waited = false;
      /* 修：原版同时调 /engine/ready + /engine/health（前者要打 /v1/models 慢且冗余），
         改为只调 /engine/health——它返回 status+progress，一次够用。 */
      for(var wi=0;wi<150;wi++){
        await new Promise(function(r){ setTimeout(r, 2000); });
        try {
          var hh = await api("GET", "/engine/health").catch(function(){ return null; });
          if(hh && hh.status === "ok"){ waited = true; break; }
        } catch(e){}
        if(wi%15===14) tstat.textContent = "模型加载中 " + Math.round((wi+1)*2/60) + " 分钟…";
      }
      if(!waited){ think.remove(); throw new Error("引擎启动超时，请到控制台查看引擎状态"); }
      tstat.textContent = "思考中…";
    }
    var port = st.port || 1919;
    var body = chatParams();
    body.messages = [];
    /* 从引擎 /v1/models 获取真实模型名而非本地路径 */
    var realModel = "default";
    try {
      var mr = await fetch("/v1/models", {signal:AbortSignal.timeout(3000)}).then(x=>x.json());
      if(mr.data && mr.data[0] && mr.data[0].id) realModel = mr.data[0].id;
    } catch(e){}
    body.model = realModel;
    var sysEl = $("pSystem");
    var sys = sysEl ? String(sysEl.value || "").trim() : "";
    if(sys) body.messages.push({ role: "system", content: sys });
    curChat.messages.forEach(function(m){ body.messages.push({ role: m.role, content: m.content }); });
    body.stream = true;
    /* 发起请求；503(加载中)与连接失败自动重试，最长约 8 分钟 */
    var resp = null, lastErr = "";
    for(var attempt=0; attempt<160; attempt++){
      try {
        resp = await fetch("/v1/chat/completions", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
        });
        if(resp.status === 503){
          var tj = null; try{ tj = await resp.json(); }catch(_e){}
          lastErr = (tj && tj.error && tj.error.message) || "model is loading";
          if(/loading/i.test(lastErr)){
            tstat.textContent = "模型加载中，请稍候…";
            await new Promise(function(r){ setTimeout(r, 3000); });
            continue;
          }
        }
        break;
      } catch(e){
        resp = null; lastErr = e.message;
        tstat.textContent = "等待引擎响应…";
        await new Promise(function(r){ setTimeout(r, 3000); });
      }
    }
    if(!resp || !resp.ok){
      think.remove();
      throw new Error("引擎未就绪" + (lastErr ? "（" + lastErr + "）" : "") + "，请到控制台确认模型已启动");
    }
    /* 流式读取：统计首字延迟与解码速度 */
    think.remove();
    var sBub = el("div", "msg assistant");
    var sAns = el("div", "msg-ans", "");
    var ftcaret = el("span", "ft-caret", "▌");
    sBub.appendChild(sAns);
    var sr = null;
    box.appendChild(sBub); ftMaybeScroll(box);
    var t0 = performance.now(), tFirst = 0, tLast = 0;
    var txt = "", reason = "", tokCount = 0, srvUsage = null;
    var reader = resp.body.getReader(), dec = new TextDecoder(), buf = "";
    while(true){
      var rd = await reader.read();
      if(rd.done) break;
      buf += dec.decode(rd.value, {stream:true});
      var nl;
      while((nl = buf.indexOf("\n")) >= 0){
        var line = buf.slice(0, nl).trim(); buf = buf.slice(nl+1);
        if(!line || line.charAt(0) === ":" || line.indexOf("data:") !== 0) continue;
        var payload = line.slice(5).trim();
        if(payload === "[DONE]") continue;
        var obj = null; try{ obj = JSON.parse(payload); }catch(e){}
        if(!obj) continue;
        if(obj.usage) srvUsage = obj.usage;
        var ch0 = obj.choices && obj.choices[0];
        var d = ch0 && (ch0.delta || ch0.message);
        if(d && (d.content || d.reasoning_content)){
          if(!tFirst){ tFirst = performance.now(); }
          tLast = performance.now(); tokCount++;
          if(d.reasoning_content){
            reason += d.reasoning_content;
            if(!sr) sr = makeReasonBlock("");
            if(!sr.wrap.parentNode) sBub.insertBefore(sr.wrap, sAns);
            sr.body.textContent = reason;
            sr.body.classList.add("open");
            sr.head.textContent = "思考中…";
            sr.body.scrollTop = sr.body.scrollHeight;
          }
          if(d.content){
            if(txt === "" && sr){ sr.body.classList.remove("open"); sr.head.textContent = "思考过程 ▸"; }
            txt += d.content;
            sAns.textContent = txt;
            sAns.appendChild(ftcaret);   // 尾部光标随内容推进
          }
          ftMaybeScroll(box);//ft
        }
      }
    }
    var endT = performance.now();
    var totalTok = (srvUsage && srvUsage.completion_tokens) || Math.max(tokCount, Math.round((txt.length + reason.length)/2));
    var firstTokS = tFirst ? ((tFirst - t0)/1000).toFixed(2) : "-";
    var decWin = (tFirst && tLast > tFirst) ? (tLast - tFirst)/1000 : 0;
    var decodeTps = (decWin > 0 && totalTok > 1) ? ((totalTok-1)/decWin).toFixed(1) : "-";
    var elapsed = ((endT - t0)/1000).toFixed(1);
    var reply = txt || (reason ? "【思考】" + reason : "(空响应)");
    curChat.messages.push({ role: "assistant", content: reply, reasoning: reason || undefined, timing: { firstTok: firstTokS, decodeTps: decodeTps, totalTok: totalTok, elapsed: elapsed } });
    storeChats();
    renderMessages();
  } catch(e){
    think.remove();
    box.appendChild(msgBubble("assistant", "⚠ " + e.message));
    ftMaybeScroll(box);//ft
  } finally { sending = false; }
}
$("sendBtn").addEventListener("click", sendMessage);
$("chatInput").addEventListener("keydown", function(e){
  if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); sendMessage(); }
});
$("chatInput").addEventListener("input", function(){
  this.style.height = "auto";
  this.style.height = Math.min(140, this.scrollHeight) + "px";
});

/* ═══════════ 日志 ═══════════ */
var logTimer = null;
async function pollLogs(){
  try {
    var d = await api("GET", "/engine/logs/snapshot?limit=400");
    var lines = (d.lines || []).slice(-400);
    var apiLines = [], srvLines = [];
    lines.forEach(function(l){
      /* 启发式分类：仅当行内出现 HTTP 动词、HTTP/x.x 协议标记或 1xx-5xx 状态码 + 已知短语时归 API；
         避免误把纯数值统计（如 "118 MiB"、"453836 tokens"）归为 API 请求 */
      if(/\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s|HTTP\/\d|\b[1-5]\d{2}\s+(OK|Not Found|Bad Request|Unauthorized|Forbidden|Internal Server|Service Unavailable|Too Large|Continue|Accepted|Created|No Content|Found|Moved|See Other)\b/i.test(l)) apiLines.push(l);
      else srvLines.push(l);
    });
    var ab = $("logApiBox"), sb = $("logServerBox");
    ab.textContent = apiLines.length ? apiLines.join("\n") : "（暂无 API 请求日志）";
    sb.textContent = srvLines.length ? srvLines.join("\n") : "（暂无服务器日志）";
    ab.scrollTop = ab.scrollHeight; sb.scrollTop = sb.scrollHeight;
  } catch(e){}
}
setInterval(function(){
  if($("page-logs").classList.contains("active")) pollLogs();
}, 3000);

/* ═══════════ 应用页复制按钮 ═══════════ */
document.querySelectorAll(".copy-btn").forEach(function(b){
  b.addEventListener("click", function(){
    navigator.clipboard.writeText(b.dataset.copy).then(function(){ toast("已复制", true); });
  });
});

/* ═══════════ 设置页 ═══════════ */
$("sTheme").addEventListener("change", function(){
  settings.sTheme = this.value;
  _persistSettings(settings);
  applyTheme();
});
$("saveSettingsBtn").addEventListener("click", function(){
  saveSettings();
  $("saveMsg").textContent = "✓ 已保存 " + new Date().toLocaleTimeString();
  setTimeout(function(){ $("saveMsg").textContent = ""; }, 2500);
});
$("setupBtn").addEventListener("click", function(){
  alert("初始设置向导：\n\n1. 在「存储类」确认模型目录与镜像\n2. 在「引擎类」选择 MoE 模式与稠密 FFN 引擎\n3. 到「模型库」下载推荐模型\n4. 回到「控制台」点「适配判断」再「启动」");
});
$("tourBtn").addEventListener("click", function(){
  alert("快速导览：\n\n📦 模型库 —— 下载与管理模型\n🖥️ 控制台 —— 启停/切换引擎、资源监控、适配判断\n💬 对话 —— 与本地模型聊天，可调采样参数\n🧩 应用 —— 三种标准 API 入口与 Agent 工具\n📋 日志 —— API 请求与服务器状态\n⚙️ 设置 —— 服务/引擎/外观/引导/存储");
});


/* ---------- 本地目录：原生浏览 / 添加 / 移除 ---------- */
function waitForBridge(timeoutMs){
  return new Promise(function(resolve){
    var t0 = Date.now();
    (function poll(){
      if(window.pywebview && window.pywebview.api && window.pywebview.api.select_folder) return resolve(true);
      if(Date.now() - t0 > timeoutMs) return resolve(false);
      setTimeout(poll, 200);
    })();
  });
}
async function browseFolder(){ return openPicker("localPathInput", true); }
async function openPicker(inputId, autoAdd){
  var input = $(inputId);
  var hasBridge = await waitForBridge(1500);
  var picked = null;
  try {
    if(hasBridge){
      picked = await window.pywebview.api.select_folder();
    } else {
      toast("打开系统文件夹选择器…");
      var r = await api("POST", "/models/browse-folder");
      picked = r.path;
    }
  } catch(e){ toast("浏览失败：" + e.message, false); return; }
  if(!picked) return;
  input.value = picked;
  if(autoAdd) addCustomDir(picked);
}
async function addCustomDir(raw){
  var path = (raw !== undefined ? raw : $("localPathInput").value).trim();
  if(!path){ toast("请先输入或选择路径", false); return; }
  try {
    var r = await api("POST", "/models/dir/add", { path: path });
    $("localPathInput").value = "";
    if(r.count > 0) toast("已添加，识别到 " + r.count + " 个模型", true);
    else toast("目录已添加（未发现模型文件）", false);
    renderDirChips();
loadModelDirFromDaemon(); refreshLocalModels();
  } catch(e){ toast(e.message, false); }
}
async function removeCustomDir(path){
  try {
    await api("POST", "/models/dir/remove", { path: path });
    toast("已移除该目录", true);
    renderDirChips();
loadModelDirFromDaemon(); refreshLocalModels();
  } catch(e){ toast(e.message, false); }
}
async function renderDirChips(){
  var box = $("dirChips");
  box.textContent = "";
  var dirs = [];
  try { dirs = (await api("GET", "/models/dirs")).custom || []; } catch(e){}
  dirs.forEach(function(d){
    var chip = el("span", "dir-chip");
    chip.appendChild(el("span", "chip-path", d));
    var x = el("button", "", "×");
    x.type = "button"; x.title = "移除";
    x.addEventListener("click", function(){ removeCustomDir(d); });
    chip.appendChild(x);
    box.appendChild(chip);
  });
}
$("browseBtn").addEventListener("click", browseFolder);
$("addPathBtn").addEventListener("click", function(){ addCustomDir(); });
$("mdBrowseBtn").addEventListener("click", async function(){
  await openPicker("sModelDir", false);
  saveSettings();
});
$("cleanupCacheBtn").addEventListener("click", async function(){
  try {
    var r = await api("POST", "/models/cache/cleanup", {});
    if(r.removed && r.removed.length){
      toast("已清理 " + r.removed.length + " 个无效缓存，释放 " + fmtGB(r.freed) + " GB", true);
      refreshLocalModels();
    } else {
      toast("未发现无效缓存", true);
    }
  } catch(e){ toast("清理失败：" + e.message, false); }
});
$("localPathInput").addEventListener("keydown", function(e){ if(e.key === "Enter") addCustomDir(); });
renderDirChips();
loadModelDirFromDaemon();

/* ═══════════ 启动 ═══════════ */
/* 先 await loadSettings()（pywebview js_api 是异步的），拿到持久化值后再
   applyTheme/fillSettingsForm，否则表单会先渲染默认值。 */
(function(){
  var boot = loadSettings();
  function afterLoad(){
    applyTheme();
    fillSettingsForm();
    syncAdvEngineRows();                               /* 高级引擎参数随 MoE 后端联动显隐/启用 */
    var _mrNode = $("sMemoryRatio");
    if(_mrNode) _mrNode.dispatchEvent(new Event("input")); /* 回填后同步显存占比滑杆数值显示 */
  }
  if(boot && boot.then) boot.then(afterLoad, afterLoad);
  else afterLoad();
  /* 模型目录/镜像以服务端真值校准（不阻塞设置表单渲染） */
  loadModelDirFromDaemon();
  setTimeout(loadModelDirFromDaemon, 800);
})();
loadChats();
renderHistory(); renderMessages();
refreshLocalModels();
refreshEngineState();
pollSys().then(function(){
  renderRecommendations();
  refreshLocalModels();
  refreshEngineState();
});
refreshChatModels();

/* ═══ FT_BACK_LOGIC：harness 返回逻辑（ESC 关抽屉 / 系统返回键联动） ═══ */
(function(){
  var isOpen=function(){ var d=$("paramDrawer"); return !!(d && d.classList.contains("open")); };
  var closeDrawer=function(){ var d=$("paramDrawer"); if(d) d.classList.remove("open"); };
  document.addEventListener("keydown", function(e){
    if(e.key==="Escape" && isOpen()){ e.preventDefault(); closeDrawer(); history.back(); }
  });
  var tb=$("toggleParamsBtn");
  if(tb) tb.addEventListener("click", function(){
    setTimeout(function(){ if($("paramDrawer").classList.contains("open")) history.pushState({ftDrawer:1}, ""); }, 0);
  });
  window.addEventListener("popstate", function(){ if(isOpen()) closeDrawer(); });
})();

/* ═══ FT_ENGINE_STATE：单一事实源状态机（2 秒一拍） ═══
   OFF 未加载 · LOADING 装载中 · READY 就绪 · ERROR 异常 · STOPPING 停止中 · DOWN 守护断连 */
function _ftShortModel(m){
  var s = String(m || "");
  var base = s.split(/[\\/]/).pop();
  return base || s || "运行中";
}
window._ftSyncSwitchBtns = function(curId){
  document.querySelectorAll("button[data-ft-switch]").forEach(function(b){
    b.disabled = !!curId && b.dataset.ftSwitch === curId;
  });
};


window._ftBusy = { active:false, model:null };
window._ftEng  = { st:"OFF", label:"未加载模型", tip:"" };
window._ftPending = null;   /* 点击乐观态：模型 id，落定后自动清除 */
function ftClassify(st, h){
  if(!st) return ["DOWN","连接断开","守护进程无响应，将自动重试"];
  var starting=!!st.starting, stopping=!!st.stopping, running=!!st.running;
  var S=function(m){ return m ? _ftShortModel(m) : "引擎"; };
  if(stopping && !running) return ["OFF","未加载模型","已停止"];
  if(stopping)             return ["STOPPING","停止中…",S(st.model)];
  if(!running && !starting)return ["OFF","未加载模型","引擎未运行"];
  if(starting)             return ["LOADING",S(st.model)+" 启动中…","正在拉起引擎进程"];
  if(!h)                   return ["LOADING",S(st.model)+" 加载中…","健康状态未知"];
  if(h.reachable === false)return ["LOADING",S(h.model||st.model)+" 加载中…","端口尚未监听 · 冷启动属正常"];
  if(h.status === "loading"){
    /* 权重装载期：以字节进度给真实反馈，而不是假就绪 */
    var p = h.progress || {};
    var tip;
    if(p.totalBytes > 0){
      var pct = Math.round((p.doneBytes||0) / p.totalBytes * 100);
      tip = "权重装载 " + pct + "%（" + fmtGB(p.doneBytes||0) + "/" + fmtGB(p.totalBytes) + " GB）";
    } else {
      tip = ({ weights:"装载权重", graphs:"捕获 CUDA graph", warmup:"预热推理", other:"初始化后端" })[h.phase] || "初始化后端";
    }
    return ["LOADING", S(h.model||st.model)+" 装载中…", tip];
  }
  if(h.status === "error") return ["ERROR","引擎异常",h.message||"backend error"];
  /* 唯一真就绪：引擎自报 status==="ok"。旧字段 h.ready 仅作兼容 */
  if(h.status === "ok")    return ["READY",S(h.model||st.model),"运行中 · 端口 "+(h.port||1919)];
  if(h.ready === false)    return ["LOADING",S(h.model||st.model)+" 收尾中…","预热 / CUDA graph 捕获"];
  if(h.status && h.status !== "ok") return ["LOADING",S(h.model||st.model)+" 装载中…","阶段: "+h.status];
  return ["READY",S(h.model||st.model),"运行中 · 端口 "+(h.port||1919)];
}
function ftApplyLock(){
  var busy = !!_ftBusy.active;
  document.querySelectorAll("button").forEach(function(b){
    var t=(b.textContent||"").trim();
    var isStart=b.classList.contains("btn-success")&&/^启动$|^已启动$|^加载中/.test(t);
    var isSwitch=t==="切换";
    if(isStart){
      b.disabled=busy;
      b.title=busy?("状态切换中（"+(_ftBusy.model?_ftShortModel(_ftBusy.model):"引擎")+"），请稍候"):"";
    } else if(isSwitch){
      /* 锁与「当前模型不可切」规则合一：busy 或 本行即当前模型 都禁用 */
      b.disabled=busy||(!!window._ftCurModelId&&b.dataset.ftSwitch===window._ftCurModelId&&!busy);
      b.title=b.disabled?(busy?"状态切换中，请稍候":"当前已加载此模型"):"";
    }
  });
}
function ftRender(e){
  var clsMap={READY:"on",LOADING:"load",STOPPING:"load",ERROR:"err",DOWN:"err"};
  var txtMap={OFF:"未运行",READY:"运行中",LOADING:"加载中…",STOPPING:"停止中…",ERROR:"引擎异常",DOWN:"守护异常"};
  var box=$("modelLive");
  if(box){
    /* 状态类挂在容器 .model-live 上（与基础样式选择器 .model-live.load .live-dot 匹配） */
    box.classList.remove("on","load","err");
    var bc=clsMap[e.st]; if(bc) box.classList.add(bc);
    var ne=$("modelLiveName"); if(ne) ne.textContent=e.label;
    box.title=e.tip||e.label;
  }
  var bar=$("engineState");
  if(bar){
    var d=bar.querySelector(".dot"); if(d) d.className="dot "+(clsMap[e.st]||"off");
    var tx=$("engineStateText"); if(tx) tx.textContent=txtMap[e.st]||e.st;
    var mt=$("engineMeta");
    if(mt) mt.textContent=(e.st==="READY"||e.st==="LOADING"||e.st==="ERROR")?(e.tip||""):"";
    var sb=$("startBtn"), xb=$("stopBtn");
    if(sb){
      sb.disabled=(e.st!=="OFF"&&e.st!=="ERROR")||_ftBusy.active;
      sb.textContent=e.st==="READY"?"已启动":(e.st==="OFF"?"启动引擎":(e.st==="ERROR"?"重新启动":"加载中…"));
    }
    if(xb) xb.disabled=!(e.st==="READY"||e.st==="LOADING"||e.st==="ERROR"||e.st==="STOPPING");
    if(e.st==="READY"){ try{ _ftSyncChatHeader(); }catch(_){}}
  }
}
async function ftTick(){
  var st=null,h=null;
  try{ st=await api("GET","/engine/status"); }catch(_){}
  if(st){ try{ h=await api("GET","/engine/health"); }catch(_){} }
  var c=ftClassify(st,h);
  window._ftCurModelId=(st&&st.running)?(st.model||null):null;
  window._ftSyncSwitchBtns(window._ftCurModelId);
  if(window._ftPending&&(c[0]==="READY"||c[0]==="OFF"||c[0]==="ERROR")) window._ftPending=null;
  var busy=(c[0]==="LOADING"||c[0]==="STOPPING"||((c[0]!=="READY")&&!!window._ftPending&&true));
  busy=busy||!!(window._ftPending&&c[0]!=="READY"&&c[0]!=="OFF"&&c[0]!=="ERROR");
  var m=window._ftPending||(((c[0]==="LOADING"||c[0]==="STOPPING")&&st)?(st.model||null):null);
  if(_ftBusy.active!==busy||_ftBusy.model!==(m||null)){ _ftBusy.active=busy; _ftBusy.model=m||null; }
  ftApplyLock();
  window._ftSyncSwitchBtns(window._ftCurModelId);   // 锁后再同步，防覆盖
  window._ftEng={st:c[0],label:c[1],tip:c[2]};
  ftRender(window._ftEng);
}
setInterval(ftTick,2000);
ftTick();


/* ═══ FT_SELECT_HARNESS：下拉选项卡对齐（渐进增强，值/事件全回写） ═══ */
(function(){
  function build(sel){
    if(sel.__ftSel) return;
    sel.__ftSel = true;
    var wrap = document.createElement("div");
    wrap.className = "ft-select";
    sel.parentNode.insertBefore(wrap, sel);
    wrap.appendChild(sel);
    sel.classList.add("ft-native");
    sel.style.position = "absolute"; sel.style.opacity = "0";
    sel.style.pointerEvents = "none"; sel.style.width = "1px"; sel.style.height = "1px";

    var btn = document.createElement("button");
    btn.type = "button"; btn.className = "ft-select-btn";
    btn.innerHTML = '<span></span><i class="chev"></i>';
    var menu = document.createElement("div"); menu.className = "ft-select-menu";
    wrap.appendChild(btn); wrap.appendChild(menu);
    /* 程序化赋值(如镜像回填/引擎模型同步)不触发 change——用心跳把真实值刷到按钮上。
       sel 可能在重渲染时从 DOM 移除，原生 setInterval 不会自动停；用 visibilitychange 兜底
       且在 select 断开时主动清理。 */
    var __ftWatch = setInterval(function(){
      try{
        if(!document.body.contains(sel)){ clearInterval(__ftWatch); return; }
        if(isOpen()) return;
        var o0 = sel.selectedOptions && sel.selectedOptions[0];
        var t0v = o0 ? String(o0.textContent||"").trim() : "";
        var cur = btn.firstChild ? btn.firstChild.textContent : "";
        if(t0v && cur !== t0v){ btn.firstChild.textContent = t0v; btn.classList.remove("ft-ph"); }
      }catch(_e){}
    }, 400);

    function items(){
      return Array.prototype.slice.call(sel.options).map(function(o){
        return { v:o.value, t:o.textContent, dis:o.disabled };
      });
    }
    function label(){
      var o = sel.selectedOptions && sel.selectedOptions[0];
      var t = o ? String(o.textContent || "").trim() : "";
      if(!t){ t = sel.getAttribute("placeholder") || "请选择…"; btn.classList.add("ft-ph"); }
      else btn.classList.remove("ft-ph");
      btn.firstChild.textContent = t;
    }
    function renderMenu(){
      var cur = sel.value;
      menu.innerHTML = "";
      items().forEach(function(it){
        var d = document.createElement("div");
        d.className = "ft-select-item" + (it.v === cur ? " sel" : "");
        d.textContent = it.t;
        if(it.dis) return;
        d.addEventListener("click", function(ev){
          ev.stopPropagation();
          if(sel.value !== it.v){ sel.value = it.v; sel.dispatchEvent(new Event("change")); }
          close();
        });
        menu.appendChild(d);
      });
      label();
    }
    function open(){
      closeAll();
      renderMenu();
      wrap.classList.add("open");
      // 贴近视口底部时向上翻
      var r = wrap.getBoundingClientRect();
      wrap.classList.toggle("up", r.bottom + 270 > window.innerHeight && r.top > 280);
    }
    function close(){ wrap.classList.remove("open"); }
    function isOpen(){ return wrap.classList.contains("open"); }

    btn.addEventListener("click", function(e){ e.stopPropagation(); isOpen() ? close() : open(); });
    menu.addEventListener("click", function(e){ e.stopPropagation(); });
    sel.addEventListener("change", function(){ renderMenu(); });
    // 动态 options（如模型列表刷新）自动重建
    new MutationObserver(function(){ if(isOpen()) renderMenu(); else label(); })
      .observe(sel, { childList:true });

    sel.__ftApi = { open:open, close:close, isOpen:isOpen };
    renderMenu();
  }
  function closeAll(){
    document.querySelectorAll(".ft-select.open").forEach(function(w){ w.classList.remove("open"); });
  }
  function enhance(root){
    (root || document).querySelectorAll("select").forEach(function(s){
      if(!s.closest(".ft-select")) build(s);
    });
  }
  // ESC：先关菜单，不落到抽屉逻辑
  document.addEventListener("keydown", function(e){
    if(e.key === "Escape"){
      var any = false;
      document.querySelectorAll(".ft-select.open").forEach(function(w){ w.classList.remove("open"); any = true; });
      if(any) e.stopPropagation();
    }
  }, true);
  document.addEventListener("click", closeAll);
  window.addEventListener("blur", closeAll);

  if(document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", function(){ enhance(); });
  } else { enhance(); }
  // 动态插入的 select（模型行/设置区重渲染）自动增强
  var __ftDynT = null;
  var __ftDynObs = new MutationObserver(function(){
    clearTimeout(__ftDynT);
    __ftDynT = setTimeout(function(){ enhance(); }, 120);
  });
  __ftDynObs.observe(document.body, { childList:true, subtree:true });
})();

/* ═══ FT_RELOAD_HINTS：引擎级参数标注「需重载」，统一 prepend 到 set-row 最左侧 ═══ */
(function(){
  var IDS = ["sMoeBackend","sDenseEngine","sIgpuService","sIgpuNoFallback","sKvDevice","sKvQuant","sMtp","sMtpK","sMtpIgpuFc","sMtpIgpuVerifyGraph","sCtFp8",
             "sConcurrency","sCpuThreads","sVramBudget",
             "sMoeCacheSize","sMoeCpuThreads","sNumTokens","sMemoryRatio","sMaxRunningReq","sKvReserveTokens","sDisableMoePrefillOverlap"];
  var TIP = "引擎启动配置，修改后需重启引擎才会生效";
  function inject(){
    IDS.forEach(function(id){
      var elx = document.getElementById(id);
      if(!elx) return;
      var row = elx.closest(".set-row");
      if(!row) return;
      if(row.querySelector(".ft-reload-hint")) return;
      var tag = document.createElement("span");
      tag.className = "ft-reload-hint";
      tag.textContent = "需重载";
      tag.title = TIP;
      /* prepend 到 row 最左侧（在所有现有子节点之前）*/
      row.insertBefore(tag, row.firstChild);
    });
  }
  if(document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", inject);
  else inject();
})();

/* ── 镜像/存储目录即改即存（v2 兜底，文件尾绑定） ── */
(function(){
  function __ftQuickSync(){ try{ saveSettings(); }catch(_e){} }
  ["sMirror","sModelDir"].forEach(function(id){
    var n = document.getElementById(id);
    if(n && !n.__ftSync){ n.__ftSync = true; n.addEventListener("change", __ftQuickSync); }
  });
})();

/* ============ 内置转换：MXFP4/原始 -> NVFP4 ============ */
var _ftConvertTimer = null;
async function convertToNvfp4(id){
  if(!id || typeof id !== "string") return;
  var msg = "把 " + id + " 转换为 NVFP4？\n输出到同级 -NVFP4 目录，约需数分钟到数十分钟。";
  if(!window.confirm(msg)) return;
  var r;
  try{ r = await api("POST", "/models/convert-nvfp4", { path: id }); }
  catch(e){ alert("启动转换失败：" + e.message); return; }
  if(r && r.error){ alert(r.error); return; }
  toast("转换已开始：" + (r.out || id));
  if(_ftConvertTimer) clearInterval(_ftConvertTimer);
  _ftConvertTimer = setInterval(pollConvertJobs, 3000);
  pollConvertJobs();
}
async function pollConvertJobs(){
  var r;
  try{ r = await api("POST", "/models/convert-nvfp4/status", {}); }catch(e){ return; }
  var jobs = (r && r.active) || [];
  var box = document.querySelector(".dl-jobs") || (typeof $ === "function" ? $("downloadJobs") : null);
  if(!box) return;
  var node = box.querySelector(".ft-convert-job");
  if(!jobs.length){
    if(node) node.remove();
    if(_ftConvertTimer){ clearInterval(_ftConvertTimer); _ftConvertTimer = null; }
    if(box.hidden === false && !box.children.length) box.hidden = true;
    return;
  }
  if(!node){
    node = document.createElement("div");
    node.className = "dl-job ft-convert-job";
    box.hidden = false;
    box.appendChild(node);
  }
  var j = jobs[jobs.length-1];
  var pct = j.total > 0 ? Math.round(j.done*100/j.total) : 0;
  var label = j.status === "done" ? "✅ 转换完成"
            : j.status === "error" ? ("❌ " + (j.message||"失败"))
            : ("⏳ " + (j.message||"转换中") + (j.total>0 ? " (" + pct + "%)" : ""));
  var srcName = (j.src||"").split("/").pop();
  var outName = (j.out||"").split("/").pop();
  node.innerHTML = "<b>NVFP4 转换</b> " + srcName + " &rarr; " + outName + "<br>" + label;
  if(j.status === "done" || j.status === "error"){
    if(_ftConvertTimer){ clearInterval(_ftConvertTimer); _ftConvertTimer = null; }
    if(j.status === "done") refreshLocalModels();
  }
}
if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", pollConvertJobs);
else pollConvertJobs();
