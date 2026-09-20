/* RAG Agent 前端逻辑 —— 无框架、无构建、无 CDN。
 *
 * 三条必须处理的、本项目特有的情况：
 *   ① 异步摄取的 job 会长时间停在 queued —— 因为没有 worker 在消费。
 *      根因通常是 REPOSITORY_BACKEND=memory（根本没有任务表），
 *      或者 worker 进程没起。默认界面上"转圈"和"真卡住"长得一样，
 *      所以这里显式做超时判定并给出排查提示。
 *   ② /chat 在 LLM_PROVIDER=none 时返回 503（problem+json），
 *      必须把 detail 原样显示出来，否则用户只看到一个红框。
 *   ③ SSE 必须用 fetch + ReadableStream 读 —— EventSource 只支持 GET，
 *      而 /chat 是 POST。见 streamChat()。
 */

const $ = (sel) => document.querySelector(sel);
const api = (p) => new URL(p, location.origin).toString();

// job 停在 queued 超过这个时长就判定为"没有 worker 在消费"
const STALL_MS = 15000;
const POLL_MS = 1500;

let threadId = null;      // 同一 threadId 共享对话上下文
let llmOK = false;        // /readyz 报的 llm 组件状态
let repoBackend = '';     // 'postgres' | 'memory'

// ======================================================================
// 工具
// ======================================================================
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/** 文件大小。★ 不能用 (bytes/1024).toFixed(0) —— 一个 200 字节的 Markdown
 *  会显示成 "0 KB"，看起来像文件是空的。小于 1KB 就直接给字节数。 */
function fmtSize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

/** 从 Problem Details 里取出给人看的那句话。 */
function problemText(payload, fallback) {
  if (!payload) return fallback;
  if (typeof payload === 'string') return payload;
  return payload.detail || payload.title || fallback;
}

async function readError(res) {
  let body = null;
  try { body = await res.json(); } catch { /* 非 JSON 响应 */ }
  return problemText(body, `${res.status} ${res.statusText}`);
}

/** 把查询切成高亮用的词。
 *  中文没有空格，整串当词是匹配不上的，所以对 CJK 连续段取**二元组**——
 *  "为什么需要重排" → 为什/什么/么需/需要/要重/重排，命中率足够且不会碎成单字。 */
function queryTerms(q) {
  const terms = new Set();
  for (const m of q.matchAll(/[A-Za-z0-9_]+/g)) {
    if (m[0].length >= 2) terms.add(m[0]);
  }
  for (const m of q.matchAll(/[一-龥]+/g)) {
    const run = m[0];
    if (run.length === 1) { terms.add(run); continue; }
    for (let i = 0; i + 2 <= run.length; i++) terms.add(run.slice(i, i + 2));
  }
  return [...terms].sort((a, b) => b.length - a.length);
}

function highlight(text, terms) {
  if (!terms.length) return esc(text);
  const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  return text.split(re)
    .map((part, i) => (i % 2 ? `<mark>${esc(part)}</mark>` : esc(part)))
    .join('');
}

/** 极简 Markdown → HTML。
 *
 *  LLM 的输出天然是 Markdown（标题、粗体、行内代码、列表），直接 textContent
 *  打进气泡会把 `**` 和反引号原样显示出来 —— 演示时很扎眼。
 *  这里只认最常见的几种，够用就好：不引 CDN（离线可用是这前端的硬约束），
 *  也不做完整的 CommonMark 解析。
 *
 *  ★ 必须先 esc() 再套正则：顺序反了就是 XSS（模型输出里可能带尖括号）。
 *    转义后文本里只剩 `&lt;` 这类实体，而 `**` / 反引号不受影响，正则可以安全跑。
 *
 *  citeIndices 是本次回答真实存在的引用编号 —— 只有它们才渲染成可点的角标，
 *  否则模型随口写的 `[99]` 会变成指向空处的死链。
 */
function mdToHtml(src, citeIndices = new Set()) {
  const inline = (s) => s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d+)\](?!\()/g, (m, n) => (
      citeIndices.has(Number(n))
        ? `<a class="cite-ref" href="#cite-${n}" data-cite="${n}">[${n}]</a>`
        : m));

  const out = [];
  let list = null;   // 'ul' | 'ol' | null
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };

  for (const raw of esc(src).split('\n')) {
    const line = raw.trimEnd();
    if (!line.trim()) { closeList(); continue; }

    let m;
    if ((m = line.match(/^#{1,6}\s+(.*)$/))) {
      closeList();
      out.push(`<h4>${inline(m[1])}</h4>`);
    } else if ((m = line.match(/^\s*[-*]\s+(.*)$/))) {
      if (list !== 'ul') { closeList(); out.push('<ul>'); list = 'ul'; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*\d+\.\s+(.*)$/))) {
      if (list !== 'ol') { closeList(); out.push('<ol>'); list = 'ol'; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^&gt;\s?(.*)$/))) {
      closeList();
      out.push(`<blockquote>${inline(m[1])}</blockquote>`);
    } else {
      closeList();
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  return out.join('');
}

// ======================================================================
// 健康状态
// ======================================================================
async function refreshHealth() {
  const box = $('#health-pills');
  try {
    const r = await fetch(api('/readyz'));
    const d = await r.json();
    llmOK = false;
    repoBackend = '';
    box.innerHTML = d.components.map((c) => {
      if (c.name === 'llm') llmOK = c.ok;
      if (c.name === 'repository') repoBackend = c.detail;
      return `<span class="pill ${c.ok ? 'ok' : 'bad'}" title="${esc(c.detail)}">
                <i></i>${esc(c.name)}</span>`;
    }).join('');

    $('#brand-dot').className = 'dot ' + (d.components.find((c) => c.name === 'vectorstore')?.ok ? 'ok' : 'bad');
    renderChatBanner();
  } catch (e) {
    box.innerHTML = `<span class="pill bad"><i></i>无法连接</span>`;
    $('#brand-dot').className = 'dot bad';
  }
}

// ======================================================================
// 上传 —— 同步路径
// ======================================================================
function uploadRow(name) {
  const el = document.createElement('div');
  el.className = 'doc';
  el.innerHTML = `<div class="doc-top">
      <span class="doc-title">${esc(name)}</span>
      <span class="tag running">上传中</span>
    </div>
    <div class="doc-meta"><span class="stage muted small">—</span></div>`;
  $('#uploads').prepend(el);
  return {
    tag: (cls, text) => { el.querySelector('.tag').className = `tag ${cls}`; el.querySelector('.tag').textContent = text; },
    stage: (html) => { el.querySelector('.stage').innerHTML = html; },
  };
}

async function uploadOne(file, sync) {
  const row = uploadRow(file.name);
  const fd = new FormData();
  fd.append('file', file);

  let res;
  try {
    res = await fetch(api(`/api/v1/documents?sync=${sync}`), { method: 'POST', body: fd });
  } catch (e) {
    row.tag('failed', '网络错误'); row.stage(esc(String(e)));
    return;
  }

  if (!res.ok) {
    row.tag('failed', `HTTP ${res.status}`);
    row.stage(esc(await readError(res)));
    return;
  }

  const d = await res.json();

  if (sync) {
    row.tag(d.status === 'succeeded' ? 'succeeded' : 'failed', d.status);
    row.stage(`${d.chunk_count} 块 · ${d.vectors_written} 向量 · ${d.duration_ms}ms`
      + (d.deduplicated ? ' · 内容重复，已跳过' : ''));
    refreshDocs();
    return;
  }

  // ---- 异步：轮询 job，并判定"卡住" ----
  row.tag('queued', '排队中');
  row.stage(`job #${d.job_id} · 等待 worker 领取…`);
  const started = Date.now();
  let warned = false;

  while (true) {
    await new Promise((r) => setTimeout(r, POLL_MS));
    let job;
    try {
      const jr = await fetch(api(`/api/v1/jobs/${d.job_id}`));
      if (!jr.ok) { row.tag('failed', `HTTP ${jr.status}`); row.stage(esc(await readError(jr))); return; }
      job = await jr.json();
    } catch (e) {
      row.tag('failed', '轮询失败'); row.stage(esc(String(e)));
      return;
    }

    const waited = ((Date.now() - started) / 1000).toFixed(0);

    if (job.status === 'succeeded') {
      row.tag('succeeded', '已完成');
      const r = job.result || {};
      row.stage(`${r.chunk_count ?? 0} 块 · ${r.vectors_written ?? 0} 向量 · ${r.duration_ms ?? 0}ms`);
      refreshDocs();
      return;
    }
    if (job.status === 'failed') {
      row.tag('failed', '失败');
      row.stage(`第 ${job.attempt}/${job.max_attempts} 次尝试：${esc(job.error || '未知错误')}`);
      refreshDocs();
      return;
    }

    if (job.status === 'running') {
      row.tag('running', '解析中');
      row.stage(`job #${job.id} · 正在解析（第 ${job.attempt}/${job.max_attempts} 次）`);
    } else {
      row.tag(warned ? 'stalled' : 'queued', warned ? '疑似卡住' : '排队中');
      row.stage(`job #${job.id} · 已等待 ${waited}s`);
    }

    // ★ 关键判定：queued 超过 STALL_MS 且从未变成 running
    if (!warned && job.status === 'queued' && Date.now() - started > STALL_MS) {
      warned = true;
      row.stage(`job #${job.id} · 已等待 ${waited}s 仍未开始 —— `
        + `没有 worker 在消费队列。检查 worker 进程是否在跑，`
        + `以及 <code>REPOSITORY_BACKEND</code> 是不是 <code>memory</code>。`);
    }
  }
}

async function handleFiles(files) {
  const sync = $('#sync').checked;
  for (const f of files) await uploadOne(f, sync);
}

// ======================================================================
// 文档列表
// ======================================================================
async function refreshDocs() {
  const box = $('#docs');
  let d;
  try {
    const r = await fetch(api('/api/v1/documents?limit=100'));
    if (!r.ok) { box.innerHTML = `<div class="empty">读取失败：${esc(await readError(r))}</div>`; return; }
    d = await r.json();
  } catch (e) {
    box.innerHTML = `<div class="empty">无法连接 API</div>`;
    return;
  }

  if (!d.items.length) {
    box.innerHTML = `<div class="empty">还没有文档。<br>上传一个 PDF / Markdown 试试。</div>`;
    return;
  }

  box.innerHTML = d.items.map((x) => `
    <div class="doc">
      <div class="doc-top">
        <span class="doc-title">${esc(x.title)}</span>
        <span class="tag ${esc(x.status)}">${esc(x.status)}</span>
        <button class="doc-del" data-del="${x.id}" title="删除">×</button>
      </div>
      <div class="doc-meta">
        <span>${x.chunk_count} 块</span>
        ${x.page_count ? `<span>${x.page_count} 页</span>` : ''}
        ${x.size_bytes ? `<span>${fmtSize(x.size_bytes)}</span>` : ''}
        <span class="muted">#${x.id}</span>
      </div>
      ${x.error_message ? `<div class="doc-meta" style="color:var(--err)">${esc(x.error_message)}</div>` : ''}
    </div>`).join('');

  box.querySelectorAll('[data-del]').forEach((b) => {
    b.onclick = async () => {
      const id = b.dataset.del;
      if (!confirm(`删除文档 #${id}？对应的向量也会一并删除。`)) return;
      const r = await fetch(api(`/api/v1/documents/${id}`), { method: 'DELETE' });
      if (!r.ok) alert(await readError(r));
      refreshDocs();
    };
  });
}

// ======================================================================
// 检索
// ======================================================================
function legBadge(kind, rank) {
  if (!rank) return `<span class="leg none">${kind === 'd' ? '稠密' : '稀疏'} —</span>`;
  const label = kind === 'd' ? '稠密' : '稀疏';
  return `<span class="leg ${kind === 'd' ? 'dense' : 'sparse'}">${label} #${rank}</span>`;
}

async function doSearch() {
  const q = $('#q').value.trim();
  if (!q) return;
  const out = $('#search-out');
  out.innerHTML = `<div class="empty">检索中…</div>`;

  const body = {
    query: q,
    top_k: 8,
    use_dense: $('#use-dense').checked,
    use_sparse: $('#use-sparse').checked,
    use_rerank: $('#use-rerank').checked,
  };

  let d;
  try {
    const r = await fetch(api('/api/v1/retrieve'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) { out.innerHTML = `<div class="banner err">${esc(await readError(r))}</div>`; return; }
    d = await r.json();
  } catch (e) {
    out.innerHTML = `<div class="banner err">请求失败：${esc(String(e))}</div>`;
    return;
  }

  const g = d.diagnostics || {};
  const rec = g.recall || {};
  const terms = queryTerms(q);

  const diag = `<div class="diag">
      <span>稠密召回 <b>${rec.dense ?? '—'}</b></span>
      <span>稀疏召回 <b>${rec.sparse ?? '—'}</b></span>
      <span>融合后 <b>${g.fused ?? '—'}</b></span>
      <span>返回 <b>${g.returned ?? d.chunks.length}</b></span>
      <span>重排 <b>${g.rerank ? '开' : '关'}</b></span>
      ${g.missing_in_postgres ? `<span style="color:var(--warn)">正文缺失 <b>${g.missing_in_postgres}</b></span>` : ''}
    </div>`;

  if (!d.chunks.length) {
    out.innerHTML = diag + `<div class="empty">没有召回任何内容。<br>
      ${Object.values(rec).every((v) => !v) ? '知识库里可能还没有文档，或者这三路检索都被关掉了。' : ''}</div>`;
    return;
  }

  out.innerHTML = diag + d.chunks.map((c) => `
    <div class="hit">
      <div class="hit-head">
        <span class="rank">${c.rank}</span>
        <span class="legs">
          ${legBadge('d', c.dense_rank)}
          ${legBadge('s', c.sparse_rank)}
        </span>
        <span class="score">${c.dense_score != null ? `余弦 ${Number(c.dense_score).toFixed(3)} · ` : ''}rrf ${Number(c.score).toFixed(4)}${c.rerank_score != null ? ` · rerank ${Number(c.rerank_score).toFixed(3)}` : ''}</span>
      </div>
      <div class="hit-label">${esc(c.doc_title)}${c.section_path ? ` · ${esc(c.section_path)}` : ''}${c.page_start ? ` · p${c.page_start}` : ''}</div>
      <div class="hit-content">${highlight(c.content, terms)}</div>
    </div>`).join('');
}

// ======================================================================
// 问答（SSE）
// ======================================================================
function renderChatBanner() {
  const box = $('#chat-banner');
  if (!box) return;
  const parts = [];
  if (repoBackend === 'memory') {
    parts.push(`当前 <code>REPOSITORY_BACKEND=memory</code>：正文、任务表、对话断点全在进程内，
      重启即丢，<b>也没有任务队列</b>。异步上传会永远停在 queued。`);
  }
  if (!llmOK) {
    parts.push(`未配置 <code>LLM_PROVIDER</code>，问答接口返回 503。检索功能不受影响。
      要启用问答，在 <code>.env</code> 里设
      <code>LLM_PROVIDER=openai</code>、<code>LLM_API_BASE</code>、<code>LLM_API_KEY</code>、<code>LLM_MODEL</code>。`);
  }
  box.innerHTML = parts.length
    ? `<div class="banner warn">${parts.join('<br><br>')}</div>`
    : '';
}

/** 解析 SSE 字节流。事件之间以空行分隔，块内是 `event: X` / `data: {...}`。 */
async function streamChat(question, bubble, citeBox) {
  const res = await fetch(api('/api/v1/chat'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, thread_id: threadId, stream: true }),
  });

  // ★ 503（未配 LLM）走的是普通 JSON，不是 SSE —— 必须先判 res.ok，
  //   否则会把 problem+json 当成事件流去解析，得到一个空答案。
  if (!res.ok) {
    bubble.innerHTML = `<div class="banner err">${esc(await readError(res))}</div>`;
    return;
  }

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let answer = '';
  const citeIdx = new Set();   // 本次回答真实存在的引用编号，供 mdToHtml 判定角标是否可点

  const handle = (raw) => {
    let name = 'message';
    const dataLines = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) name = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return;
    let d;
    try { d = JSON.parse(dataLines.join('\n')); } catch { return; }

    if (name === 'start') {
      threadId = d.thread_id;
    } else if (name === 'citations') {
      const cs = d.citations || [];
      if (cs.length) {
        for (const c of cs) citeIdx.add(Number(c.index));
        citeBox.classList.remove('hidden');
        citeBox.innerHTML = `<div class="cites-head">引用 ${cs.length} 处</div>` + cs.map((c) => `
          <div class="cite" id="cite-${esc(c.index)}">
            <b>[${esc(c.index)}] ${esc(c.label)}</b>
            ${c.quote ? `<span class="quote">${esc(c.quote)}</span>` : ''}
          </div>`).join('');
      }
    } else if (name === 'delta') {
      // 流式期间保持纯文本：每来一个 delta 就重跑一遍 Markdown 解析
      // 既浪费又会让未闭合的 `**` 闪来闪去。收尾时再整体渲染。
      answer += d.text || '';
      bubble.textContent = answer;
      bubble.parentElement.scrollIntoView({ block: 'end', behavior: 'smooth' });
    } else if (name === 'done') {
      if (answer.trim()) {
        bubble.innerHTML = mdToHtml(answer, citeIdx);
        bubble.classList.add('md');
      }
      if (d.route === 'refuse') {
        // 拒答：不显示"引用已核验"（没有任何引用），直接把判定理由摆出来，
        // 否则用户只会看到一段"无法回答"却不知道为什么。
        bubble.insertAdjacentHTML('beforeend',
          `<div class="banner warn" style="margin-top:8px">知识库里没有与该问题相关的内容${
            d.grade_reason ? `：${esc(d.grade_reason)}` : ''}${
            d.retries ? `（已重试 ${d.retries} 次改写查询）` : ''}</div>`);
        return;
      }
      const bits = [];
      if (d.verified) bits.push('引用已核验');
      else bits.push('⚠ 引用未通过核验');
      if (d.route) bits.push(`路径 ${d.route}`);
      if (d.retries) bits.push(`重试 ${d.retries} 次`);
      bubble.insertAdjacentHTML('beforeend',
        `<div class="small muted" style="margin-top:8px">${esc(bits.join(' · '))}</div>`);
    } else if (name === 'error') {
      bubble.innerHTML = `<div class="banner err">${esc(d.message || '生成失败')}</div>`;
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      handle(buf.slice(0, i));
      buf = buf.slice(i + 2);
    }
  }
  if (buf.trim()) handle(buf);
}

async function doAsk() {
  const q = $('#question').value.trim();
  if (!q) return;
  $('#question').value = '';

  const out = $('#chat-out');
  const wrap = document.createElement('div');
  wrap.className = 'msg';
  wrap.innerHTML = `<div class="msg-q">${esc(q)}</div>
    <div class="msg-a"></div>
    <div class="cites hidden"></div>`;
  out.appendChild(wrap);
  out.scrollTop = out.scrollHeight;

  const bubble = wrap.querySelector('.msg-a');
  const citeBox = wrap.querySelector('.cites');
  const btn = $('#ask');
  btn.disabled = true;
  bubble.innerHTML = '<span class="cursor">&nbsp;</span>';

  try {
    await streamChat(q, bubble, citeBox);
  } catch (e) {
    bubble.innerHTML = `<div class="banner err">${esc(String(e))}</div>`;
  } finally {
    btn.disabled = false;
  }
}

// ======================================================================
// 事件绑定
// ======================================================================
function init() {
  // 上传
  $('#pick').onclick = () => $('#file').click();
  $('#file').onchange = (e) => { handleFiles([...e.target.files]); e.target.value = ''; };
  const drop = $('#drop');
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.remove('over');
  }));
  drop.addEventListener('drop', (e) => {
    if (e.dataTransfer?.files?.length) handleFiles([...e.dataTransfer.files]);
  });

  // 检索
  $('#go').onclick = doSearch;
  $('#q').onkeydown = (e) => { if (e.key === 'Enter') doSearch(); };

  // 重排开关：没配重排器时禁用并说明。
  // ★ 不禁用的话，用户勾着"重排"却看不到任何排序变化，会以为功能是坏的 ——
  //   实际是 RERANK_PROVIDER=none，压根没加载模型。
  fetch(api('/readyz')).then((r) => r.json()).then((d) => {
    const rr = d.components.find((c) => c.name === 'reranker');
    if (rr && rr.ok) return;
    $('#use-rerank').checked = false;
    $('#use-rerank').disabled = true;
    $('#rerank-note').textContent = rr
      ? `（重排不可用：${rr.detail}）`
      : '（未配置重排器：RERANK_PROVIDER=none）';
  }).catch(() => {});

  // 问答
  $('#ask').onclick = doAsk;
  $('#question').onkeydown = (e) => { if (e.key === 'Enter') doAsk(); };
  $('#new-thread').onclick = () => {
    threadId = null;
    $('#chat-out').innerHTML = '';
  };

  // 页签
  $('#tabs').onclick = (e) => {
    const t = e.target.closest('.tab');
    if (!t) return;
    document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x === t));
    $('#pane-search').classList.toggle('hidden', t.dataset.tab !== 'search');
    $('#pane-chat').classList.toggle('hidden', t.dataset.tab !== 'chat');
  };

  // 刷新
  $('#refresh-docs').onclick = refreshDocs;
  $('#health').onclick = refreshHealth;
  $('#sync').onchange = () => {
    $('#sync').parentElement.title = $('#sync').checked
      ? '在请求里直接解析（分钟级，会占住一个请求）'
      : '只入库并返回 job_id，解析交给 worker 进程';
  };
  $('#sync').onchange();

  refreshHealth();
  refreshDocs();
  setInterval(refreshHealth, 10000);

  // 空态提示：不写的话检索区就是一片空白，演示时不知道从哪下手
  $('#search-out').innerHTML = `<div class="empty">
    输入问题后检索。每张卡片上的<b>「稠密 #n / 稀疏 #n」</b>角标是两个召回通道各自的名次 ——<br>
    看懂这一对数字，就看懂了混合检索为什么要融合。
  </div>`;

  // ?q=xxx 直接出结果、?tab=chat 直接开问答页：
  // 演示时甩一条链接过去就行，不用现场打字、不用点页签
  const params = new URLSearchParams(location.search);
  const tab0 = params.get('tab');
  if (tab0 === 'chat' || tab0 === 'search') {
    document.querySelector(`.tab[data-tab="${tab0}"]`).click();
  }
  const q0 = params.get('q');
  if (q0) { $('#q').value = q0; doSearch(); }

  // ?ask=xxx 直接切到问答页并把问题发出去。与 ?q= 同理：
  // 演示时甩一条链接就能出答案，不用现场打字（LLM 没配时也只是显示提示条）
  const ask0 = params.get('ask');
  if (ask0) {
    if (tab0 !== 'chat') document.querySelector('.tab[data-tab="chat"]').click();
    $('#question').value = ask0;
    doAsk();
  }
}

document.addEventListener('DOMContentLoaded', init);
