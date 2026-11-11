const cases = [
  {
    title: '商品服务报错', service: 'product-catalog',
    steps: [
      ['容器状态', '确认异常容器', '商品目录容器的健康检查失败。先确认服务可用性，再查看错误率是否同时上升。'],
      ['Prometheus 指标', '错误率在五分钟内升高', '服务的 up 指标为 0，同期错误率从 1% 升至 62%。下一步检查请求在哪一跳失败。'],
      ['Jaeger 链路', '找到请求中断的位置', '五条链路都在 GetProduct 调用处返回 UNAVAILABLE，没有继续进入下游服务。'],
      ['服务日志', '定位故障注入开关', '日志显示 productCatalogFailure 已启用，商品目录服务因此主动返回模拟的 5xx 错误。'],
      ['处置路由 · 转人工', '将证据交给值班人员', '该服务不在允许自动重启的名单内。权限检查拦截了重启，Agent 选择通过飞书转交人工处理。']
    ]
  },
  {
    title: '推荐服务内存泄漏', service: 'recommendation',
    steps: [
      ['内存指标', '内存逼近容器上限', '推荐服务已用 940 MiB，接近 1024 MiB 的容器上限，用量仍在增长。'],
      ['变更记录', '沿近期提交排查', '近期变更为推荐逻辑加入缓存。Agent 将排查范围缩小到新增的缓存实现。'],
      ['代码检查', '发现没有上限的缓存', '_POPULAR_CACHE 字典既没有容量限制，也没有淘汰策略，条目会持续积累。'],
      ['容器事件', '核对内存耗尽记录', '过去一小时出现三次 OOMKill。容器事件与内存持续增长的观察一致。'],
      ['处置路由 · 恢复并修复', '重启恢复服务，转入代码修复', '重启后内存降至 120 MiB。Agent 同时选择发起代码修复 PR，处理缓存增长的根因。']
    ]
  },
  {
    title: '广告服务 CPU 限流', service: 'ad',
    steps: [
      ['容器指标', 'CPU 使用率接近饱和', '广告服务的 CPU 使用率达到 99.8%。Agent 继续检查容器是否触及资源配额。'],
      ['Prometheus 指标', '确认持续限流', 'CPU 限流比例连续十分钟保持在 100%，资源配额成为下一步排查的重点。'],
      ['工具权限', '容器内探查被拦截', '权限检查阻止了 docker exec，Agent 转而选择允许执行的资源调整操作。'],
      ['处置路由 · 自动恢复', '在允许范围内调整配额', '将广告服务的配额调至 2 个 CPU 和 1024 MiB 内存。这条模拟轨迹进入自动处置路由。']
    ]
  }
];

const signals = [
  {
    short: '错误率', title: '商品服务错误率', value: '62', unit: '%', note: '5 分钟内从 1% 升至 62%', fixture: 'mock_dep_clean',
    records: ['product-catalog   Up 2 hours (unhealthy)', 'up{job="product-catalog"} = 0 · 错误率 1% → 62%', 'GetProduct → UNAVAILABLE · 5 条链路', 'productCatalogFailure = true', 'restart_instance 被拦截 → feishu_online_op']
  },
  {
    short: '内存', title: '推荐服务内存占用', value: '940', unit: 'MiB', note: '容器内存上限为 1024 MiB', fixture: 'mock_hybrid_leak',
    records: ['recommendation   MEM 940 MiB / 1024 MiB', 'abc1234 · cache popular items in-memory', '_POPULAR_CACHE · 无过期时间或容量限制', 'OOMKilled × 3 · 过去 1 小时', '重启后 120 MiB → auto_remediated_and_code_fix_pr']
  },
  {
    short: 'CPU', title: '广告服务 CPU 使用率', value: '99.8', unit: '%', note: 'CPU 限流比例连续 10 分钟达到 100%', fixture: 'mock_hookdeny_res',
    records: ['ad   CPU 99.8% · MEM 480 MiB / 512 MiB', 'CPU throttling = 100% · 10 分钟', 'docker exec → PreToolUse 拒绝', '--cpus 2 --memory 1024m → auto_remediated']
  }
];
const byId = id => document.getElementById(id);
const dialog = byId('stage');
const bench = byId('workbench');
const play = byId('play');
let caseIndex = 0;
let stepIndex = 0;
let timer = null;
let opener = null;

function addButton(container, label, action) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  button.addEventListener('click', action);
  container.append(button);
  return button;
}

function stopPlayback() {
  clearTimeout(timer);
  timer = null;
  const last = stepIndex === cases[caseIndex].steps.length - 1;
  play.textContent = last ? '↺ 重新播放' : '▶ 播放回放';
}

function togglePlayback() {
  if (timer !== null) {
    stopPlayback();
    return;
  }
  if (stepIndex === cases[caseIndex].steps.length - 1) goToStep(0);
  play.textContent = 'Ⅱ 暂停回放';
  function advance() {
    goToStep(stepIndex + 1);
    if (stepIndex < cases[caseIndex].steps.length - 1) timer = setTimeout(advance, 6500);
    else stopPlayback();
  }
  timer = setTimeout(advance, 6500);
}

function goToStep(index) {
  const item = cases[caseIndex];
  if (index < 0 || index >= item.steps.length) return;
  stepIndex = index;
  const [tool, title, text] = item.steps[index];
  byId('tool').textContent = tool;
  byId('step-title').textContent = title;
  byId('observation').textContent = text;
  byId('record').textContent = signals[caseIndex].records[index];
  byId('progress').textContent = '步骤 ' + (index + 1) + ' / ' + item.steps.length;
  byId('previous').disabled = index === 0;
  byId('next').disabled = index === item.steps.length - 1;
  byId('timeline').querySelectorAll('button').forEach((button, i) => {
    if (i === index) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
    button.classList.toggle('done', i < index);
  });
  const content = byId('step-content');
  content.classList.remove('arriving');
  void content.offsetWidth;
  content.classList.add('arriving');
}

function seek(index) {
  goToStep(index);
  stopPlayback();
}

function chooseCase(index) {
  caseIndex = index;
  const item = cases[index];
  const signal = signals[index];
  byId('service').textContent = item.service;
  byId('plot-service').textContent = item.service;
  byId('signal-title').textContent = signal.title;
  const unit = document.createElement('small');
  unit.textContent = signal.unit;
  byId('signal-value').replaceChildren(signal.value, unit);
  byId('signal-note').textContent = signal.note;
  byId('source').href = 'https://github.com/lane-cv/Sentinel-agentic-rl/blob/main/data/cold_start/trajectories/mock/' + signal.fixture + '.json';
  for (const container of [byId('case-buttons'), byId('signals')]) {
    [...container.children].forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
  }
  byId('timeline').replaceChildren(...item.steps.map(([tool], i) => {
    const row = document.createElement('li');
    const button = addButton(row, '', () => seek(i));
    button.className = 'timeline-btn';
    const number = document.createElement('span');
    number.className = 'index';
    number.textContent = String(i + 1);
    const name = document.createElement('span');
    name.className = 'step-name';
    name.textContent = tool;
    button.append(number, name);
    return row;
  }));
  seek(0);
  animateField();
}

function openStage(event) {
  if (typeof dialog.showModal !== 'function') return;
  event.preventDefault();
  opener = event.currentTarget;
  stopPlayback();
  byId('stage-slot').append(bench);
  byId('expand').hidden = true;
  dialog.showModal();
  byId('close-stage').focus();
}

byId('present').addEventListener('click', openStage);
byId('expand').addEventListener('click', openStage);
byId('close-stage').addEventListener('click', () => dialog.close());
dialog.addEventListener('close', () => {
  stopPlayback();
  byId('replay-home').append(bench);
  byId('expand').hidden = false;
  opener?.focus({preventScroll:true});
});
function handleShortcut(event) {
  if (event.altKey || event.ctrlKey || event.metaKey || event.defaultPrevented) return;
  if (event.key === 'ArrowRight') seek(stepIndex + 1);
  else if (event.key === 'ArrowLeft') seek(stepIndex - 1);
  else if (event.key.toLowerCase() === 'k' && !event.repeat) togglePlayback();
  else return;
  event.preventDefault();
}
document.addEventListener('keydown', event => {
  if (dialog.open || bench.contains(document.activeElement)) handleShortcut(event);
});
document.addEventListener('visibilitychange', () => { if (document.hidden) stopPlayback(); });
byId('previous').addEventListener('click', () => seek(stepIndex - 1));
byId('next').addEventListener('click', () => seek(stepIndex + 1));
play.addEventListener('click', togglePlayback);
cases.forEach((item, i) => {
  addButton(byId('case-buttons'), item.title, () => chooseCase(i));
  addButton(byId('signals'), signals[i].short, () => chooseCase(i));
});
for (const id of ['case-buttons', 'signals', 'controls']) byId(id).hidden = false;
byId('expand').hidden = typeof dialog.showModal !== 'function';

const canvas = byId('field');
const ctx = canvas.getContext('2d');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let frame = 0;

function drawField(phase) {
  if (!ctx) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = Math.round(width * dpr);
  const h = Math.round(height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.setTransform(w / 660, 0, 0, h / 300, 0, 0);
  ctx.clearRect(0, 0, 660, 300);
  const ridgeX = [.65,.72,.59][caseIndex];
  const ridgeWidth = [.15,.22,.26][caseIndex];
  function point(u, v) {
    const x = 22 + 565 * u + 81 * v;
    const across = (u - ridgeX - .035 * Math.sin(v * 5)) / ridgeWidth;
    const depth = (v - .51) / .32;
    const crest = Math.exp(-across * across - depth * depth);
    const ripple = Math.sin(u * 23 + v * 7 - phase * 2) * Math.sin(Math.PI * u);
    const y = 160 - 60 * u + 124 * v - crest * (103 + 14 * phase) - ripple * (5 + 6 * crest);
    return [x,y];
  }
  for (let row = 0; row < 48; row++) {
    const v = row / 47;
    const gradient = ctx.createLinearGradient(20,0,650,0);
    gradient.addColorStop(0,'rgba(128,128,214,.28)');
    gradient.addColorStop(.45,'rgba(165,153,255,.88)');
    const warmth = Math.exp(-Math.pow((v - .49) / .17, 2));
    gradient.addColorStop(ridgeX, 'rgb(' + Math.round(164 + 91*warmth) + ',' + Math.round(146 + 25*warmth) + ',' + Math.round(243 - 106*warmth) + ')');
    gradient.addColorStop(.96,'rgba(156,158,239,.56)');
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const [x,y] = point(i/100,v);
      if (i === 0) ctx.moveTo(x,y);
      else ctx.lineTo(x,y);
    }
    ctx.strokeStyle = gradient;
    ctx.lineWidth = row % 5 === 0 ? 1.5 : 1.05;
    ctx.stroke();
  }
  const sweep = .12 + (ridgeX - .12) * phase;
  const [px,py] = point(sweep,.5);
  ctx.fillStyle = '#ffd2b9';
  ctx.beginPath();
  ctx.arc(px,py,3.2,0,2*Math.PI);
  ctx.fill();
  ctx.strokeStyle = '#ffbc99ad';
  ctx.beginPath();
  ctx.arc(px,py,7,0,2*Math.PI);
  ctx.stroke();
}

function animateField() {
  cancelAnimationFrame(frame);
  if (reducedMotion.matches || document.hidden) { drawField(1); return; }
  const start = performance.now();
  function tick(now) {
    const phase = Math.min((now - start) / 1400, 1);
    drawField(phase);
    if (phase < 1) frame = requestAnimationFrame(tick);
  }
  frame = requestAnimationFrame(tick);
}
if ('ResizeObserver' in window) new ResizeObserver(() => drawField(1)).observe(canvas);
else window.addEventListener('resize', () => drawField(1));
reducedMotion.addEventListener('change', () => { cancelAnimationFrame(frame); drawField(1); });
chooseCase(0);
const copyButton = byId('copy');
copyButton.hidden = false;
let copyFeedback;
copyButton.addEventListener('click', async () => {
  const code = byId('commands');
  let copied = false;
  try {
    await navigator.clipboard.writeText(code.textContent);
    copied = true;
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    try { copied = document.execCommand('copy'); } catch { copied = false; }
    if (copied) selection.removeAllRanges();
  }
  const message = copied ? '已复制' : '已选中，请手动复制';
  byId('copy-status').textContent = message;
  copyButton.textContent = message;
  clearTimeout(copyFeedback);
  copyFeedback = setTimeout(() => { copyButton.textContent = '复制命令'; }, 2600);
});
