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

const byId = id => document.getElementById(id);
const caseButtons = byId('case-buttons');
let selectedCase = 0;
let selectedStep = 0;

function renderStep() {
  const item = cases[selectedCase];
  const [tool, title, evidence] = item.steps[selectedStep];
  byId('service').textContent = item.service;
  byId('progress').textContent = '步骤 ' + (selectedStep + 1) + ' / ' + item.steps.length;
  byId('tool').textContent = tool;
  byId('step-title').textContent = title;
  byId('observation').textContent = evidence;
  byId('previous').disabled = selectedStep === 0;
  byId('next').disabled = selectedStep === item.steps.length - 1;
  [...caseButtons.children].forEach((button, index) => button.setAttribute('aria-pressed', String(index === selectedCase)));
  byId('segments').replaceChildren(...item.steps.map((_, index) => {
    const bar = document.createElement('i');
    if (index === selectedStep) bar.className = 'current';
    if (index < selectedStep) bar.className = 'past';
    return bar;
  }));
}

cases.forEach((item, index) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'case-btn';
  const label = document.createElement('span');
  label.textContent = item.title;
  const service = document.createElement('small');
  service.textContent = item.service;
  label.append(service);
  const arrow = document.createElement('span');
  arrow.className = 'chevron';
  arrow.setAttribute('aria-hidden', 'true');
  arrow.textContent = '↗';
  button.append(label, arrow);
  button.addEventListener('click', () => {
    selectedCase = index;
    selectedStep = 0;
    renderStep();
  });
  caseButtons.append(button);
});
byId('previous').addEventListener('click', () => { selectedStep--; renderStep(); });
byId('next').addEventListener('click', () => { selectedStep++; renderStep(); });
caseButtons.hidden = false;
byId('controls').hidden = false;
renderStep();

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

const canvas = byId('topology');
const context = canvas.getContext('2d');

function drawTopology() {
  if (!context) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 3);
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const scale = Math.min(canvas.width / 640, canvas.height / 316);
  context.setTransform(scale, 0, 0, scale, (canvas.width - 640 * scale) / 2, (canvas.height - 316 * scale) / 2);
  const project = (x, z, h = 0) => [314 + (x - z) * 40, 174 + (x + z) * 18 - h];
  function shape(points, stroke, fill, width = 1) {
    context.beginPath();
    context.moveTo(...points[0]);
    points.slice(1).forEach(point => context.lineTo(...point));
    if (fill) {
      context.closePath();
      context.fillStyle = fill;
      context.fill();
    }
    if (stroke) {
      context.strokeStyle = stroke;
      context.lineWidth = width;
      context.stroke();
    }
  }
  const glow = context.createRadialGradient(314, 144, 0, 314, 144, 210);
  glow.addColorStop(0, '#293e2a');
  glow.addColorStop(1, '#14251c');
  context.fillStyle = glow;
  context.fillRect(-200, -100, 1040, 516);
  for (let i = -18; i <= 18; i++) {
    const line = i % 2 === 0 ? '#364b3b' : '#2c4031';
    shape([project(i / 2, -12), project(i / 2, 12)], line, null, .72);
    shape([project(-12, i / 2), project(12, i / 2)], line, null, .72);
  }
  const links = [
    [[-3,-1],[-3,0],[0,0]],
    [[-1,-3],[-1,-1],[0,-1],[0,0]],
    [[0,0],[1,0],[1,-2],[3,-2]],
    [[0,0],[0,1],[2,1],[2,3]],
    [[0,0],[-1,0],[-1,2],[-2,2]],
    [[-2,2],[-2,3.8],[0,3.8]],
    [[3,-2],[3,0],[4,0]]
  ];
  links.forEach((path, i) => {
    const points = path.map(([x,z]) => project(x,z));
    shape(points, '#112219', null, 6);
    shape(points, i < 2 ? '#b4de6b' : '#769f78', null, i < 2 ? 2.1 : 1.5);
  });
  const nodes = [
    {x:-3,z:-1,h:22}, {x:-1,z:-3,h:15}, {x:0,z:0,h:50,active:true},
    {x:3,z:-2,h:26}, {x:-2,z:2,h:14}, {x:0,z:3.8,h:16},
    {x:2,z:3,h:20}, {x:4,z:0,h:15}
  ];
  function block(node, radius, base, top, colors) {
    const {x,z} = node;
    const ring = height => [[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b]) => project(x + a * radius, z + b * radius, height));
    const low = ring(base);
    const high = ring(top);
    shape([high[3], high[2], low[2], low[3]], colors[3], colors[0], 1.2);
    shape([high[2], high[1], low[1], low[2]], colors[3], colors[1], 1.2);
    shape(high, colors[3], colors[2], 1.2);
  }
  nodes.sort((a,b) => a.x + a.z - b.x - b.z).forEach(node => {
    const {x,z,h,active} = node;
    const edge = active ? '#d3fa86' : '#83a78b';
    const palette = active ? ['#8fb64b','#759940','#bfef6a',edge] : ['#375540','#294534','#486a50',edge];
    block(node, active ? .77 : .50, 0, 4, ['#243a2a','#213427','#334b34',active ? '#87a951' : '#536e57']);
    block(node, active ? .64 : .40, 5, h, palette);
    if (active) {
      for (const level of [13, 22, 31, 40]) {
        shape([project(x-.64,z+.64,level),project(x+.64,z+.64,level),project(x+.64,z-.64,level)], '#648743', null, 1);
      }
      block(node, .4, h, h + 3, ['#a3d45c','#8dc34f','#d7ff90','#ecffb7']);
    } else {
      shape([project(x-.20,z+.10,h+1),project(x+.08,z+.10,h+1)], '#afc9ac', null, 1.4);
    }
  });
  shape([[314,117],[345,67],[377,67]], '#ceef98', null, 1.2);
  context.fillStyle = '#daff9c';
  context.fillRect(312,115,4,4);
  context.fillStyle = '#213629';
  context.strokeStyle = '#718a5e';
  context.lineWidth = 1;
  context.fillRect(377,54,141,26);
  context.strokeRect(377,54,141,26);
  context.font = '600 13px Menlo, Consolas, monospace';
  context.fillStyle = '#d5f9a7';
  context.fillText('product-catalog',388,71);
}
if ('ResizeObserver' in window) {
  new ResizeObserver(drawTopology).observe(canvas);
} else {
  window.addEventListener('resize', drawTopology);
}
drawTopology();
