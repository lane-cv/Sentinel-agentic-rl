# Sentinel agent

Sentinel agent 是一个命令行故障诊断与修复工具。给它一份告警 JSON，诊断 Agent 会结合告警内容、排查手册和当前可用的监控数据，输出疑似故障服务、根因、证据和置信度。比如，同样是广告服务 CPU 升高，Agent 可以查询请求量和 CPU 指标，判断两者是否同步变化，再决定如何处置。

信息类告警直接生成报告。需要线上处置时，Agent 可以对允许的目标执行白名单内的操作，例如重启单个容器。需要人工判断或诊断置信度不足时，系统生成飞书卡片。对于代码缺陷，修复 Agent 可以检出目标仓库、修改代码、运行回归测试，并在验证通过后提交拉取请求（PR）。

## 安装与自检

以下命令适用于 macOS 的终端，使用 Python 3.11。在项目根目录执行，后续示例沿用同一个虚拟环境。

```bash
cd "$HOME/Desktop/Sentinel agent"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m eval.run --dry-run
```

安装完成后，最后两条命令可以离线运行。单测应显示 `5 passed`。评测自检会逐项检查 13 个场景引用的告警文件，最后显示 `[eval] dry-run PASSED`。

## 运行第一次诊断

**配置模型。** 项目通过 Claude Agent SDK 调用支持 Anthropic Messages API 和工具调用的模型服务。先准备可访问的模型服务，再创建本机配置文件。已有 `.env` 时直接编辑即可。

```bash
cp -n .env.example .env
```

在 `.env` 中设置以下四项。程序启动时会自动读取，终端中已导出的同名变量优先。

| 变量 | 如何填写 |
| --- | --- |
| `AIOPS_LLM_BASE_URL` | 模型服务的根地址。示例值为 `http://localhost:8000`，应指向已启动的服务。 |
| `AIOPS_MODEL` | 服务实际提供的模型名。模板中的 `opus` 需要与服务端配置匹配。 |
| `AIOPS_LLM_API_KEY` | 服务使用 API Key 鉴权时填入密钥，否则将该项设为空值。 |
| `AIOPS_LLM_AUTH_TOKEN` | 服务使用 Bearer Token 鉴权时填入令牌，否则将该项设为空值。 |

**运行一个信息类告警。** 样例中的商品目录服务请求量小幅上涨，错误率和延迟仍在正常范围。这次运行临时关闭历史工单检索和自动线上处置，只需连接模型即可体验诊断流程。

```bash
AIOPS_RAG_ENABLED=0 AIOPS_REMEDIATION_ENABLED=0 \
  python -m agent.run --alert alerts/s6_info.json --json-out reports/s6_info.json
```

终端会打印 `ROUTE` 和诊断摘要，完整报告写入 `reports/s6_info.json`。这个样例的预期路由是 `info_only`。查看 `diagnosis.evidence` 可了解模型引用了哪些证据。如果结果是 `feishu_low_confidence`，先看 `diagnosis.confidence` 和 `meta.degraded`，再检查模型地址、鉴权和结构化输出。`FEISHU_WEBHOOK_URL` 留空时，生成的卡片会打印到终端。

## 用 Docker 复现 CPU 告警

要让 Agent 查到真实指标，可以启动 OpenTelemetry Demo 并注入广告服务 CPU 故障。先启动 Docker，确认 `git` 和 `docker compose` 可用，再从项目根目录执行。

```bash
./scripts/fetch-otel-demo.sh
docker compose up -d
docker compose ps
./scripts/inject.sh s2
```

服务就绪后，让演示流量运行约 2–3 分钟，再执行诊断。Prometheus 默认监听 `localhost:9090`，Jaeger 监听 `localhost:16686`，flagd 的开关查询接口监听 `localhost:8016`。

```bash
AIOPS_RAG_ENABLED=0 AIOPS_REMEDIATION_ENABLED=0 \
  python -m agent.run --alert alerts/s2.json --json-out reports/s2.json
```

演示结束后，重置故障开关并停止容器。

```bash
./scripts/reset.sh
docker compose down
```

## 评测与扩展

- **评测诊断效果。** 配好模型后，运行 `python -m eval.run --only s6_info`，即可对照预期检查一次真实诊断。运行 `python -m eval.multi_run --only s6_info` 可重复评测，逐次结果和汇总指标会写入 `reports/`。场景及预期见 [eval/expected.json](eval/expected.json)。
- **排查手册。** 诊断 Agent 从 `.claude/skills/` 加载四份手册，分别处理依赖故障、CPU 异常、内存泄漏和队列积压。可以从 [CPU 排查手册](.claude/skills/resource-cpu/SKILL.md) 查看上述示例的查询步骤。
- **自动处置。** `AIOPS_REMEDIATION_ENABLED` 默认是 `1`，`AIOPS_REMEDIATION_TARGETS` 指定允许操作的服务。默认后端是 Docker，也支持 `AIOPS_BACKEND=k8s`。具体允许的命令见 [处置白名单](agent/core/remediation.py)。
- **历史工单检索。** 在 `.env` 中设置 `AIOPS_RAG_ENABLED=1`，并提供可访问的 Milvus 和向量模型。上述 Compose 环境包含 Milvus，默认地址是 `http://localhost:19530`。首次使用需能加载 `BAAI/bge-small-zh-v1.5`。
- **自动提交修复 PR（可选）。** 启用时，在 `.env` 中添加 `GITHUB_UPSTREAM_OWNER`（上游用户或组织）、`GITHUB_REPO`（仓库名）和有权限的 `GITHUB_TOKEN`。同仓提交时将 `GITHUB_FORK_OWNER` 留空，跨仓提交时填入 fork 的所有者。`GITHUB_DEFAULT_BRANCH` 默认是 `master`，应与目标分支一致。配置完成后，修复 Agent 会在 `workspace/` 下检出代码，验证修复后向上游发起 PR。入口见 [代码修复 Agent](agent/agents/code_fix.py)。

初始化专用故障演练仓库时，`scripts/seed-github.sh` 从同一份 `.env` 读取仓库配置，使用已登录的 `gh` 账号创建或更新公开仓库，并强制推送演练历史。请将目标设为专门用于重建的仓库。
