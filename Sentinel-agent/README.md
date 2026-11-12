# Sentinel agent

Sentinel agent 是面向值班工程师的命令行故障诊断工具。输入一份告警 JSON，它会结合排查手册、监控指标和调用链，输出疑似根因、判断依据和处置建议。遇到代码回归时，修复 Agent 可以检出服务仓库、运行回归测试，并在验证通过后提交修复 PR。

**[在线体验告警回放](https://lane-cv.github.io/Sentinel-agent/)** · [快速开始](#快速开始) · [故障演练](#故障演练) · [可选集成](#可选集成)

## 告警如何处置

| 告警样例 | 诊断时看什么 | 预期处置 |
| --- | --- | --- |
| [商品目录流量上涨](alerts/s6_info.json) | 请求量比上周同期高约 3%，错误率和 p99 延迟仍在目标范围内 | 生成 `info_only` 报告 |
| [广告服务 CPU 异常](alerts/s2.json) | 对齐请求量、CPU 和 p99 延迟的时间窗口，检查故障开关 | 仅对允许的目标执行白名单操作，其余交给值班人员 |
| [推荐排序回归](alerts/s10_ranking.json) | 热门商品在发版后沉底。用相同输入验证排序方向 | 修改排序逻辑，验证目标用例通过且没有新增失败后提交 PR |

仓库提供依赖故障、CPU 异常、内存泄漏和队列积压四份[排查手册](.claude/skills/)。诊断结果包含证据和置信度。置信度低于阈值（默认 `0.6`）时，系统将事件转交人工。诊断期间执行过的低风险操作会写入报告，便于值班人员复核。

## 快速开始

### 1. 安装并自检

准备 Git、Python 3.11 和 Bash 兼容终端。已有本地仓库时，直接从项目根目录创建虚拟环境。

```bash
git clone https://github.com/lane-cv/Sentinel-agent.git
cd Sentinel-agent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

安装依赖后，可以离线运行单测并检查 13 个评测场景引用的告警文件。

```bash
python -m pytest -q
python -m eval.run --dry-run
```

自检成功时，评测命令最后一行会显示 `[eval] dry-run PASSED`。

### 2. 连接模型服务

准备一个兼容 Anthropic Messages API 且支持工具调用的模型服务。从仓库根目录复制配置模板，再编辑 `.env`。终端中导出的同名变量优先。

```bash
cp -n .env.example .env
```

| 变量 | 填写内容 |
| --- | --- |
| `AIOPS_LLM_BASE_URL` | 模型服务的根地址 |
| `AIOPS_MODEL` | 服务端提供的模型名称 |
| `AIOPS_LLM_API_KEY` / `AIOPS_LLM_AUTH_TOKEN` | 根据服务的鉴权方式填写 API Key 或 Bearer Token，将不用的一项留空 |

### 3. 运行第一条告警

以下命令临时关闭历史工单检索和自动线上处置。用商品目录的流量样例即可体验诊断，无需启动 Docker 或配置 GitHub。

```bash
AIOPS_RAG_ENABLED=0 AIOPS_REMEDIATION_ENABLED=0 \
  python -m agent.run --alert alerts/s6_info.json --json-out reports/s6_info.json
```

这个样例的预期终端输出节选如下。

```text
走向 ROUTE: info_only
```

打开 `reports/s6_info.json`，先读 `diagnosis.evidence` 了解判断依据。如果路由为 `feishu_low_confidence`，再查看 `diagnosis.confidence` 和 `meta.degraded`，区分证据不足与模型调用失败。

## 故障演练

使用 Docker Compose 启动 OpenTelemetry Demo，给广告服务注入 CPU 故障。先启动 Docker，再在项目根目录执行以下命令。

```bash
./scripts/fetch-otel-demo.sh
docker compose up -d
docker compose ps
./scripts/inject.sh s2
```

等演示流量运行约 2–3 分钟，再运行诊断。可以对照 Prometheus（`localhost:9090`）中的 CPU 和请求量、Jaeger（`localhost:16686`）中的慢调用，以及 flagd（`localhost:8016`）中的开关状态。[CPU 排查手册](.claude/skills/resource-cpu/SKILL.md)

```bash
AIOPS_RAG_ENABLED=0 AIOPS_REMEDIATION_ENABLED=0 \
  python -m agent.run --alert alerts/s2.json --json-out reports/s2.json
```

结束后重置故障开关并停止容器。

```bash
./scripts/reset.sh
docker compose down
```

## 场景评测

连接模型后，用同一场景做单次检查或多轮评测。多轮结果写入 `reports/`，可用来比较路由准确率、修复验证结果和耗时。各场景的输入文件和预期路由见[评测清单](eval/expected.json)。

```bash
python -m eval.run --only s6_info
python -m eval.multi_run --only s6_info
```

## 可选集成

| 能力 | 如何启用 |
| --- | --- |
| 飞书通知 | 在 `.env` 设置 `FEISHU_WEBHOOK_URL` 接收人工复核卡片。留空时，卡片内容打印到终端。 |
| 自动处置 | `AIOPS_REMEDIATION_ENABLED` 默认开启，`AIOPS_REMEDIATION_TARGETS` 指定允许操作的服务。默认后端为 Docker，使用 Kubernetes 时设置 `AIOPS_BACKEND=k8s`。[处置白名单](agent/core/remediation.py) |
| 历史工单检索 | 开启 `AIOPS_RAG_ENABLED=1`，准备 Milvus 和向量模型。Compose 环境提供 Milvus（默认 `http://localhost:19530`），默认模型为 `BAAI/bge-small-zh-v1.5`。[工单存取实现](agent/integrations/memory.py) |

<details>
<summary><strong>在服务仓库提交修复 PR</strong></summary>

在 `.env` 中添加 `GITHUB_UPSTREAM_OWNER`（目标仓库所属用户或组织）、`GITHUB_REPO`（仓库名）和有推送与创建 PR 权限的 `GITHUB_TOKEN`。同仓提交时将 `GITHUB_FORK_OWNER` 留空，通过 fork 提交时填入 fork 所属账号。`GITHUB_DEFAULT_BRANCH` 默认是 `master`，目标分支为 `main` 时需要显式设置。修复 Agent 在 `workspace/` 下检出代码并运行验证。[修复流程](agent/agents/code_fix.py)

如需重建专用演练仓库，`scripts/seed-github.sh` 会读取同一份配置，使用已登录的 `gh` 账号创建或更新公开仓库，并强制推送演练历史。请将目标设为专门用于重建的仓库。

</details>
