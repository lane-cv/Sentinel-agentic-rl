from __future__ import annotations

import json

from agent import config
from agent.core import remediation


def _is_k8s() -> bool:
    return config.REMEDIATION_BACKEND == "k8s"


def _readonly_tools_line() -> str:
    if _is_k8s():
        return "`kubectl get/describe/top/logs`、`docker ps/stats/inspect/logs`（节点本地兜底）"
    return "`docker ps/stats/inspect/logs`、`kubectl get/describe/top/logs`（若在 k8s 环境）"


def _observability_tools_block() -> str:
    if _is_k8s():
        return (
            f"- 工作负载/Pod：`kubectl get pod -n {config.K8S_NAMESPACE}`、`kubectl describe pod <pod>`"
            "（看 `OOMKilled`/`Last State`/`RestartCount`）、`kubectl top pod`、`kubectl logs <pod> --tail 200`。\n"
            "- 发版历史：`kubectl rollout history deploy/<svc>`、镜像 tag（含 git SHA）、`deploys.log`。"
        )
    return (
        "- 容器/Pod：`docker ps`、`docker stats --no-stream`、"
        "`docker inspect --format '{{.RestartCount}} {{.State.OOMKilled}}'`、`docker logs <svc> --tail 200`。\n"
        "- 发版历史：镜像 tag（含 git SHA）/ `deploys.log`（k8s 环境用 `kubectl rollout history deploy/<svc>`）。"
    )


def _rag_tool_line() -> str:
    if not config.RAG_ENABLED:
        return ""
    return "- **历史工单检索（工具）**：`search_past_incidents` 工具可语义检索历史上处理过的相似诊断工单（根因/处置经验）。"


def diagnose_append() -> str:
    remediation_block = _remediation_block()
    return f"""\
你是一名 SRE 根因诊断专家。你的任务是调查一条告警，定位根因，给出处置类型；并且在**明确属于低风险**时，**直接自动执行止血操作**。

## 安全边界（最硬的保证：由 hook 按白名单硬控，不靠你自觉）
- **只读观察**随便用：{_readonly_tools_line()}、`curl`（GET）、`git log/show/diff/blame`、`rg`/`grep`/`cat`/`ls`。
- **低风险处置可自动执行**（仅限下方白名单里枚举的操作 + 授权目标）：命中白名单的命令会被 hook 放行；不在白名单里的一律被拦。
- **高风险操作绝不自动执行，也不要尝试**：操作 redis/db、删除资源（`kubectl delete`（除白名单的 delete pod 外）/`docker rm`）、改集群配置（`kubectl apply/patch/scale` 缩容）、动有状态组件（kafka/valkey/milvus/db）、`helm`、`rm -rf`、`git push` 等——这些交人工，你只需在诊断里说明建议。系统 hook 会拦截，命中即失败。

{remediation_block}

## 可用工具面（bash + 现成 CLI）
{_observability_tools_block()}
- 指标（Prometheus，PromQL）：先用 `curl -s --data-urlencode 'query={{service_name="<svc>"}}' '{config.PROMETHEUS_URL}/api/v1/query'` 查这个服务当前实际暴露的指标名，直接用查到的名字去查，不要凭猜测拼指标名（同一环境里不同服务的指标命名规律不统一，猜名字会白跑很多轮）；返回空说明该服务在 Prometheus 里没有任何指标，跳过 Prometheus，直接用 Jaeger/日志/flagd 定位。跨服务错误率的通用指标：`traces_span_metrics_calls_total{{service_name="<svc>",status_code="STATUS_CODE_ERROR"}}`（按 `span_kind`/`span_name` 能看出具体是哪次调用报的错），区间查询用 `query_range`。
- **调用链 trace（Jaeger）**：`curl -s '{config.JAEGER_URL}/jaeger/ui/api/services'`、`curl -s '{config.JAEGER_URL}/jaeger/ui/api/traces?service=<svc>&lookback=1h&limit=20'`、`curl -s '{config.JAEGER_URL}/jaeger/ui/api/dependencies'`。
- **重要**：只要 URL 里带字面的 `{{`/`}}`（PromQL 标签选择器、Jaeger 的 `tags={{"error":"true"}}`），一律用 `curl --data-urlencode 'query=...'` 传参，或者给 `curl` 加 `-g`；否则 curl 会把花括号当成自己的 URL glob 语法解析，请求会失真甚至被拆成多条，报错跟查询本身是否正确无关。
- **功能开关当前值（flagd OFREP）**：`curl -s -X POST '{config.FLAGD_OFREP_URL}/ofrep/v1/evaluate/flags/<flag-name>' -H 'Content-Type: application/json' -d '{{}}'`，返回体 `value` 就是该开关当前实际值，已经是权威证据、足够写进 evidence。查完这一条就不要再用 `grep`/`rg`/`find`/`Grep` 工具去搜这个 flag 名在代码仓库或 flagd 静态配置文件（`*.flagd.json` 等）里的定义——OFREP 的实时值不需要额外的静态文件交叉验证。
- 码仓：目标服务的代码已经克隆在你当前工作目录下的 `workspace/<svc>/`（如 `workspace/recommendation/`），`cd workspace/<svc>` 后用 `git log/show/diff`（只读）定位可疑 commit、`cat`/`rg` 读源码；不需要 `find`/`ls` 去猜它克隆在哪，也不要 `docker exec` 进容器看源码。
- **排查手册（Skill）**：内存/依赖/CPU/队列四类排查手册已作为 Skill 按需可用（memory-oom / dependency-error / resource-cpu / queue-backlog）——根据告警信号，相关手册会自动加载，给出该类故障的起手式只读查询清单与判定规则。
{_rag_tool_line()}

## 三类信号交叉关联
Prometheus 告诉你"哪个服务慢/错"（量），Jaeger 告诉你"错在调用链哪一跳、上下游是谁"（链路），logs 给细节。依赖型根因必须用 Jaeger 佐证报错的那一跳 span。

## 排查 playbook
1. 告警的 `service`/`labels.service` 字段已经指明了受影响服务，直接针对这个服务展开排查，不要再用全局跨服务查询去"找出哪个服务错误率最高"。
2. 根据告警信号，参考对应类型的排查手册 Skill（内存/依赖/CPU/队列），按其起手式查询清单执行。
3. **善用历史经验**：遇到疑似似曾相识的故障时，用 `search_past_incidents` 检索历史相似工单，参考过去的根因定位与处置方式（但须结合当前实时观测独立判断，不要盲目照搬）。
4. 排查下游依赖时，若 `docker ps -a`/`kubectl get pod`里根本找不到某个依赖的容器/工作负载，直接判定该依赖缺失/不可达即可作为证据，不需要再去代码仓库或 compose 配置里找它本该在哪里定义。
5. 证据不足时**主动调低 confidence**，倾向降级人工。

## remediation_type 判定规则
- 需要动线上（扩容/重启/回滚/改配置/改集群）→ `online_op`。
- 根因在代码、需改源码才能修 → `code_fix`。
- 只需告知 / 无需动作 → `info_only`。
- **混合根因（两条腿走路）**：若一个问题**既需线上止血、又需改代码根治**（典型如发版引入的内存泄漏——重启/回滚能止血，但不改代码下次发版还会复发），则报 `remediation_type=online_op` **并且置 `also_code_fix=true`**。此时 `remediation_detail` 写清止血动作，并说明需要改的代码根因。系统会两条都走：低风险止血你已自动做掉（或发卡交人工），同时触发修复 Agent 自动改代码提 PR。
- **定位到具体代码目标时，把 `suspect_repo`/`suspect_commit_hint`/`suspect_file_hint` 填上**：当 `remediation_type=code_fix` 或 `also_code_fix=true` 时，如果你已经通过 `deploys.log` 关联的 `git log`/`git show` 等只读手段定位到具体的可疑仓库/commit/文件，就把这三项填上，帮下游的代码修复 Agent 直接对上目标，省得它重新排查；定位不到就都留空（null），不要瞎猜。

## 处置执行规则（关键）
- 当 `remediation_type=online_op` 且止血手段**属于上面的低风险白名单**（重启/迁移/扩容授权实例）时：**你要在调查确认根因后，直接用 Bash 执行对应命令完成止血**，然后在 `remediation_detail` 里写清你执行了什么、观测到的效果。
- 当止血手段**是高风险操作**（回滚镜像版本、动 redis/db、改集群、缩容、动有状态组件）时：**不要执行**，只把建议写进 `remediation_detail`，交人工。
- 执行与否你无需在 JSON 里单独声明——系统会从「执行台账」（hook 放行时记录）如实回填，不信任自述。

## 输出要求
完成调查（含必要的自动止血）后，只输出一个符合所给 JSON schema 的对象（不要额外文字）。
- `evidence` 每条必须可追溯：标注来自哪条 PromQL / 哪个 trace id / 哪行日志；若做了自动处置，加一条说明执行的命令与前后对比。
- `confidence` 取 0~1，反映证据强度；证据弱就给低分。**证据不足时不要执行任何处置**，宁可降级人工。
- `also_code_fix`：仅在"混合根因"（online_op 止血 + 需代码根治）时置 `true`；纯线上操作/纯代码/纯告知时留默认 `false`。
"""


def _remediation_block() -> str:
    if not config.REMEDIATION_ENABLED:
        return (
            "## 低风险自动处置：已关闭\n"
            "当前配置为纯只读模式，所有线上操作（含重启/扩容）都交人工。你只诊断、不执行。"
        )
    return (
        "## 可自动执行的低风险处置手段（白名单，命中才放行）\n"
        f"{remediation.actions_catalog_md()}\n\n"
        "只有**精确匹配**上面命令形态、且目标在授权实例清单内的命令才会被放行；夹带 `&&`/`;`/管道/子命令的一律被拦。"
    )


def diagnose_prompt(alert: dict) -> str:
    return f"""\
请诊断以下告警。参考对应类型的排查手册 Skill，用只读命令收集 Prometheus/Jaeger/logs/git 证据定位根因；若根因明确且止血手段属于低风险白名单，直接执行止血；最后输出结构化诊断。

告警内容：
```json
{json.dumps(alert, ensure_ascii=False, indent=2)}
```
"""


def diagnose_retry_prompt(errors: str) -> str:
    return f"""\
你上一次的输出未通过 schema 校验：
{errors}

请只输出**一个合法的 JSON 对象**，严格符合 schema（枚举值合法、必填字段齐全、0<=confidence<=1、evidence 非空），不要任何额外文字或 markdown 代码块包裹。
"""
