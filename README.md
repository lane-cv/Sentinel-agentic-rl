# Sentinel Agentic RL

Sentinel 用大语言模型排查服务告警。诊断 Agent 从指标、调用链和日志中寻找证据，给出疑似根因，再决定生成报告、执行白名单操作、转交值班人员或发起代码修复。训练管线把这些排查步骤转成监督微调（SFT）样本，并用 GRPO 对工具调用和最终诊断打分。

**[交互演示](https://lane-cv.github.io/Sentinel-agentic-rl/)** · [快速开始](#快速开始) · [用模型诊断告警](#用模型诊断告警) · [训练与部署](#训练与部署) · [下载 LoRA 权重](https://cloud.tsinghua.edu.cn/f/99e43c53c0ae49129463/)

## 一次诊断

在仓库自带的模拟案例中，`product-catalog` 的错误率在五分钟内从 1% 升至 62%。最近五条 Jaeger 调用链都在 `GetProduct` 返回 `UNAVAILABLE`，容器日志则记录了 `productCatalogFailure` 故障注入开关生效。Agent 据此将问题定位到商品目录服务。该服务超出自动处置白名单，重启请求被拦截，事件通过 `feishu_online_op` 路由交给值班人员。[查看完整轨迹](data/cold_start/trajectories/mock/mock_dep_clean.json)

## 快速开始

准备 Git、Bash 和 Python 3.11。下面的示例在 CPU 上使用模拟轨迹，首次安装依赖需要联网，无需模型密钥或 GPU。

```bash
git clone https://github.com/lane-cv/Sentinel-agentic-rl.git
cd Sentinel-agentic-rl
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash smoke_test/run_smoke.sh
```

最后一行包含 `烟雾测试全部步骤跑通`。默认示例生成 30 条种子告警，读取 5 条模拟轨迹，筛选后将其中 3 条拆成 14 条 SFT 样本。仅安装根目录依赖时，告警去重使用词法相似度。

| 文件 | 看什么 |
| --- | --- |
| `smoke_test/out/aiops_cold_start_sft.jsonl` | 每行是一条训练样本，可查看同一次排查在不同步骤可见的对话前缀。 |
| `smoke_test/out/baseline_comparison.json` | 五个带标签样例的评分结果。`evaluation_mode` 为 `scripted_mock`，脚本端点可以读取标准答案，适合核对评分逻辑。 |

## 用模型诊断告警

在同一虚拟环境中，从仓库根目录复制配置模板。

```bash
cp -n Sentinel-agent/.env.example Sentinel-agent/.env
```

在 `Sentinel-agent/.env` 中设置 `AIOPS_LLM_BASE_URL` 和 `AIOPS_MODEL`，指向兼容 Anthropic Messages API 且支持工具调用的模型服务。按服务的鉴权方式填写 `AIOPS_LLM_API_KEY` 或 `AIOPS_LLM_AUTH_TOKEN`，将另一项留空。

先诊断一条信息类告警。商品目录服务的请求量小幅上涨，错误率和延迟仍在目标范围内，预期路由是 `info_only`。这条命令临时关闭历史工单检索和自动处置。

```bash
(cd Sentinel-agent && AIOPS_RAG_ENABLED=0 AIOPS_REMEDIATION_ENABLED=0 \
  python -m agent.run --alert alerts/s6_info.json --json-out reports/s6_info.json)
```

打开 `Sentinel-agent/reports/s6_info.json`，从 `diagnosis.evidence` 核对依据，再看 `diagnosis.confidence` 和 `route` 了解置信度与处置结果。要采集演练环境的指标或启用历史工单检索，参见 [Agent 运行指南](Sentinel-agent/README.md)。

## 评测

| 要验证什么 | 从仓库根目录运行 |
| --- | --- |
| 单元测试 | `python -m pytest -q` |
| 13 个场景的告警文件是否齐全 | `(cd Sentinel-agent && python -m eval.run --dry-run)` |
| 前缀切分如何改变样本数 | `python -m eval.ablations.no_prefix_split` 对比五条模拟轨迹，加 `--show-reference` 可查看参考值。 |
| 脚本化端点的评分逻辑 | `python -m eval.run_baseline_comparison` |

配置模型后，可以在同一条告警上重复诊断。`s6_info` 默认运行三次，逐次记录和汇总指标 `metrics.json` 写入 `Sentinel-agent/reports/` 下的运行目录。[场景与预期结果](Sentinel-agent/eval/expected.json)

```bash
(cd Sentinel-agent && AIOPS_RAG_ENABLED=0 AIOPS_REMEDIATION_ENABLED=0 \
  python -m eval.multi_run --only s6_info)
```

## 训练与部署

### 使用已有权重

运行下载脚本会从清华云盘获取 ZIP，校验 SHA-256，再将 `sft-v8`、`sft-v9`、`grpo-step10` 和 `grpo-step15` 四组 LoRA adapter 解包到 `weights/`。加载时使用 Qwen3.5-9B 纯文本底座及对应 tokenizer。[权重下载页](https://cloud.tsinghua.edu.cn/f/99e43c53c0ae49129463/)

```bash
bash artifacts/download_weights.sh
```

### 重建训练集

完整数据需要一份包含指定版本的 Git 数据仓库。把示例路径替换为该仓库的实际路径，也可以通过 `SENTINEL_DATA_REPO` 指定本地仓库，或用 `SENTINEL_DATA_REMOTE` 指定远端。[数据版本与 SHA-256 校验值](data/prepare_training_data.sh)

```bash
bash data/prepare_training_data.sh /path/to/data-repository
```

| 输出文件 | 内容 |
| --- | --- |
| `data/cold_start/aiops_cold_start_sft_v9.jsonl` | 由 63 条筛选后的诊断轨迹拆出的 667 条 SFT 样本 |
| `grpo/verl_data/train.jsonl` | 74 条 GRPO 训练告警 |
| `grpo/verl_data/held_out.jsonl` | 22 条留出评测告警 |

### GPU 训练与推理

在配有 CUDA 的 Linux 机器上准备 GPU 版 PyTorch 和支持 Qwen3.5 的 Transformers。以下命令从仓库根目录运行，将 `/path/to/...` 替换为实际路径。

1. **导出纯文本底座。** 导出后，将输出目录填入 SFT 配置的 `model_name_or_path`。[SFT 配置](sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml)

   ```bash
   python sft/load_text_only.py --source-checkpoint /path/to/qwen3.5-9b --output-dir /path/to/text-base
   ```

2. **运行 LoRA SFT。** 安装 LLaMA-Factory 后指定源码目录。训练产物位于该目录下的 `saves/aiops-diagnose-qwen3.5-9b/lora/sft/`。

   ```bash
   LLAMAFACTORY_DIR=/path/to/LLaMA-Factory bash sft/train_sft.sh
   ```

3. **合并 adapter。** 安装 PEFT 后，选择上一步的训练产物或下载的 adapter。下面使用 `sft-v9`。

   ```bash
   python deploy/merge_lora.py --base-checkpoint /path/to/text-base \
     --adapter-dir weights/sft-v9 --output-dir /path/to/merged-model
   ```

4. **运行 GRPO。** 安装 veRL 和 vLLM，配置到诊断环境的 SSH 连接（`AIOPS_TUNNEL_HOST`、`AIOPS_TUNNEL_PORT`、`AIOPS_TUNNEL_USER`），并通过 `AIOPS_MAC_AGENT_DIR` 指定远端 Agent 目录。[工具调用与远端执行](verl_adapter/rollout_worker.py)

   ```bash
   MODEL_PATH=/path/to/merged-model bash grpo/train_grpo.sh
   ```

要提供推理服务，将 `MODEL_DIR` 指向要部署的合并模型。`SERVED_MODEL_NAME` 设置客户端请求时使用的模型名。

```bash
MODEL_DIR=/path/to/merged-model SERVED_MODEL_NAME=sentinel bash deploy/vllm_serve.sh
```

数据生成、奖励计算和训练时的工具适配可分别从以下目录阅读。[数据管线](data/) · [奖励函数](reward/) · [veRL 适配层](verl_adapter/)
