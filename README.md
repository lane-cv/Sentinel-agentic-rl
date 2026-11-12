# Sentinel Agentic RL

Sentinel 从一条服务告警开始诊断。Agent 查询日志、指标和历史工单，输出疑似故障服务、根因、证据和置信度，再根据诊断结果选择记录报告、转人工、执行白名单操作或发起代码修复。训练代码把多步排查轨迹转成监督微调（SFT）样本，并为 GRPO 计算工具调用、诊断结果和各步骤对最终结果的贡献。

## 本地快速运行

在项目根目录打开 Bash，使用 Python 3.11 执行以下命令。首次安装依赖需要联网，安装后可在 CPU 上用模拟告警跑通数据处理和评测。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash smoke_test/run_smoke.sh
```

看到“烟雾测试全部步骤跑通”即表示流程完成。默认生成 30 条种子告警和 5 条模拟轨迹，筛选后留下 3 条轨迹，拆成 14 条 SFT 样本。训练样本写入 `smoke_test/out/aiops_cold_start_sft.jsonl`。同目录的 `baseline_comparison.json` 记录脚本化端点在 5 个样例上的表现，用于检验评分逻辑。

运行单元测试，并检查诊断 Agent 的 13 个场景是否都能找到告警文件：

```bash
python -m pytest -q
(cd Sentinel-agent && python -m eval.run --dry-run)
```

### 验证评分与前缀切分

`python -m eval.run_baseline_comparison` 使用带标签的样例和能读取标准答案的脚本化端点。报告中的 `evaluation_mode` 为 `scripted_mock`，可用于检查路由准确率等指标的计算。

`python -m eval.ablations.no_prefix_split` 对全部 5 条模拟轨迹分别统计切分前后的样本数。加 `--show-reference` 可另外查看引用的参考值。

## 用模型诊断一条告警

从项目根目录复制配置模板：

```bash
cp -n Sentinel-agent/.env.example Sentinel-agent/.env
```

编辑 `Sentinel-agent/.env`，把 `AIOPS_LLM_BASE_URL` 设为支持 Anthropic Messages API 和工具调用的模型服务地址，把 `AIOPS_MODEL` 设为该服务识别的模型名。按服务的鉴权方式填写 `AIOPS_LLM_API_KEY` 或 `AIOPS_LLM_AUTH_TOKEN`，将未使用的一项置空。

下面的样例临时关闭历史工单检索和自动线上处置，使用商品目录服务的信息类告警。命令会调用配置的模型，并把结果写到 `Sentinel-agent/reports/s6_info.json`。查看报告的 `route`、`diagnosis.evidence` 和 `diagnosis.confidence` 可核对路由、证据和置信度。

```bash
(cd Sentinel-agent && AIOPS_RAG_ENABLED=0 AIOPS_REMEDIATION_ENABLED=0 \
  python -m agent.run --alert alerts/s6_info.json --json-out reports/s6_info.json)
```

要让 Agent 查看真实的演示环境指标，可在装有 Docker Compose 的机器上，进入 `Sentinel-agent/`，先运行 `bash scripts/fetch-otel-demo.sh`，再运行 `docker compose up -d`。使用历史工单检索还需要安装 `Sentinel-agent/requirements.txt`、配置可访问的 Milvus 并加载向量模型。

要记录配置模型在同一场景的多次诊断结果，可在完成上述配置后运行：

```bash
(cd Sentinel-agent && AIOPS_RAG_ENABLED=0 AIOPS_REMEDIATION_ENABLED=0 \
  python -m eval.multi_run --only s6_info)
```

该场景默认运行三次。可在 `Sentinel-agent/reports/` 下查看每次诊断的 JSONL 记录和汇总文件 `metrics.json`。

## 准备训练数据

CPU 烟雾测试使用仓库自带的模拟样例。重建完整训练集需要一份包含指定数据版本的 Git 仓库。版本号和校验值见[数据准备脚本](data/prepare_training_data.sh)。

```bash
bash data/prepare_training_data.sh /path/to/data-repository
```

也可通过 `SENTINEL_DATA_REPO` 指定本地数据仓库，或通过 `SENTINEL_DATA_REMOTE` 指定提供该版本的远端。

脚本从 63 条已筛选轨迹重建 SFT 样本，并按已有划分准备 GRPO 数据。三个文件均通过 SHA-256 校验后才写入项目。

| 输出文件 | 条数 |
| --- | ---: |
| `data/cold_start/aiops_cold_start_sft_v9.jsonl` | 667 |
| `grpo/verl_data/train.jsonl` | 74 |
| `grpo/verl_data/held_out.jsonl` | 22 |

## GPU 训练与部署

| 可选资源 | 链接 |
| --- | --- |
| LoRA 权重（SFT、GRPO，共四组） | [清华云盘下载 ZIP](https://cloud.tsinghua.edu.cn/f/99e43c53c0ae49129463/) |

在项目根目录运行下面的命令。脚本从清华云盘下载 ZIP，校验 SHA-256 后将四组 adapter 放入 `weights/`。加载时搭配 Qwen3.5-9B 纯文本底座及其 tokenizer。

```bash
bash artifacts/download_weights.sh
```

在配有 CUDA 的 Linux 机器上准备 GPU 版 PyTorch、支持 Qwen3.5 的 Transformers，以及各阶段使用的框架。以下入口从项目根目录运行。

| 阶段 | 入口与输入 |
| --- | --- |
| 准备底座 | 用 `python sft/load_text_only.py --source-checkpoint <原始模型目录> --output-dir <纯文本底座目录>` 导出纯文本模型，并将输出路径填入 [SFT 配置](sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml) 的 `model_name_or_path`。 |
| LoRA SFT | 安装 LLaMA-Factory，将 `LLAMAFACTORY_DIR` 指向其源码目录。运行 `bash sft/train_sft.sh` 注册数据集并训练，adapter 保存在该目录下的 `saves/aiops-diagnose-qwen3.5-9b/lora/sft/`。 |
| 合并权重 | 安装 PEFT，运行 `python deploy/merge_lora.py --base-checkpoint <纯文本底座目录> --adapter-dir <adapter目录> --output-dir <合并模型目录>`。 |
| GRPO | 安装 veRL 和 vLLM，将 `MODEL_PATH` 指向合并模型，配置训练机到诊断环境的 SSH 连接（`AIOPS_TUNNEL_HOST`、`AIOPS_TUNNEL_PORT`、`AIOPS_TUNNEL_USER`）及远端 Agent 目录（`AIOPS_MAC_AGENT_DIR`），再运行 `bash grpo/train_grpo.sh`。 |
| 启动推理服务 | 将 `MODEL_DIR` 指向待部署的合并模型，运行 `bash deploy/vllm_serve.sh`。服务名由 `SERVED_MODEL_NAME` 设置。 |

告警生成与轨迹转换代码在 [data](data/)，逐步奖励和轨迹奖励在 [reward](reward/)，veRL 工具与观察结果的适配代码在 [verl_adapter](verl_adapter/)。
