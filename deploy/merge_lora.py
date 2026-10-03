from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED_CAUSAL_LM_CLASS = "Qwen3_5ForCausalLM"


def _check_merge_dependencies():
    try:
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "[merge_lora] 当前环境没有安装 `transformers`，本脚本无法运行。\n"
            "  pip install -U transformers\n"
            "装好之后重新运行本脚本。"
        ) from exc

    causal_lm_cls = getattr(transformers, EXPECTED_CAUSAL_LM_CLASS, None)
    if causal_lm_cls is None:
        available_qwen3 = sorted(name for name in dir(transformers) if "Qwen3" in name)
        qwen3_symbols_msg = str(available_qwen3) if available_qwen3 else (
            "(一个都没有，说明这个 transformers 版本还完全不认识 Qwen3.5，可能太旧)"
        )
        raise SystemExit(
            f"[merge_lora] 本仓库的设计假设是 transformers 提供 `{EXPECTED_CAUSAL_LM_CLASS}` 这个"
            "纯文本因果语言模型类（跟 sft/load_text_only.py 剥离视觉塔时用的类必须是同一个，"
            "否则 base_model 的模块结构跟 LoRA adapter 训练时不一致），但当前安装的 "
            f"transformers=={transformers.__version__} 里找不到这个类。\n"
            f"  当前环境里名字含 'Qwen3' 的类/符号: {qwen3_symbols_msg}\n"
            "  排查步骤同 sft/load_text_only.py 的报错信息：先确认 transformers 版本，再查 "
            "Qwen3.5 官方 model card 上的真实类名，如果类名变了就同步改这里的 "
            "EXPECTED_CAUSAL_LM_CLASS（以及 sft/load_text_only.py 里的同名常量，两处必须一致）。"
        )

    try:
        import peft
    except ImportError as exc:
        raise SystemExit(
            "[merge_lora] 当前环境没有安装 `peft`，本脚本无法运行。\n"
            "  pip install -U peft\n"
            "装好之后重新运行本脚本。"
        ) from exc

    if not hasattr(peft, "PeftModel"):
        raise SystemExit(
            f"[merge_lora] 当前安装的 `peft`（版本 {getattr(peft, '__version__', '未知')}）"
            "里找不到 `PeftModel`，这个安装可能损坏或者版本极其古老。\n"
            "  pip install -U --force-reinstall peft"
        )

    return transformers, causal_lm_cls, peft


def _import_torch_or_die():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise SystemExit(
            "[merge_lora] 没有安装 `torch`，请先 `pip install torch`（配合目标 GPU 的 CUDA 版本）。"
        ) from exc


def _validate_paths(base_checkpoint: str, adapter_dir: Path, output_dir: Path) -> None:
    adapter_dir = adapter_dir.expanduser()
    if not adapter_dir.exists():
        raise SystemExit(
            f"[merge_lora] --adapter-dir 指向的目录不存在: {adapter_dir}\n"
            "  这里应该填 LLaMA-Factory LoRA SFT 训练产出的 output_dir（PEFT 标准格式，"
            "目录里应该有 adapter_config.json + adapter_model.safetensors），"
            "见 sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 的 `output_dir` 字段。"
        )
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise SystemExit(
            f"[merge_lora] --adapter-dir={adapter_dir} 存在，但里面没有 adapter_config.json，"
            "不像一份 PEFT 标准格式的 LoRA adapter 目录。检查是不是指错了目录"
            "（比如指到了训练 output_dir 的上一级，或者指到了某个 checkpoint-N 子目录才对）。"
        )

    looks_like_local_path = base_checkpoint.startswith(("/", "./", "../", "~"))
    if looks_like_local_path and not Path(base_checkpoint).expanduser().exists():
        raise SystemExit(
            f"[merge_lora] --base-checkpoint 看起来是本地路径但不存在: {base_checkpoint}\n"
            "  这里应该填 sft/load_text_only.py --output-dir 产出的 text-only checkpoint 目录，"
            "不是原始的 Qwen3.5-9B 多模态 checkpoint。"
        )

    if output_dir.expanduser().exists() and any(output_dir.expanduser().iterdir()):
        raise SystemExit(
            f"[merge_lora] --output-dir={output_dir} 已存在且非空，为避免覆盖已有 merge 产物，"
            "请换一个空目录，或者先手动清空/搬走这个目录再重跑。"
        )


def merge_lora_and_save(
    base_checkpoint: str,
    adapter_dir: Path,
    output_dir: Path,
    dtype: str,
    attn_implementation: str,
    trust_remote_code: bool,
) -> None:
    transformers, causal_lm_cls, peft = _check_merge_dependencies()
    torch = _import_torch_or_die()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype not in dtype_map:
        raise SystemExit(f"[merge_lora] 不支持的 --dtype={dtype!r}，可选: {list(dtype_map)}")

    print(
        f"[merge_lora] 用 {EXPECTED_CAUSAL_LM_CLASS}.from_pretrained('{base_checkpoint}', "
        f"torch_dtype={dtype}, attn_implementation={attn_implementation!r}) 加载 text-only 底座……"
    )
    base_model = causal_lm_cls.from_pretrained(
        base_checkpoint,
        torch_dtype=dtype_map[dtype],
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )

    print(f"[merge_lora] 用 PeftModel.from_pretrained 加载 LoRA adapter: {adapter_dir} ……")
    peft_model = peft.PeftModel.from_pretrained(base_model, str(adapter_dir))

    print(
        "[merge_lora] 调用 merge_and_unload() 把 LoRA 权重合并进底座权重……\n"
        "  预期行为：合并后返回一个普通 transformers 模型（不再有 PEFT 专属方法），"
        "参数量跟底座本身基本相同（LoRA 增量被吸收进原始权重矩阵，不额外增加参数）。"
    )
    merged_model = peft_model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge_lora] 保存合并后的单体 checkpoint 到 {output_dir} ……")
    merged_model.save_pretrained(output_dir)

    print("[merge_lora] 保存 tokenizer（跟底座一致，不需要改动）……")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_checkpoint, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(output_dir)

    print(
        f"[merge_lora] 完成。合并后的 checkpoint 已写入 {output_dir}\n"
        "  下一步：把这个目录路径填进 deploy/vllm_serve.sh 的 MODEL_DIR，用 vLLM 起服务。"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "把 sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 训练产出的 LoRA adapter "
            "合并到纯文本 Qwen3.5-9B 底座，"
            "得到一份 vLLM 可以直接 --model 加载的单体 checkpoint。"
        )
    )
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        help=(
            "text-only 底座路径，即 sft/load_text_only.py --output-dir 的产物目录（绝对路径），"
            "不是原始 Qwen3.5-9B 多模态 checkpoint。"
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        type=Path,
        help=(
            "LoRA adapter 目录（PEFT 标准格式），即 "
            "sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 里 `output_dir` 字段指向的训练产出。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="merge 后的单体 checkpoint 输出目录，交给 deploy/vllm_serve.sh 使用。",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="加载/保存精度，默认 bf16（与项目的 SFT 配置一致）。",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help=(
            "transformers `from_pretrained(attn_implementation=...)` 的取值，默认 "
            "'flash_attention_2'。如果目标机器没装 flash-attn，改成 'sdpa'。"
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Qwen3.5 是新架构，发布初期大概率仍需要 trust_remote_code=True（默认开启）。",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _validate_paths(args.base_checkpoint, args.adapter_dir, args.output_dir)

    merge_lora_and_save(
        base_checkpoint=args.base_checkpoint,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "[merge_lora] 未提供参数，仅做一次 transformers/peft 环境自检：\n"
            "  用法: python3 deploy/merge_lora.py --base-checkpoint <text-only 底座路径> "
            "--adapter-dir <LoRA adapter 目录> --output-dir <merge 产物输出目录>"
        )
        _check_merge_dependencies()
        print("[merge_lora] 环境自检通过：transformers 里找到了 Qwen3_5ForCausalLM，且 peft 已安装。")
        sys.exit(0)
    main()
