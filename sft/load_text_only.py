from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED_CAUSAL_LM_CLASS = "Qwen3_5ForCausalLM"
EXPECTED_CONDITIONAL_GEN_CLASS = "Qwen3_5ForConditionalGeneration"


def _check_transformers_support():
    try:
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "[load_text_only] 当前环境没有安装 `transformers`，本脚本无法运行。\n"
            "  pip install -U transformers\n"
            "装好之后重新运行本脚本；如果装的版本仍然没有 Qwen3_5ForCausalLM，"
            "见下面『版本不匹配』分支的排查建议。"
        ) from exc

    causal_lm_cls = getattr(transformers, EXPECTED_CAUSAL_LM_CLASS, None)
    if causal_lm_cls is not None:
        return transformers, causal_lm_cls

    available_qwen3 = sorted(name for name in dir(transformers) if "Qwen3" in name)
    has_conditional_gen_only = EXPECTED_CONDITIONAL_GEN_CLASS in available_qwen3

    hint = ""
    if has_conditional_gen_only:
        hint = (
            f"\n  检测到 `{EXPECTED_CONDITIONAL_GEN_CLASS}`（多模态类）存在但纯文本类不存在——"
            "说明这个 transformers 版本目前只暴露了多模态入口，还没有（或者改名成了别的）"
            "对应的纯文本因果 LM 类。"
        )

    if available_qwen3:
        qwen3_symbols_msg = str(available_qwen3)
    else:
        qwen3_symbols_msg = "(一个都没有，说明这个 transformers 版本还完全不认识 Qwen3.5，可能太旧)"

    raise SystemExit(
        f"[load_text_only] 本仓库的设计假设是 transformers 提供 `{EXPECTED_CAUSAL_LM_CLASS}` 这个"
        "纯文本因果语言模型类（用于剥离 Qwen3.5-9B 的视觉塔），但当前安装的 "
        f"transformers=={transformers.__version__} 里找不到这个类。{hint}\n"
        f"  当前环境里名字含 'Qwen3' 的类/符号: {qwen3_symbols_msg}\n"
        "  这份设计文档写于 Qwen3.5 正式发布前后，官方最终定案的类名/模块路径有被调整的风险。"
        "排查步骤：\n"
        "    1) 确认 transformers.__version__ 是否 >= Qwen3.5 发布时对应的版本号；\n"
        "    2) 查 Qwen3.5 官方 model card / HF Hub 页面上给出的 `from_pretrained` 示例代码，"
        "确认现在的纯文本类到底叫什么；\n"
        "    3) 如果类名变了，把本文件顶部的 `EXPECTED_CAUSAL_LM_CLASS` 改成新的真实类名再重跑，"
        "本脚本其余逻辑（config 补丁 / save_pretrained 流程）不需要改。"
    )


def strip_vision_and_resave(
    source_checkpoint: str,
    output_dir: Path,
    dtype: str,
    attn_implementation: str,
    trust_remote_code: bool,
    delete_source_after_load: bool = False,
) -> None:
    transformers, causal_lm_cls = _check_transformers_support()
    torch = _import_torch_or_die()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype not in dtype_map:
        raise SystemExit(f"[load_text_only] 不支持的 --dtype={dtype!r}，可选: {list(dtype_map)}")

    print(
        f"[load_text_only] 用 {EXPECTED_CAUSAL_LM_CLASS}.from_pretrained('{source_checkpoint}', "
        f"torch_dtype={dtype}, attn_implementation={attn_implementation!r}) 加载纯文本主干……\n"
        "  预期行为：checkpoint 里属于视觉塔/图文融合层的权重会被忽略/报 unexpected keys 警告——"
        "这是『视觉塔剥离』本身预期要发生的事，不是 bug。如果反而报 missing keys（语言主干权重缺失），"
        "说明这份 checkpoint 的分片布局跟纯文本类的期望不匹配，需要先手动检查 checkpoint 的 "
        "`model.safetensors.index.json`。"
    )
    model = causal_lm_cls.from_pretrained(
        source_checkpoint,
        torch_dtype=dtype_map[dtype],
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )

    if delete_source_after_load:
        print(
            "[load_text_only] --delete-source-after-load 已开启：clone 所有 parameters/buffers "
            "切断跟源文件 mmap 的引用关系（否则 rmtree 之后磁盘空间不会真正释放）……"
        )
        with torch.no_grad():
            for _name, _param in model.named_parameters():
                _param.data = _param.data.clone()
            for _name, _buf in model.named_buffers():
                if _buf is not None:
                    _buf.data = _buf.data.clone()

    print("[load_text_only] 加载纯文本 tokenizer（不加载 image processor/processor_config）……")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        source_checkpoint, trust_remote_code=trust_remote_code
    )

    if delete_source_after_load:
        source_path = Path(source_checkpoint)
        if source_path.is_dir():
            import gc
            import shutil

            gc.collect()
            print(
                f"[load_text_only] --delete-source-after-load 已开启：模型+tokenizer 已完整加载"
                f"进内存，现在删除本地源目录 {source_path} 以腾出磁盘空间……"
            )
            shutil.rmtree(source_path)
        else:
            print(
                f"[load_text_only] --delete-source-after-load 已开启，但 '{source_checkpoint}' "
                "不是本地目录（可能是 Hub id），跳过删除。"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load_text_only] 保存 text-only 语言主干到 {output_dir} ……")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(
        f"[load_text_only] 完成。text-only checkpoint 已写入 {output_dir}\n"
        "  下一步：把 sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 里的 "
        "`model_name_or_path` 占位路径改成这个目录的真实绝对路径。"
    )


def _import_torch_or_die():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise SystemExit(
            "[load_text_only] 没有安装 `torch`，请先 `pip install torch`（配合目标 GPU 的 CUDA 版本）。"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "把一份 Qwen3.5-9B 多模态 checkpoint 用 Qwen3_5ForCausalLM 剥离视觉塔后 "
            "保存为纯文本 checkpoint。"
        )
    )
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        help=(
            "原始 Qwen3.5-9B 多模态 checkpoint 路径或 HF Hub 模型 id，例如 "
            "'/data/models/Qwen3.5-9B' 或 'Qwen/Qwen3.5-9B'（占位，具体 Hub id 以官方发布为准）。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="剥离视觉塔后的 text-only checkpoint 输出目录。",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="加载/保存精度，默认使用 bf16。",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help=(
            "transformers `from_pretrained(attn_implementation=...)` 的取值，默认 'flash_attention_2'。"
            "如果目标机器没装 flash-attn，改成 'sdpa'。"
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Qwen3.5 是新架构，发布初期大概率仍需要 trust_remote_code=True（默认开启）。",
    )
    parser.add_argument(
        "--delete-source-after-load",
        action="store_true",
        default=False,
        help=(
            "模型+tokenizer 完整加载进内存后，删除本地 --source-checkpoint 目录再落盘 output-dir。"
            "磁盘空间不够同时容纳『原始 checkpoint + text-only 输出』时使用，Hub id 来源会自动跳过。"
        ),
    )
    args = parser.parse_args()

    strip_vision_and_resave(
        source_checkpoint=args.source_checkpoint,
        output_dir=args.output_dir,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        delete_source_after_load=args.delete_source_after_load,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "[load_text_only] 未提供参数，仅做一次 transformers/Qwen3_5ForCausalLM 环境自检：\n"
            "  用法: python3 sft/load_text_only.py --source-checkpoint <path_or_hub_id> "
            "--output-dir <path>"
        )
        _check_transformers_support()
        print("[load_text_only] 环境自检通过：transformers 里找到了 Qwen3_5ForCausalLM。")
        sys.exit(0)
    main()
