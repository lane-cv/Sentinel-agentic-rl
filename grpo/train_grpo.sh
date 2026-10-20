#!/usr/bin/env bash

set -xeuo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/aiops-qwen3.5-9b-v9-merged}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

export AIOPS_MAC_REPO_ROOT="${AIOPS_MAC_REPO_ROOT:-Desktop/Sentinel-agentic rl}"
export AIOPS_MAC_AGENT_DIR="${AIOPS_MAC_AGENT_DIR:-$AIOPS_MAC_REPO_ROOT/Sentinel-agent}"

train_batch_size=${TRAIN_BATCH_SIZE:-74}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}

ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-${train_batch_size}}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
rollout_log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
ref_log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

TRAIN_FILES=${TRAIN_FILES:-grpo/verl_data/train.jsonl}
VAL_FILES=${VAL_FILES:-grpo/verl_data/held_out.jsonl}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
rollout_n=${ROLLOUT_N:-6}
max_assistant_turns=${MAX_ASSISTANT_TURNS:-12}

lora_rank=${LORA_RANK:-16}
lora_alpha=${LORA_ALPHA:-32}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-5}

REWARD_FN_PATH=${REWARD_FN_PATH:-grpo/verl_reward_adapter.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}

PROJECT_NAME=${PROJECT_NAME:-aiops-diagnosis-grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_9b_lora_grpo_$(date +%Y%m%d_%H%M)}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.use_shm=True \
    \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=verl_adapter/tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu} \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu} \
    \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name="${REWARD_FN_NAME}" \
    \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    "$@"
