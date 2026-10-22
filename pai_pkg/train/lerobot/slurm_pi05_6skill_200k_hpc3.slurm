#!/bin/bash
#SBATCH --job-name=pi05_6skill_200k
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:8
#SBATCH --exclude=ACD1-3,ACD1-9,ACD1-50,ACD1-54,ACD1-56,ACD1-58
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=5-00:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/logs/pi05_6skill_200k_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/logs/pi05_6skill_200k_%j.err
set -euo pipefail

ROOT=/data/user/leviccdong/EKSF
TRAIN_ROOT=$ROOT/code/smolvla_train
PY=$ROOT/env_smolvla_plate062/bin/python
LERO=$TRAIN_ROOT/lerobot
BASE=$ROOT/models/pi05_base
DATA=$ROOT/data/pai_huanggua/pai_huanggua_6skill_lerobot_v3
OUT=$ROOT/outputs/pi05_pai_huanggua_6skill_200k_8gpu

test -x "$PY"
test -f "$BASE/model.safetensors"
test -f "$DATA/meta/info.json"
test ! -e "$OUT"

unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
export PYTHONPATH="$LERO/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$ROOT/env_smolvla_plate062/bin:/usr/bin:/bin"
export HF_HUB_CACHE="$TRAIN_ROOT/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$LERO"
# π0.5 6-skill full FT: 8xH100 DDP, effective batch 64 (8卡 x batch4 x 累积2)
torchrun --nproc-per-node=8 --standalone --rdzv-backend=c10d \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=astribot/pai_huanggua_6skill \
  --dataset.root="$DATA" \
  --dataset.video_backend=pyav \
  --dataloader_multiprocessing_context=fork \
  --policy.type=pi05 \
  --policy.pretrained_path="$BASE" \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name=pi05_pai_huanggua_6skill_200k \
  --batch_size=4 \
  --num_workers=4 \
  --steps=200000 \
  --save_freq=5000 \
  --seed=1000 \
  --optimizer.lr=0.00001 \
  --optimizer.betas=[0.9,0.95] \
  --optimizer.weight_decay=0.01 \
  --optimizer.grad_clip_norm=1.0 \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=1000 \
  --scheduler.num_decay_steps=200000 \
  --scheduler.peak_lr=0.00001 \
  --scheduler.decay_lr=0.000001 \
  --parallelism.dp_replicate=8 \
  --parallelism.dp_shard=1 \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps=2 \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --wandb.enable=false
