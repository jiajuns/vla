#!/usr/bin/env bash
# π0.5 实时提示词交互部署 (人工判断版)
set -euo pipefail
cd "$(dirname "$0")"
PY="/home/chefmate/miniconda3/envs/pi05/bin/python"
HOST="${HOST:-192.168.0.11}"
PORT="${PORT:-6200}"
CKPT="${CKPT:-/home/chefmate/Astribot/grasp_deploy/checkpoints/pi05_pai_huanggua_8skill_200k/120000/pretrained_model}"
TASK="${TASK:-pick up the knife and move it to the first cutting position}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "=== π0.5 实时提示词交互 ==="
echo "  ckpt = $CKPT"
echo "  task = $TASK"
exec "$PY" -u grasp_policy_client_live.py \
  --ckpt "$CKPT" --host "$HOST" --port "$PORT" --task "$TASK" \
  --smooth-alpha 0 --no-gripper-clamp "$@"
