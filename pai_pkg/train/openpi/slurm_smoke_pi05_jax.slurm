#!/bin/bash
#SBATCH --job-name=pi05_jax_smoke
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --exclude=ACD1-3,ACD1-9,ACD1-50,ACD1-54,ACD1-56,ACD1-58
#SBATCH --output=/data/user/leviccdong/EKSF/logs/pi05_jax_smoke_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/logs/pi05_jax_smoke_%j.err
set -euo pipefail

source /data/user/leviccdong/EKSF/code/openpi_jax/run_env.sh
cd /data/user/leviccdong/EKSF/code/openpi_jax
module load uv/0.9.13

uv run scripts/train.py pai_huanggua_6skill --exp-name=smoke --overwrite   --num_train_steps=5 --batch_size=2 --fsdp_devices=1 --num_workers=2
