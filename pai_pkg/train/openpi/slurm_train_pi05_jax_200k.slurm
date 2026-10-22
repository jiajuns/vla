#!/bin/bash
#SBATCH --job-name=pi05_jax_200k
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=ACD1-3,ACD1-9,ACD1-50,ACD1-54,ACD1-56,ACD1-58
#SBATCH --output=/data/user/leviccdong/EKSF/logs/pi05_jax_200k_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/logs/pi05_jax_200k_%j.err
set -euo pipefail

source /data/user/leviccdong/EKSF/code/openpi_jax/run_env.sh
cd /data/user/leviccdong/EKSF/code/openpi_jax
module load uv/0.9.13

uv run scripts/train.py pai_huanggua_6skill --exp-name=200k_8gpu --overwrite
