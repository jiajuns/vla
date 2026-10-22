#!/bin/bash
#SBATCH --job-name=convert_v21
#SBATCH --partition=acd_u
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/logs/convert_v21_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/logs/convert_v21_%j.err
set -euo pipefail

source /data/user/leviccdong/EKSF/code/openpi_jax/run_env.sh
cd /data/user/leviccdong/EKSF/code/openpi_jax
module load uv/0.9.13

uv run scripts/convert_pai_huanggua_to_lerobot_v21.py
