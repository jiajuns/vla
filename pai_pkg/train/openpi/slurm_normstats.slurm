#!/bin/bash
#SBATCH --job-name=normstats_jax
#SBATCH --partition=acd_u
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/leviccdong/EKSF/logs/normstats_jax_%j.out
#SBATCH --error=/data/user/leviccdong/EKSF/logs/normstats_jax_%j.err
set -euo pipefail

source /data/user/leviccdong/EKSF/code/openpi_jax/run_env.sh
cd /data/user/leviccdong/EKSF/code/openpi_jax
module load uv/0.9.13

uv run scripts/compute_norm_stats.py --config-name pai_huanggua_6skill
