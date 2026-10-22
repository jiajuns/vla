#!/bin/bash
# openpi JAX 训练环境变量
export OPENPI_DATA_HOME=/data/user/leviccdong/EKSF/cache/openpi
export HF_LEROBOT_HOME=/data/user/leviccdong/EKSF/lerobot_hf_home
export HF_HOME=/data/user/leviccdong/EKSF/hf_cache
export JAX_COMPILATION_CACHE_DIR=/data/user/leviccdong/EKSF/cache/jax_compilation
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
mkdir -p "$OPENPI_DATA_HOME" "$HF_LEROBOT_HOME" "$HF_HOME" "$JAX_COMPILATION_CACHE_DIR"
