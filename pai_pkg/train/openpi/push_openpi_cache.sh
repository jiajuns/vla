#!/bin/bash
# 推送 pi05_base JAX 权重 + big_vision tokenizer 缓存到 HPC3
set -e
rsync -az --info=progress2 -e "ssh -o ConnectTimeout=25" \
  /home/chefmate/.cache/openpi/ leviccdong@hpc3login.hpc.hkust-gz.edu.cn:/data/user/leviccdong/EKSF/cache/openpi/
ssh -o ConnectTimeout=25 hpc3 'du -sh /data/user/leviccdong/EKSF/cache/openpi; ls /data/user/leviccdong/EKSF/cache/openpi/openpi-assets/checkpoints/pi05_base/params | head; test -f /data/user/leviccdong/EKSF/cache/openpi/big_vision/paligemma_tokenizer.model && echo TOKENIZER_OK'
