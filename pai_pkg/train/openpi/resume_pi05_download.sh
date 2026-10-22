#!/bin/bash
# 断点续传 pi05_base params (GCS direct, curl -C -)
set -u
B="https://storage.googleapis.com/openpi-assets"
D=/home/chefmate/.cache/openpi/openpi-assets/checkpoints/pi05_base/params.partial
LOG=/home/chefmate/Astribot/grasp_deploy/resume_pi05.log

dl() {
  local name="$1" size="$2"
  local dst="$D/$name"
  mkdir -p "$(dirname "$dst")"
  if [ -f "$dst" ] && [ "$(stat -c%s "$dst")" = "$size" ]; then
    echo "SKIP $name (complete)" >> "$LOG"
    return 0
  fi
  local tmp=""
  for cand in "${dst}.gstmp" "${dst}_.gstmp"; do
    if [ -f "$cand" ]; then tmp="$cand"; break; fi
  done
  if [ -z "$tmp" ]; then tmp="$dst"; fi
  local cur=0
  [ -f "$tmp" ] && cur=$(stat -c%s "$tmp")
  if [ "$cur" -ge "$size" ] && [ "$tmp" != "$dst" ]; then
    mv "$tmp" "$dst"; echo "PROMOTE $name" >> "$LOG"; return 0
  fi
  echo "FETCH $name (have $cur / need $size)" >> "$LOG"
  if ! curl -sL --retry 5 --retry-delay 5 -C "$cur" -o "$tmp" "$B/$name"; then
    echo "FAIL $name" >> "$LOG"; return 1
  fi
  if [ "$(stat -c%s "$tmp")" != "$size" ]; then
    echo "SIZEMISMATCH $name got $(stat -c%s "$tmp")" >> "$LOG"; return 1
  fi
  if [ "$tmp" != "$dst" ]; then mv "$tmp" "$dst"; fi
  echo "DONE $name" >> "$LOG"
}

dl "checkpoints/pi05_base/params/_CHECKPOINT_METADATA" 258 || exit 1
dl "checkpoints/pi05_base/params/_METADATA" 21402 || exit 1
dl "checkpoints/pi05_base/params/_sharding" 11420 || exit 1
dl "checkpoints/pi05_base/params/array_metadatas/process_0" 8856 || exit 1
dl "checkpoints/pi05_base/params/commit_success.txt" 0 || exit 1
dl "checkpoints/pi05_base/params/d/1c4302d2d2000b5f3eb4fa1350fdef9a" 2132 || exit 1
dl "checkpoints/pi05_base/params/manifest.ocdbt" 117 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/0832cad6c37f82d4eedd897dcbb8da9d" 2240590484 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/247d4b7c814d8b1a23fa8a20f36a88f7" 34945019 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/35a545f74995e511808d4f94dfbef3b6" 70041571 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/73bbae8a4deba6498bf07d96b215a574" 2935855 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/7bc9d3296d23a6fb83a6b3778ac6e964" 2240315383 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/7d78f38c5d8be1eea31406644dde9bd6" 133821 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/828bee85475e37c61e1cc19e32d1c5ef" 929 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/8c5d7070ea57bdce2f0a19f95b8a21b4" 3077149148 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/b4349aaadb7dfa45c3a53fc67c04b8f6" 1120156687 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/caf01a82962cbd4651563d2ac1063e0b" 27445152 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/deefd3c43390a50472cbcd317b0fff58" 2393670849 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/d/ec484cf8f02dcf59e1892180f0862e40" 1234292390 || exit 1
dl "checkpoints/pi05_base/params/ocdbt.process_0/manifest.ocdbt" 458 || exit 1

echo "ALL_DONE" >> "$LOG"
