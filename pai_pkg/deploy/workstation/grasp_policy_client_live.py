#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""π0.5 实时提示词交互客户端 (人工判断版)

运行中随时在终端:
  1/2/3/4 + 回车  -> 切换到预置 skill 提示词
  任意文本 + 回车  -> 立即替换当前提示词 (下一个 chunk 生效)
  Ctrl-C          -> 退出

预置 skill (拍黄瓜 4-skill):
  1: pick up the knife and move it above the cucumber
  2: cut the cucumber into pieces and return the knife to its original position
  3: transfer all cucumber pieces from the white cutting board into the bowl
  4: pick up the sauce bowl, pour the sauce into the cucumber bowl, and return the sauce bowl to its original position
"""
import argparse
import base64
import json
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

CAMERA_KEYS = {
    "head_rgbd": "observation.images.base_0_rgb",
    "left_wrist_rgbd": "observation.images.left_wrist_0_rgb",
    "right_wrist_rgbd": "observation.images.right_wrist_0_rgb",
}
GRIPPER_DIMS = (14, 22)

PRESETS = {
    "1": "pick up the knife and move it to the first cutting position",
    "2": "make the first cut through the cucumber, lift the knife, and move it to the next cutting position",
    "3": "make the second cut through the cucumber, lift the knife, and move it to the next cutting position",
    "4": "make the third cut through the cucumber, lift the knife, and move it to the next cutting position",
    "5": "make the final cut through the cucumber and lift the knife",
    "6": "return the knife to its original position",
    "7": "transfer all cucumber pieces from the white cutting board into the bowl",
    "8": "pick up the sauce bowl, pour the sauce into the cucumber bowl, and return the sauce bowl to its original position",
}


def send_json(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(f"{len(data):010d}".encode("utf-8") + data)


def recv_json(sock, timeout=30.0):
    sock.settimeout(timeout)
    lb = b""
    while len(lb) < 10:
        chunk = sock.recv(10 - len(lb))
        if not chunk:
            raise ConnectionError("server disconnected")
        lb += chunk
    n = int(lb.decode("utf-8"))
    d = b""
    while len(d) < n:
        chunk = sock.recv(min(1 << 20, n - len(d)))
        if not chunk:
            raise ConnectionError("server disconnected")
        d += chunk
    return json.loads(d.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.0.11")
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--ckpt", default="/home/chefmate/Data/vla/outputs/pi05_pai_4skill_fullft/checkpoints/040000/pretrained_model")
    ap.add_argument("--task", default="pick up the knife and move it above the cucumber")
    ap.add_argument("--horizon", type=int, default=40,
                    help="每个 chunk 实际执行的步数 (发送前已丢弃前10步对齐). 40=默认50步chunk, 10=每步replan(旧4skill行为)")
    ap.add_argument("--smooth-alpha", type=float, default=0.0,
                    help="动作低通滤波系数 (0=不平滑, 越大越稳)")
    ap.add_argument("--no-gripper-clamp", action="store_true")
    args = ap.parse_args()

    current_task = [args.task]

    # ---- 后台 stdin 监听线程: 实时改提示词 ----
    def stdin_reader():
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            line = line.strip()
            if not line:
                continue
            if line in PRESETS:
                current_task[0] = PRESETS[line]
            else:
                current_task[0] = line
            print(f"\n[TASK] -> {current_task[0]}", flush=True)
            print("[提示] 继续输入可再改; 1-6 切预置 skill; Ctrl-C 退出", flush=True)

    threading.Thread(target=stdin_reader, daemon=True).start()

    # ---- 加载 π0.5 ----
    print(f"loading policy from {args.ckpt} ...", flush=True)
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    ptype = json.load(open(os.path.join(args.ckpt, "config.json")))["type"]
    policy = get_policy_class(ptype).from_pretrained(args.ckpt).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    print("policy loaded", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for attempt in range(30):
        try:
            sock.connect((args.host, args.port))
            break
        except OSError:
            print(f"waiting robot server ... {attempt+1}/30", flush=True)
            time.sleep(2)
    else:
        raise RuntimeError("cannot connect")
    print(f"connected {args.host}:{args.port}", flush=True)
    print("=" * 60, flush=True)
    print("实时提示词模式:", flush=True)
    for k, v in PRESETS.items():
        print(f"  {k}: {v}", flush=True)
    print("输入任意英文指令回车 = 立即替换当前提示词", flush=True)
    print(f"当前: {current_task[0]}", flush=True)
    print("=" * 60, flush=True)

    policy.reset()
    n_chunks = 0
    last_task = None
    smoother_prev = None
    try:
        while True:
            frame = recv_json(sock, timeout=30.0)
            if current_task[0] != last_task:
                last_task = current_task[0]
                policy.reset()
                print(f"[TASK] 生效: {last_task}", flush=True)
            if frame.get("want_chunk"):
                batch = {"observation.state": torch.from_numpy(np.asarray(frame["state"], dtype=np.float32)).float(),
                         "task": current_task[0]}
                for src, dst in CAMERA_KEYS.items():
                    jb = frame["images"].get(src)
                    if jb:
                        img = cv2.imdecode(np.frombuffer(base64.b64decode(jb), np.uint8), cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img = np.zeros((256, 256, 3), dtype=np.uint8)
                    if n_chunks % 30 == 0:
                        print(f"[img] {dst}: mean={img.mean():.1f} max={img.max():.0f} {'BLACK!' if img.mean() < 5 else 'ok'}", flush=True)
                    t = torch.from_numpy(img).float() / 255.0
                    batch[dst] = t.permute(2, 0, 1).contiguous()
                processed = pre(batch)
                with torch.inference_mode():
                    t0 = time.time()
                    chunk = policy.predict_action_chunk(processed)
                    infer_ms = (time.time() - t0) * 1000.0
                acts = post(chunk)[0].detach().cpu().numpy().astype(np.float32)
                out = []
                for a in acts:
                    if not args.no_gripper_clamp:
                        a = a.copy()
                        for d in GRIPPER_DIMS:
                            a[d] = float(np.clip(a[d], 0.0, 100.0))
                    out.append(a.tolist())
                out = out[10:10 + args.horizon]  # 桥接衔接对齐 + 可调执行步数
                if args.smooth_alpha > 0:
                    if smoother_prev is None:
                        smoother_prev = np.asarray(out, dtype=np.float32)
                    else:
                        smoothed = args.smooth_alpha * smoother_prev + (1.0 - args.smooth_alpha) * np.asarray(out, dtype=np.float32)
                        out = smoothed.tolist()
                        smoother_prev = smoothed
                send_json(sock, {"chunk": out, "infer_ms": round(infer_ms, 1)})
                n_chunks += 1
                if n_chunks % 30 == 0:
                    print(f"[infer] chunk #{n_chunks} | {infer_ms:.0f} ms | task: {current_task[0][:60]}", flush=True)
            else:
                send_json(sock, {"ack": 1})
    except KeyboardInterrupt:
        print("\n[exit] 已退出", flush=True)
    except ConnectionError as e:
        print(f"[exit] {e}", flush=True)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
