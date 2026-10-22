#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astribot S1 增强采集记录器 v2 (D_demo / D_event)

三路 RGBD 同时采集, 自适应相机流格式:
  - color: SDK need_decode=True 解码(兼容 jpeg/raw 两态), 统一转 jpeg 字节存 HDF5
  - depth: 原始字节流原样保存 + 每相机格式标记 (head=raw uint16 512x720 mm;
           wrist=jpeg 压缩流, 离线解码脚本处理)
  - joints: current/desired position(25) + velocity(25) + torque(25)
  - poses: chassis/torso/armL/armR/head (xyz+xyzw, 7)
  - 时间戳 + session 元数据 (session_id/task_id/operator/...)

用法 (Orin):
  cd ~/Downloads/astribot_sdk_aarch64 && source env.sh
  python3 -u ~/grasp_deploy/orin_collect_rgbd.py \
      --session-id s1_20260902_a --task-id pick_green_object_place_in_wooden_box_v1 \
      --operator zhangsan --episode 1 --out-dir ~/grasp_deploy/collect_rgbd
  回车开始 / 再回车结束
"""
import argparse
import os
import threading
import time

import cv2
import h5py
import numpy as np

from astribot_sdk.core.astribot_api.astribot_client import Astribot

CAMERAS = ["head_rgbd", "left_wrist_rgbd", "right_wrist_rgbd"]
HEAD_DEPTH_SHAPE = (512, 720)

STATE_COMPONENTS = [
    ("astribot_chassis", 3), ("astribot_torso", 4),
    ("astribot_arm_left", 7), ("astribot_gripper_left", 1),
    ("astribot_arm_right", 7), ("astribot_gripper_right", 1),
    ("astribot_head", 2),
]
STATE_NAMES = [n for n, _ in STATE_COMPONENTS]
STATE_DIM = 25
POSE_PARTS = ["astribot_chassis", "astribot_torso", "astribot_arm_left",
              "astribot_arm_right", "astribot_head"]


def flatten_joints(joints, expect=STATE_DIM):
    q = []
    for comp in joints:
        q.extend(np.asarray(comp, dtype=np.float64).reshape(-1).tolist())
    if len(q) != expect:
        q = (q + [0.0] * expect)[:expect]
    return np.asarray(q, dtype=np.float64)


def decode_color(msg_format, raw_bytes):
    """把 SDK 回调数据统一解码为 BGR ndarray。"""
    if msg_format.lower() == "jpeg":
        img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        return img
    # raw: 按长度猜 YUV 布局
    d = np.frombuffer(raw_bytes, np.uint8)
    if d.size == 512 * 720 * 2:                      # head YUV422
        return cv2.cvtColor(d.reshape(512, 720, 2), cv2.COLOR_YUV2BGR_YUYV)
    if d.size == 640 * 480 * 3 // 2:                 # wrist NV12
        return cv2.cvtColor(d.reshape(480 * 3 // 2, 640), cv2.COLOR_YUV2BGR_NV12)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--task-text", default="pick green object and place into wooden box")
    ap.add_argument("--operator", default="unknown")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out-dir", default="~/grasp_deploy/collect_rgbd")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--robot-serial", default="unknown")
    ap.add_argument("--sdk-version", default="unknown")
    ap.add_argument("--behavior-source", default="teleop_expert")
    ap.add_argument("--test-seconds", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.task_id}_ep{args.episode:03d}.hdf5")
    if os.path.exists(out_path):
        print(f"[collect] REFUSE to overwrite {out_path}")
        return 1

    print("[collect] initializing SDK ...")
    robot = Astribot()
    robot.activate_camera()
    time.sleep(3)

    latest = {c: {"color": None, "color_ts": -1.0, "depth": None, "depth_fmt": None, "depth_ts": -1.0}
              for c in CAMERAS}
    lock = threading.Lock()

    def cb_color(cam):
        def cb(topic_name, msg, w, h, arr):
            try:
                if isinstance(arr, np.ndarray) and arr.ndim == 3:
                    img = arr  # SDK 已解码 BGR
                else:
                    img = decode_color(msg.format, bytes(msg.data))
                if img is None:
                    return
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    with lock:
                        latest[cam]["color"] = buf.tobytes()
                        latest[cam]["color_ts"] = time.time()
            except Exception:
                pass
        return cb

    def cb_depth(cam):
        def cb(topic_name, msg, w, h, arr):
            try:
                raw = bytes(msg.data)
                with lock:
                    if cam == "head_rgbd" and len(raw) == HEAD_DEPTH_SHAPE[0] * HEAD_DEPTH_SHAPE[1] * 2:
                        # 头部深度: raw uint16 512x720 mm, 已实测验证
                        latest[cam]["depth"] = np.frombuffer(raw, np.uint16).reshape(HEAD_DEPTH_SHAPE).copy()
                        latest[cam]["depth_fmt"] = "raw_uint16_mm"
                    else:
                        # 腕部 depth_compress 实测为彩色流(非深度), 原样保存待 GPU 路径补齐
                        latest[cam]["depth"] = raw
                        latest[cam]["depth_fmt"] = "vendor_stream_unverified"
                    latest[cam]["depth_ts"] = time.time()
            except Exception:
                pass
        return cb

    for cam in CAMERAS:
        robot.register_image_callback(cam, "color", cb_color(cam), need_decode=True)
        robot.register_image_callback(cam, "depth", cb_depth(cam), need_decode=False)
    print("[collect] callbacks registered (3x color decoded + 3x depth raw)")
    # 等待每路相机 color+depth 首帧就绪 (相机流启动有延迟)
    deadline = time.time() + 30
    while time.time() < deadline:
        with lock:
            missing = [f"{c}:{k}" for c in CAMERAS for k in ("color", "depth")
                       if (k == "color" and latest[c]["color"] is None)
                       or (k == "depth" and latest[c]["depth"] is None)]
        if not missing:
            break
        time.sleep(1)
        print(f"[collect] waiting first frames: {missing}", flush=True)
    print(f"[collect] ready (missing={missing if 'missing' in dir() else 'none'})")

    BUF = []
    recording = threading.Event()
    stop = threading.Event()

    def loop():
        period = 1.0 / args.fps
        while not stop.is_set():
            if recording.is_set():
                with lock:
                    colors = {c: latest[c]["color"] for c in CAMERAS}
                    if any(v is None for v in colors.values()):
                        time.sleep(0.005)
                        continue
                    cts = {c: latest[c]["color_ts"] for c in CAMERAS}
                    depths = {c: latest[c]["depth"] for c in CAMERAS}
                    dfmts = {c: latest[c]["depth_fmt"] for c in CAMERAS}
                    dts = {c: latest[c]["depth_ts"] for c in CAMERAS}
                try:
                    jpos = flatten_joints(robot.get_current_joints_position(STATE_NAMES))
                    jcmd = flatten_joints(robot.get_desired_joints_position(STATE_NAMES))
                    jvel = flatten_joints(robot.get_current_joints_velocity(STATE_NAMES))
                    jtor = flatten_joints(robot.get_current_joints_torque(STATE_NAMES))
                except Exception as e:
                    print(f"[collect] state read warn: {e}", flush=True)
                    time.sleep(period)
                    continue
                try:
                    poses = robot.get_current_cartesian_pose(POSE_PARTS, frame="chassis")
                    poses = [np.asarray(p, dtype=np.float64).reshape(-1)[:7] for p in poses]
                except Exception:
                    poses = [np.zeros(7) for _ in POSE_PARTS]
                BUF.append((time.time(), colors, cts, depths, dfmts, dts, jpos, jcmd, jvel, jtor, poses))
            time.sleep(period)

    t = threading.Thread(target=loop, daemon=True)
    t.start()

    if args.test_seconds > 0:
        print(f"[collect] test mode: recording {args.test_seconds}s ...")
        recording.set()
        time.sleep(args.test_seconds)
    else:
        input(f"[collect] 场景就绪后按回车开始录制 (ep {args.episode}) ...")
        recording.set()
        print("[collect] RECORDING ... 再按回车停止")
        input()
    recording.clear()
    time.sleep(0.6)
    stop.set()

    if len(BUF) == 0:
        print("[collect] ERROR: no frames captured, aborting")
        return 2
    print(f"[collect] recorded {len(BUF)} frames, writing HDF5 ...")

    f = h5py.File(out_path, "w")
    meta = dict(session_id=args.session_id, task_id=args.task_id, task_text=args.task_text,
                operator=args.operator, episode=args.episode, behavior_source=args.behavior_source,
                robot_serial=args.robot_serial, sdk_version=args.sdk_version, fps=args.fps,
                state_dim=STATE_DIM, depth_unit="mm (head raw uint16 512x720; wrist jpeg 压缩流, 离线解码)",
                state_components=";".join(f"{n}:{d}" for n, d in STATE_COMPONENTS),
                pose_order="chassis,torso,arm_left,arm_right,head (xyz+qxyzw)")
    for k, v in meta.items():
        f.attrs[k] = v

    T = len(BUF)
    f.create_dataset("time", data=np.array([b[0] for b in BUF], dtype=np.float64))

    ig = f.create_group("images_dict")
    for cam in CAMERAS:
        g = ig.create_group(cam)
        ds = g.create_dataset("rgb", (T,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
        g.create_dataset("rgb_size", data=np.array([len(b[1][cam]) for b in BUF], dtype=np.float64))
        g.create_dataset("rgb_timestamp", data=np.array([b[2][cam] for b in BUF], dtype=np.float64))
        g.create_dataset("depth_timestamp", data=np.array([b[5][cam] for b in BUF], dtype=np.float64))
        if cam == "head_rgbd":
            dd = g.create_dataset("depth", (T, HEAD_DEPTH_SHAPE[0], HEAD_DEPTH_SHAPE[1]), dtype=np.uint16)
            for i, b in enumerate(BUF):
                dd[i] = b[3][cam] if isinstance(b[3][cam], np.ndarray) else np.zeros(HEAD_DEPTH_SHAPE, np.uint16)
        else:
            dd = g.create_dataset("depth", (T,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            g.create_dataset("depth_size", data=np.array([len(b[3][cam] or b"") for b in BUF], dtype=np.float64))
            for i, b in enumerate(BUF):
                dd[i] = np.frombuffer(b[3][cam] or b"", np.uint8)
        for i, b in enumerate(BUF):
            ds[i] = np.frombuffer(b[1][cam], np.uint8)
        f.attrs[cam + "_depth_format"] = str(BUF[0][4][cam])

    jg = f.create_group("joints_dict")
    jg.create_dataset("joints_position_state", data=np.stack([b[6] for b in BUF]), dtype=np.float64)
    jg.create_dataset("joints_position_command", data=np.stack([b[7] for b in BUF]), dtype=np.float64)
    jg.create_dataset("joints_velocity_state", data=np.stack([b[8] for b in BUF]), dtype=np.float64)
    jg.create_dataset("joints_torque_state", data=np.stack([b[9] for b in BUF]), dtype=np.float64)
    jg.create_dataset("state_timestamp", data=np.array([b[0] for b in BUF]), dtype=np.float64)
    jg.create_dataset("command_timestamp", data=np.array([b[0] for b in BUF]), dtype=np.float64)

    pg = f.create_group("poses_dict")
    for j, part in enumerate(POSE_PARTS):
        pg.create_dataset(part, data=np.stack([b[10][j] for b in BUF]), dtype=np.float64)
        pg.create_dataset(part + "_timestamp", data=np.array([b[0] for b in BUF]), dtype=np.float64)

    f.close()
    print(f"[collect] saved: {out_path} ({T} frames)")
    if args.test_seconds <= 0:
        print("[collect] 下一步: 填 episode_outcomes.csv (task_success/terminal_reason/...)")


if __name__ == "__main__":
    main()
