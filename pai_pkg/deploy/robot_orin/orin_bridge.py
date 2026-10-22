#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astribot S1 grasp VLA 部署 — Orin 桥接服务 (运行在机器人 Orin 上, ROS2 SDK)

职责:
  1) SDK 订阅机器人相机 (head_rgbd / left_wrist_rgbd / right_wrist_rgbd)
  2) 读取 25 维关节状态 (chassis3 + torso4 + armL7 + gripL1 + armR7 + gripR1 + head2)
  3) TCP 发送 {images(base64), state} 给推理机 (192.168.0.100), 接收 25 维 action
  4) 执行指令 (默认 DRY-RUN, --execute 才真实下发)
  5) 实时显示 + 录像 + CSV (可选)

用法 (在 Orin 上):
  cd ~/Downloads/astribot_sdk_aarch64 && source env.sh
  python3 /home/astribot/grasp_deploy/orin_bridge.py            # dry-run
  python3 /home/astribot/grasp_deploy/orin_bridge.py --execute  # 真实控制!
"""
import argparse
import base64
import csv
import json
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np

from astribot_sdk.core.astribot_api.astribot_client import Astribot

STATE_COMPONENTS = [
    ("astribot_chassis", 3),
    ("astribot_torso", 4),
    ("astribot_arm_left", 7),
    ("astribot_gripper_left", 1),
    ("astribot_arm_right", 7),
    ("astribot_gripper_right", 1),
    ("astribot_head", 2),
]
STATE_NAMES = [n for n, _ in STATE_COMPONENTS]
STATE_DIM = 25
CAMERA_NAMES = ["head_rgbd", "left_wrist_rgbd", "right_wrist_rgbd"]
IMG_SIZE = 256
DT = 0.033


def resize_with_pad(img, size=IMG_SIZE):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    rh, rw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - rh) // 2, (size - rw) // 2
    canvas[top:top + rh, left:left + rw] = resized
    return canvas


def send_json(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(f"{len(data):010d}".encode("utf-8") + data)


def recv_json(sock, timeout=10.0):
    sock.settimeout(timeout)
    lb = b""
    while len(lb) < 10:
        chunk = sock.recv(10 - len(lb))
        if not chunk:
            raise ConnectionError("client disconnected")
        lb += chunk
    n = int(lb.decode("utf-8"))
    d = b""
    while len(d) < n:
        chunk = sock.recv(min(65536, n - len(d)))
        if not chunk:
            raise ConnectionError("client disconnected")
        d += chunk
    return json.loads(d.decode("utf-8"))


class Display:
    def __init__(self, enable):
        self.enable = enable
        self.history = []
        self._canvas = None
        self._ok = False
        # 无 DISPLAY 时自动跳过(SSH 会话下 Qt 会 abort, 不能依赖 try/except)
        if enable and not os.environ.get("DISPLAY"):
            print("[display] no DISPLAY env, auto-disabling GUI (use RustDesk/NoMachine desktop to see the window)", flush=True)
            enable = False
        if enable:
            try:
                cv2.namedWindow("Grasp VLA Orin Bridge", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Grasp VLA Orin Bridge", 1400, 620)
                self._ok = True
            except Exception as e:  # noqa: BLE001
                print(f"[display] unavailable: {e}")

    def update(self, images, state, action, step, infer_ms, dry_run):
        if not self._ok:
            return -1
        panels = []
        for cam in CAMERA_NAMES:
            img = images.get(cam)
            if img is None or img.size == 0:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            if img.shape[0] != IMG_SIZE:
                img = resize_with_pad(img)
            label = np.zeros((28, IMG_SIZE, 3), dtype=np.uint8)
            cv2.putText(label, cam, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            panels.append(np.vstack([label, img]))
        left = np.hstack(panels)
        info = np.zeros((left.shape[0], 520, 3), dtype=np.uint8)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1) if action is not None else None
        lines = [f"step: {step}", f"infer: {infer_ms:.0f} ms",
                 f"mode: {'DRY-RUN (no commands)' if dry_run else 'EXECUTE'}",
                 "armL: " + " ".join(f"{x:+.2f}" for x in state[7:14]),
                 f"gripL: {state[14]:.1f}%  armR: " + " ".join(f"{x:+.2f}" for x in state[15:22]),
                 f"gripR: {state[22]:.1f}%  head: " + " ".join(f"{x:+.2f}" for x in state[23:25])]
        if action is not None:
            lines += [f"A gripL: {action[14]:.1f}%  A gripR: {action[22]:.1f}%",
                      "A armL: " + " ".join(f"{x:+.2f}" for x in action[7:14])]
        y = 24
        for line in lines:
            cv2.putText(info, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            y += 22
        self.history.append(action.copy() if action is not None else None)
        self.history = self.history[-300:]
        plot = np.zeros((left.shape[0] - y, 520, 3), dtype=np.uint8)
        ph, pw = plot.shape[:2]
        for j, (d, color, name) in enumerate([(14, (0, 255, 0), "gripL"), (22, (255, 200, 0), "gripR")]):
            pts = []
            for x, h in enumerate(self.history):
                if h is None:
                    continue
                v = h[d]
                pts.append((int(x / 300 * pw), int(ph - (np.clip(v, 0, 100)) / 100 * ph)))
            if len(pts) > 1:
                cv2.polylines(plot, [np.array(pts)], False, color, 1)
            cv2.putText(plot, name, (8, 18 + j * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        info = np.vstack([info[:y], plot])
        canvas = np.hstack([left, info])
        cv2.imshow("Grasp VLA Orin Bridge", canvas)
        key = cv2.waitKey(1)
        self._canvas = canvas
        return key


class Bridge:
    CMD_FREQ = 250.0  # SDK set_joints_position 期望的调用频率

    def __init__(self, execute, display_enable, gripper_binary=False, gripper_threshold=50.0):
        self.execute = execute
        self.gripper_binary = gripper_binary
        self.gripper_threshold = gripper_threshold
        self.astribot = None
        self.images = {}
        self._last_gripper = {"left": None, "right": None}
        self._latest_names = None
        self._latest_cmd = None
        self._cmd_lock = threading.Lock()
        self._cmd_thread = None
        self._stop = threading.Event()
        self.display = Display(display_enable)
        print("[bridge] connecting SDK ...", flush=True)
        self.astribot = Astribot(freq=self.CMD_FREQ)
        print("[bridge] SDK connected", flush=True)
        self._setup_cameras()
        if self.execute:
            self._cmd_thread = threading.Thread(target=self._command_loop, daemon=True)
            self._cmd_thread.start()
            print(f"[bridge] 250Hz command loop started", flush=True)

    def _command_loop(self):
        """以 250Hz 重复下发最新目标, 匹配 SDK 期望的调用频率。"""
        period = 1.0 / self.CMD_FREQ
        while not self._stop.is_set():
            with self._cmd_lock:
                names, cmd = self._latest_names, self._latest_cmd
            if names is not None:
                try:
                    self.astribot.set_joints_position(names, cmd, control_way="filter", use_wbc=False, add_default_torso=False)
                except Exception as e:  # noqa: BLE001
                    print(f"[exec] set_joints_position error: {e}", flush=True)
            time.sleep(period)

    def stop(self):
        self._stop.set()

    def _image_callback(self, topic_name, msg, width, height, array):
        camera_name = self.astribot.get_camera_name_from_topic_name(topic_name)
        if msg.format.lower() == "jpeg":
            self.images[camera_name] = array

    def _setup_cameras(self):
        print("[bridge] activating cameras ...", flush=True)
        print(self.astribot.activate_camera(), flush=True)
        deadline = time.time() + 60
        while time.time() < deadline:
            info = self.astribot.get_cameras_info()
            ok = all(info.get(n, {}).get("activate", False) for n in CAMERA_NAMES)
            if ok:
                break
            # SDK 状态字段可能滞后: 只要回调里已收到实际图像帧就算就绪
            got_frames = all(len(self.images.get(n, b"")) > 0 for n in CAMERA_NAMES)
            if got_frames:
                print("[bridge] camera status fields lagging, but frames are flowing - proceeding", flush=True)
                break
            print("waiting cameras:", {k: v.get("activate") for k, v in info.items()}, flush=True)
            time.sleep(2.0)
        for name in CAMERA_NAMES:
            sub = self.astribot.register_image_callback(name, "color", self._image_callback, need_decode=True)
            if sub is not None:
                print(f"[bridge] subscribed {name}", flush=True)

    def get_state(self):
        joints = self.astribot.get_current_joints_position(STATE_NAMES)
        if joints is None:
            return np.zeros(STATE_DIM, dtype=np.float32)
        q = []
        for comp in joints:
            q.extend(np.asarray(comp, dtype=np.float32).reshape(-1).tolist())
        if len(q) != STATE_DIM:
            print(f"[bridge] WARN state dim {len(q)} != {STATE_DIM}", flush=True)
            q = (q + [0.0] * STATE_DIM)[:STATE_DIM]
        return np.asarray(q, dtype=np.float32)

    def get_images(self):
        return {cam: self.images.get(cam) for cam in CAMERA_NAMES}

    def execute_action(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if len(a) < STATE_DIM:
            a = np.pad(a, (0, STATE_DIM - len(a)))
        a = a[:STATE_DIM]
        cmd = {
            "astribot_arm_left": a[7:14],
            "astribot_arm_right": a[15:22],
            "astribot_head": a[23:25],
            "astribot_torso": a[3:7],
        }
        if self.gripper_binary:
            # 旧二值化模式(可选): 阈值开合
            for side, idx in (("left", 14), ("right", 22)):
                closed = bool(a[idx] >= self.gripper_threshold)
                if self._last_gripper[side] != closed:
                    name = self.astribot.effector_left_name if side == "left" else self.astribot.effector_right_name
                    try:
                        if closed:
                            self.astribot.set_effector_max_force([name], max_force=[80.0])
                            self.astribot.close_effector([name], duration=0.05)
                        else:
                            self.astribot.open_effector([name], duration=0.05)
                        self._last_gripper[side] = closed
                        print(f"[exec] gripper {side}: {'CLOSE' if closed else 'OPEN'}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"[exec] gripper {side} failed: {e}", flush=True)
        else:
            # 连续模式(默认): 夹爪直接下发模型输出的 0-100% 实际值
            cmd["astribot_gripper_left"] = a[14:15]
            cmd["astribot_gripper_right"] = a[22:23]
        names = list(cmd.keys())
        cmd_list = [v.tolist() for v in cmd.values()]
        # 交给 250Hz 线程重复下发 (匹配 SDK 调用频率, 消除 frequency 警告)
        with self._cmd_lock:
            self._latest_names = names
            self._latest_cmd = cmd_list
        return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--gripper-binary", action="store_true",
                    help="夹爪二值化(旧模式); 默认连续输出模型实际值")
    ap.add_argument("--gripper-threshold", type=float, default=50.0, help="夹爪闭合阈值 (%), 仅二值化模式用")
    ap.add_argument("--replan-horizon", type=int, default=50,
                    help="每块实际执行的步数, 之后重新规划 (10=每333ms重规划, 50=整块)")
    ap.add_argument("--min-buffer", type=int, default=10,
                    help="缓冲低于该步数时请求新 chunk (0=执行完整个 chunk 后再请求推理, 推理期间机器人保持最后目标等待)")
    ap.add_argument("--no-images", action="store_true",
                    help="跳过图像编码/发送 (replay/纯动作回放时大幅提速)")
    ap.add_argument("--settle-frames", type=int, default=0,
                    help="chunk 执行完后额外等待 N 帧让关节稳定, 再取状态发 want (消除块边界回退, 建议 3)")
    args = ap.parse_args()

    if args.execute:
        print("\n" + "=" * 70)
        print("  !!! EXECUTE MODE: 将真实控制机器人关节 !!!")
        print("  确认: 工作区已清空 / 急停与操作员就位")
        print("=" * 70 + "\n")
        time.sleep(3)

    bridge = Bridge(execute=args.execute, display_enable=not args.no_display,
                    gripper_binary=args.gripper_binary, gripper_threshold=args.gripper_threshold)

    video_writer, csv_file, csv_writer = None, None, None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        video_writer = cv2.VideoWriter(os.path.join(args.save_dir, "deploy_rollout.mp4"),
                                       cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (1400, 620))
        csv_file = open(os.path.join(args.save_dir, "actions.csv"), "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "ts"] + [f"state_{i}" for i in range(STATE_DIM)] + [f"action_{i}" for i in range(STATE_DIM)])

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.host, args.port))
    server_socket.listen(4)
    print(f"[bridge] listening on {args.host}:{args.port}, waiting for policy client ...", flush=True)

    while True:  # 循环接受客户端; 仅当某次会话真正跑满 max_steps 才退出
        conn, addr = server_socket.accept()
        print(f"[bridge] policy client connected: {addr}", flush=True)
        completed = False
        try:
            completed = _serve_client(conn, addr, bridge, args, video_writer, csv_writer, csv_file)
        except (ConnectionError, BrokenPipeError, ConnectionResetError, KeyboardInterrupt) as e:
            print(f"[bridge] client {addr} gone: {e}", flush=True)
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        if completed:
            break

    bridge.stop()
    if video_writer is not None:
        video_writer.release()
    if csv_file is not None:
        csv_file.close()
    print("[bridge] shutdown", flush=True)
    return


def _serve_client(conn, addr, bridge, args, video_writer, csv_writer, csv_file):
    """服务一个客户端; 返回 True 表示该会话完整结束(跑满 max_steps 或用户按 q)。

    协议 v2: 桥接本地维护动作缓冲, 按 30Hz 逐帧取出执行;
    缓冲低于阈值时在帧里带 want_chunk=true, 客户端回传整块(50步)动作。
    推理延迟被预取吸收, 机器人动作流不再有周期卡顿。
    """
    import collections
    buffer = collections.deque()
    MIN_BUFFER = args.min_buffer
    KEEP = args.replan_horizon  # 每块只保留前 N 步执行, 之后重新规划 (默认 50=整块)
    step = 0
    action = None
    infer_ms = 0.0
    completed = False
    try:
        while True:
            if args.max_steps and step >= args.max_steps:
                print(f"[bridge] reached max_steps={args.max_steps}", flush=True)
                completed = True
                break
            t0 = time.time()
            state = bridge.get_state()
            images = {} if args.no_images else bridge.get_images()

            # 取出下一个待执行动作 (缓冲空则保持上一目标, 由 250Hz 线程继续维持)
            if buffer:
                action = np.asarray(buffer.popleft(), dtype=np.float32)
            if action is not None:
                names = bridge.execute_action(action)
                if step % 30 == 0:
                    mode = "DRY-RUN (not executed)" if not args.execute else "EXECUTED"
                    print(f"[step {step}] action -> {names} | {mode}", flush=True)
                if csv_writer is not None:
                    csv_writer.writerow([step, f"{time.time():.4f}"] + state.tolist() + action.tolist())
                    if step % 10 == 0:
                        csv_file.flush()

            encoded = {}
            if not args.no_images:
                for cam in CAMERA_NAMES:
                    img = images.get(cam)
                    if img is None or img.size == 0:
                        continue
                    small = resize_with_pad(img)
                    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        encoded[cam] = base64.b64encode(buf).decode("utf-8")

            want = (len(buffer) <= MIN_BUFFER) if MIN_BUFFER == 0 else (len(buffer) < MIN_BUFFER)
            if want and args.settle_frames > 0:
                # 关节稳定后再取状态: 消除跟踪滞后导致的块边界动作回退
                for _ in range(args.settle_frames):
                    time.sleep(1.0 / 30.0)
                state = bridge.get_state()
            send_json(conn, {"images": encoded, "state": state.tolist(), "step": step,
                             "ts": time.time(), "want_chunk": want})
            try:
                reply = recv_json(conn, timeout=10.0)
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as e:
                print(f"[bridge] bad client data: {e}; closing connection", flush=True)
                raise ConnectionError("protocol error") from e

            if "chunk" in reply and reply["chunk"]:
                kept = reply["chunk"][:KEEP]
                buffer.extend(kept)
                infer_ms = float(reply.get("infer_ms", 0.0))
                if len(kept) < len(reply["chunk"]):
                    print(f"[bridge] buffer +{len(kept)}/{len(reply['chunk'])} (replan horizon {KEEP}, total {len(buffer)}), infer {infer_ms:.0f} ms", flush=True)
                else:
                    print(f"[bridge] buffer +{len(kept)} (total {len(buffer)}), infer {infer_ms:.0f} ms", flush=True)
            elif reply.get("action") is not None:
                buffer.append(reply["action"])
                infer_ms = float(reply.get("infer_ms", 0.0))

            key = bridge.display.update(images, state, action, step, infer_ms, dry_run=not args.execute)
            if key in (ord("q"), 27):
                completed = True
                break
            if video_writer is not None and bridge.display._canvas is not None:
                video_writer.write(bridge.display._canvas)
            elapsed = time.time() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)
            step += 1
    except KeyboardInterrupt:
        print("\n[bridge] interrupted", flush=True)
        raise
    except ConnectionError:
        raise
    return completed


if __name__ == "__main__":
    main()
