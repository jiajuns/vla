#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回到拍黄瓜任务初始位姿 (基于拍黄瓜数据集 80 个 episode 第一帧状态均值)。

用法 (在 Orin 上, 需 WebUI 已切「终端」控制模式):
  cd ~/Downloads/astribot_sdk_aarch64 && source env.sh
  python3 ~/grasp_deploy/go_home_pai.py
"""
import argparse
import time

import numpy as np

from astribot_sdk.core.astribot_api.astribot_client import Astribot

# 拍黄瓜数据集 80 集第一帧 state 均值 (chassis3+torso4+armL7+gripL1+armR7+gripR1+head2)
HOME_STATE = [
    -0.0002, -0.0001, 0.0004,              # chassis (不控制)
    0.5958, -1.1951, 0.5946, 0.0055,       # torso
    0.2543, -0.0472, -1.4603, 1.6525, -0.3600, 0.2603, 0.0340,  # armL
    -0.0519,                                # gripL (≈0 = 张开)
    -0.0979, -0.0749, 1.4404, 1.6413, 0.4298, 0.0819, -0.0584,  # armR
    -0.0406,                                # gripR (≈0 = 张开)
    -0.0078, 0.8752,                        # head
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0, help="移动时长(秒)")
    ap.add_argument("--no-gripper", action="store_true", help="不动夹爪")
    ap.add_argument("--no-head", action="store_true", help="不动头部")
    ap.add_argument("--no-torso", action="store_true", help="不动躯干")
    args = ap.parse_args()

    print("[go_home_pai] connecting SDK ...", flush=True)
    a = Astribot(freq=250)
    print("[go_home_pai] SDK connected", flush=True)

    names = []
    cmds = []
    if not args.no_torso:
        names.append("astribot_torso")
        cmds.append(HOME_STATE[3:7])
    names.append("astribot_arm_left")
    cmds.append(HOME_STATE[7:14])
    names.append("astribot_arm_right")
    cmds.append(HOME_STATE[15:22])
    if not args.no_head:
        names.append("astribot_head")
        cmds.append(HOME_STATE[23:25])

    print("[go_home_pai] 目标状态 (80集第一帧均值):", flush=True)
    for n, c in zip(names, cmds):
        print(f"  {n}: {np.round(c, 3).tolist()}", flush=True)

    print(f"[go_home_pai] moving ({args.duration}s) ...", flush=True)
    a.move_joints_position(names, cmds, duration=args.duration, use_wbc=False, add_default_torso=False)
    print("[go_home_pai] 到位", flush=True)

    if not args.no_gripper:
        print("[go_home_pai] 张开夹爪 ...", flush=True)
        a.open_effector(["astribot_gripper_left", "astribot_gripper_right"], duration=0.3)
        time.sleep(0.5)
        print("[go_home_pai] 夹爪已张开", flush=True)
    print("[go_home_pai] DONE", flush=True)


if __name__ == "__main__":
    main()
