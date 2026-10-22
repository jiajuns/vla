#!/usr/bin/env python3
"""拍黄瓜 8-skill 数据集构建: 80 长集 + 8skill 标注 -> 独立 skill episodes
每段一个 episode: task=对应 prompt, frame_index/timestamp 从0重来, stats 重算
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

SRC = Path("/home/chefmate/Data/vla/datasets/pai_huanggua_lerobot_v3")
LBL = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_8skill")
OUT = Path("/home/chefmate/Data/vla/datasets/pai_huanggua_8skill_lerobot_v3")
IMG = 256

CAM_COLS = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]


class VideoReader:
    def __init__(self, cam_col):
        self.files = sorted((SRC / "videos" / cam_col / "chunk-000").glob("*.mp4"))
        self.caps = [cv2.VideoCapture(str(f)) for f in self.files]
        self.offsets = []
        total = 0
        for c in self.caps:
            self.offsets.append(total)
            total += int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total = total
        self.pos = -1
        self.cur_chunk = 0

    def read(self, gf):
        if gf == self.pos + 1:
            ok, img = self.caps[self.cur_chunk].read()
            if ok:
                self.pos = gf
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.cur_chunk += 1
            if self.cur_chunk < len(self.caps):
                ok, img = self.caps[self.cur_chunk].read()
                if ok:
                    self.pos = gf
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None
        ci = 0
        for i, off in enumerate(self.offsets):
            if gf >= off:
                ci = i
        local = gf - self.offsets[ci]
        self.caps[ci].set(cv2.CAP_PROP_POS_FRAMES, local)
        ok, img = self.caps[ci].read()
        if not ok:
            return None
        self.cur_chunk = ci
        self.pos = gf
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists")

    df = pq.read_table(SRC / "data/chunk-000/file-000.parquet").to_pandas()
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    ep_meta = pq.read_table(SRC / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    ep_meta = ep_meta.sort_values("episode_index").reset_index(drop=True)

    # 切分计划: (global_start, global_end, skill_id, prompt)
    segments = []
    SKIP_EPS = {5}  # ep005 录制数据本身有问题, 整条剔除
    for ep in range(80):
        if ep in SKIP_EPS:
            continue
        labels = json.loads((LBL / f"labels_ep{ep:03d}.json").read_text())
        start_g = int(ep_meta.iloc[ep]["dataset_from_index"])
        T = int(ep_meta.iloc[ep]["length"])
        bounds = [0] + [a["end_frame"] + 1 for a in labels[:-1]] + [T]
        for k, lab in enumerate(labels):
            s, e = bounds[k], bounds[k + 1]
            if e <= s:
                continue
            segments.append((start_g + s, start_g + e, lab["skill_id"], lab["prompt"]))
    print(f"segments: {len(segments)}")

    from lerobot.datasets import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig

    features = {
        c: {"dtype": "video", "shape": (IMG, IMG, 3), "names": ["height", "width", "channels"]}
        for c in CAM_COLS
    }
    features["observation.state"] = {"dtype": "float32", "shape": (25,), "names": ["qpos"]}
    features["action"] = {"dtype": "float32", "shape": (25,), "names": ["action"]}

    dataset = LeRobotDataset.create(
        repo_id="astribot/pai_huanggua_8skill",
        fps=30,
        features=features,
        root=OUT,
        robot_type="astribot_s1",
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=23, preset="fast"),
    )

    readers = {c: VideoReader(c) for c in CAM_COLS}
    t0 = time.time()
    done = 0
    skill_frames = {}
    for s, e, sid, prompt in segments:
        sub = df.iloc[s:e]
        for _, row in sub.iterrows():
            gf = int(row["index"])
            frame = {
                "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                "action": np.asarray(row["action"], dtype=np.float32),
                "task": prompt,
            }
            for c in CAM_COLS:
                img = readers[c].read(gf)
                if img is None:
                    raise RuntimeError(f"frame {gf} missing in {c}")
                frame[c] = img
            dataset.add_frame(frame)
        dataset.save_episode()
        skill_frames[sid] = skill_frames.get(sid, 0) + (e - s)
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(segments)} episodes ({time.time()-t0:.0f}s)", flush=True)

    dataset.finalize()
    print(f"DONE. {len(segments)} episodes at {OUT} ({time.time()-t0:.0f}s)")
    for k, v in sorted(skill_frames.items()):
        print(f"  {k}: {v} frames")


if __name__ == "__main__":
    main()
