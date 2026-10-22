#!/usr/bin/env python3
"""拍黄瓜 6-skill: LeRobot v3.0 -> v2.1 数据集转换（供 openpi JAX 训练使用）。

Reads the v3.0 dataset built by the PyTorch fork (parquet + per-cam mp4 videos,
global frame order aligned with the "index" column) and rewrites it in the
LeRobot v2.1 layout that openpi's pinned lerobot expects.

Run inside the openpi venv:
    HF_LEROBOT_HOME=/data/user/leviccdong/EKSF/lerobot_hf_home         uv run scripts/convert_pai_huanggua_to_lerobot_v21.py
"""

import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SRC = Path("/data/user/leviccdong/EKSF/data/pai_huanggua/pai_huanggua_6skill_lerobot_v3")
OUT_HOME = "/data/user/leviccdong/EKSF/lerobot_hf_home"
REPO_ID = "astribot/pai_huanggua_6skill_v21"
IMG = 256

CAM_COLS = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]

# The lerobot import reads HF_LEROBOT_HOME at import time.
os.environ["HF_LEROBOT_HOME"] = OUT_HOME


class AvReader:
    """Sequential mp4 frame reader with random-access fallback."""

    def __init__(self, cam_col: str):
        import av

        self.av = av
        self.files = sorted((SRC / "videos" / cam_col / "chunk-000").glob("*.mp4"))
        if not self.files:
            raise FileNotFoundError(f"no videos for {cam_col}")
        self.offsets = []
        total = 0
        for f in self.files:
            with av.open(str(f)) as c:
                n = c.streams.video[0].frames
            self.offsets.append(total)
            total += n
        self.total = total
        self.pos = -1
        self.cur = -1
        self.container = None
        self.it = None

    def read(self, gf: int):
        if gf == self.pos + 1 and self.it is not None:
            try:
                fr = next(self.it)
                self.pos = gf
                return fr.to_ndarray(format="rgb24")
            except StopIteration:
                self.container.close()
                self.it = None

        idx = 0
        for i, off in enumerate(self.offsets):
            if gf >= off:
                idx = i
        local = gf - self.offsets[idx]

        if self.cur != idx or self.it is None:
            if self.container is not None:
                self.container.close()
            self.cur = idx
            self.container = self.av.open(str(self.files[idx]))
        self.it = self.container.decode(video=0)

        fr = None
        for j, f in enumerate(self.it):
            if j == local:
                fr = f
                break
        if fr is None:
            raise RuntimeError(f"frame {gf} missing in {self.files[idx]}")
        self.pos = gf
        return fr.to_ndarray(format="rgb24")


def main() -> None:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    print("HF_LEROBOT_HOME:", HF_LEROBOT_HOME)

    df = pq.read_table(SRC / "data/chunk-000/file-000.parquet").to_pandas()
    tasks = pq.read_table(SRC / "meta/tasks.parquet").to_pandas()
    task_texts = {int(r.task_index): r.task for r in tasks.itertuples()}
    print(f"frames: {len(df)}, episodes: {df.episode_index.nunique()}, tasks: {len(task_texts)}")

    ep = df["episode_index"].to_numpy()
    gf = df["index"].to_numpy().astype(np.int64)
    ti = df["task_index"].to_numpy().astype(np.int64)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    assert state.shape == (len(df), 25) and action.shape == (len(df), 25)

    features = {
        c: {"dtype": "image", "shape": (IMG, IMG, 3), "names": ["height", "width", "channel"]}
        for c in CAM_COLS
    }
    features["observation.state"] = {"dtype": "float32", "shape": (25,), "names": ["qpos"]}
    features["action"] = {"dtype": "float32", "shape": (25,), "names": ["action"]}

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        robot_type="astribot_s1",
        fps=30,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    readers = {c: AvReader(c) for c in CAM_COLS}
    for c, rd in readers.items():
        print(f"  {c}: {rd.total} frames in {len(rd.files)} file(s)")
    n_frames = len(df)
    if any(rd.total != n_frames for rd in readers.values()):
        raise RuntimeError("video frame count mismatch with parquet rows")

    t0 = time.time()
    cur_ep = -1
    done = 0
    for i in range(n_frames):
        if ep[i] != cur_ep:
            if cur_ep != -1:
                dataset.save_episode()
                done += 1
            cur_ep = ep[i]
            prompt = task_texts[int(ti[i])]
        frame = {
            "observation.state": state[i],
            "action": action[i],
            "task": prompt,
        }
        for c in CAM_COLS:
            frame[c] = readers[c].read(int(gf[i]))
        dataset.add_frame(frame)
        if i % 20000 == 0 and i > 0:
            print(f"  {i}/{n_frames} frames ({time.time() - t0:.0f}s)", flush=True)

    dataset.save_episode()
    done += 1
    print(f"DONE: {done} episodes in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
