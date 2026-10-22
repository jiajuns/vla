#!/usr/bin/env python3
"""拍黄瓜 skill 区间标注工具 - 后端 (仅标准库, 无依赖)
服务: 头部视频流 + episode 帧范围 + 标注 JSON 读写
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DATASET = Path("/home/chefmate/Data/vla/datasets/pai_huanggua_lerobot_v3")
VIDEO = DATASET / "videos/observation.images.base_0_rgb/chunk-000/file-000.mp4"
LABEL_DIR = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels")
LABEL_DIR.mkdir(parents=True, exist_ok=True)
TOOL_DIR = Path(__file__).parent
FPS = 30.0

SKILLS = [
    ("K1_PREPARE_KNIFE", "pick up the knife"),
    ("K2_CUT_1", "move the knife to the first cutting position, make the first cut through the cucumber, and lift the knife"),
    ("K3_CUT_2", "move the knife to the next cutting position, make the second cut through the cucumber, and lift the knife"),
    ("K4_CUT_3", "move the knife to the next cutting position, make the third cut through the cucumber, and lift the knife"),
    ("K5_CUT_4", "move the knife to the next cutting position, make the fourth cut through the cucumber, and lift the knife"),
    ("K6_RETURN_KNIFE", "return the knife to its original position"),
    ("S7_TRANSFER_CUCUMBER", "transfer all cucumber pieces from the white cutting board into the bowl"),
    ("S8_POUR_SAUCE", "pick up the sauce bowl, pour the sauce into the cucumber bowl, and return the sauce bowl to its original position"),
]


def load_episodes():
    import pyarrow.parquet as pq
    df = pq.read_table(DATASET / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    df = df.sort_values("episode_index")
    eps = []
    for _, row in df.iterrows():
        start = int(row["dataset_from_index"])  # 该集起始帧(数据集全局index)
        length = int(row["length"])
        eps.append({
            "ep": int(row["episode_index"]),
            "start_frame": start,
            "length": length,
            "start_s": round(start / FPS, 3),
            "dur_s": round(length / FPS, 3),
        })
    return eps


EPISODES = load_episodes()
VIDEO_SIZE = VIDEO.stat().st_size

# 逐帧服务: 多 chunk VideoCapture + 顺序缓存
import cv2 as _cv2
_CHUNK_FILES = sorted(VIDEO.parent.glob("*.mp4"))
_chunk_offsets = []   # 每个 chunk 的全局起始帧
_chunk_caps = {}
_chunk_pos = {}
_frame_cache = {}  # 全局帧号 -> jpeg bytes (LRU)
_CACHE_MAX = 400


def _init_chunks():
    total = 0
    for p in _CHUNK_FILES:
        _chunk_offsets.append(total)
        c = _cv2.VideoCapture(str(p))
        n = int(c.get(_cv2.CAP_PROP_FRAME_COUNT))
        c.release()
        total += n
    print(f"video chunks: {len(_CHUNK_FILES)} files, {total} frames total", flush=True)


def get_frame_jpeg(global_frame):
    if not _chunk_offsets:
        _init_chunks()
    if global_frame in _frame_cache:
        return _frame_cache[global_frame]
    # 定位 chunk
    ci = 0
    for i, off in enumerate(_chunk_offsets):
        if global_frame >= off:
            ci = i
    local = global_frame - _chunk_offsets[ci]
    if ci not in _chunk_caps:
        _chunk_caps[ci] = _cv2.VideoCapture(str(_CHUNK_FILES[ci]))
        _chunk_pos[ci] = -1
    cap = _chunk_caps[ci]
    if local == _chunk_pos[ci] + 1:
        ok, img = cap.read()          # 顺序读, 快
    else:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, local)
        ok, img = cap.read()
    if not ok or img is None:
        return None
    _chunk_pos[ci] = local
    ok, buf = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    data = buf.tobytes()
    _frame_cache[global_frame] = data
    if len(_frame_cache) > _CACHE_MAX:
        for k in list(_frame_cache)[: _CACHE_MAX // 2]:
            _frame_cache.pop(k, None)
    return data


def load_labels(ep):
    p = LABEL_DIR / f"labels_ep{ep:03d}.json"
    if p.exists():
        return json.loads(p.read_text())
    return []


def load_labels5(ep):
    p = LABEL_DIR / f"labels5_ep{ep:03d}.json"
    if p.exists():
        return json.loads(p.read_text())
    return []


def get_s3_segment(ep):
    """从旧 4-skill 标注取 S3 段 (帧号, 半开区间)。"""
    labs = load_labels(ep)
    for seg in labs:
        if seg["skill_id"].startswith("S3"):
            return int(seg["start_frame"]), int(seg["end_frame"])
    return None


def get_knife_phase(ep):
    """刀阶段范围 = 旧 S1 起点 -> 旧 S2 终点 (拿起刀 到 放回刀)。"""
    labs = load_labels(ep)
    s1 = next((x for x in labs if x["skill_id"].startswith("S1")), None)
    s2 = next((x for x in labs if x["skill_id"].startswith("S2")), None)
    if s1 is None or s2 is None:
        return None
    return {"start_frame": int(s1["start_frame"]), "end_frame": int(s2["end_frame"])}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = (TOOL_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/fine.html":
            html = (TOOL_DIR / "fine.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/s2.html":
            html = (TOOL_DIR / "s2.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/s2v2.html":
            html = (TOOL_DIR / "s2v2.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/skills":
            self._send(200, json.dumps(SKILLS).encode())
        elif self.path == "/api/episodes":
            self._send(200, json.dumps(EPISODES).encode())
        elif self.path.startswith("/api/labels_s2v2"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            d = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_s2v2")
            sp = d / f"labels_s2v2_ep{ep:03d}.json"
            data = json.loads(sp.read_text()) if sp.exists() else []
            self._send(200, json.dumps(data).encode())
        elif self.path.startswith("/api/knifephase"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            self._send(200, json.dumps(get_knife_phase(ep)).encode())
        elif self.path.startswith("/api/status_s2v2"):
            d = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_s2v2")
            status = {e["ep"]: (d / f"confirmed_s2v2_ep{e['ep']:03d}.json").exists()
                      for e in EPISODES}
            done = sum(1 for v in status.values() if v)
            self._send(200, json.dumps({"done": done, "total": len(EPISODES), "per_ep": status}).encode())
        elif self.path.startswith("/api/status_s2"):
            s2_dir = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_s2")
            status = {e["ep"]: (s2_dir / f"confirmed_s2_ep{e['ep']:03d}.json").exists()
                      for e in EPISODES}
            done = sum(1 for v in status.values() if v)
            self._send(200, json.dumps({"done": done, "total": len(EPISODES), "per_ep": status}).encode())
        elif self.path.startswith("/api/s2segment"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            labs = load_labels(ep)
            s2 = next((x for x in labs if x.get("skill_id", "").startswith("S2")), None)
            self._send(200, json.dumps(s2).encode())
        elif self.path.startswith("/api/labels_s2"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            s2_dir = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_s2")
            sp = s2_dir / f"labels_s2_ep{ep:03d}.json"
            data = json.loads(sp.read_text()) if sp.exists() else []
            self._send(200, json.dumps(data).encode())
        elif self.path.startswith("/api/labels_fine"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            fine_dir = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_fine")
            fp = fine_dir / f"labels_fine_ep{ep:03d}.json"
            data = json.loads(fp.read_text()) if fp.exists() else []
            self._send(200, json.dumps(data).encode())
        elif self.path.startswith("/api/s3seg"):
            # S3 二次切分数据: 返回旧标注的 S3 段 + 已有 labels5
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            seg = get_s3_segment(ep)
            labs5 = load_labels5(ep)
            self._send(200, json.dumps({"s3": seg, "labels5": labs5}).encode())
        elif self.path.startswith("/api/annotations"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            self._send(200, json.dumps(load_labels(ep)).encode())
        elif self.path.startswith("/api/status"):
            # 5-skill 二次切分完成状态: confirmed5 标记
            status = {e["ep"]: (LABEL_DIR / f"confirmed5_ep{e['ep']:03d}.json").exists()
                      for e in EPISODES}
            done = sum(1 for v in status.values() if v)
            self._send(200, json.dumps({"done": done, "total": len(EPISODES), "per_ep": status}).encode())
        elif self.path.startswith("/api/frame"):
            # ?ep=N&f=M -> 该集第 M 帧的 JPEG
            m = re.search(r"ep=(\d+)&f=(\d+)", self.path)
            ep = int(m.group(1)); f = int(m.group(2))
            e = EPISODES[ep]
            gf = e["start_frame"] + f
            if f < 0 or f >= e["length"]:
                self._send(400, b"frame out of range")
                return
            data = get_frame_jpeg(gf)
            if data is None:
                self._send(500, b"frame read failed")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/video":
            # 支持 Range 的 mp4 流
            rng = self.headers.get("Range")
            start, end = 0, VIDEO_SIZE - 1
            if rng:
                m = re.match(r"bytes=(\d+)-(\d*)", rng)
                if m:
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), VIDEO_SIZE - 1)
            length = end - start + 1
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{VIDEO_SIZE}")
            self.end_headers()
            with open(VIDEO, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if self.path.startswith("/api/labels_s2v2"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            d = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_s2v2")
            d.mkdir(parents=True, exist_ok=True)
            (d / f"labels_s2v2_ep{ep:03d}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
            (d / f"confirmed_s2v2_ep{ep:03d}.json").write_text(json.dumps({"confirmed": True, "n": len(data)}))
            self._send(200, json.dumps({"ok": True, "ep": ep, "n": len(data)}).encode())
        if self.path.startswith("/api/labels_s2"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            s2_dir = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_s2")
            s2_dir.mkdir(parents=True, exist_ok=True)
            (s2_dir / f"labels_s2_ep{ep:03d}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
            (s2_dir / f"confirmed_s2_ep{ep:03d}.json").write_text(json.dumps({"confirmed": True, "n": len(data)}))
            self._send(200, json.dumps({"ok": True, "ep": ep, "n": len(data)}).encode())
        if self.path.startswith("/api/labels_fine"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            fine_dir = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_fine")
            fine_dir.mkdir(parents=True, exist_ok=True)
            (fine_dir / f"labels_fine_ep{ep:03d}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
            (fine_dir / f"confirmed_fine_ep{ep:03d}.json").write_text(json.dumps({"confirmed": True, "n": len(data)}))
            self._send(200, json.dumps({"ok": True, "ep": ep, "n": len(data)}).encode())
        if self.path.startswith("/api/labels5"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            (LABEL_DIR / f"labels5_ep{ep:03d}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=1))
            (LABEL_DIR / f"confirmed5_ep{ep:03d}.json").write_text(
                json.dumps({"confirmed": True, "n": len(data)}))
            self._send(200, json.dumps({"ok": True, "ep": ep, "n": len(data)}).encode())
        elif self.path.startswith("/api/annotations"):
            ep = int(re.search(r"ep=(\d+)", self.path).group(1))
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n))
            (LABEL_DIR / f"labels_ep{ep:03d}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=1))
            # 用户手动保存 -> 标记为已确认
            (LABEL_DIR / f"confirmed_ep{ep:03d}.json").write_text(
                json.dumps({"confirmed": True, "n": len(data)}))
            self._send(200, json.dumps({"ok": True, "ep": ep, "n": len(data)}).encode())
        else:
            self._send(404, b"not found")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"标注服务: http://localhost:{port}  (80 episodes, video {VIDEO_SIZE/1e6:.0f}MB)")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
