#!/usr/bin/env python3
"""合并旧S1 + S2V2刀阶段标注(4刀+RETURN) + 旧S3/S4 -> 完整 8-skill 标注
结构: K1_PREPARE(旧S1) | K2_CUT_1 | K3_CUT_2 | K4_CUT_3 | K5_CUT_4 | K6_RETURN | S7_TRANSFER | S8_POUR
输出: pai_huanggua_labels_8skill/labels_epNNN.json
"""
import json
from pathlib import Path

LBL4 = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels")
LBL_K = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_s2v2")
OUT = Path("/home/chefmate/Astribot/grasp_deploy/pai_huanggua_labels_8skill")
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = {
    "K1_PREPARE_KNIFE": "pick up the knife and move it to the first cutting position",
    "K2_CUT_1": "make the first cut through the cucumber, lift the knife, and move it to the next cutting position",
    "K3_CUT_2": "make the second cut through the cucumber, lift the knife, and move it to the next cutting position",
    "K4_CUT_3": "make the third cut through the cucumber, lift the knife, and move it to the next cutting position",
    "K5_CUT_4": "make the fourth cut through the cucumber, lift the knife, and move it to the next cutting position",
    "K6_CUT_5": "make the fifth cut through the cucumber, lift the knife, and move it to the next cutting position",
    "K7_CUT_6": "make the sixth cut through the cucumber, lift the knife, and move it to the next cutting position",
    "K8_FINAL_CUT": "make the final cut through the cucumber and lift the knife",
    "K9_RETURN_KNIFE": "return the knife to its original position",
    "S7_TRANSFER_CUCUMBER": "transfer all cucumber pieces from the white cutting board into the bowl",
    "S8_POUR_SAUCE": "pick up the sauce bowl, pour the sauce into the cucumber bowl, and return the sauce bowl to its original position",
}
MIN_FRAMES = 3  # 短于此的段丢弃 (如某 demo 只切了 3 刀)


SKIP_EPS = {5}  # ep005 录制数据本身有问题, 整条剔除


def main():
    stats = {}
    problems = []
    missing = [ep for ep in range(80) if ep not in SKIP_EPS and not (LBL_K / f"labels_s2v2_ep{ep:03d}.json").exists()]
    if missing:
        print(f"缺少 {len(missing)} 集的 S2V2 标注: {missing}")
        return
    for ep in range(80):
        if ep in SKIP_EPS:
            continue
        old4 = sorted(json.loads((LBL4 / f"labels_ep{ep:03d}.json").read_text()),
                      key=lambda s: s["start_frame"])
        s1 = next(s for s in old4 if s["skill_id"].startswith("S1"))
        s3 = next(s for s in old4 if s["skill_id"].startswith("S3"))
        s4 = next(s for s in old4 if s["skill_id"].startswith("S4"))
        kf = json.loads((LBL_K / f"labels_s2v2_ep{ep:03d}.json").read_text())
        kf = sorted(kf, key=lambda s: s["start_frame"])
        kf = [x for x in kf if x["end_frame"] - x["start_frame"] >= MIN_FRAMES]

        # 校验: 刀阶段段连续覆盖, 无重叠无缝隙
        prev_end = None
        for sg in kf:
            if prev_end is not None and sg["start_frame"] != prev_end:
                problems.append(f"ep{ep}: gap/overlap at {prev_end} -> {sg['start_frame']}")
            prev_end = sg["end_frame"]

        out = [{
            "skill_id": "K1_PREPARE_KNIFE", "prompt": PROMPTS["K1_PREPARE_KNIFE"],
            "start": s1["start"], "end": s1["end"],
            "start_frame": s1["start_frame"], "end_frame": s1["end_frame"],
        }]
        for sg in kf:
            sid = sg["skill_id"]
            out.append({
                "skill_id": sid, "prompt": PROMPTS[sid],
                "start": sg["start"], "end": sg["end"],
                "start_frame": sg["start_frame"], "end_frame": sg["end_frame"],
            })
        out.append({
            "skill_id": "S7_TRANSFER_CUCUMBER", "prompt": PROMPTS["S7_TRANSFER_CUCUMBER"],
            "start": s3["start"], "end": s3["end"],
            "start_frame": s3["start_frame"], "end_frame": s3["end_frame"],
        })
        out.append({
            "skill_id": "S8_POUR_SAUCE", "prompt": PROMPTS["S8_POUR_SAUCE"],
            "start": s4["start"], "end": s4["end"],
            "start_frame": s4["start_frame"], "end_frame": s4["end_frame"],
        })
        (OUT / f"labels_ep{ep:03d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        for sg in out:
            stats[sg["skill_id"]] = stats.get(sg["skill_id"], 0) + 1

    print("merged 80 episodes ->", OUT)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v} segments")
    if problems:
        print("PROBLEMS:")
        for p in problems[:20]:
            print(" ", p)


if __name__ == "__main__":
    main()
