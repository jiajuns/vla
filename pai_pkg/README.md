# 拍黄瓜 Multi-Skill VLA（π0.5）训练与部署包

任务：Astribot S1 拍黄瓜全流程（拿刀 → 切黄瓜 → 放回刀 → 转移 → 倒酱）多技能视觉语言动作模型。
本包包含两条训练路线（LeRobot PyTorch fork / OpenPI JAX）+ 部署代码 + 数据采集与切分工具。

---

## 1. 技能定义（8-skill / 10 prompts）

25 维状态与动作布局（绝对关节，30Hz）：
chassis(3) + torso(4) + armL(7) + gripL(1) + armR(7) + gripR(1) + head(2)

| skill_id | prompt（task 文本，训练与部署完全一致） |
|---|---|
| K1_PREPARE_KNIFE | pick up the knife and move it to the first cutting position |
| K2_CUT_1 | make the first cut through the cucumber, lift the knife, and move it to the next cutting position |
| K3_CUT_2 | make the second cut through the cucumber, lift the knife, and move it to the next cutting position |
| K4_CUT_3 | make the third cut through the cucumber, lift the knife, and move it to the next cutting position |
| K5_CUT_4 | make the fourth cut through the cucumber, lift the knife, and move it to the next cutting position |
| K6_CUT_5 | make the fifth cut through the cucumber, lift the knife, and move it to the next cutting position |
| K7_CUT_6 | make the sixth cut through the cucumber, lift the knife, and move it to the next cutting position |
| K8_FINAL_CUT | make the final cut through the cucumber and lift the knife |
| K9_RETURN_KNIFE | return the knife to its original position |
| S7_TRANSFER_CUCUMBER | transfer all cucumber pieces from the white cutting board into the bowl |
| S8_POUR_SAUCE | pick up the sauce bowl, pour the sauce into the cucumber bowl, and return the sauce bowl to its original position |

当前 8-skill 数据集（pai_huanggua_8skill_lerobot_v3）：79 集（ep005 因录制数据问题被剔除）、
631 段、147422 帧、实际出现的 task 共 10 个（K7_CUT_6 无人打出来，未进数据集）。

---

## 2. 机器与路径总览

| 机器 | 角色 | 关键路径 |
|---|---|---|
| HPC3 登录节点 hpc3login.hpc.hkust-gz.edu.cn | 训练（H100）| 见第 6/7 节 |
| 工作站（推理服务器）| 模型推理 | /home/chefmate/Astribot/grasp_deploy/ |
| Orin astribot@192.168.0.11 | 机器人桥接 | /home/astribot/grasp_deploy/ |

HPC3 环境：
- 账号 leviccdong，分区 acd_u（8×H100 80GB/节点，2TB 内存）
- Python：/data/user/leviccdong/EKSF/env_smolvla_plate062/bin/python（PyTorch 路线）
- LeRobot fork：/data/user/leviccdong/EKSF/code/smolvla_train/lerobot（0.6.2 定制版，含 ot_train 定制训练循环）
- OpenPI JAX：/data/user/leviccdong/EKSF/code/openpi_jax（uv 管理）
- 数据：/data/user/leviccdong/EKSF/data/pai_huanggua/、输出：.../EKSF/outputs/、日志：.../EKSF/logs/
- JAX 缓存：.../EKSF/cache/openpi（pi05_base 权重 + tokenizer）、v2.1 数据集：.../EKSF/lerobot_hf_home

---

## 3. 数据采集（遥操作录制）

在 Orin 上（WebUI 切「终端」控制模式）：

\`\`\`bash
ssh astribot@192.168.0.11
cd ~/Downloads/astribot_sdk_aarch64 && source env.sh
python3 -u ~/grasp_deploy/orin_collect_rgbd.py \
    --session-id s1_20261009_a --task-id pai_huanggua_v1 \
    --operator yourname --episode 1 --out-dir ~/grasp_deploy/collect_rgbd
# 回车开始录制, 再回车结束
\`\`\`

- 三路相机（head/left_wrist/right_wrist，256×256）+ 25 维关节状态（current/desired/velocity/torque）
- 每个 demo 录完整流程：拿起刀 → 切 N 刀（N=2~6）→ 放回刀 → 转移黄瓜 → 倒酱
- 录制产物经 convert_grasp_to_lerobot.py 类脚本转为 LeRobot v3 原始数据集
  （pai_huanggua_lerobot_v3，80 个长 episode，含 index/frame_index/episode_index 与视频分块）

---

## 4. 数据集切分（打标签 → 合并 → 建数据集）

### 4.1 标注工具（切分刀阶段）

启动（工作站，pi05 conda 环境）：

\`\`\`bash
cd /home/chefmate/Astribot/grasp_deploy/pai_label_tool
/home/chefmate/miniconda3/envs/pi05/bin/python server.py 8765
# 浏览器打开 http://127.0.0.1:8765/s2v2.html
\`\`\`

键盘操作（页面内）：
- A = 闭合一个"循环刀"（下切+抬起+左移到下一位置）——按刀数灵活打，2 刀打 2 次、4 刀打 4 次
- F = 闭合"最后一刀"（下切+抬起完成）——每集必按，不按无法保存
- R = 闭合 RETURN（刀放回原位）——可选，缺省到刀阶段末尾
- 空格/回车 = 保存并跳下一集；Backspace = 撤销；←/→ 逐帧；P 播放/暂停；边界白线可拖动微调
- 页面时间轴：白色=循环刀边界，黄色=最后一刀(F)，紫色=RETURN(R)，RETURN 段固定显示在最后

切分范围 = 旧 4-skill 标注的 S2 段（刀已在黄瓜上方 到 刀放回原位）。
PREPARE 不重新打，合并时沿用旧 S1 段（0 → 刀在黄瓜上方）。

标注保存到：pai_huanggua_labels_s2v2/labels_s2v2_epNNN.json（每段含 skill_id/prompt/start_frame/end_frame）。

### 4.2 合并标签

\`\`\`bash
cd /home/chefmate/Astribot/grasp_deploy
/home/chefmate/miniconda3/envs/pi05/bin/python merge_8skill_labels.py
# 输出 pai_huanggua_labels_8skill/labels_epNNN.json（79 集, 跳过 ep005）
\`\`\`

merge 脚本内含 SKIP_EPS={5}（ep005 录制数据有问题，整条剔除）与 MIN_FRAMES=3（丢弃空段）。

### 4.3 构建 LeRobot v3 数据集

\`\`\`bash
/home/chefmate/miniconda3/envs/pi05/bin/python build_8skill_dataset.py
# 输出 /home/chefmate/Data/vla/datasets/pai_huanggua_8skill_lerobot_v3 （约 40 分钟, ~800MB）
\`\`\`

每段成一个独立 episode（task=对应 prompt，帧号/时间戳从 0 重算，视频重编码 h264）。

### 4.4 推上 HPC3

\`\`\`bash
rsync -av /home/chefmate/Data/vla/datasets/pai_huanggua_8skill_lerobot_v3/ \
  leviccdong@hpc3login.hpc.hkust-gz.edu.cn:/data/user/leviccdong/EKSF/data/pai_huanggua/pai_huanggua_8skill_lerobot_v3/
\`\`\`

---

## 5. 训练路线 A：LeRobot PyTorch fork（当前主力，已训 200k）

\`\`\`bash
cd /data/user/leviccdong/EKSF/code/smolvla_train
sbatch slurm_pi05_8skill_200k_hpc3.slurm
\`\`\`

脚本要点（train/lerobot/slurm_pi05_8skill_200k_hpc3.slurm）：
- 8×H100，--mem=1000G（8 rank 全参初始化需要 ~465GB 内存，512G 会 OOM）
- --exclude=ACD1-3,ACD1-9,ACD1-50,ACD1-54,ACD1-56,ACD1-58（这 6 个节点 CUDA 驱动损坏，cuInit 报 802）
- 全参微调（freeze_vision_encoder=false, train_expert_only=false），从 pi05_base 起训
- 200000 步；batch 4/卡 × 8 卡 × grad_accum 2 = 有效 64；bf16；gradient checkpointing
- 学习率由 pi05 preset 覆盖：peak 2.5e-5，warmup 1000，cosine 30k 步衰减到 2.5e-6
- 每 5000 步存 checkpoint（每个约 40GB，其中 pretrained_model 16GB 可直接部署）
- 实测 0.58 s/step → 200k 步约 33 小时；wandb/eval 关闭
- 输出：/data/user/leviccdong/EKSF/outputs/pi05_pai_huanggua_8skill_200k_8gpu/

重要：本 fork 的 pi05 n_action_steps=50（不是官方 10），部署时必须 --replan-horizon 50。
fork 的定制点：--policy.pretrained_path、--dataset.eval_split、--parallelism.dp_replicate/dp_shard、
--accelerator.gradient_accumulation.steps、--wandb.enable、ot_train 训练循环（含 loss 子项统计）。
fork 与训练环境在 HPC3 上：code/smolvla_train/lerobot（src 用 PYTHONPATH 引用）+ env_smolvla_plate062。

---

## 6. 训练路线 B：OpenPI JAX（备选路线）

### 6.1 前置：pi05_base JAX 权重 + tokenizer（22G）

\`\`\`bash
# 工作站断点续传下载（GCS）
bash train/openpi/resume_pi05_download.sh
# 推送 HPC3
bash train/openpi/push_openpi_cache.sh
\`\`\`

注意：下载脚本会把权重放到 .../pi05_base/params.partial 目录，**必须改名为 params** 并删除残留 params.lock：
\`\`\`bash
cd ~/.cache/openpi/openpi-assets/checkpoints/pi05_base && mv params.partial params && rm -f params.lock
# HPC3 侧同样操作: /data/user/leviccdong/EKSF/cache/openpi/openpi-assets/checkpoints/pi05_base/
\`\`\`

### 6.2 数据集 v3 → v2.1 转换（openpi 锁定的 lerobot 版本）

\`\`\`bash
# 工作站本地（HF_LEROBOT_HOME=/home/chefmate/lerobot_hf_home）跑转换脚本，
# SRC 指 v3 数据集, REPO_ID 对应 astribot/pai_huanggua_8skill_v21（用 8skill 需先改脚本路径）
# 转换产物 rsync 到 HPC3: /data/user/leviccdong/EKSF/lerobot_hf_home/
# openpi 配置项在 src/openpi/training/config.py（name="pai_huanggua_6skill" 为样例）
\`\`\`

### 6.3 依赖安装（重要：必须在登录节点，计算节点无外网）

\`\`\`bash
ssh leviccdong@hpc3login.hpc.hkust-gz.edu.cn
cd /data/user/leviccdong/EKSF/code/openpi_jax
module load uv/0.9.13 && module load acd
uv sync        # .venv 建在共享盘, 计算节点直接可用
\`\`\`

坑：在计算节点上 uv run 会尝试下载 python-build-standalone 而超时失败（无外网），
所以在 slurm 脚本里直接用 .venv 的 python 而不是 uv run。

### 6.4 归一化统计 + 冒烟 + 训练

\`\`\`bash
sbatch slurm_normstats.slurm      # 重算 norm stats（8-skill 需先加 config 项）
sbatch slurm_smoke_pi05_jax.slurm # 冒烟: 1 卡训 5 步
sbatch slurm_train_pi05_jax_200k.slurm
\`\`\`

slurm 配额（acd_u 分区硬限制）：每 GPU ≤12 核、≤240G 内存；账号 QOS 总 GPU 上限 16 卡
（被占满时新 GPU 申请 PENDING，Reason=QOSMaxGRESPerUser）。

---

## 7. 部署

### 7.1 拉 checkpoint

\`\`\`bash
rsync -av leviccdong@hpc3login.hpc.hkust-gz.edu.cn: \
  /data/user/leviccdong/EKSF/outputs/pi05_pai_huanggua_8skill_200k_8gpu/checkpoints/120000/pretrained_model/ \
  /home/chefmate/Astribot/grasp_deploy/checkpoints/pi05_pai_huanggua_8skill_200k/120000/pretrained_model/
# 只需 pretrained_model（16GB, fp32 safetensors），不需要整个 checkpoint（40GB）
\`\`\`

### 7.2 机器人端（Orin，192.168.0.11）

\`\`\`bash
ssh astribot@192.168.0.11
cd ~/Downloads/astribot_sdk_aarch64 && source env.sh
# 1) 归位（拍黄瓜初始位姿, 夹爪张开）
python3 ~/grasp_deploy/go_home_pai.py
# 2) 启动桥接（真实控制, 6200 端口, replan-horizon 50 与 pi05 n_action_steps=50 对齐）
python3 ~/grasp_deploy/orin_bridge.py --execute --replan-horizon 50
# 看到 "listening on 0.0.0.0:6200, waiting for policy client ..." 即就绪
\`\`\`

### 7.3 工作站端（推理服务器）

\`\`\`bash
cd /home/chefmate/Astribot/grasp_deploy
bash start_pi05_live.sh    # CKPT 环境变量可覆盖默认 checkpoint
\`\`\`

运行中按键（8-skill preset）：1=拿起刀 2/3/4=第1/2/3刀(切+抬+左移) 5=最后一刀(切+抬)
6=放回刀 7=转移 8=倒酱；输入任意英文文本=自定义 prompt 立即生效；Ctrl-C 退出。
客户端 chunk=50 步（丢弃前 10 步对齐），执行 40 步后重新推理（--horizon 40）。

---

## 8. 已知坑（Troubleshooting）

1. **HPC3 坏节点**：ACD1-3/9/50/54/56/58 CUDA 损坏（cuInit 802，nvidia-smi 正常但任何 CUDA 程序无法初始化，
   10 月 6 日驱动 610.43.02 重装所致）。现象：GPU 0 MiB、主进程纯 CPU 慢跑。所有 slurm 脚本已 --exclude。
2. **8 卡全参 OOM**：512G 内存不够 8 rank 初始化（每个 rank ~58GB），必须 --mem=1000G。
3. **uv 在计算节点下载超时**：计算节点无外网，uv run 会卡在 python-build-standalone 下载。
   解决：登录节点 module load acd + uv sync；slurm 里直接调 .venv/bin/python。
4. **params.partial 目录名**：下载脚本把 JAX 权重存进 params.partial，openpi 期望 params，需改名。
5. **n_action_steps=50**：本 fork pi05 为 50（非官方 10），部署端 --replan-horizon 50 必须匹配，
   否则动作对不上（只执行了 10 步就重新推理，动作断档）。
6. **磁盘空间**：工作站 Data 盘与根盘易满（vla/outputs 553G 历史输出可清）；
   HPC3 /data 12P 充足。
7. **端口 6200**：工作站客户端连 Orin 桥接的固定端口；"Connection refused" = 机器人端桥接没启动。
8. **QOS 配额**：16 卡/用户满时任何新 GPU 作业 PENDING（Reason=QOSMaxGRESPerUser），等释放即可。
