# 拍黄瓜 Multi-Skill VLA（π0.5）

Astribot S1 拍黄瓜任务的训练、部署、采集与标注代码。代码和数据集均公开，可直接分享和下载。

## 代码与文档

- [完整使用文档（8 大节）](pai_pkg/README.md)
- [原始代码压缩包](pai_huanggua_vla_package.zip)：保留服务器原件，48,489 字节。
- [机器人与工作站部署](pai_pkg/deploy/)
- [标注、标签合并与数据集构建](pai_pkg/data/)
- [LeRobot 与 OpenPI 训练脚本](pai_pkg/train/)

`pai_pkg/` 保留压缩包中的原始目录和文件内容。文档及脚本中的服务器路径是原环境示例，使用时请按自己的环境调整。

## 公开数据集

大体积数据托管于 Hugging Face，GitHub 保存代码和原始代码包。

| 数据集 | 集数 / 段数 | 帧数 | 任务文本数 | 源文件总大小 |
| --- | ---: | ---: | ---: | ---: |
| [8-skill 训练数据](https://huggingface.co/datasets/sunjiajun1/pai_huanggua_8skill_lerobot_v3) | 631 段，来自 79 集 | 147,422 | 10 | 813,308,354 字节 |
| [原始完整流程数据](https://huggingface.co/datasets/sunjiajun1/pai_huanggua_lerobot_v3) | 80 集 | 150,046 | 1 | 851,921,655 字节 |

两套数据均为 LeRobot v3 格式：30 Hz、25 维状态与动作，以及头部、左腕、右腕三路 256×256 RGB 视频。8-skill 版本已剔除原始 ep005；实际任务文本数为 10，详见完整文档。原始数据集还附有服务器上已有的 4-skill 标注；S2V2 和 8-skill 手工标注 JSON 不在本次取得的源目录中，最终 8-skill 数据集已包含切分后的 episode 与 task 元数据。

安装 Hugging Face CLI 后下载：

```bash
python -m pip install -U huggingface_hub

hf download sunjiajun1/pai_huanggua_8skill_lerobot_v3 \
  --repo-type dataset --local-dir datasets/pai_huanggua_8skill_lerobot_v3

hf download sunjiajun1/pai_huanggua_lerobot_v3 \
  --repo-type dataset --local-dir datasets/pai_huanggua_lerobot_v3
```

公开下载无需账号授权。数据集各自附有源文件 SHA-256 清单，便于核对文件完整性。
