#!/usr/bin/env bash
# ============================================================
# 启动脚本 1: 机器人桥接服务 (真实 SDK 模式)
# 用法:
#   DRY-RUN 验证(不下发指令, 推荐第一次跑):   bash start_robot_server.sh
#   正式执行(控制机器人!):  I_UNDERSTAND_ROBOT_WILL_MOVE=1 bash start_robot_server.sh execute
# ============================================================
set -eo pipefail

MODE="${1:-dryrun}"
SDK_ROOT="${SDK_ROOT:-/home/chefmate/Astribot/astribot_sdk}"
SDK_PYTHON_BIN="${SDK_PYTHON_BIN:-/home/chefmate/miniconda3/envs/astribot/bin/python}"
ROS_MASTER_URI="${ROS_MASTER_URI:-http://192.168.0.10:11311}"
ROS_IP="${ROS_IP:-192.168.0.100}"
PORT="${PORT:-5000}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_DIR="${SAVE_DIR:-/home/chefmate/Astribot/grasp_deploy/runs/$(date +%Y%m%d_%H%M%S)}"

cd "$(dirname "$0")"

unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
export ROS_DISTRO="noetic"
export PATH="/home/chefmate/miniconda3/bin:/usr/bin:/bin:/usr/local/bin"
export ROBOT_TYPE="${ROBOT_TYPE:-S1}"
export ROS_MASTER_URI ROS_IP
export ROS_HOME="/home/chefmate/.ros_grasp_deploy"
mkdir -p "$ROS_HOME"

if [ -f /opt/ros/noetic/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/noetic/setup.bash
fi
# shellcheck disable=SC1091
[ -f "$SDK_ROOT/env.sh" ] && source "$SDK_ROOT/env.sh"

echo "===================================================="
echo "Grasp VLA 机器人桥接服务"
echo "  ROS_MASTER_URI = $ROS_MASTER_URI"
echo "  ROS_IP         = $ROS_IP"
echo "  PORT           = $PORT"
echo "  MAX_STEPS      = $MAX_STEPS"
echo "  SAVE_DIR       = $SAVE_DIR"
echo "===================================================="

EXTRA=()
if [ "$MODE" = "execute" ]; then
    if [ "${I_UNDERSTAND_ROBOT_WILL_MOVE:-0}" != "1" ]; then
        echo "错误: 执行模式需要显式确认。"
        echo "请先清空工作区、确认急停与操作员就位, 然后用:"
        echo "  I_UNDERSTAND_ROBOT_WILL_MOVE=1 bash start_robot_server.sh execute"
        exit 2
    fi
    EXTRA+=(--execute)
else
    echo "DRY-RUN 模式: 只读状态/相机, 不会给机器人下发任何指令"
fi

mkdir -p "$SAVE_DIR"
exec "$SDK_PYTHON_BIN" -u grasp_robot_server.py \
    --port "$PORT" \
    --max-steps "$MAX_STEPS" \
    --save-dir "$SAVE_DIR" \
    "${EXTRA[@]}"
