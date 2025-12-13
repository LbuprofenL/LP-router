#!/bin/bash

# --- 配置 ---
# 使用你拉取好的、兼容的镜像名
DOCKER_IMAGE="vllm/vllm-openai:v0.8.5.post1" 

# 定义要测试的参数
CONCURRENCY_LEVELS=(16 32 64 96 128 160 192 256)
MAX_TOKENS_LIST=(128 256 512)
MODEL_NAME="qwen3-8b"
MODEL_PATH="/models/qwen3-8b"
GPU_NAME="RTX-4090"

# --- 准备工作 ---
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 自动生成输出文件名：包含模型名、GPU名和时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="performance_${MODEL_NAME}_${GPU_NAME}_${TIMESTAMP}.csv"

# 确保结果目录存在
mkdir -p "$PROJECT_ROOT/data/experiments"

echo "输出文件: $OUTPUT_FILE"

# --- 启动一个后台运行的、持久化的容器 ---
echo "正在启动一个后台实验容器..."
CONTAINER_ID=$(docker run \
  -d \
  --gpus all \
  --name automated-vllm-runner \
  --entrypoint /bin/bash \
  -v /data/lxl/models/LP-router:/models \
  -v "$PROJECT_ROOT:/app" \
  -w /app \
  $DOCKER_IMAGE \
  -c "tail -f /dev/null") # 使用 tail 命令让容器保持运行

echo "在容器内安装 pynvml 依赖..."
docker exec "$CONTAINER_ID" pip install nvidia-ml-py

# 检查容器是否成功启动
if [ -z "$CONTAINER_ID" ]; then
    echo "错误：容器启动失败！"
    exit 1
fi
echo "实验容器已启动，ID: $CONTAINER_ID"
# 等待几秒钟确保容器内部服务完全就绪
sleep 5

# --- 循环遍历所有参数组合并执行测试 ---
for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
  for tokens in "${MAX_TOKENS_LIST[@]}"; do
    echo "======================================================"
    echo "正在运行: 并发数=$concurrency, Tokens=$tokens"
    echo "======================================================"
    
    # 使用 docker exec 在正在运行的容器内执行命令
    docker exec $CONTAINER_ID python3 /app/scripts/calibrate_hardware.py \
      --gpu $GPU_NAME \
      --model-path /models/$MODEL_NAME \
      --num-requests $concurrency \
      --max-tokens $tokens \
      --output-csv /app/data/experiments/$OUTPUT_FILE
    
  done
done

# --- 清理工作 ---
echo "所有实验完成，正在停止并删除容器..."
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo "======================================================"
echo "实验数据已保存在: $PROJECT_ROOT/data/experiments/$OUTPUT_FILE"
echo "======================================================"