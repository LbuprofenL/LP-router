#!/bin/bash

# --- 配置 ---
# 使用你拉取好的、兼容的镜像名
DOCKER_IMAGE="vllm/vllm-openai:v0.8.5.post1" 

# 定义要测试的参数
CONCURRENCY_LEVELS=(16 32 64)
MAX_TOKENS_LIST=(128 256 512)
MODEL_NAME="qwen3-8b"
MODEL_PATH="/models/qwen3-8b"
GPU_NAME="RTX-4090"
OUTPUT_FILE="performance_data_qwen3.csv"

# --- 准备工作 ---
# 确保结果和模型目录存在
mkdir -p results

# 如果旧的结果文件存在，先删除
rm -f $OUTPUT_FILE

# --- 启动一个后台运行的、持久化的容器 ---
echo "正在启动一个后台实验容器..."
CONTAINER_ID=$(docker run \
  -d \
  --gpus all \
  --name automated-vllm-runner \
  --entrypoint /bin/bash \
  -v /data/lxl/models/roofline:/models \
  -v /data/lxl/projects/roofline/experiment_data_collector.py:/app/experiment.py \
  -v /data/lxl/projects/roofline/results:/app/results \
  $DOCKER_IMAGE \
  -c "tail -f /dev/null") # 使用 tail 命令让容器保持运行

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
    docker exec $CONTAINER_ID python3 /app/experiment.py \
      --gpu $GPU_NAME \
      --model-path $MODEL_PATH \
      --num-requests $concurrency \
      --max-tokens $tokens \
      --output-csv /app/results/$OUTPUT_FILE
    
  done
done

# --- 清理工作 ---
echo "所有实验完成，正在停止并删除容器..."
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo "实验数据已保存在主机的: $OUTPUT_FILE"