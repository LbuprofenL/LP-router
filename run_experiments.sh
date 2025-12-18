#!/bin/bash

# 定义要测试的策略
STRATEGIES=("Random" "Round-Robin" "LP-Router")
USER_COUNT=50
DURATION="1m" # 每次压测 1 分钟

for strat in "${STRATEGIES[@]}"; do
    echo "=== Running Experiment: $strat ==="
    
    # 1. 启动 Server (后台运行)
    # 确保你的 server.py 接受 --strategy 参数
    python src/server.py --strategy $strat > server.log 2>&1 &
    SERVER_PID=$!
    
    # 等待 Server 启动
    sleep 5
    
    # 2. 运行 Locust (无头模式，不打开网页)
    echo "Starting Locust..."
    locust -f scripts/locustfile.py \
           --headless \
           -u $USER_COUNT -r 10 \
           --run-time $DURATION \
           --host http://localhost:8000 \
           --csv data/results/locust_$strat
           
    # 3. 停止 Server
    kill $SERVER_PID
    
    # 4. 重命名结果文件 (Server 记录的详细 CSV)
    # 假设 logger 默认写到 experiment_results.csv
    mv data/results/experiment_results.csv data/results/final_log_$strat.csv
    
    echo "=== Finished $strat ==="
    sleep 2
done