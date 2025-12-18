from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import argparse
import uvicorn
import logging
import time

# 导入你的核心模块
import sys
import os
# 先添加项目根目录到 Python 路径，然后才能导入 src 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import ExperimentLogger
from src.router import LPRouter

# === 全局变量声明 ===
# 为了让 FastAPI 的路由函数能访问到，我们在模块层级声明这些变量
# 真正的初始化会在 main 函数中根据命令行参数执行
router = None
csv_logger = None

csv_logger = ExperimentLogger(f"results_{int(time.time())}.csv")

app = FastAPI(title="LP-Router Server")

# 初始化 Router
# 这里模拟你的集群配置
cluster_config = {
    0: "RTX-4090",
    1: "RTX-4090",
    2: "A100" # 假设你有这三张卡
}
router = LPRouter(cluster_config)

class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    max_tokens: Optional[int] = 128
    # 可选：你可以让 Client 传 prompt_len，或者在 Server 端算
    # 为了简单，我们这里假设 Client 传了 prompt_len (模拟 DistilBERT 的结果)
    prompt_len: Optional[int] = 128 

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """
    接收请求 -> 路由决策 -> 返回决策结果 (暂不转发给 vLLM，先测调度逻辑)
    """
    start_time = time.time()
    
    # 1. 构造 Router 需要的 Request 对象
    # 注意：这里我们假设 DistilBERT 已经把 Prompt 长度算好了
    router_req = {
        "id": f"req_{int(start_time*1000)}",
        "prompt_len": req.prompt_len,
        "output_len": req.max_tokens,
        "model": req.model
    }
    
    # 2. 调用 Router 进行调度
    # schedule 接口目前是设计为批处理的，我们这里传单条列表
    target_slo = 200.0
    decisions = router.schedule([router_req], slo_ms=target_slo)
    
    decision = decisions.get(router_req["id"])

    if not decision:
        raise HTTPException(status_code=500, detail="Scheduling failed")
    
    csv_logger.log(router_req, decision, slo_ms=target_slo)
    # 3. (未来工作) 这里应该根据 decision['gpu_id'] 转发给对应的 vLLM
    # 现在我们直接返回决策结果，方便调试
    
    return {
        "id": router_req["id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "message": {"role": "assistant", "content": "This is a mock response."},
                "finish_reason": "stop"
            }
        ],
        # 调试信息：告诉你这个请求被分到了哪里
        "debug_router": {
            "assigned_gpu": decision["gpu_id"],
            "gpu_model": decision["gpu_model"],
            "predicted_latency_ms": decision["expected_latency"],
            "strategy": decision.get("strategy", "Unknown")
        }
    }

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Server")

    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="LP-Router Server")
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="LP-Router", 
        choices=["LP-Router", "Random", "Round-Robin", "LOR"],
        help="Scheduling strategy to use"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind"
    )
    args = parser.parse_args()

    logger.info(f"Starting server with Strategy: {args.strategy}")

    # --- 初始化全局组件 ---
    
    # 1. 硬件配置 (模拟)
    cluster_config = {
        0: "RTX-4090",
        1: "RTX-4090",
        2: "A100"
    }
    
    # 2. 初始化 Router (传入命令行指定的策略)
    # 这里的 router 会修改上面定义的全局变量
    router = LPRouter(cluster_config, strategy=args.strategy)
    
    # 3. 初始化 Logger
    # 文件名加上策略和时间戳，防止覆盖
    log_filename = f"experiment_{args.strategy}_{int(time.time())}.csv"
    csv_logger = ExperimentLogger(log_filename)
    logger.info(f"Logging results to data/results/{log_filename}")

    # --- 启动 Uvicorn ---
    uvicorn.run(app, host=args.host, port=args.port)