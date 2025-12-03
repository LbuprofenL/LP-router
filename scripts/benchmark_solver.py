"""
benchmark_solver_v2.py: 修正后的求解器基准测试
主要改进：
1. 修正了物理量级（SLO从28ms改为合理的数值，或模拟TTFT）。
2. 引入软约束（Slack），防止因单一请求超时导致全盘无解。
3. 更换为 SCIP 后端。
"""
import argparse
import random
import time
from ortools.linear_solver import pywraplp

# --- 1. 修正物理参数 ---
# 假设场景：优化 TTFT (首字延迟)
# 4090 Prefill 速度约为 3000 tokens/s -> 0.33 ms/token
# A100 Prefill 速度约为 6000 tokens/s -> 0.16 ms/token
DEVICE_CONFIG = {
    "A100": {
        "vram_limit": 80_000, 
        "prefill_ms_per_token": 0.15,  # A100 处理 Prompt 极快
        "overhead_ms": 10.0,           # 基础开销
        "power_base": 300.0,           # 基础功率 W
        "energy_per_req": 50.0,        # 假设单位能耗 (简化模型)
    },
    "Ada": {
        "vram_limit": 48_000,
        "prefill_ms_per_token": 0.25,
        "overhead_ms": 12.0,
        "power_base": 150.0,
        "energy_per_req": 30.0,
    },
    "4090": {
        "vram_limit": 24_000,
        "prefill_ms_per_token": 0.35,
        "overhead_ms": 15.0,
        "power_base": 100.0,
        "energy_per_req": 20.0,
    },
}

# 假设 Token 占用 KV Cache (MB)
TOKEN_TO_VRAM_MB = 1.5 
# 目标 SLO: TTFT < 200ms
TARGET_SLO_MS = 200.0 

def generate_requests(num_requests: int, low: int, high: int):
    requests = []
    for idx in range(num_requests):
        # 这里随机生成的是 Input Prompt Length (影响 TTFT)
        prompt_len = random.randint(low, high)
        # 还要考虑 Output Length (影响显存占用)
        output_len = random.randint(128, 512) 
        requests.append({
            "id": f"req_{idx:02d}", 
            "prompt_len": prompt_len,
            "total_tokens_for_mem": prompt_len + output_len
        })
    return requests

def build_and_solve(requests, slo_target=TARGET_SLO_MS):
    # --- 2. 建议使用 SCIP 后端 ---
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("SCIP 不可用，回退到 CBC...")
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")

    # 变量定义
    x = {} # x[req, device] = 0 or 1
    slack = {} # slack[req] = 超时的时间量 (ms) >= 0
    
    infinity = solver.infinity()

    for req in requests:
        # 定义松弛变量，允许违约，但要惩罚
        slack[req["id"]] = solver.NumVar(0, infinity, f"slack_{req['id']}")
        
        for device in DEVICE_CONFIG:
            x[(req["id"], device)] = solver.BoolVar(f"x_{req['id']}_{device}")

    # 约束 1: 每个请求必须且只能分配给一个设备
    for req in requests:
        solver.Add(sum(x[(req["id"], device)] for device in DEVICE_CONFIG) == 1)

    # 约束 2: 显存容量限制 (硬约束)
    for device, config in DEVICE_CONFIG.items():
        solver.Add(
            sum(
                x[(req["id"], device)] * req["total_tokens_for_mem"] * TOKEN_TO_VRAM_MB
                for req in requests
            ) <= config["vram_limit"]
        )

    # 约束 3: 延迟软约束 (Latency <= SLO + slack)
    # 这里的 Latency 模型修改为 TTFT
    for req in requests:
        prompt_len = req["prompt_len"]
        
        # 实际延迟表达式
        latency_expr = sum(
            x[(req["id"], device)] * (config["prefill_ms_per_token"] * prompt_len + config["overhead_ms"])
            for device, config in DEVICE_CONFIG.items()
        )
        
        # 核心逻辑：延迟 - 松弛变量 <= SLO
        # 如果延迟(150) <= SLO(200)，slack可以是0
        # 如果延迟(250) <= SLO(200)，slack必须至少是50
        solver.Add(latency_expr - slack[req["id"]] <= slo_target)

    # --- 3. 目标函数：最小化 (总能耗 + 惩罚系数 * 总违约时间) ---
    objective = solver.Objective()
    
    PENALTY_WEIGHT = 100.0 # 1ms 的超时相当于消耗 100J 的能量 -> 迫使 Solver 尽量不超时
    
    total_energy_expr = 0
    total_slack_expr = 0
    
    for req in requests:
        # 累加能耗
        for device, config in DEVICE_CONFIG.items():
            # 简化能耗模型
            objective.SetCoefficient(x[(req["id"], device)], config["energy_per_req"])
        
        # 累加惩罚
        objective.SetCoefficient(slack[req["id"]], PENALTY_WEIGHT)

    objective.SetMinimization()

    # 开始计时
    start = time.time()
    status = solver.Solve()
    elapsed_ms = (time.time() - start) * 1000

    return status, elapsed_ms, solver, x, slack

def pretty_print(status, elapsed_ms, solver, assignments, slacks, requests):
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        print(f"✅ 求解成功! 耗时: {elapsed_ms:.2f} ms")
        violation_count = 0
        
        print(f"{'ReqID':<8} | {'Prompt':<6} | {'Device':<8} | {'Latency':<8} | {'Slack(ms)':<8}")
        print("-" * 50)
        
        for i, req in enumerate(requests):
            if i >= 10: break # 只打前10个
            chosen_device = "N/A"
            pred_lat = 0
            
            for device, config in DEVICE_CONFIG.items():
                if assignments[(req["id"], device)].solution_value() > 0.5:
                    chosen_device = device
                    pred_lat = config["prefill_ms_per_token"] * req["prompt_len"] + config["overhead_ms"]
                    break
            
            s_val = slacks[req["id"]].solution_value()
            if s_val > 0.001:
                violation_count += 1
                
            print(f"{req['id']:<8} | {req['prompt_len']:<6} | {chosen_device:<8} | {pred_lat:<8.1f} | {s_val:<8.1f}")
            
        print(f"\n总计 SLO 违约请求数: {violation_count} / {len(requests)}")
        
    else:
        print(f"❌ 求解失败，状态码: {status}")

def main():
    args = argparse.Namespace(num_requests=50, token_min=128, token_max=2048, seed=42)
    print(f"=== Benchmark: {args.num_requests} Requests (SCIP + Soft Constraints) ===")
    requests = generate_requests(args.num_requests, args.token_min, args.token_max)
    status, elapsed, solver, x, slack = build_and_solve(requests)
    pretty_print(status, elapsed, solver, x, slack, requests)

if __name__ == "__main__":
    main()