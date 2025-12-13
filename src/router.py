# src/router.py
import time
import logging
import numpy as np
from typing import List, Dict, Tuple
from ortools.linear_solver import pywraplp

# 引入你的模块（使用包内相对导入，避免路径问题）
from .controller import GPUController
# from .predictor import LatencyPredictor # TODO: 真正实现时取消注释

# --- Mock Predictor (如果你还没写完 src/predictor.py，暂时用这个顶替) ---
class MockPredictor:
    def __init__(self):
        # 这里填入我们刚才标定好的参数！
        self.params = {
            "RTX-4090": {"alpha": 1.0, "beta": 0.015, "bw": 852.77, "flops": 330000}, # Beta=15ms
            "A100":     {"alpha": 1.0, "beta": 0.010, "bw": 1935.0, "flops": 312000},
        }
    
    def predict(self, req, gpu_name):
        # 简化的 Roofline 计算 (假设 Qwen-8B)
        # AI ~ 60 (Decode)
        input_len = req.get("prompt_len", 128)
        output_len = req.get("output_len", 128)
        
        # 简单模拟：Latency = (Token * 0.01s) * alpha + beta
        p = self.params.get(gpu_name, self.params["RTX-4090"])
        
        # 区分 Prefill 和 Decode (粗略)
        t_prefill = (input_len * 0.0005) # 假设值
        t_decode = (output_len * 0.01)   # 假设值
        
        t_theo = t_prefill + t_decode
        t_real = p["alpha"] * t_theo + p["beta"]
        
        # 估算 AI (用于 DVFS 决策)
        # AI = FLOPs / Bytes. Decode 阶段 AI 很低 (~10-20)
        estimated_ai = 20.0 if output_len > input_len else 150.0 
        
        # 估算能耗 (Power * Time)
        # 假设 4090 平均 300W, A100 平均 250W
        power_avg = 300.0 if "4090" in gpu_name else 250.0
        energy = power_avg * t_real
        
        return t_real, energy, estimated_ai

# -----------------------------------------------------------------------

logger = logging.getLogger("LPRouter")

class LPRouter:
    def __init__(self, hardware_config: Dict):
        """
        初始化路由调度器
        :param hardware_config: 定义集群有哪些卡，例如 {"gpu_0": "RTX-4090", "gpu_1": "A100"}
        """
        self.hardware_map = hardware_config
        self.controller = GPUController()
        
        # 尝试加载真实 Predictor，否则使用 Mock
        try:
            from .predictor import LatencyPredictor
            self.predictor = LatencyPredictor("configs/gpu_specs.json", "configs/model_specs.json")
            logger.info("Loaded real LatencyPredictor.")
        except ImportError:
            self.predictor = MockPredictor()
            logger.warning("Using MockPredictor (src.predictor not found).")

        # 求解器参数
        self.solver_timeout_ms = 20  # 超过 20ms 还没解出来就降级

    def schedule(self, requests: List[Dict], slo_ms: float = 200.0) -> Dict:
        """
        核心调度入口
        :param requests: 请求列表 [{"id": "req1", "prompt_len": 128, ...}, ...]
        :param slo_ms: 目标延迟 (TTFT 或 Total Latency)
        :return: 分配方案 {"req1": {"gpu_id": 0, "power_limit": 300}, ...}
        """
        start_time = time.time()
        
        # 1. 准备数据矩阵
        #    Cost Matrix: Energy[i][j], Latency[i][j]
        req_ids = [r["id"] for r in requests]
        gpu_ids = list(self.hardware_map.keys())
        
        cost_energy = np.zeros((len(requests), len(gpu_ids)))
        constraint_latency = np.zeros((len(requests), len(gpu_ids)))
        predicted_ais = np.zeros((len(requests), len(gpu_ids))) # 用于 DVFS 决策
        
        for i, req in enumerate(requests):
            for j, gid in enumerate(gpu_ids):
                gpu_model = self.hardware_map[gid]
                # 调用预测器 (这是论文的核心贡献点！)
                lat, eng, ai = self.predictor.predict(req, gpu_model)
                
                constraint_latency[i, j] = lat * 1000.0 # 转为 ms
                cost_energy[i, j] = eng
                predicted_ais[i, j] = ai

        # 2. 调用 Solver 求解 (ILP)
        assignments = self._solve_ilp_ortools(req_ids, gpu_ids, cost_energy, constraint_latency, slo_ms)
        
        # 3. 后处理：生成决策并加入 DVFS 逻辑
        final_decision = {}
        
        for req_idx, gpu_idx in assignments.items():
            req_id = req_ids[req_idx]
            gpu_id = gpu_ids[gpu_idx]
            ai = predicted_ais[req_idx, gpu_idx]
            
            # === 论文 4.3.3 DVFS 策略 ===
            # 如果判定为 Memory Bound (AI < 30 且是 Decode 阶段)，可以安全降功耗
            # 4090 的 Ridge Point 很高，但在 Decode 阶段 AI 极低
            power_cap = None
            if "RTX-4090" in self.hardware_map[gpu_id]:
                if ai < 50: # Memory Bound 阈值 (需要实验标定)
                    power_cap = 300 # 降频节能模式
                else:
                    power_cap = 450 # 性能模式
            
            # 注意：实际控制通常是异步的，或者按 Batch 统一设置。
            # 这里为了演示，将指令附带在决策中
            final_decision[req_id] = {
                "gpu_id": gpu_id,
                "gpu_model": self.hardware_map[gpu_id],
                "expected_latency": constraint_latency[req_idx, gpu_idx],
                "set_power_limit": power_cap
            }

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Scheduling completed in {elapsed:.2f}ms. Assigned {len(final_decision)}/{len(requests)} requests.")
        
        return final_decision

    def _solve_ilp_ortools(self, req_ids, gpu_ids, energy_matrix, latency_matrix, slo_ms):
        """
        使用 OR-Tools 进行整数规划求解
        目标: Min(Total Energy)
        约束: Latency <= SLO (软约束)
        """
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            logger.warning("SCIP solver unavailable, falling back to Greedy.")
            return self._solve_greedy(req_ids, gpu_ids, energy_matrix, latency_matrix, slo_ms)

        # 变量定义
        x = {} # x[i, j] = 1
        num_reqs = len(req_ids)
        num_gpus = len(gpu_ids)
        infinity = solver.infinity()
        
        # 引入松弛变量 slack[i] 以避免无解 (Hard Constraint -> Soft Constraint)
        slack = {} 

        for i in range(num_reqs):
            slack[i] = solver.NumVar(0, infinity, f"slack_{i}")
            for j in range(num_gpus):
                x[i, j] = solver.BoolVar(f"x_{i}_{j}")

        # 约束 1: 每个请求必须分配给 1 个 GPU
        for i in range(num_reqs):
            solver.Add(sum(x[i, j] for j in range(num_gpus)) == 1)

        # 约束 2: 延迟约束 (Latency <= SLO + slack)
        for i in range(num_reqs):
            # sum(x[i,j] * lat[i,j]) <= SLO + slack[i]
            solver.Add(
                sum(x[i, j] * latency_matrix[i, j] for j in range(num_gpus)) 
                <= slo_ms + slack[i]
            )

        # 约束 3 (隐式): 显存容量/Batch容量约束 (这里暂时略过，需 Predictor 提供显存预估)
        
        # 目标函数: Min (Total Energy + Penalty * Total Slack)
        objective = solver.Objective()
        PENALTY_WEIGHT = 500.0 # 1ms 的违约相当于 500J 的惩罚，迫使尽量满足 SLO
        
        for i in range(num_reqs):
            objective.SetCoefficient(slack[i], PENALTY_WEIGHT)
            for j in range(num_gpus):
                objective.SetCoefficient(x[i, j], energy_matrix[i, j])
        
        objective.SetMinimization()
        
        # 设置求解超时
        solver.set_time_limit(self.solver_timeout_ms)

        status = solver.Solve()
        
        assignments = {}
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            for i in range(num_reqs):
                for j in range(num_gpus):
                    if x[i, j].solution_value() > 0.5:
                        assignments[i] = j
                        break
        else:
            logger.warning("Solver failed or timed out. Falling back to Greedy.")
            return self._solve_greedy(req_ids, gpu_ids, energy_matrix, latency_matrix, slo_ms)
            
        return assignments

    def _solve_greedy(self, req_ids, gpu_ids, energy_matrix, latency_matrix, slo_ms):
        """
        降级策略: 贪心算法
        逻辑: 只要满足 SLO，就选能耗最低的。如果都不能满足，选延迟最低的。
        """
        assignments = {}
        for i in range(len(req_ids)):
            best_gpu = -1
            min_energy = float('inf')
            min_latency = float('inf')
            best_latency_gpu = -1
            
            # 遍历所有 GPU
            for j in range(len(gpu_ids)):
                lat = latency_matrix[i, j]
                eng = energy_matrix[i, j]
                
                # 记录最快 GPU 以防万一
                if lat < min_latency:
                    min_latency = lat
                    best_latency_gpu = j
                
                # 检查是否满足 SLO
                if lat <= slo_ms:
                    if eng < min_energy:
                        min_energy = eng
                        best_gpu = j
            
            # 决策
            if best_gpu != -1:
                assignments[i] = best_gpu
            else:
                # 此时所有卡都超时，只能选最快的（牺牲能效保 SLO）
                assignments[i] = best_latency_gpu
                
        return assignments

# 使用示例 (仅用于测试)
if __name__ == "__main__":
    # 定义一个异构集群
    cluster_config = {
        0: "RTX-4090",
        1: "RTX-4090", # 假设有两张 4090
        2: "A100"      # 一张 A100
    }
    
    router = LPRouter(cluster_config)
    
    # 模拟一批请求
    mock_requests = [
        {"id": "req_0", "prompt_len": 128, "output_len": 128}, # 短任务
        {"id": "req_1", "prompt_len": 2048, "output_len": 512} # 长任务
    ]
    
    decision = router.schedule(mock_requests, slo_ms=200)
    
    import json
    print(json.dumps(decision, indent=2))