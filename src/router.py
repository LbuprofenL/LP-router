# src/router.py
import time
import random
import logging
import numpy as np
from typing import List, Dict, Tuple
from ortools.linear_solver import pywraplp

# 引入你的模块（使用包内相对导入，避免路径问题）
from .controller import GPUController
from .solver import RouterSolver
from .predictor import LatencyPredictor

logger = logging.getLogger("LPRouter")

class LPRouter:
    def __init__(self, hardware_config, strategy="LP-Router"):
        """
        初始化路由调度器
        :param hardware_config: 集群配置 {0: "RTX-4090", 1: "A100", ...}
        :param strategy: 调度策略 "LP-Router" | "Random" | "Round-Robin" | "LOR"
        """
        self.hardware_map = hardware_config
        self.strategy = strategy
        
        # --- 核心组件初始化 ---
        if self.strategy == "LP-Router":
            self.solver = RouterSolver()
            self.predictor = LatencyPredictor()
            logger.info("LP-Router mode initialized with Solver and Predictor.")
        else:
            logger.info(f"Router initialized in Baseline mode: {self.strategy}")

        # --- 基线策略状态变量 ---
        self.rr_index = 0 # 轮询指针
        # LOR (Least Outstanding Requests) 计数器
        # 在真实系统中这里应该对接 vLLM 的实时队列长度，仿真时我们用累计分配数模拟
        self.lor_counts = {gid: 0 for gid in hardware_config.keys()}

    def schedule(self, requests: List[Dict], slo_ms: float = 200.0) -> Dict:
        """
        调度入口：根据策略分发到不同的处理逻辑
        """
        if not requests:
            return {}

        if self.strategy == "Random":
            return self._solve_random(requests)
        elif self.strategy == "Round-Robin":
            return self._solve_rr(requests)
        elif self.strategy == "LOR":
            return self._solve_lor(requests)
        else:
            # 默认走我们的核心算法
            return self._solve_smart(requests, slo_ms)

    def _solve_smart(self, requests, slo_ms):
        """
        核心算法：预测 -> 构建矩阵 -> ILP求解
        """
        start_time = time.time()
        
        req_ids = [r["id"] for r in requests]
        gpu_ids = list(self.hardware_map.keys())
        num_reqs = len(requests)
        num_gpus = len(gpu_ids)
        
        # 1. 准备数据矩阵
        cost_energy = np.zeros((num_reqs, num_gpus))
        constraint_latency = np.zeros((num_reqs, num_gpus))
        
        for i, req in enumerate(requests):
            model_name = req.get("model", "qwen3-8b") # 获取模型名
            
            for j, gid in enumerate(gpu_ids):
                gpu_name = self.hardware_map[gid]
                
                # 调用预测器
                # predict 返回 (latency_ms, energy_joules)
                lat_ms, eng_j = self.predictor.predict(req, gpu_name, model_name)
                
                constraint_latency[i, j] = lat_ms
                cost_energy[i, j] = eng_j

        # 2. 调用 Solver 求解
        # 注意：这里调用的是 src/solver.py 中封装好的 solve 方法
        assignments = self.solver.solve(req_ids, gpu_ids, cost_energy, constraint_latency, slo_ms)
        
        # 3. 容错处理：如果 ILP 求解失败 (比如超时或无解)，降级为 LOR
        if assignments is None:
            logger.warning(f"ILP Solver failed for batch {req_ids[0]}... Fallback to LOR.")
            return self._solve_lor(requests)

        # 4. 结果包装
        final_decision = {}
        for req_idx, gpu_idx in assignments.items():
            req_id = req_ids[req_idx]
            gpu_id = gpu_ids[gpu_idx]
            
            # 更新 LOR 计数 (保持状态同步)
            self.lor_counts[gpu_id] += 1
            
            final_decision[req_id] = {
                "gpu_id": gpu_id,
                "gpu_model": self.hardware_map[gpu_id],
                "expected_latency": constraint_latency[req_idx, gpu_idx],
                "strategy": "ILP_Energy_Optimized"
            }

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"[LP-Router] Batch scheduled in {elapsed:.2f}ms")
        return final_decision

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

    def _solve_random(self, requests):
        final_decision = {}
        gpu_ids = list(self.hardware_map.keys())
        for req in requests:
            gid = random.choice(gpu_ids)
            self.lor_counts[gid] += 1
            final_decision[req["id"]] = {
                "gpu_id": gid, 
                "gpu_model": self.hardware_map[gid],
                # 基线策略不做预测，这里用 None 占位，后续由 logger 兼容处理
                "expected_latency": None,
                "strategy": "Random"
            }
        return final_decision

    def _solve_rr(self, requests):
        final_decision = {}
        gpu_ids = list(self.hardware_map.keys())
        num_gpus = len(gpu_ids)
        for req in requests:
            gid = gpu_ids[self.rr_index % num_gpus]
            self.rr_index += 1
            self.lor_counts[gid] += 1
            final_decision[req["id"]] = {
                "gpu_id": gid,
                "gpu_model": self.hardware_map[gid],
                # 基线策略不做预测，这里用 None 占位，后续由 logger 兼容处理
                "expected_latency": None,
                "strategy": "Round-Robin"
            }
        return final_decision

    def _solve_lor(self, requests):
        """
        Least Outstanding Requests (模拟)
        选择当前分配数最少的节点
        """
        final_decision = {}
        for req in requests:
            # 找到计数最小的 GPU ID
            best_gid = min(self.lor_counts, key=self.lor_counts.get)
            
            self.lor_counts[best_gid] += 1
            final_decision[req["id"]] = {
                "gpu_id": best_gid,
                "gpu_model": self.hardware_map[best_gid],
                # 基线策略不做预测，这里用 None 占位，后续由 logger 兼容处理
                "expected_latency": None,
                "strategy": "LOR"
            }
        return final_decision

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