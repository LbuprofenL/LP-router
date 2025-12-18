from ortools.linear_solver import pywraplp
import logging

logger = logging.getLogger("RouterSolver")
"""
RouterSolver: 封装 OR-Tools 进行路由优化求解
"""


class RouterSolver:
    """
    路由求解器类，使用 OR-Tools 进行优化
    """
    
    def __init__(self, backend="SCIP"):
        """
        初始化求解器
        :param backend: OR-Tools 求解器后端，支持 "CBC", "SCIP", "GLOP" 等
        """
        self.backend = backend
        self.solver = None
    
    def solve(self, req_ids, gpu_ids, energy_matrix, latency_matrix, slo_ms, penalty_weight=100.0, timeout_ms=20):
        """
        Args:
            req_ids: 请求ID列表
            gpu_ids: GPU ID列表
            energy_matrix: shape [num_reqs, num_gpus] 的能耗矩阵
            latency_matrix: shape [num_reqs, num_gpus] 的延迟矩阵 (ms)
            slo_ms: 目标 SLO (ms)
        Returns:
            assignments: {req_idx: gpu_idx} 字典
        """
        # 1. 创建求解器
        solver = pywraplp.Solver.CreateSolver(self.backend)
        if not solver:
            logger.warning(f"{self.backend} backend not available, falling back to GLOP (LP only) or CBC.")
            solver = pywraplp.Solver.CreateSolver("CBC")
            if not solver: return None # 彻底失败

        # 2. 变量定义
        num_reqs = len(req_ids)
        num_gpus = len(gpu_ids)
        infinity = solver.infinity()
        
        x = {} # x[req, gpu] = 0 or 1
        slack = {} # slack[req] >= 0 (超时惩罚)

        for i in range(num_reqs):
            slack[i] = solver.NumVar(0, infinity, f"slack_{i}")
            for j in range(num_gpus):
                x[i, j] = solver.BoolVar(f"x_{i}_{j}")

        # 3. 约束
        # C1: 每个请求必须且只能分配给一个 GPU
        for i in range(num_reqs):
            solver.Add(sum(x[i, j] for j in range(num_gpus)) == 1)

        # C2: 延迟软约束 (Latency <= SLO + slack)
        for i in range(num_reqs):
            solver.Add(
                sum(x[i, j] * latency_matrix[i, j] for j in range(num_gpus)) 
                <= slo_ms + slack[i]
            )

        # 4. 目标函数: Min (Total Energy + Penalty * Total Slack)
        objective = solver.Objective()
        for i in range(num_reqs):
            objective.SetCoefficient(slack[i], penalty_weight)
            for j in range(num_gpus):
                objective.SetCoefficient(x[i, j], energy_matrix[i, j])
        objective.SetMinimization()

        # 5. 求解控制
        solver.set_time_limit(timeout_ms) # 毫秒级超时控制
        status = solver.Solve()

        # 6. 结果解析
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            assignments = {}
            for i in range(num_reqs):
                for j in range(num_gpus):
                    if x[i, j].solution_value() > 0.5:
                        assignments[i] = j
                        break
            return assignments
        else:
            return None # 求解失败，由 Router 降级处理
    
    def add_constraint(self, constraint):
        """
        添加约束条件
        
        Args:
            constraint: 约束条件
        """
        raise NotImplementedError("需要实现 add_constraint 方法")

