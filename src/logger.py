import csv
import time
import os

class ExperimentLogger:
    def __init__(self, filename="experiment_results.csv"):
        self.filepath = os.path.join("data", "results", filename)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        # 初始化 CSV Header
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "req_id", "strategy", 
                    "prompt_len", "output_len", 
                    "assigned_gpu", "predicted_latency", "predicted_energy",
                    "slo_ms", "slo_violation"
                ])

    def log(self, req, decision, slo_ms):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)

            # 检查是否违约 (Simulation based)
            latency = decision.get("expected_latency", None)

            # 兼容基线策略：可能没有真实预测延迟
            if isinstance(latency, (int, float)):
                violation = 1 if latency > slo_ms else 0
                latency_str = f"{latency:.2f}"
            else:
                violation = 0  # 对于没有预测延迟的情况，视为未违约 / 未知
                latency_str = "NaN"

            writer.writerow([
                time.time(),
                req["id"],
                decision.get("strategy", "Unknown"),
                req["prompt_len"],
                req["output_len"],
                decision["gpu_model"],
                latency_str,
                "0.00", # TODO: 如果 Router 返回了能耗，填在这里
                slo_ms,
                violation
            ])