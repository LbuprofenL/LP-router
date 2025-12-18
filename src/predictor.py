# src/predictor.py
import json
import os
import logging

logger = logging.getLogger("Predictor")

class LatencyPredictor:
    def __init__(self, gpu_specs_path="configs/gpu_specs.json", model_specs_path="configs/model_specs.json", calibration_path="data/calibration/calibration.json"):
        # 1. 加载静态规格配置
        self._load_json(gpu_specs_path, "gpu_specs")
        self._load_json(model_specs_path, "model_specs")
        
        # 2. 加载动态标定参数 (如果还没有，就用默认值)
        # 这里的 alpha/beta 是你通过 scripts/calibrate_hardware.py 跑出来的
        if os.path.exists(calibration_path):
            with open(calibration_path, 'r') as f:
                self.calibration = json.load(f)
            logger.info(f"Loaded calibration data from {calibration_path}")
        else:
            logger.warning(f"Calibration file {calibration_path} not found. Using default fallback.")
            self.calibration = {
                "RTX-4090": {"alpha": 1.0, "beta": 0.0},
                "A100":     {"alpha": 0.8, "beta": 0.0}, # 假设 A100 更快
            }
    def _load_json(self, path, attr_name):
        try:
            with open(path, 'r') as f:
                setattr(self, attr_name, json.load(f))
        except FileNotFoundError:
            logger.error(f"Config file not found: {path}")
            setattr(self, attr_name, {})
    def _estimate_flops_and_bytes(self, arch, prompt_len, output_len, batch_size=1):
            B = float(batch_size)
            S_prompt = float(prompt_len)
            S_decode = float(output_len)

            # 提取架构参数
            H = float(arch["hidden_size"])
            L = float(arch["num_hidden_layers"])
            d_ff = float(arch["intermediate_size"])
            bytes_per_elem = float(arch.get("bytes_per_elem", 2))
            num_q_heads = float(arch.get("num_attention_heads", 32))
            num_kv_heads = float(arch.get("num_key_value_heads", num_q_heads))
            
            # 计算 derived 参数
            head_dim = H / num_q_heads
            kv_dim = head_dim * num_kv_heads

            
            # sum of context lengths
            sum_T_prefill = S_prompt * (S_prompt + 1.0) / 2.0
            # 注意：这里计算的是生成 S_decode 个 token 过程中的累积长度
            sum_T_decode = S_decode * S_prompt + S_decode * (S_decode + 1.0) / 2.0
            total_tokens = S_prompt + S_decode

            # FLOPs Calculation
            flops_no_T_per_layer_per_token = 8.0 * B * H * H + 4.0 * B * H * d_ff
            flops_no_T_total = L * flops_no_T_per_layer_per_token * total_tokens
            flops_T_total = L * 4.0 * B * H * (sum_T_prefill + sum_T_decode)
            total_flops = flops_no_T_total + flops_T_total

            # Bytes Calculation (HBM Access)
            # 1. Weights
            params_per_layer = 4.0 * H * H + 2.0 * H * d_ff
            bytes_weights_per_layer_per_token = params_per_layer * bytes_per_elem
            bytes_weights_total = L * bytes_weights_per_layer_per_token * total_tokens

            # 2. KV Cache
            bytes_kv_prefill_per_layer = 2.0 * B * S_prompt * kv_dim * bytes_per_elem
            bytes_kv_decode_per_layer = 2.0 * B * kv_dim * bytes_per_elem * (
                S_decode +
                S_decode * S_prompt +
                S_decode * (S_decode + 1.0) / 2.0
            )
            bytes_kv_total = L * (bytes_kv_prefill_per_layer + bytes_kv_decode_per_layer)

            total_bytes = bytes_weights_total + bytes_kv_total
            
            return total_flops, total_bytes

    def predict(self, req, gpu_name, model_name):
        """
        核心预测接口：Router 调用此函数
        """
        # 1. 获取模型架构和 GPU 规格
        model_spec = self.model_specs.get(model_name)
        gpu_spec = self.gpu_specs.get(gpu_name)
        
        if not model_spec or not gpu_spec:
            return 10000.0, 0.0 # 缺失数据时返回极大惩罚
            
        # 2. 从 req 对象中提取参数
        # 注意：这里解决了你的参数不一致问题
        prompt_len = req.get("prompt_len", 128)
        
        # TODO: 这里之后会接入你的 DistilBERT 预测结果
        # 现在暂时用 req 里自带的，或者默认值
        output_len = req.get("output_len", 128) 

        # 3. 调用核心估算函数 (默认 batch_size=1)
        total_flops, total_bytes = self._estimate_flops_and_bytes(
            arch=model_spec["arch"], 
            prompt_len=prompt_len, 
            output_len=output_len, 
            batch_size=1
        )

        # 4. Roofline 计算 (Latency = max(Compute, Memory))
        peak_flops = gpu_spec["peak_flops_fp16"] * 1e9
        peak_bw = gpu_spec["mem_bandwidth_gb_s"] * 1e9
        
        time_compute = total_flops / peak_flops
        time_memory = total_bytes / peak_bw
        t_theo = max(time_compute, time_memory) # 理论时间 (秒)

        # 5. 线性修正 (Alpha/Beta)
        calib = self.calibration.get(gpu_name, {"alpha": 1.0, "beta": 0.0})
        t_pred = calib["alpha"] * t_theo + calib["beta"]

        # 6. 能耗估算 (Energy = Power * Time)
        # 简单模型：使用 GPU 的 TDP 或 平均观测功率
        # TODO: 这里的 300/250 应该从 calibration.json 里读，这里为了代码跑通先写个默认值
        avg_power = 300.0 if "4090" in gpu_name else 250.0 
        energy_pred = avg_power * t_pred

        return t_pred * 1000.0, energy_pred # 返回 ms, Joules
        
    def predict_output_length(self, prompt: str) -> int:
        """
        [占位符] DistilBERT 预测逻辑
        """
        return 128