from locust import HttpUser, task, between
import random

# 模拟 ShareGPT 的长度分布 (简化版)
DISTRIBUTION = [
    (128, 0.5),   # 50% 短任务 (4090 擅长)
    (512, 0.3),   # 30% 中等
    (2048, 0.2),  # 20% 长任务 (必须给 A100)
]

class LLMUser(HttpUser):
    wait_time = between(0.1, 0.5) # 模拟泊松到达

    @task
    def chat_request(self):
        # 随机生成 Input/Output 长度
        r = random.random()
        length_conf = 128
        if r > 0.8: length_conf = 2048
        elif r > 0.5: length_conf = 512
        
        # 构造请求
        payload = {
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "x" * length_conf}], # 模拟长度
            "max_tokens": 128, # Output 长度
            "prompt_len": length_conf # 显式传给 Router，模拟 Predictor 结果
        }
        
        with self.client.post("/v1/chat/completions", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                # 可以在这里解析 debug_router 字段，统计路由情况
                pass
            else:
                response.failure("Failed")