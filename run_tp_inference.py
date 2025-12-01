import time
import argparse
import asyncio
import os
import numpy as np
import matplotlib.pyplot as plt

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

# --- 1. 硬件和模型规格配置 ---
# 在这里添加你想要测试的GPU和模型
GPU_SPECS = {
    "RTX-4090": {
        "name": "NVIDIA GeForce RTX 4090",
        "peak_flops_fp16": 82600,  # 82.6 TFLOP/s in GFLOP/s
        "mem_bandwidth_gb_s": 1008   # 1008 GB/s
    },
    "A100-80GB": {
        "name": "NVIDIA A100-80GB",
        "peak_flops_fp16": 312000,  # 312 TFLOP/s in GFLOP/s for FP16 Tensor Core
        "mem_bandwidth_gb_s": 1935   # 1935 GB/s
    },
    "H100-80GB": {
        "name": "NVIDIA H100-80GB",
        "peak_flops_fp16": 989000,  # 989 TFLOP/s in GFLOP/s for FP16 Tensor Core
        "mem_bandwidth_gb_s": 3350   # 3350 GB/s
    }
}

MODEL_SPECS = {
    "opt-1.3b": {"params_b": 1.3, "bytes_per_param": 2}, # FP16 uses 2 bytes
}

# --- 2. Roofline 绘图函数 ---
def plot_roofline(gpu_spec, performance_point, save_path="roofline_plot.png"):
    """
    绘制Roofline图并标出性能点
    """
    peak_flops = gpu_spec["peak_flops_fp16"]
    mem_bandwidth = gpu_spec["mem_bandwidth_gb_s"]
    ai_point, gflops_point = performance_point

    # 计算拐点
    ridge_point = peak_flops / mem_bandwidth

    # 生成绘图数据
    ai_space = np.logspace(-1, 4, 100)
    mem_roof = ai_space * mem_bandwidth
    compute_roof = np.full_like(ai_space, peak_flops)
    effective_roof = np.minimum(mem_roof, compute_roof)

    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 绘制屋顶
    plt.plot(ai_space, effective_roof, color='black', linewidth=3, label='Effective Roofline')
    plt.axvline(x=ridge_point, linestyle='--', color='gray', label=f'Ridge Point ({ridge_point:.2f} FLOPs/Byte)')

    # 绘制性能点
    plt.scatter([ai_point], [gflops_point], color='red', s=200, marker='*', zorder=5, label='LLM Performance Point')
    
    # 标注性能点坐标
    plt.annotate(f'({ai_point:.2f}, {gflops_point:,.0f})',
                 (ai_point, gflops_point),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center',
                 fontsize=12,
                 color='red')

    # 图表格式
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Roofline Model for {gpu_spec["name"]}', fontsize=16)
    plt.xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=12)
    plt.ylabel('Performance (GFLOP/s)', fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    # 保存图表
    plt.savefig(save_path, dpi=300)
    print(f"Roofline图已保存到: {save_path}")
    plt.show()

# --- 3. 性能点计算函数 ---
def calculate_performance_point(model_spec, workload_args, total_time):
    """根据估算值计算性能点坐标"""
    # 估算总工作量
    model_params = model_spec["params_b"] * 1e9
    bytes_per_param = model_spec["bytes_per_param"]
    total_generated_tokens = workload_args.num_requests * workload_args.max_tokens

    # FLOPs ≈ 2 * Params * Total_Tokens (解码阶段估算)
    total_flops = 2 * model_params * total_generated_tokens
    
    # Bytes ≈ Params * Bytes_per_Param * Total_Tokens (极简估算)
    # 这是一个非常粗略的下限，假设主要数据移动是权重
    total_bytes_moved = model_params * bytes_per_param * total_generated_tokens

    if total_time == 0 or total_bytes_moved == 0:
        return 0, 0

    # 计算性能指标
    gflops_per_sec = (total_flops / 1e9) / total_time
    arithmetic_intensity = total_flops / total_bytes_moved
    
    print("\n--- 性能点计算结果 ---")
    print(f"估算总FLOPs: {total_flops / 1e12:.2f} TFLOPs")
    print(f"估算总数据移动量: {total_bytes_moved / 1e12:.2f} TB")
    print(f"实测吞吐量 (GFLOP/s): {gflops_per_sec:,.2f}")
    print(f"实测算术强度 (FLOPs/Byte): {arithmetic_intensity:.2f}")

    return arithmetic_intensity, gflops_per_sec

# --- 4. 主实验函数 ---
async def main(args: argparse.Namespace):
    """主实验和绘图函数"""
    # 检查硬件和模型是否在配置中
    if args.gpu not in GPU_SPECS:
        raise ValueError(f"错误: 未知的GPU型号 '{args.gpu}'. 可选项: {list(GPU_SPECS.keys())}")
    if args.model_name not in MODEL_SPECS:
        raise ValueError(f"错误: 未知的模型 '{args.model_name}'. 可选项: {list(MODEL_SPECS.keys())}")

    gpu_spec = GPU_SPECS[args.gpu]
    model_spec = MODEL_SPECS[args.model_name]
    
    print(f"--- 正在为 {gpu_spec['name']} 上的 {args.model_name} 模型运行测试 ---")
    
    # 配置并启动vLLM引擎
    engine_args = AsyncEngineArgs(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        disable_log_requests=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # 准备负载
    sampling_params = SamplingParams(max_tokens=args.max_tokens)
    prompts = ["Tell me a story about AI."] * args.num_requests
    
    # 预热
    print("--- 正在预热... ---")
    await engine.generate(prompts[0], sampling_params, "warmup_0")
    print("预热完成。")
    
    # 执行测试
    print(f"--- 正在执行并发测试 (请求数: {args.num_requests}, Token数: {args.max_tokens}) ---")
    start_time = time.time()
    
    tasks = [
        engine.generate(prompts[i], sampling_params, f"test_{i}")
        for i in range(args.num_requests)
    ]
    # 在这个简化版本中，我们不等待每个请求的详细输出，只关心总时间
    await asyncio.gather(*[task async for task in stream] for stream in tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"--- 测试完成，总耗时: {total_time:.2f} 秒 ---")
    
    # 计算性能点
    performance_point = calculate_performance_point(model_spec, args, total_time)
    
    # 绘制Roofline图
    plot_roofline(gpu_spec, performance_point, save_path=args.output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM性能测试与Roofline可视化脚本")
    
    parser.add_argument("--gpu", type=str, required=True, choices=list(GPU_SPECS.keys()),
                        help="正在测试的GPU型号")
    parser.add_argument("--model-name", type=str, required=True, choices=list(MODEL_SPECS.keys()),
                        help="正在测试的模型名称 (用于查找参数)")
    parser.add_argument("--model-path", type=str, default="/models/Qwen-7B-Chat",
                        help="挂载到容器内的模型文件路径")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="张量并行的GPU数量")
    parser.add_argument("--num-requests", type=int, default=32, help="模拟的并发请求数量")
    parser.add_argument("--max-tokens", type=int, default=256, help="每个请求生成的最大Token数")
    parser.add_argument("--output-image", type=str, default="roofline_result.png",
                        help="保存Roofline图的文件名")
                        
    args = parser.parse_args()
    # vLLM的异步引擎需要在事件循环中运行
    asyncio.run(main(args))