import time
import argparse
import asyncio
import csv
import os

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams
from transformers import AutoTokenizer

async def main(args: argparse.Namespace):
    """
    使用流式生成法分离并发场景下的prefill与decode壁钟时间：
    - Prefill壁钟时间：从开始到所有请求首次产生token的最晚时间
    - Decode壁钟时间：总耗时减去prefill壁钟时间
    """
    print("--- [1/4] 配置vLLM异步引擎 ---")
    
    engine_args = AsyncEngineArgs(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        disable_log_requests=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    print("--- [2/4] 准备实验负载 ---")
    sampling_params = SamplingParams(max_tokens=args.max_tokens)
    prompts = ["Tell me a story about AI."] * args.num_requests
    # 计算单条prompt的token数量（与推理一致的tokenizer与特殊符号设置）
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True
    )
    prompt_tokens_per_request = len(tokenizer.encode(prompts[0], add_special_tokens=True))

    print(f"模型路径: {args.model_path}")
    print(f"并发请求数: {args.num_requests}")
    print(f"最大输出Token数: {args.max_tokens}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")

    # --- [3/4] 预热 ---
    print("--- [3/4] 正在进行预热... ---")
    warmup_gen = engine.generate(prompts[0], sampling_params, "warmup_0")
    async for _ in warmup_gen:
        pass
    print("预热完成。")

    # --- [4/4] 执行并发测试（单次流式生成内分离prefill/decode） ---
    print("--- [4/4] 正在执行并发性能测试... ---")

    start_time = time.time()

    first_token_times = [None] * args.num_requests
    done_times = [None] * args.num_requests

    async def consume_and_measure(gen, idx):
        first_seen = False
        async for output in gen:
            if not first_seen:
                outs = getattr(output, "outputs", []) or []
                if len(outs) > 0:
                    first_candidate = outs[0]
                    token_ids = getattr(first_candidate, "token_ids", []) or []
                    if len(token_ids) >= 1:
                        first_seen = True
                        first_token_times[idx] = time.time()
            if getattr(output, "finished", False):
                done_times[idx] = time.time()
        if done_times[idx] is None:
            done_times[idx] = time.time()

    tasks = []
    for i in range(args.num_requests):
        request_id = f"req_{i}"
        gen = engine.generate(prompts[i], sampling_params, request_id)
        tasks.append(asyncio.create_task(consume_and_measure(gen, i)))

    await asyncio.gather(*tasks)

    end_time = time.time()

    latest_first_token_time = max(t for t in first_token_times if t is not None)
    prefill_wall_time = latest_first_token_time - start_time
    total_time = end_time - start_time
    decode_wall_time = total_time - prefill_wall_time

    print(f"\n--- 测试完成 ---")
    print(f"Prefill阶段壁钟耗时: {prefill_wall_time:.4f} 秒")
    print(f"Decode阶段壁钟耗时: {decode_wall_time:.4f} 秒")
    print(f"总壁钟耗时: {total_time:.4f} 秒")

    # --- 保存结果到CSV文件 ---
    csv_file = args.output_csv
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "gpu_name",
                "model_name",
                "tensor_parallel_size",
                "num_concurrent_requests",
                "max_output_tokens",
                "prompt_tokens_per_request",
                "prefill_wall_time_seconds",
                "decode_wall_time_seconds",
                "total_time_seconds"
            ])
        
        writer.writerow([
            args.gpu,
            os.path.basename(args.model_path),
            args.tensor_parallel_size,
            args.num_requests,
            args.max_tokens,
            prompt_tokens_per_request,
            f"{prefill_wall_time:.4f}",
            f"{decode_wall_time:.4f}",
            f"{total_time:.4f}"
        ])
    
    print(f"分别统计的性能数据已追加到: {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM性能数据收集脚本")
    
    parser.add_argument("--gpu", type=str, required=True, help="正在测试的GPU型号 (例如 RTX-4090)")
    parser.add_argument("--model-path", type=str, required=True, help="挂载到容器内的模型路径")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="张量并行的GPU数量")
    parser.add_argument("--num-requests", type=int, default=16, help="模拟的并发请求数量")
    parser.add_argument("--max-tokens", type=int, default=128, help="每个请求生成的最大Token数")
    parser.add_argument("--output-csv", type=str, default="raw_performance_data.csv", help="保存原始数据的CSV文件名")
                        
    args = parser.parse_args()
    asyncio.run(main(args))