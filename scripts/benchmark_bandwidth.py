import torch
import time
import numpy as np

def benchmark_bandwidth_pro(size_gb=2, num_iters=100):
    print(f"--- [Pro] Memory Bandwidth Benchmark (Size: {size_gb} GB) ---")
    
    # 1. 准备数据 (2GB float32)
    # 确保显存足够，不够就自动降级
    try:
        ele_num = int(size_gb * 1024**3 / 4)
        a = torch.ones(ele_num, device='cuda', dtype=torch.float32)
        b = torch.zeros(ele_num, device='cuda', dtype=torch.float32)
    except torch.cuda.OutOfMemoryError:
        print("警告: 显存不足，尝试降低测试规模到 512MB...")
        size_gb = 0.5
        ele_num = int(size_gb * 1024**3 / 4)
        a = torch.ones(ele_num, device='cuda', dtype=torch.float32)
        b = torch.zeros(ele_num, device='cuda', dtype=torch.float32)

    print(f"Testing with Tensor size: {size_gb} GB")
    
    # 2. 预热 (Warmup)
    # 让 GPU 频率拉升到 P0 状态
    print("Warmup (10 iters)...")
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()

    # 3. 精确测量 (Per-iteration Measurement)
    # 使用 torch.cuda.Event 进行 GPU 侧计时，消除 Python 开销
    times_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Running {num_iters} iterations with cuda.Event timing...")
    
    for i in range(num_iters):
        start_event.record()
        b.copy_(a) # Device-to-Device Copy
        end_event.record()
        
        # 等待该次操作完成
        end_event.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event) # 返回毫秒
        times_ms.append(elapsed_time_ms)

    # 4. 统计分析
    times_ms = np.array(times_ms)
    # 每次拷贝涉及 读+写，所以数据量是 2 * size
    total_bytes_per_iter = 2 * size_gb * 1024**3 
    
    # 转换为 GB/s
    # Bandwidth = (Bytes / 1024^3) / (ms / 1000)
    bandwidths = (total_bytes_per_iter / 1024**3) / (times_ms / 1000)
    
    mean_bw = np.mean(bandwidths)
    std_bw = np.std(bandwidths)
    median_bw = np.median(bandwidths)
    max_bw = np.max(bandwidths)
    min_bw = np.min(bandwidths)
    
    print("\n--- Results ---")
    print(f"Mean Bandwidth:   {mean_bw:.2f} GB/s")
    print(f"Median Bandwidth: {median_bw:.2f} GB/s")
    print(f"Std Deviation:    {std_bw:.2f} GB/s ({(std_bw/mean_bw)*100:.2f}%)")
    print(f"Min / Max:        {min_bw:.2f} / {max_bw:.2f} GB/s")
    
    theoretical_peak = 1008.0
    print(f"\nTheoretical Peak: {theoretical_peak} GB/s")
    print(f"Efficiency:       {mean_bw/theoretical_peak*100:.1f}%")
    
    # 导师点评
    if std_bw > 50:
        print("\n⚠️ 波动过大！检查是否有其他进程抢占显存或功耗受限。")
    elif mean_bw < 800:
        print("\n⚠️ 带宽异常偏低！检查 PCIe 槽位或是否开启了 ECC。")
    else:
        print("\n✅ 数据稳健。")

if __name__ == "__main__":
    benchmark_bandwidth_pro()