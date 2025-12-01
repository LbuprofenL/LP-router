import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 硬件和模型规格配置---
GPU_SPECS = {
    "RTX-4090": {
        "name": "NVIDIA GeForce RTX 4090",
        "peak_flops_fp16": 330000,
        "mem_bandwidth_gb_s": 1008
    }
}

MODEL_SPECS = {
    "facebook-opt-1.3b": {
        "params_b": 1.3,
        "bytes_per_param": 2,
        "arch": {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 8192,
            "bytes_per_elem": 2
        }
    },
    "qwen3-8b": {
        "params_b": 8.2,
        "bytes_per_param": 2,
        "arch": {
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 22016,
            "bytes_per_elem": 2
        }
    },
}

def estimate_flops_and_bytes_decoder(num_concurrent_requests,
                                     prompt_tokens_per_request,
                                     max_output_tokens,
                                     arch):
    """
    论文级近似：算子级 FLOPs 与 HBM Bytes 估算，用于 decoder-only 模型。
    """
    B = float(num_concurrent_requests)
    S_prompt = float(prompt_tokens_per_request)
    S_decode = float(max_output_tokens)

    H = float(arch["hidden_size"])
    L = float(arch["num_hidden_layers"])
    d_ff = float(arch["intermediate_size"])
    bytes_per_elem = float(arch.get("bytes_per_elem", 2))

    # sum of context lengths
    sum_T_prefill = S_prompt * (S_prompt + 1.0) / 2.0
    sum_T_decode = S_decode * S_prompt + S_decode * (S_decode + 1.0) / 2.0
    total_tokens = S_prompt + S_decode

    flops_no_T_per_layer_per_token = 8.0 * B * H * H + 4.0 * B * H * d_ff
    flops_no_T_total = L * flops_no_T_per_layer_per_token * total_tokens
    flops_T_total = L * 4.0 * B * H * (sum_T_prefill + sum_T_decode)
    total_flops = flops_no_T_total + flops_T_total

    params_per_layer = 4.0 * H * H + 2.0 * H * d_ff
    bytes_weights_per_layer_per_token = params_per_layer * bytes_per_elem
    bytes_weights_total = L * bytes_weights_per_layer_per_token * total_tokens

    bytes_kv_prefill_per_layer = 2.0 * B * S_prompt * H * bytes_per_elem
    bytes_kv_decode_per_layer = 2.0 * B * H * bytes_per_elem * (
        S_decode +
        S_decode * S_prompt +
        S_decode * (S_decode + 1.0) / 2.0
    )
    bytes_kv_total = L * (bytes_kv_prefill_per_layer + bytes_kv_decode_per_layer)

    total_bytes = bytes_weights_total + bytes_kv_total
    return total_flops, total_bytes

def calculate_performance_point(model_spec, workload_params, total_time):
    """根据估算值和CSV中的数据计算性能点"""
    model_params = model_spec["params_b"] * 1e9
    bytes_per_param = model_spec["bytes_per_param"]
    total_generated_tokens = workload_params["num_concurrent_requests"] * workload_params["max_output_tokens"]

    if model_spec.get("arch"):
        total_flops, total_bytes_moved = estimate_flops_and_bytes_decoder(
            workload_params["num_concurrent_requests"],
            workload_params.get("prompt_tokens_per_request", 0),
            workload_params["max_output_tokens"],
            model_spec["arch"]
        )
    else:
        total_flops = 2 * model_params * total_generated_tokens
        total_bytes_moved = model_params * bytes_per_param * total_generated_tokens

    if total_time == 0 or total_bytes_moved == 0:
        return 0, 0

    gflops_per_sec = (total_flops / 1e9) / total_time
    arithmetic_intensity = total_flops / total_bytes_moved
    
    return arithmetic_intensity, gflops_per_sec

def plot_roofline(gpu_spec, performance_points, save_path):
    """
    绘制Roofline图并标出所有从CSV读取的性能点
    """
    peak_flops = gpu_spec["peak_flops_fp16"]
    mem_bandwidth = gpu_spec["mem_bandwidth_gb_s"]
    ridge_point = peak_flops / mem_bandwidth

    # max_ai_data = performance_points["ai"].max()
    # ai_limit = max(max_ai_data * 1.2, ridge_point * 1.2, 50)

    # ai_space = np.linspace(0, ai_limit, 200)
    ai_space = np.linspace(0, 100, 10)
    mem_roof = ai_space * mem_bandwidth
    compute_roof = np.full_like(ai_space, peak_flops)
    effective_roof = np.minimum(mem_roof, compute_roof)

    plt.figure(figsize=(14, 9))
    sns.set_theme(style="whitegrid", context="talk")
    
    plt.plot(
        ai_space,
        effective_roof,
        color='black',
        linewidth=3,
        label='Effective Roofline',
        zorder=1
    )
    plt.axvline(
        x=ridge_point,
        linestyle='--',
        color='gray',
        label=f'Ridge Point ({ridge_point:.2f} FLOPs/Byte)',
        zorder=1
    )
    
    # 使用seaborn绘制散点图，可以根据并发数或Token数自动着色
    sns.scatterplot(
        data=performance_points,
        x='ai',
        y='gflops',
        hue='num_concurrent_requests',  # 或 'max_output_tokens'
        size='max_output_tokens',       # 或 'num_concurrent_requests'
        style='model_name',
        palette='viridis',
        sizes=(60, 240),
        alpha=0.75,
        edgecolor='black',
        linewidth=0.4,
        zorder=5
    )
    
    # 为每个点添加标签
    for i, point in performance_points.iterrows():
        # 只给少量代表性点加标签，避免严重遮挡
        if i % 2 != 0:
            continue
        plt.annotate(
            f"{int(point['num_concurrent_requests'])}x{int(point['max_output_tokens'])}",
            (point['ai'], point['gflops']),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=9,
            alpha=0.9
        )

    plt.xlim(0, 100)
    plt.title(f'Roofline Model for {gpu_spec["name"]}', fontsize=18)
    plt.xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=14)
    plt.ylabel('Performance (GFLOP/s)', fontsize=14)
    plt.legend(
        title='Experiment Parameters',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局为图例留出空间
    
    plt.savefig(save_path, dpi=300)
    print(f"Roofline图已保存到: {save_path}")
    plt.show()

def main(args):
    # 读取CSV数据
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"错误: 找不到CSV文件 '{args.csv_file}'")
        return
    print(df['gpu_name'].unique())
    # 筛选出指定GPU的数据
    df_gpu = df[df['gpu_name'] == args.gpu].copy()
    if df_gpu.empty:
        print(f"错误: CSV文件中没有找到GPU '{args.gpu}' 的数据。")
        return

    print(f"已加载 {len(df_gpu)} 条关于 {args.gpu} 的实验数据。")
    
    performance_points = []
    for index, row in df_gpu.iterrows():
        model_name_key = row['model_name']
        if model_name_key not in MODEL_SPECS:
            print(f"警告: 在MODEL_SPECS中找不到模型 '{model_name_key}'，跳过此行。")
            continue
        
        model_spec = MODEL_SPECS[model_name_key]
        workload_params = {
            "num_concurrent_requests": row['num_concurrent_requests'],
            "max_output_tokens": row['max_output_tokens'],
            "prompt_tokens_per_request": row.get('prompt_tokens_per_request', 0)
        }
        total_time = row['total_time_seconds']
        
        ai, gflops = calculate_performance_point(model_spec, workload_params, total_time)
        
        point_data = row.to_dict()
        point_data['ai'] = ai
        point_data['gflops'] = gflops
        performance_points.append(point_data)

    if not performance_points:
        print("没有可用于绘图的数据点。")
        return

    # 转换为Pandas DataFrame方便绘图
    points_df = pd.DataFrame(performance_points)
    
    # 绘图
    gpu_spec = GPU_SPECS[args.gpu]
    plot_roofline(gpu_spec, points_df, args.output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从CSV数据生成Roofline图")
    parser.add_argument("--csv-file", type=str, required=True, help="包含原始性能数据的CSV文件路径")
    parser.add_argument("--gpu", type=str, required=True, choices=list(GPU_SPECS.keys()), help="要为其绘图的GPU型号")
    parser.add_argument("--output-image", type=str, default="roofline_analysis.png", help="输出的图表文件名")
    
    args = parser.parse_args()
    main(args)