# LP-Router

LP-Router 是一个用于 LLM 推理性能分析和路由优化的研究项目。

## 项目结构

```
LP-Router/
├── configs/                 # 硬件规格、模型参数配置
│   ├── gpu_specs.json       # GPU 规格配置（峰值 FLOPS、内存带宽等）
│   └── model_specs.json     # 模型规格配置（参数量、架构参数等）
│
├── data/                    # 数据目录
│   ├── calibration/         # 离线标定数据（Roofline CSV、功耗基准）
│   └── experiments/         # 在线实验日志（包含功耗的 CSV 数据）
│
├── figures/                 # 生成的图表文件（PNG 图片）
│
├── src/                     # 核心代码模块
│   ├── __init__.py
│   ├── monitor.py           # PowerMonitor 类（封装 pynvml）
│   ├── predictor.py         # 性能预测器（DistilBERT 或线性回归）
│   └── solver.py            # RouterSolver 类（OR-Tools 优化求解）
│
├── scripts/                 # 各种运行脚本
│   ├── calibrate_hardware.py    # 硬件标定脚本（原 experiment_data_collector.py）
│   ├── plot_from_csv.py         # 从 CSV 生成 Roofline 图
│   ├── benchmark_solver.py      # 求解器基准测试（待实现）
│   ├── run_cluster_sim.py       # 集群仿真主入口（待实现）
│   ├── run_all_experiments.sh   # 批量实验脚本
│   └── run_tp_inference.py      # Tensor Parallel 推理测试
│
├── logs/                    # 运行时文本日志
│
├── requirements.txt         # Python 依赖
├── Dockerfile               # Docker 构建文件
└── README.md                # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 硬件标定

使用 `calibrate_hardware.py` 进行硬件性能标定：

```bash
python scripts/calibrate_hardware.py \
  --gpu RTX-4090 \
  --model-path /models/qwen3-8b \
  --num-requests 16 \
  --max-tokens 128 \
  --output-csv data/experiments/performance_data.csv
```

### 3. 生成 Roofline 图

使用 `plot_from_csv.py` 从实验数据生成 Roofline 图：

```bash
python scripts/plot_from_csv.py \
  --csv-file data/experiments/performance_data.csv \
  --gpu RTX-4090 \
  --output-image figures/LP-router_analysis.png
```

### 4. 批量实验

使用 `run_all_experiments.sh` 运行批量实验：

```bash
bash scripts/run_all_experiments.sh
```

## 配置说明

### GPU 规格配置 (`configs/gpu_specs.json`)

包含 GPU 的峰值性能参数：
- `peak_flops_fp16`: FP16 峰值 FLOPS（单位：GFLOP/s）
- `mem_bandwidth_gb_s`: 内存带宽（单位：GB/s）

### 模型规格配置 (`configs/model_specs.json`)

包含模型的架构参数：
- `params_b`: 参数量（单位：B，十亿）
- `bytes_per_param`: 每个参数的字节数
- `arch`: 架构详细参数（hidden_size, num_layers 等）

## 核心模块

### PowerMonitor (`src/monitor.py`)

封装 `pynvml` 进行 GPU 功耗监控：

```python
from src.monitor import PowerMonitor

monitor = PowerMonitor()
monitor.initialize_device(device_id=0)
power = monitor.get_power_usage()  # 获取当前功耗（瓦特）
```

### PerformancePredictor (`src/predictor.py`)

性能预测器基类，支持线性回归或 DistilBERT 模型。

### RouterSolver (`src/solver.py`)

路由优化求解器，使用 OR-Tools 进行优化求解。

## 数据组织

- **离线标定数据** (`data/calibration/`): Roofline CSV、功耗基准等
- **在线实验数据** (`data/experiments/`): 包含功耗信息的实验 CSV 文件

## 注意事项

- 所有脚本默认将输出保存到相应的目录（`data/experiments/` 或 `figures/`）
- 配置文件使用 JSON 格式，便于扩展和维护
- Docker 相关脚本路径已更新，确保挂载正确的目录

## 开发计划

- [ ] 实现 `benchmark_solver.py` 求解器基准测试
- [ ] 实现 `run_cluster_sim.py` 集群仿真主入口
- [ ] 完善 `PowerMonitor` 功耗监控功能
- [ ] 实现性能预测模型（线性回归/DistilBERT）
- [ ] 实现 `RouterSolver` OR-Tools 优化求解

