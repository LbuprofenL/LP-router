docker run \
  --rm \
  --gpus all \
  -it \
  --name vllm-data-collection \
  --entrypoint /bin/bash \
  -v /data/lxl/models/roofline:/models \
  -v /data/lxl/projects/roofline/experiment_data_collector.py:/app/experiment.py \
  -v /data/lxl/projects/roofline/results:/app/results \
  vllm/vllm-openai:v0.8.5.post1

# 临时
docker run -it --rm --gpus all -v /data/lxl/models/roofline:/models --entrypoint /bin/bash vllm/vllm-openai:v0.8.5.post1


# 在容器的bash里
cd /app

python3 experiment.py \
  --gpu RTX-4090 \
  --model-path /models/opt-1.3b \
  --num-requests 16 \
  --max-tokens 128 \
  --output-csv /app/results/performance_data.csv

# qwen
python3 experiment.py \
  --gpu RTX-4090 \
  --model-path /models/qwen-8b \
  --num-requests 16 \
  --max-tokens 128 \
  --output-csv /app/results/performance_data_qwen.csv

# 绘图命令
## opt-1.3b
python plot_from_csv.py \
  --csv-file ./results/final_performance_data.csv \
  --gpu RTX-4090 \
  --output-image 4090_performance_analysis_opt_v2.png
## qwen3
python plot_from_csv.py \
  --csv-file ./results/performance_data_qwen3.csv \
  --gpu RTX-4090 \
  --output-image 4090_performance_analysis_qwen3_v2.png

# run all 
