# 使用你之前确定可以用的、兼容驱动的vLLM官方基础镜像
FROM vllm/vllm-openai:v0.8.5.post1

# 设置工作目录
WORKDIR /app

# 复制我们下载好的所有Python依赖包
COPY python-packages ./python-packages
COPY requirements.txt .

# 在容器内进行完全离线的安装
# --no-index 确保它不去网上找
# --find-links 指向我们本地的包文件夹
RUN pip install --no-index --find-links=./python-packages -r requirements.txt

# 复制模型文件和你的脚本
# (这一步也可以在运行时通过卷挂载完成)
# COPY model_files ./model_files
# COPY roofline_experiment.py .

# 覆盖默认入口点，让我们能进入bash
ENTRYPOINT ["/bin/bash"]