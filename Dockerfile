# 使用官方 Python 3.12 镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements 文件
COPY server/requirements.txt /app/requirements.txt

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY server/ /app/

# 暴露端口
EXPOSE 8000

# 启动命令
# Railway 会自动设置 PORT 环境变量，uvicorn 会从环境变量读取
ENV PORT=8000
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port $PORT"]

