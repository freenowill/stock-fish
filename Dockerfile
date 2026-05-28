FROM python:3.11-slim

WORKDIR /app

# 基础依赖（gcc 只在需要编译扩展时安装）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 从本地复制 sxsc_tushare（私有包，无法从 PyPI 安装）
COPY sxsc_tushare /app/sxsc_tushare/
COPY sxsc_tushare.dist-info /app/sxsc_tushare.dist-info/
RUN pip install /app/sxsc_tushare/ 2>/dev/null || echo "sxsc_tushare 安装跳过"

COPY . .

EXPOSE 6000

CMD ["python", "app.py"]
