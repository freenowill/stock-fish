#!/bin/bash
set -e

# ==========================================
# StockFish + MiroFish 一键部署脚本
# 完整链路: 分析 -> 种子文档 -> OASIS 模拟 -> 预测报告
#
# 模式:
#   1. Docker 部署（默认）: bash run.sh
#   2. 本地直接运行:       bash run.sh --local
#   3. Docker 跳过 MiroFish: bash run.sh --no-mirofish
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
MODE="${1:-docker}"

echo "========================================"
echo "  StockFish - A 股分析 + 股价推演"
echo "========================================"
echo ""

# ---- .env 检查 ----
if [ ! -f .env ]; then
    echo "[!] .env 文件不存在"
    echo "    请创建 .env 并填入 API Key（参考 .env.example）"
    exit 1
fi

# 导出环境变量供本地模式使用
set -a; source .env; set +a

# ==========================================
# 模式 A: 本地直接运行
# ==========================================
if [ "$MODE" = "--local" ]; then
    echo "[模式] 本地直接运行"
    echo ""

    # 清理旧进程
    lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true

    # 检查依赖
    echo "[1/3] 检查 Python 依赖..."
    python -c "import flask; import openai; import sxsc_tushare" 2>/dev/null || {
        echo "  安装依赖中..."
        pip install -r requirements.txt -q
    }
    echo "  ✓ 依赖就绪"

    echo ""
    echo "[2/3] 启动 StockFish（端口 8000）..."
    python app.py &
    STOCKFISH_PID=$!
    echo "  PID: $STOCKFISH_PID"

    # 等待启动 (Flask debug 会重启子进程, 需等待 API 可用)
    echo "  等待 Flask 就绪..."
    for i in $(seq 1 20); do
        if curl -s http://localhost:8000/api/config 2>/dev/null | grep -q backend; then
            echo "  服务就绪"
            break
        fi
        sleep 2
    done

    echo ""
    echo "[3/3] 验证..."
    curl -s http://localhost:8000/api/config | python -m json.tool

    echo ""
    echo "========================================"
    echo "  启动完成！"
    echo "  StockFish: http://localhost:8000"
    echo "  PID: $STOCKFISH_PID"
    echo ""
    echo "  停止: kill $STOCKFISH_PID"
    echo "========================================"

    # 保持前台等待
    wait $STOCKFISH_PID
    exit 0
fi

# ==========================================
# 模式 B: Docker 部署
# ==========================================
echo "[1/3] 检查 Docker 环境..."
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker 未安装，请先安装 Docker"
    echo "  或使用本地模式: bash run.sh --local"
    exit 1
fi
echo "  ✓ Docker $(docker --version | cut -d' ' -f3 | tr -d ',')"

# 测试 Docker 能否拉取镜像
echo "  测试 Docker 网络..."
# 用 docker.io 上不存在于本地的镜像测试拉取
docker pull nginx:alpine --quiet 2>/dev/null && DOCKER_OK=true || DOCKER_OK=false
docker rmi nginx:alpine 2>/dev/null || true

if [ "$DOCKER_OK" = "false" ]; then
    echo "  ⚠ Docker 无法访问镜像仓库（网络限制）"
    echo "  自动切换到本地模式..."
    echo ""
    exec bash "$0" --local
    exit 0
fi
echo "  ✓ Docker 网络正常"

echo ""
echo "[2/3] 构建镜像..."
if [ "$MODE" = "--no-mirofish" ]; then
    docker compose build stockfish
    echo "  ✓ StockFish 镜像构建完成（跳过 MiroFish）"
else
    docker compose build
    echo "  ✓ 镜像构建完成"
fi

echo ""
echo "[3/3] 启动服务..."
if [ "$MODE" = "--no-mirofish" ]; then
    docker compose up -d stockfish
else
    docker compose up -d
fi
echo ""

echo "  等待服务启动..."
for i in $(seq 1 15); do
    STOCKFISH_OK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null || echo "000")
    MIROFISH_OK="000"
    if [ "$MODE" != "--no-mirofish" ]; then
        MIROFISH_OK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/health 2>/dev/null || echo "000")
    fi
    if [ "$STOCKFISH_OK" != "000" ] && { [ "$MODE" = "--no-mirofish" ] || [ "$MIROFISH_OK" != "000" ]; }; then
        break
    fi
    sleep 2
done

echo ""
echo "========================================"
echo "  部署完成！"
echo "========================================"
echo ""
echo "  StockFish (分析+桥接): http://localhost:8000"
if [ "$MODE" != "--no-mirofish" ]; then
    echo "  MiroFish (模拟引擎):   http://localhost:3000"
    echo "  MiroFish API:          http://localhost:5001"
fi
echo ""
echo "  使用:"
echo "    curl -X POST http://localhost:8000/api/analyze \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"symbol\": \"600519\"}'"
echo ""
echo "  日志: docker compose logs -f stockfish"
echo "  停止: docker compose down"
echo "========================================"
