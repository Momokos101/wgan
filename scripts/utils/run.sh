#!/bin/bash
# VCU GAN 项目运行脚本
# 自动激活虚拟环境并运行相应命令

# 获取脚本所在目录并切换到项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "错误: 虚拟环境不存在，正在创建..."
    python3 -m venv venv
    echo "虚拟环境创建完成，请先安装依赖："
    echo "  source venv/bin/activate"
    echo "  pip install tensorflow numpy"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 根据参数执行相应命令
case "$1" in
    test)
        echo "运行数据测试..."
        python scripts/vcu/test_vcu_data.py
        ;;
    train)
        echo "运行模型训练..."
        python scripts/vcu/train_vcu.py
        ;;
    generate)
        echo "运行序列生成..."
        python scripts/vcu/generate_vcu.py
        ;;
    generate-simple)
        echo "运行序列生成（简化版）..."
        python scripts/vcu/generate_vcu_simple.py
        ;;
    check)
        echo "检查环境..."
        python scripts/utils/check_environment.py
        ;;
    install)
        echo "安装依赖..."
        pip install tensorflow numpy
        ;;
    *)
        echo "用法: ./run.sh [test|train|generate|generate-simple|check|install]"
        echo ""
        echo "命令说明:"
        echo "  test           - 测试数据加载"
        echo "  train          - 训练模型"
        echo "  generate       - 生成测试序列（使用训练好的模型）"
        echo "  generate-simple - 生成测试序列（使用未训练模型）"
        echo "  check          - 检查环境"
        echo "  install        - 安装依赖"
        exit 1
        ;;
esac

# 退出虚拟环境
deactivate

