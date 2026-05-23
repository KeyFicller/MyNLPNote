#!/bin/bash

# 激活Python虚拟环境的脚本
# 使用方式：source activate_env.sh

echo "正在激活Python虚拟环境..."
source venv/bin/activate
echo "✅ 虚拟环境已激活！"
echo "当前Python路径：$(which python)"
echo "Python版本：$(python --version)"
echo ""
echo "提示：输入 'deactivate' 可退出虚拟环境"
