# PowerShell 激活 Python 虚拟环境的脚本
# 使用方式：.\activate_env.ps1

Write-Host "正在激活 Python 虚拟环境..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
Write-Host "✅ 虚拟环境已激活！" -ForegroundColor Green
Write-Host "当前 Python 路径：$(Get-Command python | Select-Object -ExpandProperty Source)"
$pythonVersion = python --version
Write-Host "Python 版本：$pythonVersion"
Write-Host ""
Write-Host "提示：输入 'deactivate' 可退出虚拟环境" -ForegroundColor Yellow
