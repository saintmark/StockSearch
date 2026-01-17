@echo off
TITLE StockSearch System Starter
echo ==========================================
echo    StockSearch 量化系统一键启动工具
echo ==========================================
echo.

:: 启动后端服务
echo [1/2] 正在新的窗口中启动 Python 后端 API 服务...
start "StockSearch-Backend" cmd /k "cd /d %~dp0server && venv\Scripts\activate && python main.py"

:: 等待 2 秒确保后端初始化
timeout /t 2 /nobreak > nul

:: 启动前端服务
echo [2/2] 正在新的窗口中启动 React 前端开发服务器...
start "StockSearch-Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ==========================================
echo 系统启动指令已发出！
echo 后端地址: http://127.0.0.1:8000
echo 前端地址: http://localhost:5173 (请见前端窗口输出)
echo ==========================================
pause
