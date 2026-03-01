@echo off
REM EdgeFlowNet 推理流水线
REM 在 conda vela 环境下执行: extract_onnx -> patch_convtranspose -> 视频光流推理
REM 用法: run_pipeline.bat [视频路径]

cd /d "%~dp0"

call conda activate vela
if errorlevel 1 (
    echo [错误] 无法激活 conda 环境 vela，请确保已安装并配置 conda
    pause
    exit /b 1
)

if "%~1"=="" (
    python run_inference_pipeline.py
) else (
    python run_inference_pipeline.py --video "%~1"
)

pause
