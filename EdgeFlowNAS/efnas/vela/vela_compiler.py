import os  # 导入操作系统相关功能的模块
import sys  # 导入系统控制相关功能的模块
import subprocess  # 导入执行子进程命令的模块
import argparse  # 导入命令行参数解析模块
import pandas as pd  # 导入数据处理库 pandas

def run_vela(tflite_path, mode="basic", output_dir="output", optimise="Performance", silent=False):  # 定义运行 Vela 编译的核心函数
    """编译 TFLite 模型并根据模式返回结果"""  # 函数功能简述
    
    # --- 基础配置 ---  # 配置部分开始
    vela_ini = os.path.join(os.path.dirname(__file__), "vela.ini")  # 获取当前目录下 vela.ini 的绝对路径
    accelerator = "ethos-u55-64"  # 指定 NPU 加速器型号为 ethos-u55-64
    sys_config = "Grove_Sys_Config"  # 指定系统配置文件中的 Grove_Sys_Config 节
    mem_mode = "Grove_Mem_Mode"  # 指定内存模式为 Grove_Mem_Mode
    
    # --- 寻找 Vela 执行文件 ---  # 寻找 vela.exe
    python_dir = os.path.dirname(sys.executable)  # 获取当前 Python 解释器的目录
    vela_exe = os.path.join(python_dir, "Scripts", "vela.exe")  # 拼接 Windows 下默认的 vela.exe 路径
    vela_cmd = vela_exe if os.path.exists(vela_exe) else "vela"  # 如果找到绝对路径则使用，否则回退到全局命令
    
    # --- 构建命令 ---  # 命令列表构建
    cmd = [
        vela_cmd, tflite_path,  # 指定编译器和输入的 tflite 文件
        "--accelerator-config", accelerator,  # 设置加速器配置
        "--config", vela_ini,  # 指定配置文件路径
        "--system-config", sys_config,  # 设置系统配置参数
        "--memory-mode", mem_mode,  # 设置内存模式参数
        "--optimise", optimise,  # 设置优化目标 (Performance 或 Size)
        "--output-dir", output_dir,  # 指定输出目录
    ]

    # 根据用户选择的模式添加参数
    if mode == "verbose":
        cmd.append("--verbose-performance")  # 开启分层性能分析
        cmd.append("--verbose-allocation")  # 开启详细分配日志

    # --- 执行编译 ---
    try:
        if not silent:
            print(f"\n>>> 正在以 [{mode}] 模式编译: {os.path.basename(tflite_path)} | 优化策略: {optimise}")
        
        # 运行 Vela 编译并捕获所有输出
        # 注意: --supported-ops-report 会抑制标准性能输出并生成通用文档，因此不应使用该标志来查看模型细节
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding="utf-8", 
            errors="replace",
            check=True
        )
        
        # --- 处理详细输出报告 (verbose 模式) ---
        if mode == "verbose":
            perf_txt_path = os.path.join(output_dir, "detailed_performance.txt")
            with open(perf_txt_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)
            print(f"[*] 详细性能报告已保存至: {perf_txt_path}")
        
        # --- 处理算子支持情况 (ops 模式) ---
        # ops 模式下，直接打印 Vela 的标准输出，其中包含算子回退警告和摘要
        if mode == "ops":
             print(result.stdout)

        # --- 始终解析基础数据 (SRAM/Time) ---
        model_name = os.path.splitext(os.path.basename(tflite_path))[0]
        csv_path = os.path.join(output_dir, f"{model_name}_summary_{sys_config}.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            sram_mb = df['sram_memory_used'].values[0] / 1024.0
            time_ms = df['inference_time'].values[0] * 1000.0
            return sram_mb, time_ms
        return None, None
        
    except Exception as e:
        print(f"编译失败: {e}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vela 编译器包装脚本 - 指定 Grove Vision AI V2 配置")
    parser.add_argument("model", help="输入的 .tflite 模型路径")
    parser.add_argument("-m", "--mode", choices=["basic", "ops", "verbose"], default="basic", 
                        help="选择输出模式: basic(基础汇总), ops(含算子报告), verbose(含详细性能分析)")
    parser.add_argument("--optimise", choices=["Performance", "Size"], default="Performance",
                        help="Vela 优化策略: Performance(性能优先，SRAM略高), Size(SRAM优先，稍慢)")
    
    args = parser.parse_args()
    
    sram, time = run_vela(args.model, mode=args.mode, optimise=args.optimise)
    
    if sram is not None:
        print(f"汇总结果: SRAM占用 = {sram:.2f} MB, 预估时间 = {time:.2f} ms")
