"""干跑验证脚本：测试 search 模块的完整 import 链和 file_io 生命周期。"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efnas.search import file_io
from efnas.search import prompts

print("=" * 50)
print("1. 模块导入验证")
print("=" * 50)
print(f"  file_io OK, HISTORY_COLUMNS = {file_io.HISTORY_COLUMNS}")
print(f"  prompts OK, 6 套提示词长度:")
print(f"    AGENT_A: {len(prompts.AGENT_A_SYSTEM)} chars")
print(f"    AGENT_B: {len(prompts.AGENT_B_SYSTEM)} chars")
print(f"    AGENT_C: {len(prompts.AGENT_C_SYSTEM)} chars")
print(f"    AGENT_D1: {len(prompts.AGENT_D1_SYSTEM)} chars")
print(f"    AGENT_D2: {len(prompts.AGENT_D2_SYSTEM)} chars")
print(f"    AGENT_D3: {len(prompts.AGENT_D3_SYSTEM)} chars")

print()
print("=" * 50)
print("2. 实验目录初始化验证")
print("=" * 50)
tmp_root = os.path.join(tempfile.gettempdir(), "efnas_dryrun_test")
exp_dir = file_io.init_experiment_dir(tmp_root, "dryrun")
print(f"  exp_dir: {exp_dir}")

print("  目录结构:")
for root, dirs, files in os.walk(exp_dir):
    level = root.replace(exp_dir, "").count(os.sep)
    indent = "    " + "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        print(f"{indent}  {f}")

print()
print("=" * 50)
print("3. 文件 I/O 生命周期验证")
print("=" * 50)

# 读取初始状态
df = file_io.read_history(exp_dir)
print(f"  history columns: {list(df.columns)}")
print(f"  history rows: {len(df)}")

assumptions = file_io.read_assumptions(exp_dir)
print(f"  assumptions: {assumptions}")

findings = file_io.read_findings(exp_dir)
print(f"  findings len: {len(findings)} chars")

strategy = file_io.read_strategy_log(exp_dir)
print(f"  strategy_log len: {len(strategy)} chars")

# 猜想增删
file_io.append_assumptions(exp_dir, [{"id": "A01", "description": "test assumption"}])
a2 = file_io.read_assumptions(exp_dir)
print(f"  after append: {a2}")

file_io.remove_assumption_by_id(exp_dir, "A01")
a3 = file_io.read_assumptions(exp_dir)
print(f"  after remove: {a3}")

nid = file_io.get_next_assumption_id(exp_dir)
print(f"  next_id: {nid}")

# Map-Reduce 模拟
file_io.write_worker_result(exp_dir, "0,1,2,0,0,1,2,1,0", {
    "arch_code": "0,1,2,0,0,1,2,1,0",
    "epe": 1.23,
    "fps": 30.5,
    "sram_kb": 128.0,
    "cycles_npu": 50000,
    "macs": 100000,
    "micro_insight": "test insight",
    "epoch": 0,
    "timestamp": "2026-03-04T12:00:00",
})
committed = file_io.collect_and_commit_worker_results(exp_dir)
print(f"  committed from tmp_workers: {committed}")

df2 = file_io.read_history(exp_dir)
print(f"  history after commit: {len(df2)} rows")
print(f"  data:\n{df2.to_string()}")

# 断点恢复 (应该没有残留)
rescued = file_io.rescue_orphaned_results(exp_dir)
print(f"  rescue (should be 0): {rescued}")

# 去重验证
evaluated = file_io.get_evaluated_arch_codes(exp_dir)
print(f"  evaluated set: {evaluated}")

# 战术日志追加
file_io.append_strategy_log(exp_dir, epoch=0, reflection_text="测试反思：全部正常")
log = file_io.read_strategy_log(exp_dir)
print(f"  strategy_log after append: {len(log)} chars")

# Findings 覆写
file_io.write_findings(exp_dir, "# Findings\n\n- 测试规则\n")
f2 = file_io.read_findings(exp_dir)
print(f"  findings after write: {len(f2)} chars")

# 验证脚本写入
script_path = file_io.write_verification_script(exp_dir, "eval_test.py", "print('hello')")
print(f"  script saved: {script_path}")
print(f"  script exists: {os.path.exists(script_path)}")

# 清理
shutil.rmtree(tmp_root)
print()
print("=" * 50)
print("4. 清理完毕，所有验证通过!")
print("=" * 50)
