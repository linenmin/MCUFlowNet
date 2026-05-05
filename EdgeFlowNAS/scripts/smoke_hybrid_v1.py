"""Phase 5 ablation 提交前的 HPC interactive smoke test.

跑三步:
  ① sandbox 子进程能正常启动 (验证 fb98ec5 的 env 修复)
  ② 5 个 LLM role 全部能调用 Gemini 3.1 Pro
  ③ AST 白名单能正确拒绝违规 import

任何一步炸了直接非零退出, 不要往下跑 slurm.
"""

import os
import sys
import traceback

# 让脚本无论从哪跑都能 import efnas
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def step1_sandbox():
    print("\n=== [1/3] sandbox subprocess ===")
    from efnas.search.sandbox import execute_verification

    code = (
        "import json\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "df = pd.DataFrame({'epe':[3.1,3.0,3.2], 'fps':[1.5,2.1,1.8]})\n"
        "print(json.dumps({'corr': float(np.corrcoef(df.epe, df.fps)[0,1]), 'n': len(df)}))\n"
    )
    r = execute_verification(code, timeout=30)
    print(f"  status      = {r['status']}")
    print(f"  returncode  = {r['returncode']}")
    print(f"  stdout      = {r['stdout'].strip()[:200]}")
    if r["stderr"]:
        print(f"  stderr tail = {r['stderr'][-200:]}")
    print(f"  parsed_json = {r['parsed_json']}")
    if r["status"] != "ok" or r["parsed_json"] is None:
        raise RuntimeError(f"sandbox smoke failed: status={r['status']}")


def step2_llm():
    print("\n=== [2/3] 5x LLM role connectivity (Gemini 3.1 Pro) ===")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not set in env")

    import yaml
    from efnas.search.llm_client import LLMClient

    with open(os.path.join(ROOT, "configs", "nsga2_v3.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    client = LLMClient(cfg)

    roles = [
        "warmstart_agent",
        "scientist_stage_a",
        "scientist_stage_b1",
        "scientist_stage_b2",
        "supervisor_agent",
    ]
    fails = []
    for r in roles:
        try:
            out = client.chat(
                role=r,
                system_prompt="You are a JSON-only echo bot. Reply with exactly the JSON object the user requests.",
                user_message=f'Reply with this exact JSON: {{"role":"{r}","ok":true}}',
                response_format={"type": "json_object"},
            )
            preview = str(out)[:90].replace("\n", " ")
            print(f"  [OK]   {r:25s}  {preview}")
        except Exception as exc:
            print(f"  [FAIL] {r:25s}  {type(exc).__name__}: {exc}")
            fails.append(r)
    if fails:
        raise RuntimeError(f"LLM smoke failed for roles: {fails}")


def step3_ast_whitelist():
    print("\n=== [3/3] AST whitelist rejects forbidden imports ===")
    from efnas.search.sandbox import execute_python

    bad_cases = [
        "import os; os.remove('/tmp/x')",
        "import subprocess; subprocess.run(['ls'])",
        "from socket import socket",
    ]
    for code in bad_cases:
        r = execute_python(code, timeout=5)
        if r["status"] != "validation_error":
            raise RuntimeError(f"whitelist failed to block: {code!r}  status={r['status']}")
        print(f"  [OK]   blocked: {code:60s}  ({r['error']})")


def main():
    try:
        step1_sandbox()
        step2_llm()
        step3_ast_whitelist()
    except Exception:
        print("\n!!! SMOKE FAILED !!!")
        traceback.print_exc()
        sys.exit(1)
    print("\n*** ALL SMOKE STEPS PASSED — safe to sbatch group b/c/d ***")


if __name__ == "__main__":
    main()
