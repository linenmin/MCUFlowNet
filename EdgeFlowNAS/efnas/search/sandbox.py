"""Phase 3.1 (search_hybrid_v1): Python 沙箱执行器.

给 Phase 3 Scientist Stage B-1 写出来的 verification code 一个隔离运行环境.
分两道防线:

1. **AST 静态导入白名单** (主防线): 在 exec 前 parse code, 拒绝任何不在白名单
   里的 import. 这是最可靠的防御 -- 攻击代码连进入子进程都进不去.

2. **subprocess 隔离 + 超时** (辅防线): 即使白名单漏过什么, subprocess 跑在
   tempdir, env 受限, 30s timeout, stdout/stderr 截断.

不是为了防恶意攻击设计 (LLM 不是对抗方); 而是为了防 Scientist agent 误产生
危险代码 (例如手滑写 ``import os; os.remove(...)`` 误删文件).
"""

import ast
import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Phase 3 决策: 沙箱 Python 库白名单
# - pandas/numpy: 数据分析必备
# - json/math/re/itertools: 标准库辅助
# - scipy/scipy.stats: 统计检验 (相关性 / t-test 等)
# - sys: 仅用于 sys.argv 读 csv 路径
ALLOWED_IMPORTS = frozenset({
    "pandas",
    "numpy",
    "json",
    "math",
    "re",
    "itertools",
    "scipy",
    "scipy.stats",
    "sys",
})


def validate_imports(code: str) -> Tuple[bool, str]:
    """AST-based import 白名单检查.

    扫描 code 里所有的 ``import X`` 和 ``from X import Y``, 拒绝任何 top-level
    模块不在 ``ALLOWED_IMPORTS`` 里的语句.

    Returns:
        (ok, error_message). ok=True 则 error_message 为空.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                top = module.split(".")[0]
                if top not in ALLOWED_IMPORTS and module not in ALLOWED_IMPORTS:
                    return False, f"disallowed import: {module}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not module:
                # `from . import foo` —— relative import, 一律拒绝
                return False, "disallowed relative import"
            top = module.split(".")[0]
            if top not in ALLOWED_IMPORTS and module not in ALLOWED_IMPORTS:
                return False, f"disallowed import from: {module}"
    return True, ""


def execute_python(
    code: str,
    *,
    args: Optional[List[str]] = None,
    timeout: int = 30,
    cwd: Optional[str] = None,
    stdout_truncate: int = 4000,
    stderr_truncate: int = 2000,
) -> Dict[str, Any]:
    """在隔离 subprocess 跑一段 Python code.

    流程:
        1. AST 验证导入白名单, 失败直接返回
        2. 写 code 到 tempfile.py
        3. subprocess 跑 ``[python, tempfile, *args]``, cwd 隔离, 限制 env
        4. timeout 超时返回特殊 status
        5. 收集 stdout/stderr, 截断超长部分
        6. 删除 tempfile (无论成败)

    Args:
        code: 要执行的 Python 源码
        args: 额外 argv (会附加在脚本路径之后, agent 可用 sys.argv[1], [2], ...)
        timeout: 最大执行时间 (秒), 超时强行 kill
        cwd: 工作目录; None 则用 tempfile 所在 tempdir (推荐)
        stdout_truncate / stderr_truncate: 输出截断长度, 防止 1MB stderr 灌爆 LLM

    Returns:
        dict 含:
            - status (str): 'ok' / 'validation_error' / 'syntax_error' /
              'nonzero_exit' / 'timeout' / 'subprocess_error'
            - returncode (int | None)
            - stdout (str): 截断后的 stdout
            - stderr (str): 截断后的 stderr
            - error (str): 仅在非 ok status 时填; 简短错误描述
    """
    args = list(args or [])

    # 防线 1: AST 验证
    ok, err = validate_imports(code)
    if not ok:
        return {
            "status": "syntax_error" if err.startswith("SyntaxError") else "validation_error",
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": err,
        }

    # 防线 2: subprocess 隔离
    tmp_dir = tempfile.mkdtemp(prefix="scientist_sandbox_")
    tmp_file = os.path.join(tmp_dir, "verification.py")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(code)

        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),  # Windows 必需
        }
        # 显式传 conda env 路径, 让子进程能找到 pandas/numpy/scipy
        for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "TMPDIR", "TEMP", "TMP"):
            if key in os.environ:
                env[key] = os.environ[key]

        try:
            result = subprocess.run(
                [sys.executable, tmp_file] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or tmp_dir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error": f"exceeded {timeout}s",
            }
        except Exception as exc:
            return {
                "status": "subprocess_error",
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error": f"{type(exc).__name__}: {exc}",
            }

        stdout = (result.stdout or "")[-stdout_truncate:]
        stderr = (result.stderr or "")[-stderr_truncate:]
        if result.returncode == 0:
            return {
                "status": "ok",
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "error": "",
            }
        return {
            "status": "nonzero_exit",
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error": f"exit code {result.returncode}",
        }
    finally:
        # 清理 tempfile (best effort)
        try:
            os.remove(tmp_file)
        except OSError:
            pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def execute_verification(
    code: str,
    *,
    args: Optional[List[str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """跑 verification code 并尝试解析 stdout 为 JSON (verification 约定).

    Wraps ``execute_python`` 然后在成功时尝试从 stdout 最后一个非空行 parse JSON,
    把它放进 result["parsed_json"]. 解析失败时 status 改为 'ok_no_json'.

    Returns:
        dict 同 ``execute_python`` 的输出, 多两个字段:
            - parsed_json (Any | None): 成功 parse 的 JSON 对象
            - status 多两种可能值: 'ok' (有 JSON), 'ok_no_json' (无 JSON 但 exit 0)
    """
    result = execute_python(code, args=args, timeout=timeout)
    if result["status"] != "ok":
        result["parsed_json"] = None
        return result

    stdout = result["stdout"]
    parsed = None
    for line in reversed(stdout.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") or line.startswith("["):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if parsed is None:
        result["status"] = "ok_no_json"
    result["parsed_json"] = parsed
    return result
