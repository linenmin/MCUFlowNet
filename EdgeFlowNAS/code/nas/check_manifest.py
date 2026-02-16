"""训练产物校验脚本。"""  # 定义模块用途
import argparse  # 导入参数解析模块
import csv  # 导入CSV模块
import json  # 导入JSON模块
from pathlib import Path  # 导入路径工具
from typing import Any, Dict, List  # 导入类型注解

from code.utils.json_io import read_json  # 导入JSON读取工具


def _required_manifest_fields() -> List[str]:  # 定义必填字段列表函数
    """返回训练清单必须包含的字段名。"""  # 说明函数用途
    return [  # 返回字段名列表
        "seed",  # 定义随机种子字段
        "config_hash",  # 定义配置哈希字段
        "git_commit",  # 定义提交哈希字段
        "input_shape",  # 定义输入尺寸字段
        "optimizer",  # 定义优化器字段
        "lr_schedule",  # 定义学习率调度字段
        "wd",  # 定义权重衰减字段
        "grad_clip",  # 定义梯度裁剪字段
    ]  # 结束字段名列表


def _check_manifest(path: Path) -> Dict[str, Any]:  # 定义manifest校验函数
    """校验manifest字段完整性。"""  # 说明函数用途
    payload = read_json(str(path))  # 读取manifest文件内容
    if not isinstance(payload, dict):  # 判断manifest是否为字典结构
        return {"ok": False, "missing": _required_manifest_fields(), "reason": "manifest_not_dict"}  # 返回结构错误结果
    missing = []  # 初始化缺失字段列表
    for field in _required_manifest_fields():  # 遍历必填字段
        if field not in payload:  # 判断字段是否存在
            missing.append(field)  # 记录缺失字段
    return {"ok": len(missing) == 0, "missing": missing, "reason": ""}  # 返回校验结果字典


def _check_fairness_counts(path: Path) -> Dict[str, Any]:  # 定义fairness校验函数
    """校验公平计数字典是否满足三选项相等。"""  # 说明函数用途
    payload = read_json(str(path))  # 读取fairness计数文件
    if not isinstance(payload, dict):  # 判断fairness文件是否为字典结构
        return {"ok": False, "block_issues": ["fairness_not_dict"]}  # 返回结构错误结果
    issues = []  # 初始化问题列表
    for block_idx in range(9):  # 遍历九个可选块
        block_key = str(block_idx)  # 计算块索引字符串
        block = payload.get(block_key, None)  # 读取块计数字典
        if not isinstance(block, dict):  # 判断块计数是否为字典结构
            issues.append(f"block_{block_key}_missing")  # 记录块缺失问题
            continue  # 跳过当前块后续校验
        values = []  # 初始化当前块计数列表
        for option in (0, 1, 2):  # 遍历三个选项
            option_key = str(option)  # 计算选项字符串键
            value = block.get(option_key, None)  # 读取选项计数值
            if value is None:  # 判断选项计数是否缺失
                issues.append(f"block_{block_key}_option_{option_key}_missing")  # 记录选项缺失问题
                values = []  # 清空计数列表避免后续比较
                break  # 退出当前块选项循环
            values.append(int(value))  # 记录选项计数值
        if values and (max(values) - min(values) != 0):  # 判断三选项计数是否严格相等
            issues.append(f"block_{block_key}_not_equal")  # 记录计数不相等问题
    return {"ok": len(issues) == 0, "block_issues": issues}  # 返回公平计数校验结果


def _check_eval_history(path: Path) -> Dict[str, Any]:  # 定义评估历史校验函数
    """校验评估历史CSV是否包含核心列。"""  # 说明函数用途
    required_headers = ["epoch", "mean_epe_12", "std_epe_12", "fairness_gap"]  # 定义必需列名
    if not path.exists():  # 判断评估历史文件是否存在
        return {"ok": False, "missing_headers": required_headers, "reason": "history_missing"}  # 返回文件缺失结果
    with path.open("r", encoding="utf-8", newline="") as handle:  # 打开CSV文件
        reader = csv.DictReader(handle)  # 创建CSV字典读取器
        headers = list(reader.fieldnames or [])  # 读取表头字段列表
        rows = list(reader)  # 读取所有数据行
    missing = []  # 初始化缺失列列表
    for header in required_headers:  # 遍历必需列名
        if header not in headers:  # 判断列名是否存在
            missing.append(header)  # 记录缺失列名
    return {"ok": len(missing) == 0 and len(rows) > 0, "missing_headers": missing, "row_count": len(rows), "reason": ""}  # 返回校验结果


def _default_artifact_paths(manifest_path: Path) -> Dict[str, Path]:  # 定义默认产物路径函数
    """根据manifest路径推断同目录产物路径。"""  # 说明函数用途
    root = manifest_path.parent  # 计算产物目录
    return {  # 返回推断路径字典
        "manifest": manifest_path,  # 返回manifest路径
        "fairness_counts": root / "fairness_counts.json",  # 返回fairness路径
        "eval_history": root / "eval_epe_history.csv",  # 返回评估历史路径
    }  # 结束路径字典


def _build_parser() -> argparse.ArgumentParser:  # 定义参数解析器构建函数
    """创建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="check EdgeFlowNAS training artifacts")  # 创建解析器对象
    parser.add_argument("--path", required=True, help="path to train_manifest.json")  # 添加manifest路径参数
    parser.add_argument("--strict", action="store_true", help="return non-zero if any check fails")  # 添加严格模式参数
    return parser  # 返回解析器对象


def main() -> int:  # 定义主函数
    """执行训练产物校验流程。"""  # 说明函数用途
    parser = _build_parser()  # 构建参数解析器
    args = parser.parse_args()  # 解析命令行参数
    manifest_path = Path(args.path)  # 构造manifest路径对象
    artifact_paths = _default_artifact_paths(manifest_path=manifest_path)  # 计算产物路径字典
    manifest_result = _check_manifest(path=artifact_paths["manifest"]) if artifact_paths["manifest"].exists() else {"ok": False, "missing": _required_manifest_fields(), "reason": "manifest_missing"}  # 执行manifest校验
    fairness_result = _check_fairness_counts(path=artifact_paths["fairness_counts"]) if artifact_paths["fairness_counts"].exists() else {"ok": False, "block_issues": ["fairness_missing"]}  # 执行fairness校验
    history_result = _check_eval_history(path=artifact_paths["eval_history"])  # 执行评估历史校验
    summary = {  # 组装汇总结果字典
        "ok": bool(manifest_result.get("ok", False) and fairness_result.get("ok", False) and history_result.get("ok", False)),  # 计算总体是否通过
        "manifest": manifest_result,  # 写入manifest校验结果
        "fairness_counts": fairness_result,  # 写入fairness校验结果
        "eval_history": history_result,  # 写入评估历史校验结果
    }  # 结束汇总字典
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # 打印结构化校验结果
    if args.strict and not summary["ok"]:  # 判断严格模式下是否失败
        return 1  # 返回失败状态码
    return 0  # 返回成功状态码


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出
