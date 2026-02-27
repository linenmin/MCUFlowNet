"""分层导入规则检查脚本。"""  # 定义模块用途

import argparse  # 导入参数解析模块
import ast  # 导入抽象语法树模块
import json  # 导入JSON模块
from pathlib import Path  # 导入路径工具
from typing import Dict, List  # 导入类型注解

import yaml  # 导入YAML解析模块


def _load_rules(path_like: str) -> List[Dict]:  # 定义规则加载函数
    """加载分层规则配置。"""  # 说明函数用途
    with Path(path_like).open("r", encoding="utf-8") as handle:  # 以UTF-8打开规则文件
        payload = yaml.safe_load(handle)  # 解析YAML内容
    return list(payload.get("rules", []))  # 返回规则列表


def _collect_imports(py_file: Path) -> List[str]:  # 定义导入收集函数
    """收集Python文件中的导入模块名。"""  # 说明函数用途
    with py_file.open("r", encoding="utf-8") as handle:  # 以UTF-8读取文件内容
        tree = ast.parse(handle.read(), filename=str(py_file))  # 解析语法树
    imports: List[str] = []  # 初始化导入列表
    for node in ast.walk(tree):  # 遍历语法树节点
        if isinstance(node, ast.Import):  # 判断是否为import语句
            for alias in node.names:  # 遍历导入别名
                imports.append(alias.name)  # 记录导入模块名
        elif isinstance(node, ast.ImportFrom):  # 判断是否为from-import语句
            if node.module is None:  # 判断模块名是否缺失
                continue  # 跳过相对空模块导入
            imports.append(node.module)  # 记录from模块名
    return imports  # 返回导入列表


def check_layering(root: str, rules_path: str) -> Dict[str, List[str]]:  # 定义分层检查函数
    """按规则检查跨层违规导入。"""  # 说明函数用途
    root_path = Path(root)  # 构造根目录路径对象
    rules = _load_rules(rules_path)  # 加载规则列表
    violations: List[str] = []  # 初始化违规列表
    for py_file in root_path.rglob("*.py"):  # 递归遍历Python文件
        rel_path = py_file.as_posix()  # 获取POSIX风格文件路径
        imports = _collect_imports(py_file)  # 收集当前文件导入
        for rule in rules:  # 遍历每条分层规则
            source = str(rule.get("source", ""))  # 读取规则源路径
            forbidden_prefixes = list(rule.get("forbidden_prefixes", []))  # 读取禁用前缀列表
            if not rel_path.startswith(source + "/"):  # 判断文件是否属于当前源层
                continue  # 不属于当前源层则跳过
            for import_name in imports:  # 遍历当前文件导入模块
                for forbidden in forbidden_prefixes:  # 遍历禁用前缀
                    if import_name.startswith(forbidden):  # 判断是否命中禁用前缀
                        violations.append(f"{rel_path} imports {import_name} forbidden_by {forbidden}")  # 记录违规项
    return {"violations": violations}  # 返回检查结果


def _build_parser() -> argparse.ArgumentParser:  # 定义参数解析器构建函数
    """创建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="check layering import rules")  # 创建解析器
    parser.add_argument("--root", default="code", help="root path to scan")  # 添加扫描根目录参数
    parser.add_argument("--rules", default="configs/layering_rules.yaml", help="yaml rules file path")  # 添加规则文件参数
    return parser  # 返回解析器


def main() -> int:  # 定义主函数
    """执行命令行入口。"""  # 说明函数用途
    parser = _build_parser()  # 构建参数解析器
    args = parser.parse_args()  # 解析命令行参数
    result = check_layering(root=args.root, rules_path=args.rules)  # 执行分层检查
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 打印检查结果
    if result["violations"]:  # 判断是否存在违规项
        return 1  # 返回失败状态码
    return 0  # 返回成功状态码


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出

