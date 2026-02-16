"""中文注释覆盖检查脚本。"""  # 定义模块用途

import argparse  # 导入参数解析模块
import json  # 导入JSON模块
from pathlib import Path  # 导入路径工具
from typing import Dict, List  # 导入类型注解


def _is_executable_line(line: str) -> bool:  # 定义可执行行判断函数
    """判断当前行是否需要注释检查。"""  # 说明函数用途
    stripped = line.strip()  # 去除两端空白字符
    if not stripped:  # 判断是否为空行
        return False  # 空行不参与检查
    if stripped.startswith("#"):  # 判断是否为纯注释行
        return False  # 注释行不参与检查
    if stripped.startswith("import ") or stripped.startswith("from "):  # 判断是否为导入行
        return False  # 导入行不参与检查
    if stripped in {"(", ")", "[", "]", "{", "}"}:  # 判断是否为纯括号换行
        return False  # 纯括号换行不参与检查
    if all(char in "[]{}()," for char in stripped):  # 判断是否为仅含括号逗号的结构行
        return False  # 结构行不参与逐行注释检查
    if stripped.startswith('"""') or stripped.startswith("'''"):  # 判断是否为文档字符串起始
        return False  # 文档字符串不参与逐行注释检查
    return True  # 其余行默认参与检查


def _has_inline_comment(line: str) -> bool:  # 定义行内注释判断函数
    """判断当前行是否包含注释符。"""  # 说明函数用途
    return "#" in line  # 通过注释符存在性判断


def check_comment_coverage(root: str) -> Dict[str, List[str]]:  # 定义注释覆盖检查函数
    """检查指定目录下Python文件的注释覆盖情况。"""  # 说明函数用途
    root_path = Path(root)  # 构造根目录路径对象
    missing: List[str] = []  # 初始化缺失列表
    for py_file in root_path.rglob("*.py"):  # 递归遍历Python文件
        with py_file.open("r", encoding="utf-8") as handle:  # 以UTF-8读取文件
            for idx, line in enumerate(handle, start=1):  # 遍历文件每一行
                if not _is_executable_line(line):  # 判断是否需要检查
                    continue  # 跳过无需检查行
                if _has_inline_comment(line):  # 判断是否已有注释
                    continue  # 已注释则跳过
                missing.append(f"{py_file}:{idx}")  # 记录缺失注释位置
    return {"missing_comment_lines": missing}  # 返回检查结果


def _build_parser() -> argparse.ArgumentParser:  # 定义参数解析器构建函数
    """创建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="check chinese inline comments for executable lines")  # 创建解析器
    parser.add_argument("--root", default="code", help="root directory to scan")  # 添加扫描根目录参数
    parser.add_argument("--strict", action="store_true", help="return non-zero when missing lines exist")  # 添加严格模式参数
    return parser  # 返回解析器


def main() -> int:  # 定义主函数
    """执行命令行入口。"""  # 说明函数用途
    parser = _build_parser()  # 构建参数解析器
    args = parser.parse_args()  # 解析命令行参数
    result = check_comment_coverage(root=args.root)  # 执行注释覆盖检查
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 打印检查结果
    if args.strict and result["missing_comment_lines"]:  # 判断严格模式且存在缺失
        return 1  # 返回失败状态码
    return 0  # 返回成功状态码


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出
