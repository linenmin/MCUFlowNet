"""9维架构编码解析器。"""  # 定义模块用途

import argparse  # 导入参数解析模块
import json  # 导入JSON模块
from typing import Dict, List  # 导入类型注解

DEPTH_LABELS = {0: "Deep1", 1: "Deep2", 2: "Deep3"}  # 定义深度标签映射
KERNEL_LABELS = {0: "7x7Conv", 1: "5x5Conv", 2: "3x3Conv"}  # 定义卷积核标签映射
BACKBONE_KEYS = ["EB0", "EB1", "DB0", "DB1"]  # 定义骨干块顺序
HEAD_KEYS = ["H0Out", "H1", "H1Out", "H2", "H2Out"]  # 定义头部块顺序


def validate_arch_code(arch_code: List[int]) -> None:  # 定义编码校验函数
    """校验架构编码合法性。"""  # 说明函数用途
    if len(arch_code) != 9:  # 检查编码长度是否为9
        raise ValueError("length=9 required")  # 抛出长度错误
    for idx, value in enumerate(arch_code):  # 遍历每个编码位
        if value not in (0, 1, 2):  # 检查取值是否在允许集合中
            raise ValueError(f"value must be in {{0,1,2}}, got idx={idx}, value={value}")  # 抛出取值错误


def decode_arch_code(arch_code: List[int]) -> Dict[str, Dict[str, str]]:  # 定义编码解码函数
    """将9维编码解码为可读结构。"""  # 说明函数用途
    validate_arch_code(arch_code)  # 执行编码合法性校验
    backbone = {}  # 初始化骨干块映射
    head = {}  # 初始化头部块映射
    for idx, key in enumerate(BACKBONE_KEYS):  # 遍历骨干块索引与名称
        backbone[key] = DEPTH_LABELS[arch_code[idx]]  # 写入骨干块深度标签
    for offset, key in enumerate(HEAD_KEYS):  # 遍历头部块偏移与名称
        head[key] = KERNEL_LABELS[arch_code[4 + offset]]  # 写入头部块卷积核标签
    return {"backbone": backbone, "head": head}  # 返回可读结构字典


def parse_arch_code_text(text: str) -> List[int]:  # 定义文本解析函数
    """将逗号分隔字符串解析为编码列表。"""  # 说明函数用途
    items = [item.strip() for item in text.split(",") if item.strip()]  # 拆分并清理文本项
    return [int(item) for item in items]  # 返回整数列表


def run_self_test() -> Dict[str, str]:  # 定义自测函数
    """运行最小自测并返回结果摘要。"""  # 说明函数用途
    example = [0, 1, 2, 0, 0, 1, 2, 1, 0]  # 定义示例编码
    decoded = decode_arch_code(example)  # 解码示例编码
    assert decoded["backbone"]["EB0"] == "Deep1"  # 断言EB0映射正确
    assert decoded["backbone"]["EB1"] == "Deep2"  # 断言EB1映射正确
    assert decoded["head"]["H1Out"] == "3x3Conv"  # 断言H1Out映射正确
    try:  # 尝试解析错误长度编码
        decode_arch_code([0] * 8)  # 输入8维编码触发异常
    except ValueError as exc:  # 捕获长度异常
        assert "length=9 required" in str(exc)  # 断言异常文案正确
    try:  # 尝试解析非法取值编码
        decode_arch_code([0, 1, 2, 0, 0, 1, 2, 1, 3])  # 输入非法取值触发异常
    except ValueError as exc:  # 捕获取值异常
        assert "value must be in {0,1,2}" in str(exc)  # 断言异常文案正确
    return {"status": "ok"}  # 返回自测成功结果


def _build_parser() -> argparse.ArgumentParser:  # 定义参数解析器构建函数
    """构建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="arch code parser for EdgeFlowNAS")  # 创建解析器
    parser.add_argument("--arch_code", default="", help="comma separated arch code, e.g. 0,1,2,0,0,1,2,1,0")  # 添加编码参数
    parser.add_argument("--self_test", action="store_true", help="run internal self test")  # 添加自测开关
    return parser  # 返回解析器


def main() -> int:  # 定义主函数
    """执行命令行入口。"""  # 说明函数用途
    parser = _build_parser()  # 构建参数解析器
    args = parser.parse_args()  # 解析命令行参数
    if args.self_test:  # 判断是否触发自测
        print(json.dumps(run_self_test(), ensure_ascii=False))  # 输出自测结果
        return 0  # 返回成功状态
    if not args.arch_code:  # 判断是否缺失编码参数
        parser.error("please provide --arch_code or --self_test")  # 报告参数错误
    arch_code = parse_arch_code_text(args.arch_code)  # 解析文本编码
    decoded = decode_arch_code(arch_code)  # 解码编码结构
    print(json.dumps(decoded, ensure_ascii=False, indent=2))  # 输出解码结果
    return 0  # 返回成功状态


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出

