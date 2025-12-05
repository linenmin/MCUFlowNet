import argparse  # 解析命令行参数
import csv  # 写入 CSV
from pathlib import Path  # 路径处理
import torch  # 张量与设备
from tqdm import tqdm  # 进度条

# 项目路径设定
CURRENT_DIR = Path(__file__).resolve().parent  # 当前文件夹
ROOT_DIR = CURRENT_DIR.parent  # MCUFlowNet 根目录
NASLIB_ROOT = ROOT_DIR / "NASLib"  # NASLib 根目录

# 确保路径可见
import sys  # sys 路径
sys.path.insert(0, str(ROOT_DIR))  # 项目根加入路径
sys.path.insert(0, str(NASLIB_ROOT))  # NASLib 加入路径

# 复用工具与 proxy 计算
from proxy_TransNAS.utils.load_model import (  # 加载工具函数
    load_transbench_classes,  # 动态加载搜索空间类
    load_transbench_api,  # 加载并缓存 TransNASBenchAPI
    make_train_loader,  # 构造 DataLoader
    truncate_loader,  # 截断批次
    set_op_indices_from_str,  # 从架构串写入 op_indices
)  # 结束导入
from proxy_TransNAS.proxies.flops import compute_flops  # FLOPs 计算


def parse_args():  # 解析命令行
    parser = argparse.ArgumentParser(description="导出指定任务/搜索空间的 FLOPs 查询表")  # 说明
    parser.add_argument("--task", choices=["autoencoder", "segmentsemantic", "normal"], default="autoencoder",
                        help="任务名称")  # 任务
    parser.add_argument("--search_space", choices=["micro", "macro"], default="micro",
                        help="搜索空间")  # 搜索空间
    parser.add_argument("--data_root", type=str, default=str(NASLIB_ROOT / "naslib" / "data"),
                        help="数据根路径（含 transnas-bench_v10141024.pth 和 taskonomy 数据）")  # 数据根
    parser.add_argument("--batch_size", type=int, default=1, help="DataLoader 的 batch 大小")  # batch 大小
    parser.add_argument("--maxbatch", type=int, default=1, help="截断的 batch 数（越小越省显存）")  # 截断批数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")  # 随机种子
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")  # 设备
    parser.add_argument("--out_dir", type=str, default=str(CURRENT_DIR / "flops_lookup"),
                        help="输出目录（会自动创建）")  # 输出目录
    parser.add_argument("--out_name", type=str, default=None,
                        help="输出文件名，不含后缀。默认自动生成")  # 输出名
    return parser.parse_args()  # 返回参数


def main():  # 主流程
    args = parse_args()  # 解析参数
    torch.manual_seed(args.seed)  # 设定种子
    # 加载类与 API
    TransMicro, TransMacro, graph_module = load_transbench_classes()  # 搜索空间类
    Metric = graph_module.Metric  # 指标枚举
    data_root = Path(args.data_root).resolve()  # 数据根
    dataset_api = load_transbench_api(data_root, args.task)  # API 字典
    api = dataset_api["api"]  # 取 API 对象
    # 选择搜索空间实例
    if args.search_space == "micro":  # 微搜索空间
        if args.task == "segmentsemantic":  # 分割任务需要 n_classes
            ss = TransMicro(dataset=args.task, create_graph=True, n_classes=17)  # 创建微搜索空间
        else:
            ss = TransMicro(dataset=args.task, create_graph=True)  # 创建微搜索空间
        arch_list = api.all_arch_dict["micro"]  # 获取所有架构串
    else:  # 宏搜索空间
        ss = TransMacro(dataset=args.task, create_graph=True)  # 创建宏搜索空间
        arch_list = api.all_arch_dict["macro"]  # 获取所有架构串
    # 构建 DataLoader，仅取前 maxbatch 个 batch 供 FLOPs 输入
    train_loader = make_train_loader(args.task, data_root, args.batch_size, args.seed)  # 构造 loader
    train_batches = truncate_loader(train_loader, args.maxbatch)  # 截断批次
    # 输出目录与文件名
    out_dir = Path(args.out_dir)  # 输出目录
    out_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
    if args.out_name is None:  # 自动命名
        out_name = f"flops_{args.search_space}_{args.task}.csv"  # 默认文件名
    else:
        out_name = f"{args.out_name}.csv"  # 自定义名
    out_path = out_dir / out_name  # 完整路径
    # 写 CSV 表头
    with open(out_path, "w", newline="") as f:  # 打开文件
        writer = csv.writer(f)  # CSV 写入器
        writer.writerow(["arch_str", "flops", "gt"])  # 表头
    # 遍历所有架构
    rows = []  # 临时缓存
    pbar = tqdm(arch_list, desc="计算 FLOPs", unit="arch")  # 进度条
    for arch_str in pbar:  # 遍历架构串
        graph = ss.clone()  # 克隆搜索空间
        graph = set_op_indices_from_str(args.search_space, graph, arch_str)  # 写入 op_indices
        graph.parse()  # 实例化
        model = graph.to(args.device)  # 模型上设备
        # 计算 FLOPs
        flops = compute_flops(model, train_batches, args.device)  # 计算 FLOPs
        # 计算 GT
        metric_name = "valid_ssim" if args.task in ["autoencoder", "normal"] else "valid_acc" if args.task == "segmentsemantic" else "valid_top1"  # 指标名
        gt = api.get_single_metric(arch_str, args.task, metric_name, mode="final")  # 查询 GT
        rows.append((arch_str, flops, gt))  # 记录结果
        # 清理
        model.cpu()  # 移回 CPU
        del model, graph  # 删除引用
        torch.cuda.empty_cache()  # 清显存
        # 分批写盘防止内存占用过大
        if len(rows) >= 500:  # 每 500 条写一次
            with open(out_path, "a", newline="") as f:  # 追加写
                writer = csv.writer(f)  # 写入器
                writer.writerows(rows)  # 写入
            rows = []  # 清空缓存
    # 写入剩余部分
    if rows:  # 还有剩余
        with open(out_path, "a", newline="") as f:  # 追加写
            writer = csv.writer(f)  # 写入器
            writer.writerows(rows)  # 写入
    # 读取一次排序并重写（按 FLOPs 升序）
    import pandas as pd  # 导入 pandas 仅用于排序
    df = pd.read_csv(out_path)  # 读入
    df = df.sort_values(by="flops")  # 按 FLOPs 排序
    df.to_csv(out_path, index=False)  # 覆盖写回
    print(f"完成，输出: {out_path}，共 {len(df)} 条")  # 完成提示


if __name__ == "__main__":  # 脚本入口
    main()  # 运行主流程

