import argparse  # 导入命令行解析
import os  # 执行本地命令
from utils import run_docker_with_cmd, create_common_parser  # 保留原docker工具以便回退

# 原先的docker调用逻辑，保留注释方便回退
# def run_test(args):
#     test_cmd = f"cd optical && python code/train.py --Dataset {args.dataset} --data_list code/dataset_paths/"
#     run_docker_with_cmd(test_cmd, args)


def run_train_local(args):  # 本地模式运行训练
    cmd = ["python", "code/train.py"]  # 调用训练脚本
    cmd += ["--Dataset", args.dataset]  # 指定数据集名（FC2/FT3D/MSCOCO）
    cmd += ["--data_list", args.data_list]  # 指定列表文件所在目录
    cmd += ["--GPUDevice", str(args.gpu_device)]  # 选择GPU编号，-1为CPU
    cmd += ["--NumEpochs", str(args.num_epochs)]  # 训练轮数
    cmd += ["--MiniBatchSize", str(args.batch_size)]  # batch大小
    cmd += ["--LR", str(args.lr)]  # 学习率
    if args.base_path:  # 可选：数据根目录（部分代码可能使用）
        cmd += ["--BasePath", args.base_path]
    if args.load_checkpoint:  # 是否从最新checkpoint恢复
        cmd += ["--LoadCheckPoint", "1"]
    print("Running:", " ".join(cmd))  # 打印命令便于排查
    return os.system(" ".join(cmd))  # 实际执行


def main():  # 入口函数
    parser = argparse.ArgumentParser(description="run train locally without docker")  # 创建解析器
    parser.add_argument("--dataset", default="FlyingChairs2", help="选择数据集：FC2/FT3D/MSCOCO")  # 数据集名
    parser.add_argument("--data_list", default="code/dataset_paths", help="数据列表所在目录")  # 列表目录
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU编号，-1为CPU")  # GPU选择
    parser.add_argument("--num_epochs", type=int, default=400, help="训练轮数")  # 训练轮数
    parser.add_argument("--batch_size", type=int, default=16, help="batch大小")  # batch大小
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")  # 学习率
    parser.add_argument("--base_path", default="", help="数据根目录（若代码需用）")  # 数据根目录
    parser.add_argument("--load_checkpoint", action="store_true", help="从最新checkpoint恢复")  # 断点续训
    args = parser.parse_args()  # 解析参数

    run_train_local(args)  # 调用本地运行


if __name__ == '__main__':  # 脚本入口
    main()  # 执行入口
