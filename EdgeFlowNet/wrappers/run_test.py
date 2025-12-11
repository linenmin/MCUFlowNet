import argparse  # 导入命令行解析库
import os  # 导入os用于执行命令
from utils import run_docker_with_cmd, create_common_parser  # 保留原docker工具以防回退使用


# 原先的docker调用逻辑，保留注释
# def run_test(args):
#     test_cmd = "cd optical && python code/test_sintel.py \
#                 --checkpoint checkpoints/best.ckpt \
#                 --uncertainity"
#     if args.dataset == 'sintel':
#         test_cmd += ' --data_list code/dataset_paths/MPI_Sintel_Final_train_list.txt'
#     if args.dataset == 'flyingchairs2':
#         test_cmd += ' --data_list code/dataset_paths/flyingchairs2.txt'
#     run_docker_with_cmd(test_cmd, args)


def run_test_local(args):  # 本地模式运行测试
    cmd = ["python", "code/test_sintel.py"]  # 本机直接运行测试脚本
    cmd += ["--checkpoint", args.checkpoint]  # 传入模型权重路径
    cmd += ["--gpu_device", str(args.gpu_device)]  # 指定GPU编号，-1为CPU
    if args.uncertainity:  # 是否开启不确定性输出
        cmd.append("--uncertainity")

    # 优先使用显式传入的data_list，否则按dataset选择默认
    if args.data_list:  # 如果用户传了自定义列表
        cmd += ["--data_list", args.data_list]
    else:  # 否则根据dataset选择内置列表
        if args.dataset == "sintel":  # 选择sintel默认列表
            cmd += ["--data_list", "code/dataset_paths/MPI_Sintel_Final_train_list.txt"]
        elif args.dataset == "flyingchairs2":  # 选择flyingchairs2默认列表
            cmd += ["--data_list", "code/dataset_paths/flyingchairs2.txt"]

    # 执行命令并返回退出码
    print("Running:", " ".join(cmd))  # 打印命令便于排查
    return os.system(" ".join(cmd))  # 实际运行 

def main():  # 入口函数
    parser = argparse.ArgumentParser(description="run test locally without docker")  # 创建解析器
    create_common_parser(parser)  # 复用通用参数（dataset等）
    parser.add_argument("--checkpoint", default="checkpoints/best.ckpt", help="模型权重路径")  # 权重路径
    parser.add_argument("--data_list", default=None, help="数据列表文件，可覆盖默认")  # 数据列表可自定义
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU编号，-1为CPU")  # GPU选择
    parser.add_argument("--uncertainity", action="store_true", help="开启不确定性输出")  # 不确定性开关
    args = parser.parse_args()  # 解析参数

    run_test_local(args)  # 调用本地运行函数

if __name__ == '__main__':  # 脚本入口
    main()  # 调用入口
