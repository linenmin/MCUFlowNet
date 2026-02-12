import argparse
import os


def run_train_local(args):
    cmd = ["python", "code/train.py"]
    cmd += ["--Dataset", args.dataset]
    cmd += ["--data_list", args.data_list]
    cmd += ["--GPUDevice", str(args.gpu_device)]
    cmd += ["--NumEpochs", str(args.num_epochs)]
    cmd += ["--MiniBatchSize", str(args.batch_size)]
    cmd += ["--LR", str(args.lr)]
    cmd += ["--network_module", args.network_module]

    if args.base_path:
        cmd += ["--BasePath", args.base_path]
    if args.load_checkpoint:
        cmd += ["--LoadCheckPoint", "1"]

    print("Running:", " ".join(cmd))
    return os.system(" ".join(cmd))


def main():
    parser = argparse.ArgumentParser(description="run train locally without docker")
    parser.add_argument("--dataset", default="FC2", help="dataset: FC2/FT3D/MSCOCO")
    parser.add_argument("--data_list", default="code/dataset_paths", help="directory containing *_train.txt and related list files")
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU index, set -1 for CPU")
    parser.add_argument("--num_epochs", type=int, default=400, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--base_path", default="", help="optional dataset root path")
    parser.add_argument("--load_checkpoint", action="store_true", help="resume from latest checkpoint in Checkpoints/")
    parser.add_argument(
        "--network_module",
        default="network.MultiScaleResNet",
        help="network module path, e.g. sramTest.network.MultiScaleResNet_bilinear",
    )
    args = parser.parse_args()

    run_train_local(args)


if __name__ == "__main__":
    main()
