import argparse
import os
import sys


def run_train_local(args):
    def _arg_present(flag_name):
        full_name = f"--{flag_name}"
        return any(arg == full_name or arg.startswith(full_name + "=") for arg in sys.argv[1:])

    summary_level = args.summary_level
    summary_every = args.summary_every
    summary_flush_every = args.summary_flush_every
    prefetch_batches = args.prefetch_batches
    skip_model_profile = args.skip_model_profile

    if args.fast_mode:
        if not _arg_present('summary_level'):
            summary_level = 'scalar'
        if not _arg_present('summary_every'):
            summary_every = 100
        if not _arg_present('summary_flush_every'):
            summary_flush_every = 200
        if not _arg_present('prefetch_batches'):
            prefetch_batches = 8
        if not _arg_present('skip_model_profile'):
            skip_model_profile = 1

    cmd = ["python", "code/train.py"]
    cmd += ["--Dataset", args.dataset]
    cmd += ["--data_list", args.data_list]
    cmd += ["--GPUDevice", str(args.gpu_device)]
    cmd += ["--NumEpochs", str(args.num_epochs)]
    cmd += ["--MiniBatchSize", str(args.batch_size)]
    cmd += ["--LR", str(args.lr)]
    cmd += ["--network_module", args.network_module]
    cmd += ["--ExperimentFileName", args.experiment_name]
    cmd += ["--summary_level", summary_level]
    cmd += ["--summary_every", str(summary_every)]
    cmd += ["--summary_flush_every", str(summary_flush_every)]
    cmd += ["--prefetch_batches", str(prefetch_batches)]
    cmd += ["--skip_model_profile", str(skip_model_profile)]

    if args.fast_mode:
        cmd += ["--fast_mode"]

    if args.base_path:
        cmd += ["--BasePath", args.base_path]
    if args.load_checkpoint:
        if not args.resume_experiment_name:
            raise SystemExit("ERROR: --load_checkpoint requires --resume_experiment_name.")
        cmd += ["--LoadCheckPoint", "1"]
        cmd += ["--ResumeExperimentFileName", args.resume_experiment_name]

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
    parser.add_argument("--experiment_name", default="default", help="save experiment folder name under Checkpoints/ and Logs/")
    parser.add_argument("--resume_experiment_name", default="", help="resume source experiment folder name under Checkpoints/")
    parser.add_argument("--fast_mode", action="store_true", help="enable speed-optimized training defaults")
    parser.add_argument("--summary_level", default="full", choices=["full", "scalar"], help="TensorBoard summary verbosity")
    parser.add_argument("--summary_every", type=int, default=1, help="write summary every N global steps")
    parser.add_argument("--summary_flush_every", type=int, default=1, help="flush summary writer every N global steps")
    parser.add_argument("--prefetch_batches", type=int, default=0, help="number of prefetched batches, 0 disables prefetch")
    parser.add_argument("--skip_model_profile", type=int, default=0, choices=[0, 1], help="skip FLOPs profiling at startup")
    parser.add_argument(
        "--network_module",
        default="network.MultiScaleResNet",
        help="network module path, e.g. sramTest.network.MultiScaleResNet_bilinear",
    )
    args = parser.parse_args()

    run_train_local(args)


if __name__ == "__main__":
    main()
