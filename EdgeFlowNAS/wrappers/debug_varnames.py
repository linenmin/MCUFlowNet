"""
诊断脚本：对比 checkpoint 变量名 vs 新建图中的变量名。
在 HPC 的 EdgeFlowNAS 目录下运行：
  python /tmp/debug_varnames.py outputs/standalone/retrain_dual_run1/model_target/checkpoints/last.ckpt
"""
import os, sys
from pathlib import Path

# sys.path setup
project_root = Path(__file__).resolve().parent
if str(project_root) == "/tmp":
    project_root = Path(os.getcwd()).resolve()
    if project_root.name == "EdgeFlowNAS":
        project_root = project_root.parent

edgeflownet_dir = project_root / "EdgeFlowNet"
edgeflownas_dir = project_root / "EdgeFlowNAS"
for p in [str(edgeflownet_dir), str(edgeflownet_dir/"code"), str(edgeflownas_dir), str(edgeflownas_dir/"code"), str(project_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", help="Path to .ckpt (without extension)")
    args = parser.parse_args()

    # 1. 列出 checkpoint 中的变量
    print("=" * 60)
    print("CHECKPOINT VARIABLES:")
    print("=" * 60)
    reader = tf.compat.v1.train.NewCheckpointReader(args.ckpt_path)
    ckpt_vars = sorted(reader.get_variable_to_shape_map().keys())
    for v in ckpt_vars:
        print(f"  {v}  shape={reader.get_variable_to_shape_map()[v]}")
    print(f"\nTotal: {len(ckpt_vars)} variables in checkpoint\n")

    # 2. 用评估逻辑构建图
    print("=" * 60)
    print("GRAPH VARIABLES (eval build):")
    print("=" * 60)
    tf.compat.v1.reset_default_graph()

    from EdgeFlowNAS.code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 416, 1024, 6], name="input_ph")
    is_training_ph = tf.compat.v1.placeholder_with_default(False, shape=[], name="is_training_ph")
    arch_code_ph = tf.constant([0,2,1,1,0,0,1,0,1], dtype=tf.int32)

    with tf.compat.v1.variable_scope("model_target"):
        model = MultiScaleResNetSupernet(
            input_ph=input_ph,
            arch_code_ph=arch_code_ph,
            is_training_ph=is_training_ph,
            num_out=4,
        )
        preds = model.build()

    graph_vars = sorted([v.name for v in tf.compat.v1.global_variables()])
    for v in graph_vars:
        print(f"  {v}")
    print(f"\nTotal: {len(graph_vars)} variables in graph\n")

    # 3. 对比
    print("=" * 60)
    print("DIFF ANALYSIS:")
    print("=" * 60)
    ckpt_set = set(ckpt_vars)
    graph_set = set(v.rstrip(":0") for v in graph_vars)

    in_ckpt_not_graph = ckpt_set - graph_set
    in_graph_not_ckpt = graph_set - ckpt_set

    if in_ckpt_not_graph:
        print(f"\n  IN CHECKPOINT but NOT in graph ({len(in_ckpt_not_graph)}):")
        for v in sorted(in_ckpt_not_graph)[:20]:
            print(f"    {v}")
        if len(in_ckpt_not_graph) > 20:
            print(f"    ... and {len(in_ckpt_not_graph)-20} more")

    if in_graph_not_ckpt:
        print(f"\n  IN GRAPH but NOT in checkpoint ({len(in_graph_not_ckpt)}):")
        for v in sorted(in_graph_not_ckpt)[:20]:
            print(f"    {v}")
        if len(in_graph_not_ckpt) > 20:
            print(f"    ... and {len(in_graph_not_ckpt)-20} more")

    if not in_ckpt_not_graph and not in_graph_not_ckpt:
        print("  PERFECT MATCH!")

if __name__ == "__main__":
    main()
