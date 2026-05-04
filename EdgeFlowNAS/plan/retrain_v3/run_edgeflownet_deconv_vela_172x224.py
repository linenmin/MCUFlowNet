"""Export original EdgeFlowNet deconv model at 172x224 and run Vela.

This is a reproducible baseline script for Retrain V3 candidate selection.
It intentionally imports the original EdgeFlowNet `code/network/MultiScaleResNet.py`
rather than the modified `sramTest/network` copy.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> int:
    tf.compat.v1.disable_eager_execution()

    edge_root = Path(r"D:\Dataset\MCUFlowNet\EdgeFlowNet")
    code_dir = edge_root / "code"
    sys.path.insert(0, str(code_dir))

    from misc.utils import AccumPreds
    from network.MultiScaleResNet import MultiScaleResNet

    height, width = 172, 224
    out_dir = Path(r"D:\Dataset\MCUFlowNet\EdgeFlowNAS\outputs\vela_edgeflownet_deconv_172x224")
    tflite_path = out_dir / "edgeflownet_deconv_original_172x224_int8.tflite"
    vela_out = out_dir / "vela_size"
    checkpoint_path = edge_root / "checkpoints" / "best.ckpt"

    out_dir.mkdir(parents=True, exist_ok=True)
    if vela_out.exists():
        shutil.rmtree(vela_out)
    vela_out.mkdir(parents=True, exist_ok=True)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, height, width, 6], name="input_image")
        model = MultiScaleResNet(
            InputPH=input_ph,
            InitNeurons=32,
            ExpansionFactor=2.0,
            NumSubBlocks=2,
            NumOut=4,
            NumBlocks=1,
            Padding="same",
        )
        outputs = model.Network()
        accum_out, _ = AccumPreds(outputs)
        final_output = accum_out[..., 0:2]

        saver = tf.compat.v1.train.Saver()
        try:
            saver.restore(sess, str(checkpoint_path))
            restore_status = "restored"
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            sess.run(tf.compat.v1.global_variables_initializer())
            restore_status = f"init_only_restore_failed: {exc}"

        converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [final_output])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        def representative_dataset():
            rng = np.random.default_rng(20260504)
            for _ in range(5):
                yield [rng.uniform(0.0, 1.0, size=[1, height, width, 6]).astype(np.float32)]

        converter.representative_dataset = representative_dataset
        tflite_path.write_bytes(converter.convert())
        sess.close()

    vela_exe = Path(sys.executable).parent / "Scripts" / "vela.exe"
    if not vela_exe.exists():
        vela_exe = Path(shutil.which("vela") or "vela")
    vela_ini = edge_root / "vela" / "vela.ini"
    cmd = [
        str(vela_exe),
        str(tflite_path),
        "--accelerator-config",
        "ethos-u55-64",
        "--config",
        str(vela_ini),
        "--system-config",
        "Grove_Sys_Config",
        "--memory-mode",
        "Grove_Mem_Mode",
        "--optimise",
        "Size",
        "--output-dir",
        str(vela_out),
        "--verbose-performance",
        "--verbose-allocation",
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    (vela_out / "detailed_performance.txt").write_text(result.stdout, encoding="utf-8")

    summary_csv = vela_out / f"{tflite_path.stem}_summary_Grove_Sys_Config.csv"
    with summary_csv.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    sram_kb = float(row["sram_memory_used"])
    inference_time_s = float(row["inference_time"])
    metrics = {
        "model": "original EdgeFlowNet MultiScaleResNet deconv",
        "input_height": height,
        "input_width": width,
        "input_shape": [1, height, width, 6],
        "output": "AccumPreds(outputs)[0][...,0:2]",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_restore_status": restore_status,
        "tflite_path": str(tflite_path),
        "vela_ini": str(vela_ini),
        "vela_optimise": "Size",
        "accelerator": "ethos-u55-64",
        "system_config": "Grove_Sys_Config",
        "memory_mode": "Grove_Mem_Mode",
        "summary_csv": str(summary_csv),
        "sram_kb": sram_kb,
        "sram_mb": sram_kb / 1024.0,
        "inference_time_ms": inference_time_s * 1000.0,
        "fps": 1.0 / inference_time_s,
        "vela_command": cmd,
    }
    metrics_path = out_dir / "edgeflownet_deconv_172x224_vela_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
