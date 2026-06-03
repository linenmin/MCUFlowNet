#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ch5 Table 5.2 (controlled block-family comparison).

Builds the VALIDATED L1 backbone (ablation_edgeflownet: bilinear decoder +
bottleneck ECA + /4 global broadcast gate, with the EdgeFlowNet 7x7/5x5 head
structure) and swaps ONLY the residual block family:
  ResNet, MBConv (E=4), MBConv (E=6), ShuffleNet
Each full backbone -> INT8 TFLite -> Vela at 156x208. The four rows differ only
in the block type; the decoder, attention modules and heads are held fixed.

Block implementations are ported verbatim from the already-validated
MultiScaleResNet_cell_33decoder model. We override ABlationEdgeFlowNetV1._res_block
so every encoder/decoder block instance is swapped, leaving the rest of the
validated graph (heads, ECA, gate, bilinear upsamplers) untouched.

Validation gate: the ResNet variant must reproduce the known ResNet+bilinear
backbone (~2.63 G MACs, 1170 KiB at 156x208), confirming the head structure and
ECA/gate insertion are correct.

Run in a conda env with TensorFlow + the Vela compiler:
  python make_thesis_ch5_table_blocks.py
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()

SRAMTEST = os.path.dirname(os.path.abspath(__file__))
EFNAS = str(Path(__file__).resolve().parents[2] / "EdgeFlowNAS")
for p in (SRAMTEST, EFNAS):
    if p not in sys.path:
        sys.path.insert(0, p)

from efnas.network.ablation_edgeflownet import ABlationEdgeFlowNetV1
from efnas.vela.vela_compiler import run_vela

H, W = 156, 208
OUT_DIR = os.path.join(SRAMTEST, "_ch5_table_tmp")
os.makedirs(OUT_DIR, exist_ok=True)

FULL_VARIANT = {"name": "full", "upsample_mode": "bilinear", "bottleneck_eca": True, "gate_4x": True}


class BlockSwapAblation(ABlationEdgeFlowNetV1):
    """Validated L1 backbone with the residual block family made swappable."""

    def __init__(self, *args, block_type="resblock", mbconv_expand=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_type = block_type
        self.mbconv_expand = int(mbconv_expand)

    def _res_block(self, inputs, filters, name):
        if self.block_type == "resblock":
            return super()._res_block(inputs, filters, name)
        if self.block_type == "mbconv":
            return self._mbconv_block(inputs, filters, name)
        if self.block_type == "shufflenet":
            return self._shufflenet_block(inputs, filters, name)
        raise ValueError(self.block_type)

    def _mbconv_block(self, inputs, filters_out, name):
        in_c = inputs.get_shape().as_list()[-1]
        hidden_c = int(in_c * self.mbconv_expand)
        with tf.compat.v1.variable_scope(name):
            out = tf.compat.v1.layers.conv2d(inputs, hidden_c, 1, 1, padding="same", use_bias=False)
            out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
            out = tf.nn.relu6(out)
            with tf.compat.v1.variable_scope("dw"):
                dw = tf.compat.v1.get_variable("dw_filter", [3, 3, hidden_c, 1],
                                               initializer=tf.compat.v1.initializers.glorot_uniform())
            out = tf.nn.depthwise_conv2d(out, dw, [1, 1, 1, 1], padding="SAME")
            out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
            out = tf.nn.relu6(out)
            out = tf.compat.v1.layers.conv2d(out, filters_out, 1, 1, padding="same", use_bias=False)
            out = tf.compat.v1.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5)
            if in_c == filters_out:
                out = tf.add(out, inputs)
            return out

    def _channel_shuffle(self, x, groups=2):
        c = x.get_shape().as_list()[-1]
        cpg = c // groups
        perm = np.zeros(c, dtype=np.int32)
        for i in range(c):
            perm[i] = (i % groups) * cpg + (i // groups)
        kw = np.zeros((1, 1, c, c), dtype=np.float32)
        for i in range(c):
            kw[0, 0, perm[i], i] = 1.0
        return tf.nn.conv2d(x, tf.constant(kw), strides=[1, 1, 1, 1], padding="VALID")

    def _shufflenet_block(self, inputs, out_c, name):
        with tf.compat.v1.variable_scope(name):
            mid_c = out_c // 2
            x1, x2 = tf.split(inputs, num_or_size_splits=2, axis=3)
            x2 = tf.compat.v1.layers.conv2d(x2, mid_c, 1, 1, padding="same", use_bias=False)
            x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
            x2 = tf.nn.relu(x2)
            with tf.compat.v1.variable_scope("dw"):
                dw = tf.compat.v1.get_variable("dw_filter", [3, 3, mid_c, 1],
                                               initializer=tf.compat.v1.initializers.glorot_uniform())
            x2 = tf.nn.depthwise_conv2d(x2, dw, [1, 1, 1, 1], padding="SAME")
            x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
            x2 = tf.compat.v1.layers.conv2d(x2, mid_c, 1, 1, padding="same", use_bias=False)
            x2 = tf.compat.v1.layers.batch_normalization(x2, momentum=0.9, epsilon=1e-5)
            x2 = tf.nn.relu(x2)
            out = tf.concat([x1, x2], axis=3)
            return self._channel_shuffle(out, 2)


def accum_preds(preds):
    acc = preds[0]
    for p in preds[1:]:
        acc = tf.compat.v1.image.resize_bilinear(acc, [p.shape.as_list()[1], p.shape.as_list()[2]],
                                                 align_corners=False, half_pixel_centers=False)
        acc = acc + p
    return acc


CONFIGS = [
    {"name": "ResNet",     "block": "resblock",   "expand": 6},
    {"name": "MBConv_E4",  "block": "mbconv",     "expand": 4},
    {"name": "MBConv_E6",  "block": "mbconv",     "expand": 6},
    {"name": "ShuffleNet", "block": "shufflenet", "expand": 6},
]


def export_tflite(cfg):
    g = tf.Graph()
    with g.as_default():
        sess = tf.compat.v1.Session(graph=g)
        inp = tf.compat.v1.placeholder(tf.float32, [1, H, W, 6], name="input")
        is_tr = tf.compat.v1.placeholder_with_default(False, shape=[], name="IsTraining")
        model = BlockSwapAblation(input_ph=inp, is_training_ph=is_tr, num_out=4,
                                  variant_config=FULL_VARIANT, init_neurons=32,
                                  expansion_factor=2.0, num_sub_blocks=2,
                                  block_type=cfg["block"], mbconv_expand=cfg["expand"])
        outs = model.build()
        final = accum_preds(outs)[..., 0:2]
        sess.run(tf.compat.v1.global_variables_initializer())
        conv = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [inp], [final])
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        conv.representative_dataset = lambda: (([np.random.rand(1, H, W, 6).astype(np.float32)]) for _ in range(5))
        tfl = conv.convert()
        sess.close()
    path = os.path.join(OUT_DIR, f"{cfg['name']}_{H}x{W}.tflite")
    with open(path, "wb") as f:
        f.write(tfl)
    return path


def main():
    rows = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        tfl = export_tflite(cfg)
        vela_dir = os.path.join(OUT_DIR, f"{cfg['name']}_vela")
        os.makedirs(vela_dir, exist_ok=True)
        sram_mb, time_ms = run_vela(tfl, mode="basic", output_dir=vela_dir, optimise="Size", silent=True)
        model_name = os.path.splitext(os.path.basename(tfl))[0]
        csv_path = os.path.join(vela_dir, f"{model_name}_summary_Grove_Sys_Config.csv")
        sram_kib = macs_g = None
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            sram_kib = float(df["sram_memory_used"].values[0])
            macs_g = float(df["nn_macs"].values[0]) / 1e9
        rows.append({"name": cfg["name"], "inf_ms": time_ms, "sram_kib": sram_kib, "macs_g": macs_g})
        print(f"  inf={time_ms} ms  sram={sram_kib} KiB  macs={macs_g} G")

    print("\n\n==== Ch5 Table 5.2 (full backbone, block-only swap, 156x208) ====")
    print(f"{'Block':<14}{'Inf/ms':>9}{'SRAM/KiB':>10}{'MACs/G':>9}")
    for r in rows:
        print(f"{r['name']:<14}{r['inf_ms']:>9.1f}{r['sram_kib']:>10.0f}{r['macs_g']:>9.2f}")


if __name__ == "__main__":
    main()
