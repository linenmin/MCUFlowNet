"""Evaluate one fixed inherited-weight V3 subnet and export search-worker artifacts."""

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from efnas.app.train_supernet_app import _load_yaml, _merge_overrides
from efnas.data.dataloader_builder import build_fc2_provider
from efnas.data.prefetch_provider import PrefetchBatchProvider
from efnas.engine.checkpoint_manager import build_checkpoint_paths
from efnas.engine.eval_step import accumulate_predictions
from efnas.engine.supernet_trainer import _resolve_output_dir
from efnas.engine.supernet_v3_evaluator import setup_supernet_v3_eval_model
from efnas.nas.arch_codec_v3 import decode_arch_code
from efnas.nas.search_space_v3 import validate_arch_code
from efnas.nas.supernet_subnet_distribution import _run_vela_for_arch
from efnas.nas.supernet_v2_rank_consistency import _evaluate_fc2_one_arch, _option_or_default, _run_bn_recalibration
from efnas.network.base_layers import BaseLayers
from efnas.utils.path_utils import ensure_directory


def _put_override(overrides: Dict[str, Any], key: str, value: Any) -> None:
    """Set one nested override only when value is provided."""
    if value is not None:
        overrides[key] = value


def _to_arch_text(arch_code: Sequence[int]) -> str:
    """Convert arch code to canonical comma-separated text."""
    return ",".join(str(int(item)) for item in arch_code)


def _parse_arch_text(raw: str) -> List[int]:
    """Parse one comma-separated V3 arch code string."""
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    arch_code = [int(token) for token in tokens]
    validate_arch_code(arch_code)
    return arch_code


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write dict rows to CSV with stable headers."""
    headers = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not headers:
            return
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Map selected CLI args onto config overrides."""
    overrides: Dict[str, Any] = {}
    _put_override(overrides, "runtime.experiment_name", args.experiment_name)
    _put_override(overrides, "runtime.seed", args.seed)
    _put_override(overrides, "data.base_path", args.base_path)
    _put_override(overrides, "data.train_dir", args.train_dir)
    _put_override(overrides, "data.val_dir", args.val_dir)
    _put_override(overrides, "train.batch_size", args.train_batch_size)
    if args.num_workers is not None:
        _put_override(overrides, "data.fc2_num_workers", int(args.num_workers))
        _put_override(overrides, "data.fc2_eval_num_workers", int(args.num_workers))
    if args.prefetch_batches is not None:
        _put_override(overrides, "data.prefetch_batches", int(args.prefetch_batches))
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    """Build module CLI parser."""
    parser = argparse.ArgumentParser(description="evaluate one fixed inherited-weight V3 subnet")
    parser.add_argument("--config", default="configs/supernet_v3_fc2_172x224.yaml", help="path to supernet V3 config yaml")
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best", help="checkpoint type")
    parser.add_argument("--fixed_arch", required=True, help="fixed V3 arch code")
    parser.add_argument("--experiment_dir", default=None, help="trained V3 supernet experiment folder containing checkpoints/")
    parser.add_argument("--output_tag", default=None, help="suffix tag for output folder")
    parser.add_argument("--output_dir", default=None, help="override output directory")

    parser.add_argument("--bn_recal_batches", type=int, default=None, help="override bn recalibration batches")
    parser.add_argument("--eval_batches_per_arch", type=int, default=None, help="deprecated and ignored")
    parser.add_argument("--max_fc2_val_samples", type=int, default=None, help="optional cap for quick FC2 pilot runs")
    parser.add_argument("--batch_size", type=int, default=None, help="FC2 evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="FC2 train/eval loader worker count")
    parser.add_argument("--prefetch_batches", type=int, default=None, help="bounded batch prefetch depth")
    parser.add_argument("--cpu_only", action="store_true", help="force CPU-only eval")

    parser.add_argument("--enable_vela", action="store_true", help="enable vela benchmark")
    parser.add_argument("--vela_mode", choices=["basic", "verbose"], default="basic", help="vela mode")
    parser.add_argument("--vela_optimise", choices=["Performance", "Size"], default="Size", help="vela optimise mode")
    parser.add_argument("--vela_limit", type=int, default=None, help="accepted for compatibility; currently unused")
    parser.add_argument("--vela_rep_dataset_samples", type=int, default=3, help="rep-dataset samples for int8 export")
    parser.add_argument("--vela_float32", action="store_true", help="export float32 tflite")
    parser.add_argument("--vela_keep_artifacts", action="store_true", help="keep vela temp folders")
    parser.add_argument("--vela_verbose_log", action="store_true", help="show vela detailed logs")

    parser.add_argument("--experiment_name", default=None, help="override runtime.experiment_name")
    parser.add_argument("--base_path", default=None, help="override data.base_path")
    parser.add_argument("--train_dir", default=None, help="override data.train_dir")
    parser.add_argument("--val_dir", default=None, help="override data.val_dir")
    parser.add_argument("--train_batch_size", type=int, default=None, help="override train.batch_size")
    parser.add_argument("--seed", type=int, default=None, help="override runtime.seed")
    return parser


def _resolve_run_output_dir(config: Dict[str, Any], output_dir_text: Optional[str], output_tag: Optional[str]) -> Path:
    """Resolve one run output directory."""
    if output_dir_text:
        return Path(str(output_dir_text)).resolve()
    output_root = _resolve_output_dir(config)
    folder_name = str(output_tag or f"fixed_arch_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return (output_root / "subnet_distribution_v3" / folder_name).resolve()


def _resolve_experiment_dir(config: Dict[str, Any], experiment_dir_text: Optional[str]) -> Path:
    """Resolve trained supernet experiment directory."""
    if experiment_dir_text:
        return Path(str(experiment_dir_text)).resolve()
    return _resolve_output_dir(config).resolve()


def _wrap_prefetch(provider: Any, prefetch_batches: int) -> Any:
    """Wrap a provider with bounded prefetch when requested."""
    if int(prefetch_batches) <= 0:
        return provider
    return PrefetchBatchProvider(provider, prefetch_batches=int(prefetch_batches))


def _close_provider(provider: Any) -> None:
    """Close provider if it exposes a close method."""
    if hasattr(provider, "close"):
        provider.close()


class _FixedSubnetForExportV3(BaseLayers):
    """Build a hard-routed V3 subnet graph for TFLite/Vela export only."""

    def __init__(self, input_ph, is_training_ph, arch_code: Sequence[int], num_out=4, init_neurons=32, expansion_factor=2.0):
        super().__init__(is_training_ph=is_training_ph)
        validate_arch_code([int(item) for item in arch_code])
        self.input_ph = input_ph
        self.arch_code = [int(item) for item in arch_code]
        self.num_out = int(num_out)
        self.init_neurons = int(init_neurons)
        self.expansion_factor = float(expansion_factor)
        self._preds = None

    def _res_block(self, inputs, filters, name):
        import tensorflow as tf

        with tf.compat.v1.variable_scope(name):
            net = self.conv_bn_relu(inputs=inputs, filters=filters, strides=(1, 1), name="conv1")
            net = self.conv(inputs=net, filters=filters, strides=(1, 1), activation=None, name="conv2")
            net = self.bn(inputs=net, name="bn2")
            net = tf.add(net, inputs, name="res_add")
            return self.relu(inputs=net, name="res_relu")

    def _deep_choice_block(self, inputs, filters, choice_idx: int, name: str):
        choice = int(choice_idx)
        if choice not in (0, 1, 2):
            raise ValueError(f"invalid deep choice: {choice}")
        import tensorflow as tf

        with tf.compat.v1.variable_scope(name):
            out1 = self._res_block(inputs=inputs, filters=filters, name="branch1_block1")

            out2 = self._res_block(inputs=inputs, filters=filters, name="branch2_block1")
            out2 = self._res_block(inputs=out2, filters=filters, name="branch2_block2")

            out3 = self._res_block(inputs=inputs, filters=filters, name="branch3_block1")
            out3 = self._res_block(inputs=out3, filters=filters, name="branch3_block2")
            out3 = self._res_block(inputs=out3, filters=filters, name="branch3_block3")
            return [out1, out2, out3][choice]

    def _conv_bn_relu_dilated(self, inputs, filters, kernel_size, dilation_rate, name):
        import tensorflow as tf

        with tf.compat.v1.variable_scope(name):
            net = tf.compat.v1.layers.conv2d(
                inputs=inputs,
                filters=int(filters),
                kernel_size=kernel_size,
                strides=(1, 1),
                dilation_rate=dilation_rate,
                padding="same",
                activation=None,
                use_bias=False,
                name="conv",
            )
            net = self.bn(inputs=net, name="bn")
            return self.relu(inputs=net, name="relu")

    def _eca_block(self, inputs, kernel_size: int = 3, name: str = "eca"):
        import tensorflow as tf

        channels = inputs.get_shape().as_list()[-1]
        if channels is None:
            raise ValueError("ECA block requires static channel dimension")
        pooled = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True, name=f"{name}_mean")
        pooled_1d = tf.reshape(pooled, [-1, 1, channels, 1], name=f"{name}_reshape_to_1d")
        attn = tf.compat.v1.layers.conv2d(
            inputs=pooled_1d,
            filters=1,
            kernel_size=(1, kernel_size),
            strides=(1, 1),
            padding="same",
            activation=None,
            name=f"{name}_conv1d",
        )
        attn = tf.reshape(attn, [-1, 1, 1, channels], name=f"{name}_reshape_back")
        attn = tf.math.sigmoid(attn, name=f"{name}_sigmoid")
        return tf.multiply(inputs, attn, name=f"{name}_scale")

    def _global_broadcast_gate(self, context_inputs, target_inputs, target_filters: int, name: str):
        import tensorflow as tf

        context = tf.reduce_mean(context_inputs, axis=[1, 2], keepdims=True, name=f"{name}_mean")
        gate = self.conv(inputs=context, filters=int(target_filters), kernel_size=(1, 1), activation=None, name=f"{name}_proj")
        gate = tf.math.sigmoid(gate, name=f"{name}_sigmoid")
        return tf.multiply(target_inputs, gate, name=f"{name}_scale")

    def _e0_choice_block(self, inputs, filters, choice_idx: int, name: str):
        choice = int(choice_idx)
        import tensorflow as tf

        with tf.compat.v1.variable_scope(name):
            conv3 = self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(2, 2), name="k3")
            conv5 = self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(5, 5), strides=(2, 2), name="k5")
            conv7 = self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(7, 7), strides=(2, 2), name="k7")
            return [conv3, conv5, conv7][choice]

    def _e1_choice_block(self, inputs, filters, choice_idx: int, name: str):
        choice = int(choice_idx)
        import tensorflow as tf

        with tf.compat.v1.variable_scope(name):
            conv3 = self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(2, 2), name="k3")
            conv5 = self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(5, 5), strides=(2, 2), name="k5")
            net = self.conv_bn_relu(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(2, 2), name="k3_down")
            dilated = self._conv_bn_relu_dilated(inputs=net, filters=filters, kernel_size=(3, 3), dilation_rate=(2, 2), name="k3_dilated")
            return [conv3, conv5, dilated][choice]

    def _head_choice_conv(self, inputs, filters, choice_idx: int, name: str):
        choice = int(choice_idx)
        import tensorflow as tf

        with tf.compat.v1.variable_scope(name):
            conv3 = self.conv(inputs=inputs, filters=filters, kernel_size=(3, 3), activation=None, name="k3")
            conv5 = self.conv(inputs=inputs, filters=filters, kernel_size=(5, 5), activation=None, name="k5")
            return [conv3, conv5][choice]

    def _head_choice_resize_conv(self, inputs, filters, choice_idx: int, name: str):
        choice = int(choice_idx)
        import tensorflow as tf

        with tf.compat.v1.variable_scope(name):
            conv3 = self.resize_conv(inputs=inputs, filters=filters, kernel_size=(3, 3), name="k3")
            conv5 = self.resize_conv(inputs=inputs, filters=filters, kernel_size=(5, 5), name="k5")
            return [conv3, conv5][choice]

    def build(self):
        """Build fixed V3 subnet graph with checkpoint-compatible scopes."""
        if self._preds is not None:
            return self._preds

        arch = self.arch_code
        c1 = int(self.init_neurons * self.expansion_factor)
        c2 = int(c1 * self.expansion_factor)
        c3 = int(c2 * self.expansion_factor)
        c4 = int(c3 / self.expansion_factor)
        c5 = int(c4 / self.expansion_factor)
        h1_filters = max(1, int(c5 / self.expansion_factor))
        h2_filters = max(1, int(h1_filters / self.expansion_factor))

        import tensorflow as tf

        with tf.compat.v1.variable_scope("supernet_backbone"):
            net = self._e0_choice_block(inputs=self.input_ph, filters=self.init_neurons, choice_idx=arch[0], name="E0")
            net = self._e1_choice_block(inputs=net, filters=c1, choice_idx=arch[1], name="E1")
            net = self._deep_choice_block(inputs=net, filters=c1, choice_idx=arch[2], name="EB0")

            net = self.conv(inputs=net, filters=c2, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down1Conv")
            net = self.bn(inputs=net, name="Down1BN")
            net = self.relu(inputs=net, name="Down1ReLU")
            net = self._deep_choice_block(inputs=net, filters=c2, choice_idx=arch[3], name="EB1")

            net = self.conv(inputs=net, filters=c3, kernel_size=(3, 3), strides=(2, 2), activation=None, name="Down2Conv")
            net = self.bn(inputs=net, name="Down2BN")
            net = self.relu(inputs=net, name="Down2ReLU")
            net_low = self._deep_choice_block(inputs=net, filters=c3, choice_idx=arch[4], name="DB0")
            net_low = self._eca_block(inputs=net_low, kernel_size=3, name="eca_bottleneck")
            bottleneck_context = net_low

            net_mid = self.resize_conv(inputs=net_low, filters=c4, kernel_size=(3, 3), name="Up1")
            net_mid = self.bn(inputs=net_mid, name="Up1BN")
            net_mid = self.relu(inputs=net_mid, name="Up1ReLU")
            net_mid = self._deep_choice_block(inputs=net_mid, filters=c4, choice_idx=arch[5], name="DB1")

            net_high = self.resize_conv(inputs=net_mid, filters=c5, kernel_size=(3, 3), name="Up2")
            net_high = self.bn(inputs=net_high, name="Up2BN")
            net_high = self.relu(inputs=net_high, name="Up2ReLU")
            net_high = self._global_broadcast_gate(
                context_inputs=bottleneck_context,
                target_inputs=net_high,
                target_filters=c5,
                name="global_gate_4x",
            )

        with tf.compat.v1.variable_scope("supernet_head"):
            out_1_4 = self._head_choice_conv(inputs=net_high, filters=self.num_out, choice_idx=arch[6], name="H0Out")
            h1 = self._head_choice_resize_conv(inputs=net_high, filters=h1_filters, choice_idx=arch[7], name="H1")
            h1 = self.bn(inputs=h1, name="H1BN")
            h1 = self.relu(inputs=h1, name="H1ReLU")
            out_1_2 = self._head_choice_conv(inputs=h1, filters=self.num_out, choice_idx=arch[8], name="H1Out")
            h2 = self._head_choice_resize_conv(inputs=h1, filters=h2_filters, choice_idx=arch[9], name="H2")
            h2 = self.bn(inputs=h2, name="H2BN")
            h2 = self.relu(inputs=h2, name="H2ReLU")
            out_1_1 = self._head_choice_conv(inputs=h2, filters=self.num_out, choice_idx=arch[10], name="H2Out")

        self._preds = [out_1_4, out_1_2, out_1_1]
        return self._preds


def _build_tflite_for_arch_v3(
    checkpoint_prefix: Path,
    tflite_path: Path,
    arch_code: Sequence[int],
    input_h: int,
    input_w: int,
    flow_channels: int,
    quantize_int8: bool,
    rep_dataset_samples: int,
) -> None:
    """Export one fixed-arch V3 subnet to TFLite for Vela profiling."""
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    graph = tf.Graph()
    with graph.as_default():
        tf.compat.v1.set_random_seed(2026)
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, int(input_h), int(input_w), 6], name="Input")
        is_training_const = tf.constant(False, dtype=tf.bool, name="IsTraining")
        with tf.compat.v1.variable_scope("shared_supernet"):
            model = _FixedSubnetForExportV3(
                input_ph=input_ph,
                is_training_ph=is_training_const,
                arch_code=arch_code,
                num_out=int(flow_channels) * 2,
                init_neurons=32,
                expansion_factor=2.0,
            )
            preds = model.build()
        preds_accum = accumulate_predictions(preds)
        output_tensor = preds_accum[..., 0:int(flow_channels)]
        saver = tf.compat.v1.train.Saver(max_to_keep=1)

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, str(checkpoint_prefix))
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_ph], [output_tensor])
            if quantize_int8:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                rng = np.random.default_rng(seed=2026)

                def representative_dataset_gen():
                    for _ in range(max(1, int(rep_dataset_samples))):
                        sample = rng.random((1, int(input_h), int(input_w), 6)).astype(np.float32)
                        yield [sample]

                converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)


def run_fixed_subnet_eval_v3(config_path: str, overrides: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run one fixed-arch V3 evaluation and export artifacts for the search worker."""
    config = _merge_overrides(_load_yaml(config_path), overrides)
    arch_code = _parse_arch_text(args.fixed_arch)
    arch_text = _to_arch_text(arch_code)
    run_output_dir = _resolve_run_output_dir(config=config, output_dir_text=args.output_dir, output_tag=args.output_tag)
    analysis_dir = Path(ensure_directory(str(run_output_dir / "analysis")))

    if bool(args.cpu_only):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    fc2_train_provider = build_fc2_provider(config=config, split="train", seed_offset=0, provider_mode="train")
    fc2_val_provider = build_fc2_provider(config=config, split="val", seed_offset=1, provider_mode="eval")
    prefetch_batches = int(config.get("data", {}).get("prefetch_batches", 0))
    fc2_train_provider = _wrap_prefetch(fc2_train_provider, prefetch_batches=prefetch_batches)
    fc2_val_provider = _wrap_prefetch(fc2_val_provider, prefetch_batches=prefetch_batches)

    experiment_dir = _resolve_experiment_dir(config=config, experiment_dir_text=args.experiment_dir)
    eval_model = setup_supernet_v3_eval_model(
        experiment_dir=experiment_dir,
        checkpoint_type=str(args.checkpoint_type),
        flow_channels=int(config.get("data", {}).get("flow_channels", 2)),
        allow_growth=not bool(args.cpu_only),
    )

    fc2_batch_size = int(_option_or_default(vars(args), "batch_size", config.get("train", {}).get("batch_size", 8)))
    bn_recal_batches = int(_option_or_default(vars(args), "bn_recal_batches", config.get("eval", {}).get("bn_recal_batches", 8)))
    max_fc2_val_samples = _option_or_default(vars(args), "max_fc2_val_samples", None)

    try:
        eval_model["saver"].restore(eval_model["sess"], str(eval_model["checkpoint_path"]))
        _run_bn_recalibration(
            eval_model=eval_model,
            train_provider=fc2_train_provider,
            arch_code=arch_code,
            num_batches=bn_recal_batches,
            batch_size=fc2_batch_size,
        )
        fc2_epe = _evaluate_fc2_one_arch(
            eval_model=eval_model,
            val_provider=fc2_val_provider,
            arch_code=arch_code,
            batch_size=fc2_batch_size,
            max_samples=max_fc2_val_samples,
        )
    finally:
        eval_model["sess"].close()
        _close_provider(fc2_train_provider)
        _close_provider(fc2_val_provider)

    record_rows = [
        {
            "sample_index": 0,
            "arch_code": arch_text,
            "decoded_json": json.dumps(decode_arch_code(list(arch_code)), ensure_ascii=False),
            "epe": float(fc2_epe),
            "fc2_eval_samples": int(len(fc2_val_provider) if max_fc2_val_samples is None else min(len(fc2_val_provider), int(max_fc2_val_samples))),
            "num_workers": int(config.get("data", {}).get("fc2_eval_num_workers", 1)),
            "prefetch_batches": int(prefetch_batches),
            "experiment_dir": str(experiment_dir),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
    ]
    _write_csv(analysis_dir / "records.csv", record_rows)

    vela_row: Dict[str, Any] = {
        "arch_code": arch_text,
        "sram_peak_mb": "",
        "inference_ms": "",
        "fps": "",
        "cycles_npu": "",
        "macs": "",
        "vela_status": "skipped",
        "vela_error": "",
    }
    if bool(args.enable_vela):
        data_cfg = config.get("data", {})
        vela_root = analysis_dir / "vela_tmp"
        arch_dir = vela_root / f"arch_{arch_text.replace(',', '')}"
        tflite_path = arch_dir / "model_int8.tflite"
        checkpoint_paths = build_checkpoint_paths(str(experiment_dir))
        checkpoint_key = "best" if str(args.checkpoint_type).strip().lower() == "best" else "last"
        _build_tflite_for_arch_v3(
            checkpoint_prefix=Path(str(checkpoint_paths[checkpoint_key])),
            tflite_path=tflite_path,
            arch_code=arch_code,
            input_h=int(data_cfg.get("input_height", 172)),
            input_w=int(data_cfg.get("input_width", 224)),
            flow_channels=int(data_cfg.get("flow_channels", 2)),
            quantize_int8=not bool(args.vela_float32),
            rep_dataset_samples=int(args.vela_rep_dataset_samples),
        )
        vela_metrics = _run_vela_for_arch(
            tflite_path=tflite_path,
            output_dir=arch_dir,
            vela_mode=str(args.vela_mode),
            vela_optimise=str(args.vela_optimise),
            vela_silent=not bool(args.vela_verbose_log),
        )
        vela_row.update(vela_metrics)
        if not bool(args.vela_keep_artifacts) and vela_root.exists():
            shutil.rmtree(vela_root, ignore_errors=True)
    _write_csv(analysis_dir / "vela_metrics.csv", [vela_row])

    summary = {
        "arch_code": arch_text,
        "fc2_epe": float(fc2_epe),
        "fc2_eval_samples": record_rows[0]["fc2_eval_samples"],
        "num_workers": record_rows[0]["num_workers"],
        "prefetch_batches": record_rows[0]["prefetch_batches"],
        "experiment_dir": str(experiment_dir),
        "vela_enabled": bool(args.enable_vela),
        "output_dir": str(run_output_dir),
        "analysis_dir": str(analysis_dir),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    """CLI entry."""
    parser = _build_parser()
    args = parser.parse_args()
    overrides = _build_overrides(args)
    result = run_fixed_subnet_eval_v3(config_path=str(args.config), overrides=overrides, args=args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
