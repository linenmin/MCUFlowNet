"""One-off Sintel-clean fine-tune for v3_light, purely to rescue the demo
visual at deploy resolution.

NOT a benchmark. Acknowledged data leakage: trains on Sintel train **clean**
pass (1041 pairs from the same 23 scenes that the eval Sintel Final pass
uses). Different visual rendering but same geometry/motion.

Pipeline:
    - Build FixedArchModelV3 (v3_light, arch_code all zeros) at 172x224 input
    - Restore pre-FT sintel_best.ckpt from retrain_v3_ft3d_run1
    - Iterate over Sintel train clean pairs, batch_size 8, 20 epochs
    - Random h-flip + small spatial jitter only (no aggressive crop at this size)
    - 80/20 random split (seeded) for FT train / FT val
    - Adam lr=1e-5, grad clip 50
    - Save best ckpt (lowest in-FT val EPE) to
      outputs/sintel_demo_ft/v3_light_172x224/checkpoints/best.ckpt
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

EFNAS_ROOT = "/mnt/d/Dataset/MCUFlowNet/EdgeFlowNAS"
if EFNAS_ROOT not in sys.path:
    sys.path.insert(0, EFNAS_ROOT)

from efnas.network.fixed_arch_models_v3 import FixedArchModelV3  # noqa: E402
from efnas.engine.train_step import (  # noqa: E402
    add_weight_decay,
    build_multiscale_uncertainty_loss,
)
from efnas.engine.eval_step import accumulate_predictions, build_epe_metric  # noqa: E402


# --- Sintel data utilities ---------------------------------------------------
def read_flo(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        assert f.read(4) == b"PIEH"
        w = np.frombuffer(f.read(4), np.int32)[0]
        h = np.frombuffer(f.read(4), np.int32)[0]
        return np.frombuffer(f.read(2 * w * h * 4), np.float32).reshape(h, w, 2)


def load_sintel_clean_pairs(sintel_root: str) -> List[Tuple[str, str, str]]:
    base = Path(sintel_root) / "training"
    pairs: List[Tuple[str, str, str]] = []
    for scene in sorted((base / "clean").iterdir()):
        if not scene.is_dir():
            continue
        frames = sorted(scene.glob("frame_*.png"))
        for i in range(len(frames) - 1):
            f1 = frames[i]
            f2 = frames[i + 1]
            flo = base / "flow" / scene.name / f1.name.replace(".png", ".flo")
            if not flo.exists():
                continue
            pairs.append((str(f1), str(f2), str(flo)))
    return pairs


def prepare_sample(img1_path, img2_path, flo_path, in_h, in_w, flow_divisor,
                   clip_val, h_flip, jitter_x, jitter_y):
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    gt = read_flo(flo_path)
    if img1 is None or img2 is None:
        return None

    # Resize image+gt to model input size.
    img1 = cv2.resize(img1, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    # Resize GT flow: spatial resize + rescale flow magnitudes.
    src_h, src_w = gt.shape[:2]
    fx = in_w / src_w
    fy = in_h / src_h
    gt_resized = cv2.resize(gt, (in_w, in_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    gt_resized[..., 0] *= fx
    gt_resized[..., 1] *= fy
    # Divide by flow_divisor (v3 training convention), then clip.
    gt_resized = np.clip(gt_resized / flow_divisor, -clip_val, clip_val)

    # Light h-flip aug
    if h_flip:
        img1 = img1[:, ::-1, :].copy()
        img2 = img2[:, ::-1, :].copy()
        gt_resized = gt_resized[:, ::-1, :].copy()
        gt_resized[..., 0] *= -1.0  # x flow negated

    # Small spatial jitter via shift (no crop — input is already small).
    if jitter_x != 0 or jitter_y != 0:
        img1 = np.roll(img1, (jitter_y, jitter_x), axis=(0, 1))
        img2 = np.roll(img2, (jitter_y, jitter_x), axis=(0, 1))
        gt_resized = np.roll(gt_resized, (jitter_y, jitter_x), axis=(0, 1))

    inp = np.concatenate([img1, img2], axis=2).astype(np.float32)
    # Match training input range: (uint8/255)*2 - 1
    inp = (inp / 255.0) * 2.0 - 1.0
    return inp, gt_resized


def make_batch(pairs, indices, in_h, in_w, flow_divisor, clip_val, augment):
    rng = np.random.default_rng()
    inps, gts = [], []
    for idx in indices:
        p1, p2, fl = pairs[idx]
        h_flip = augment and rng.random() < 0.5
        jit_x = int(rng.integers(-4, 5)) if augment else 0
        jit_y = int(rng.integers(-4, 5)) if augment else 0
        s = prepare_sample(p1, p2, fl, in_h, in_w, flow_divisor, clip_val,
                           h_flip, jit_x, jit_y)
        if s is None:
            continue
        inps.append(s[0])
        gts.append(s[1])
    if not inps:
        return None
    return np.stack(inps, 0), np.stack(gts, 0)


# --- Graph -------------------------------------------------------------------
def build_graph(arch_code: List[int], in_h: int, in_w: int, flow_channels: int = 2):
    g = tf.Graph()
    with g.as_default():
        input_ph = tf.compat.v1.placeholder(tf.float32, [None, in_h, in_w, 6], name="input")
        label_ph = tf.compat.v1.placeholder(tf.float32, [None, in_h, in_w, flow_channels], name="label")
        lr_ph = tf.compat.v1.placeholder(tf.float32, (), name="lr")
        is_training_ph = tf.compat.v1.placeholder(tf.bool, (), name="is_training")

        with tf.compat.v1.variable_scope("v3_light"):
            model = FixedArchModelV3(
                input_ph=input_ph,
                is_training_ph=is_training_ph,
                arch_code=arch_code,
                num_out=4,
                init_neurons=32,
                expansion_factor=2.0,
            )
            preds = model.build()

        loss_terms = build_multiscale_uncertainty_loss(
            preds=preds, label_ph=label_ph, num_out=flow_channels, return_terms=True
        )
        loss = loss_terms["total"]
        fwd_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith("v3_light/")]
        loss = add_weight_decay(loss, weight_decay=0.0, trainable_vars=fwd_vars)
        epe_tensor = build_epe_metric(
            pred_tensor=accumulate_predictions(preds), label_ph=label_ph, num_out=flow_channels
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_ph)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=fwd_vars)
        grads_and_vars = [(g_, v_) for g_, v_ in grads_and_vars if g_ is not None]
        bn_updates = [op for op in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                      if op.name.startswith("v3_light/")]
        with tf.control_dependencies(bn_updates):
            grads = [g_ for g_, _ in grads_and_vars]
            vars_ = [v_ for _, v_ in grads_and_vars]
            clipped, grad_norm = tf.clip_by_global_norm(grads, clip_norm=50.0)
            train_op = optimizer.apply_gradients(zip(clipped, vars_))

        scope_global_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("v3_light/")]
        saver = tf.compat.v1.train.Saver(var_list=scope_global_vars, max_to_keep=3)
    return g, sess_fields(input_ph, label_ph, lr_ph, is_training_ph,
                          loss, train_op, grad_norm, epe_tensor, saver, fwd_vars, scope_global_vars)


def sess_fields(*args):
    keys = ("input_ph", "label_ph", "lr_ph", "is_training_ph", "loss", "train_op",
            "grad_norm", "epe", "saver", "fwd_vars", "scope_vars")
    return dict(zip(keys, args))


# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel-root", default="/mnt/g/AI_thesis/datasets/MPI-Sintel-complete")
    ap.add_argument("--init-ckpt",
                    default=f"{EFNAS_ROOT}/outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1/"
                            "model_v3_light/checkpoints/sintel_best.ckpt")
    ap.add_argument("--out-dir", default=f"{EFNAS_ROOT}/outputs/sintel_demo_ft/v3_light_172x224")
    ap.add_argument("--height", type=int, default=172)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--flow-divisor", type=float, default=12.5)
    ap.add_argument("--clip-val", type=float, default=50.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    pairs = load_sintel_clean_pairs(args.sintel_root)
    print(f"[data] {len(pairs)} Sintel clean pairs")
    rng = random.Random(args.seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx = sorted(idx[:n_val])
    train_idx = sorted(idx[n_val:])
    print(f"[data] {len(train_idx)} train / {len(val_idx)} val")

    arch_code = [0] * 11  # v3_light
    g, T = build_graph(arch_code=arch_code, in_h=args.height, in_w=args.width)

    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.compat.v1.Session(graph=g, config=cfg) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # Restore only the FORWARD-PASS vars (exclude Adam slots, which the
        # source ckpt doesn't carry).
        with g.as_default():
            init_vars = [v for v in tf.compat.v1.global_variables()
                         if v.name.startswith("v3_light/")
                         and "/Adam" not in v.op.name
                         and not v.op.name.endswith("beta1_power")
                         and not v.op.name.endswith("beta2_power")]
            init_saver = tf.compat.v1.train.Saver(var_list=init_vars)
        init_saver.restore(sess, args.init_ckpt)
        print(f"[init] restored {len(init_vars)} fwd vars from {args.init_ckpt}")

        rng_np = np.random.default_rng(args.seed)
        steps_per_epoch = max(1, len(train_idx) // args.batch_size)
        best_val_epe = float("inf")
        no_improve = 0
        history = []

        with log_path.open("w") as fh:
            fh.write("epoch,train_loss,train_epe,val_epe,best_val_epe,elapsed_s\n")
            fh.flush()
            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                rng_np.shuffle(train_idx)
                ep_loss = 0.0
                ep_epe = 0.0
                count = 0
                for step in range(steps_per_epoch):
                    batch_indices = train_idx[step * args.batch_size : (step + 1) * args.batch_size]
                    batch = make_batch(pairs, batch_indices, args.height, args.width,
                                       args.flow_divisor, args.clip_val, augment=True)
                    if batch is None:
                        continue
                    inp, gt = batch
                    _, loss_val, epe_val = sess.run(
                        [T["train_op"], T["loss"], T["epe"]],
                        feed_dict={
                            T["input_ph"]: inp,
                            T["label_ph"]: gt,
                            T["lr_ph"]: args.lr,
                            T["is_training_ph"]: True,
                        },
                    )
                    ep_loss += float(loss_val)
                    ep_epe += float(epe_val)
                    count += 1

                avg_loss = ep_loss / max(1, count)
                avg_train_epe = ep_epe / max(1, count)

                # Validation
                val_steps = max(1, len(val_idx) // args.batch_size)
                v_epe = 0.0
                v_count = 0
                for vs in range(val_steps):
                    batch_indices = val_idx[vs * args.batch_size : (vs + 1) * args.batch_size]
                    batch = make_batch(pairs, batch_indices, args.height, args.width,
                                       args.flow_divisor, args.clip_val, augment=False)
                    if batch is None:
                        continue
                    inp, gt = batch
                    epe_val = sess.run(T["epe"], feed_dict={
                        T["input_ph"]: inp,
                        T["label_ph"]: gt,
                        T["is_training_ph"]: True,  # match retrain_v3 eval convention
                    })
                    v_epe += float(epe_val)
                    v_count += 1
                val_epe = v_epe / max(1, v_count)
                elapsed = time.time() - t0
                history.append({"epoch": epoch, "train_loss": avg_loss,
                                "train_epe": avg_train_epe, "val_epe": val_epe})

                marker = ""
                if val_epe < best_val_epe - 1e-4:
                    best_val_epe = val_epe
                    no_improve = 0
                    T["saver"].save(sess, str(out_dir / "checkpoints" / "best.ckpt"),
                                    write_meta_graph=False)
                    marker = " *BEST*"
                else:
                    no_improve += 1
                T["saver"].save(sess, str(out_dir / "checkpoints" / "last.ckpt"),
                                write_meta_graph=False)

                line = (f"epoch={epoch:>3} time={elapsed:.1f}s loss={avg_loss:.4f} "
                        f"train_epe={avg_train_epe:.4f} val_epe={val_epe:.4f} "
                        f"best_val_epe={best_val_epe:.4f}{marker}")
                print(line)
                fh.write(f"{epoch},{avg_loss},{avg_train_epe},{val_epe},{best_val_epe},{elapsed}\n")
                fh.flush()

                if no_improve >= args.patience:
                    print(f"[stop] no improvement for {no_improve} epochs, early stopping")
                    break

        with (out_dir / "history.json").open("w") as fh:
            json.dump({
                "args": vars(args),
                "history": history,
                "best_val_epe": best_val_epe,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }, fh, indent=2)
        print(f"[done] best_val_epe={best_val_epe:.4f}, ckpt at {out_dir / 'checkpoints' / 'best.ckpt'}")


if __name__ == "__main__":
    main()
