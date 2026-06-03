"""EdgeFlowNet mainline `MultiScaleResNet` ported into the efnas package.

This is a verbatim port of `EdgeFlowNet/code/network/MultiScaleResNet.py` and
its `BaseLayers` / `Decorators` dependencies. The variable scope names match
the published checkpoint at `EdgeFlowNet/checkpoints/best.ckpt` exactly
(e.g. `EncoderDecoderBlock0/ResNetBlock0/ConvBNReLUBlock1/...`); restoring
weights from that file into this graph works out of the box.

Two intentional differences vs the original code:

1. The module is fully self-contained. No `import network.BaseLayers` or
   `import misc.Decorators` — those helpers are included below. This avoids
   adding the EdgeFlowNet repo to `sys.path` on HPC.

2. `BN` keeps the original behaviour of NOT passing `training=` (so the
   default `training=False` is used at every site). This matches how the
   published checkpoint was actually trained — `moving_mean`/`moving_var`
   stay at their init values and act as a no-op normalizer; the affine
   transform is carried entirely by `gamma`/`beta`. Fine-tune therefore
   only updates conv kernels + gamma/beta (no BN running-stat updates).
"""
from __future__ import annotations

from functools import wraps
from typing import List

import numpy as np
import tensorflow as tf


# ----------------------------------------------------------------------------
# Decorator (matches EdgeFlowNet/code/misc/Decorators.py)
# ----------------------------------------------------------------------------
def _count_and_scope(func):
    """Open a variable_scope named `<methodName><CurrBlock><Suffix>` and bump
    `CurrBlock`. This produces the exact scope hierarchy the checkpoint
    expects, so don't rename methods or change the counter order."""
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        with tf.compat.v1.variable_scope(
            func.__name__ + str(self.CurrBlock) + str(self.Suffix)
        ):
            self.CurrBlock += 1
            return func(self, *args, **kwargs)

    return wrapped


# ----------------------------------------------------------------------------
# Base layer ops (matches EdgeFlowNet/code/network/BaseLayers.py)
# ----------------------------------------------------------------------------
class _BaseLayers:
    def __init__(self):
        self.CurrBlock = 0

    @_count_and_scope
    def ConvBNReLUBlock(self, inputs=None, filters=None, kernel_size=None,
                        strides=None, padding=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        conv = self.Conv(inputs=inputs, filters=filters, kernel_size=kernel_size,
                         strides=strides, padding=padding)
        bn = self.BN(conv)
        return self.ReLU(bn)

    @_count_and_scope
    def ConvTransposeBNReLUBlock(self, inputs=None, filters=None,
                                 kernel_size=None, strides=None, padding=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        conv = self.ConvTranspose(inputs=inputs, filters=filters,
                                  kernel_size=kernel_size, strides=strides,
                                  padding=padding)
        bn = self.BN(conv)
        return self.ReLU(bn)

    @_count_and_scope
    def Conv(self, inputs=None, filters=None, kernel_size=None, strides=None,
             padding=None, activation=None, name=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        return tf.compat.v1.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, activation=activation, name=name,
        )

    @_count_and_scope
    def ConvTranspose(self, inputs=None, filters=None, kernel_size=None,
                      strides=None, padding=None, activation=None, name=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        return tf.compat.v1.layers.conv2d_transpose(
            inputs=inputs, filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, activation=activation, name=name,
        )

    @_count_and_scope
    def BN(self, inputs=None):
        # NOTE: matches the original mainline BN call — no `training=` arg.
        # Defaults to training=False so moving_mean/var stay at init and
        # only gamma/beta act as an affine layer. See module docstring.
        return tf.compat.v1.layers.batch_normalization(inputs=inputs)

    @_count_and_scope
    def ReLU(self, inputs=None):
        return tf.compat.v1.nn.relu(inputs)


# ----------------------------------------------------------------------------
# MultiScaleResNet (verbatim port of EdgeFlowNet/code/network/MultiScaleResNet.py)
# ----------------------------------------------------------------------------
class EdgeFlowNetMainline(_BaseLayers):
    """Transpose-conv multi-scale optical-flow ResNet, AKA "mainline".

    Same as the published EdgeFlowNet `MultiScaleResNet`. With
    `NumOut=2, UncType='LinearSoftplus'` (the eval-time configuration used
    by `test_sintel.py --uncertainity`), the network produces 3 multi-scale
    predictions each with **4 channels** (2 flow + 2 uncertainty)."""

    def __init__(self, InputPH=None, Padding=None, NumOut=None,
                 InitNeurons=None, ExpansionFactor=None, NumSubBlocks=None,
                 NumBlocks=None, Suffix=None, UncType=None):
        super().__init__()
        if InputPH is None:
            raise ValueError("InputPH cannot be None")
        self.InputPH = InputPH

        if InitNeurons is None:
            InitNeurons = 37
        if ExpansionFactor is None:
            ExpansionFactor = 2.0
        if NumSubBlocks is None:
            NumSubBlocks = 2
        if NumBlocks is None:
            NumBlocks = 1
        if Suffix is None:
            Suffix = ""
        if NumOut is None:
            NumOut = 1

        self.InitNeurons = int(InitNeurons)
        self.ExpansionFactor = float(ExpansionFactor)
        self.NumSubBlocks = int(NumSubBlocks)
        self.NumBlocks = int(NumBlocks)
        self.Suffix = Suffix
        self.NumOut = int(NumOut)
        self.UncType = UncType
        self.currBlock = 0
        self.FeaturePyramid = None
        self.DropOutRate = 0.7  # unused at inference; preserved for parity

        if self.UncType in ("Aleatoric", "Inlier", "LinearSoftplus"):
            # Each output channel gets a paired uncertainty channel.
            self.NumOut *= 2

        self.kernel_size = (3, 3)
        self.strides = (2, 2)
        self.padding = "same" if Padding is None else Padding

    @_count_and_scope
    def ResBlock(self, inputs=None, filters=None, kernel_size=None,
                 strides=None, padding=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        Net = self.ConvBNReLUBlock(inputs=inputs, filters=filters,
                                   padding=padding, strides=(1, 1))
        Net = self.Conv(inputs=Net, filters=filters, padding=padding,
                        strides=(1, 1), activation=None)
        Net = self.BN(inputs=Net)
        Net = tf.add(Net, inputs)
        return self.ReLU(inputs=Net)

    @_count_and_scope
    def ResBlockTranspose(self, inputs=None, filters=None, kernel_size=None,
                          strides=None, padding=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        Net = self.ConvTransposeBNReLUBlock(inputs=inputs, filters=filters,
                                            padding=padding, strides=(1, 1))
        Net = self.ConvTranspose(inputs=Net, filters=filters, padding=padding,
                                 strides=(1, 1), activation=None)
        Net = self.BN(inputs=Net)
        Net = tf.add(Net, inputs)
        return self.ReLU(inputs=Net)

    @_count_and_scope
    def ResNetBlock(self, inputs):
        NumFilters = self.InitNeurons
        Net = self.ConvBNReLUBlock(inputs=inputs, filters=NumFilters,
                                   kernel_size=(7, 7))
        NumFilters = int(NumFilters * self.ExpansionFactor)
        Net = self.ConvBNReLUBlock(inputs=Net, filters=NumFilters,
                                   kernel_size=(5, 5))

        for _ in range(self.NumSubBlocks):
            Net = self.ResBlock(inputs=Net, filters=NumFilters)
            NumFilters = int(NumFilters * self.ExpansionFactor)
            Net = self.Conv(inputs=Net, filters=NumFilters)

        Nets: List[tf.Tensor] = []
        for _ in range(self.NumSubBlocks):
            Net = self.ResBlockTranspose(inputs=Net, filters=NumFilters)
            NumFilters = int(NumFilters / self.ExpansionFactor)
            Net = self.ConvTranspose(inputs=Net, filters=NumFilters)

        feat_low = Net
        NetOut = self.ConvTranspose(inputs=Net, filters=self.NumOut,
                                    strides=(1, 1), kernel_size=(7, 7))
        Nets.append(NetOut)

        NumFilters = int(NumFilters / self.ExpansionFactor)
        Net = self.ConvTransposeBNReLUBlock(inputs=Net, filters=NumFilters,
                                            kernel_size=(5, 5))
        feat_mid = Net
        NetOut = self.ConvTranspose(inputs=Net, filters=self.NumOut,
                                    strides=(1, 1), kernel_size=(7, 7))
        Nets.append(NetOut)

        NumFilters = int(NumFilters / self.ExpansionFactor)
        Net = self.ConvTransposeBNReLUBlock(inputs=Net, filters=NumFilters,
                                            kernel_size=(7, 7))
        feat_high = Net
        Net = self.ConvTranspose(inputs=Net, filters=self.NumOut,
                                 kernel_size=(7, 7), strides=(1, 1),
                                 activation=None)
        Nets.append(Net)
        self.FeaturePyramid = [feat_low, feat_mid, feat_high]
        return Nets

    def build(self) -> List[tf.Tensor]:
        """Return the 3 multi-scale predictions (low, mid, full).

        Each prediction has `self.NumOut` channels (already doubled if
        UncType triggered the doubling). For typical fine-tune usage
        (NumOut=2, UncType='LinearSoftplus'), each prediction is
        `[B, H, W, 4]`."""
        OutNow = self.InputPH
        for count in range(self.NumBlocks):
            with tf.compat.v1.variable_scope(
                "EncoderDecoderBlock" + str(count) + self.Suffix
            ):
                OutNow = self.ResNetBlock(OutNow)
                self.currBlock += 1
        return OutNow


# Default config that matches the published EdgeFlowNet best.ckpt.
MAINLINE_DEFAULT_CONFIG = dict(
    InitNeurons=32,           # EdgeFlowNet/code/train.py DEFAULT_INIT_NEURONS
    ExpansionFactor=2.0,
    NumSubBlocks=2,
    NumBlocks=1,
    Padding="same",
    NumOut=2,                 # logical; doubled to 4 internally via UncType
    UncType="LinearSoftplus",
)


def build_mainline(input_ph: tf.Tensor) -> EdgeFlowNetMainline:
    """Convenience builder used by `deploy_ft_trainer` and exporters."""
    model = EdgeFlowNetMainline(InputPH=input_ph, **MAINLINE_DEFAULT_CONFIG)
    return model
