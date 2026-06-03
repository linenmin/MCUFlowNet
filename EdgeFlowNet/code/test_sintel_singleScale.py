#!/usr/bin/env python3
"""
Single-Scale 测试脚本：只使用最后一个输出，不使用多尺度累加
用于对比多尺度累加 (test_sintel.py) 和单尺度输出的精度差异
"""
import tensorflow as tf
import cv2
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import glob
import re
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from misc.MiscUtils import *
from misc.FlowVisUtilsNP import *
import numpy as np
import time
import argparse
import shutil
import string
from termcolor import colored, cprint
import misc.FlowVisUtilsNP as fvu
import misc.SintelFlowViz as svu
import math as m
from tqdm import tqdm
import misc.TFUtils as tu
from misc.Decorators import *
import misc.FlowShift as fs
import importlib
from datetime import datetime
import getpass
import copy
import platform
import misc.TFUtils as tu
from scipy.ndimage import gaussian_filter
import misc.FlowPolar as fp
from misc.processor import FlowPostProcessor

# 禁止生成 pyc 文件
sys.dont_write_bytecode = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from misc.utils import get_sintel_batch, read_sintel_list, is_multiscale

# 禁用 Eager Execution 以使用 TF1 风格代码
tf.compat.v1.disable_eager_execution()


def setup_single_scale_model(InputPH, Args):
    """
    设置单尺度模型：只使用最后一个输出，不使用 AccumPreds 累加
    与 setup_full_model 的区别在于不做多尺度累加
    """
    Args.NetworkName = "Network.MultiScaleResNet"
    ClassName = Args.NetworkName.replace('Network.', '').split('Net')[0] + 'Net'
    Network = getattr(Args.Net, ClassName)
    
    VN = Network(
        InputPH=InputPH,
        InitNeurons=32,
        NumSubBlocks=2,
        Suffix='',
        NumOut=Args.NumOut,
        ExpansionFactor=2,
        UncType=None
    )
    
    prValList = VN.Network()  # 返回 [scale1, scale2, final_output] 的 list
    
    # 关键区别：只取最后一个输出，不做多尺度累加
    if isinstance(prValList, list):
        prVal = prValList[-1][..., 0:2]  # 只取最后一个输出的前 2 通道 (u, v)
        print(f"[*] 使用单尺度输出 (最后一个)，共 {len(prValList)} 个尺度可用")
    else:
        prVal = prValList[..., 0:2]
        print("[*] 模型只有单一输出")
    
    Saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    Saver.restore(sess, Args.checkpoint)
    
    return VN, prVal, sess


def TestOperation(InputPH, Args):
    """执行测试操作"""
    VN, prVal, sess = setup_single_scale_model(InputPH, Args)

    processor = FlowPostProcessor("single_scale", is_multiscale(Args))
    
    im1_filenames, im2_filenames, flo_filenames = read_sintel_list(Args)

    IBatch = np.random.rand(Args.InputPatchSize[0], Args.InputPatchSize[1], 2*Args.InputPatchSize[2])
    Label1Batch = np.random.rand(1, Args.InputPatchSize[0], Args.InputPatchSize[1], Args.NumOut)

    for i in tqdm(range(0, len(im1_filenames))):
        IBatch, Label1Batch = get_sintel_batch(im1_filenames[i], im2_filenames[i], flo_filenames[i], Args.PatchSize)
        if IBatch is None:
            continue
    
        FeedDict = {VN.InputPH: IBatch[None, ...]}
        
        prediction_full = sess.run([prVal], feed_dict=FeedDict)[0]
    
        processor.update(Label1Batch, prediction_full, Args)

    processor.print()


def main():
    """主函数"""
    Parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    Parser.add_argument('--checkpoint', default='/root/optical/models/brown/', help='Path to save checkpoints')
    Parser.add_argument('--gpu_device', type=int, default=0, help='What GPU do you want to use? -1 for CPU')
    Parser.add_argument('--patch_dim_0', default=416, type=int, help='patch size 0')
    Parser.add_argument('--patch_dim_1', default=1024, type=int, help='patch size 1')
    Parser.add_argument('--patch_channels', default=3, type=int, help='patch size channels')
    Parser.add_argument('--patch_delta', default=20, type=int, help='additional patch delta')
    Parser.add_argument('--uncertainity', action='store_true', help='is uncertainity')
    Parser.add_argument('--data_list', default='./Misc/MPI_Sintel_train_clean.txt', help='list of sintel data')
    args = Parser.parse_args()

    tu.SetGPU(args.gpu_device)

    args.Net = importlib.import_module('network.MultiScaleResNet')

    args.InputPatchSize = np.array([args.patch_dim_0, args.patch_dim_1, args.patch_channels])
    args.PatchSize = args.InputPatchSize
    args.NumOut = 2

    if args.uncertainity:
        args.NumOut = 4

    InputPH = tf.compat.v1.placeholder(
        tf.float32,
        shape=(1, args.InputPatchSize[0], args.InputPatchSize[1], 2*args.InputPatchSize[2]),
        name='Input'
    )
    
    print("=" * 50)
    print("Single-Scale Test (只使用最后一个输出)")
    print("=" * 50)
    
    TestOperation(InputPH, args)


if __name__ == '__main__':
    main()
