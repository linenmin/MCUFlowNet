"""FC2 数据集读取与采样实现。"""  # 定义模块用途

import os  # 导入系统路径模块
import random  # 导入随机模块
from pathlib import Path  # 导入路径工具
from typing import List, Optional, Tuple  # 导入类型注解

try:  # 尝试导入OpenCV模块
    import cv2  # 导入OpenCV模块
except Exception:  # 捕获OpenCV导入异常
    cv2 = None  # 回退为空模块占位
import numpy as np  # 导入NumPy模块


def _read_text_lines(path_like: str) -> List[str]:  # 定义文本读取函数
    """读取文本文件并返回非空行列表。"""  # 说明函数用途
    path = Path(path_like)  # 构造路径对象
    if not path.exists():  # 检查文件是否存在
        return []  # 文件缺失时返回空列表
    lines = []  # 初始化行列表
    with path.open("r", encoding="utf-8") as handle:  # 以UTF-8打开文件
        for raw in handle:  # 遍历原始文本行
            stripped = raw.strip()  # 去除首尾空白字符
            if not stripped:  # 判断是否为空行
                continue  # 空行时跳过
            lines.append(stripped)  # 添加有效行到列表
    return lines  # 返回有效行列表


def _read_flow_file(path_like: str) -> np.ndarray:  # 定义光流读取函数
    """读取.flo文件并返回光流数组。"""  # 说明函数用途
    with open(path_like, "rb") as handle:  # 以二进制方式打开文件
        magic = handle.read(4)  # 读取魔数头部
        if magic.decode("utf-8") != "PIEH":  # 检查魔数是否正确
            raise ValueError(f"invalid flow header: {path_like}")  # 抛出非法头部错误
        width = np.fromfile(handle, np.int32, 1).squeeze()  # 读取宽度信息
        height = np.fromfile(handle, np.int32, 1).squeeze()  # 读取高度信息
        data = np.fromfile(handle, np.float32, width * height * 2)  # 读取光流数据
        flow = data.reshape((height, width, 2)).astype(np.float32)  # 重塑为二维光流张量
    return flow  # 返回光流张量


def _resolve_sample_path(base_path: Optional[str], raw_path: str) -> str:  # 定义样本路径解析函数
    """解析样本相对路径为本地绝对路径。"""  # 说明函数用途
    candidate = Path(raw_path)  # 构造候选路径对象
    if candidate.is_absolute() and candidate.exists():  # 判断候选绝对路径是否存在
        return str(candidate)  # 返回存在的绝对路径
    if base_path:  # 判断是否提供基础路径
        root = Path(base_path)  # 构造基础路径对象
        joined = root / raw_path  # 拼接基础路径与原始路径
        if joined.exists():  # 判断拼接路径是否存在
            return str(joined)  # 返回存在的拼接路径
    return str(candidate)  # 返回原始候选字符串


def resolve_fc2_samples(data_list_dir: str, split_file_name: str, base_path: Optional[str] = None) -> List[str]:  # 定义样本解析函数
    """根据索引列表解析FC2图像路径。"""  # 说明函数用途
    data_list_root = Path(data_list_dir)  # 构造数据列表目录对象
    dirnames_path = data_list_root / "FC2_dirnames.txt"  # 计算目录名文件路径
    split_path = data_list_root / split_file_name  # 计算划分索引文件路径
    dirnames = _read_text_lines(str(dirnames_path))  # 读取目录名列表
    split_tokens = _read_text_lines(str(split_path))  # 读取划分索引列表
    indices = [int(token) for token in split_tokens]  # 将索引字符串转为整数
    samples = []  # 初始化样本路径列表
    for idx in indices:  # 遍历索引列表
        if idx < 0 or idx >= len(dirnames):  # 检查索引是否越界
            continue  # 越界时跳过当前索引
        sample = _resolve_sample_path(base_path=base_path, raw_path=dirnames[idx])  # 解析单个样本路径
        samples.append(sample)  # 添加解析后的样本路径
    return samples  # 返回样本路径列表


def _build_fc2_triplet(img0_path: str) -> Tuple[str, str, str]:  # 定义样本三元组构建函数
    """根据img_0路径推导img_1与flow路径。"""  # 说明函数用途
    img1_path = img0_path.replace("img_0", "img_1")  # 根据命名规则推导第二帧路径
    flow_path = img0_path.replace("img_0.png", "flow_01.flo")  # 根据命名规则推导光流路径
    return img0_path, img1_path, flow_path  # 返回样本三元组路径


def _random_crop_triplet(  # 定义随机裁剪函数
    img0: np.ndarray,  # 定义第一帧图像参数
    img1: np.ndarray,  # 定义第二帧图像参数
    flow: np.ndarray,  # 定义光流图像参数
    crop_h: int,  # 定义裁剪高度参数
    crop_w: int,  # 定义裁剪宽度参数
    rng: random.Random,  # 定义随机数生成器参数
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # 定义随机裁剪返回类型
    """对图像和光流执行同区域随机裁剪。"""  # 说明函数用途
    h, w = img0.shape[0], img0.shape[1]  # 读取输入图像尺寸
    if h < crop_h or w < crop_w:  # 判断输入尺寸是否小于目标裁剪尺寸
        raise ValueError(f"input too small for crop: {h}x{w} vs {crop_h}x{crop_w}")  # 抛出尺寸不足错误
    top = rng.randint(0, h - crop_h)  # 随机采样裁剪起始行
    left = rng.randint(0, w - crop_w)  # 随机采样裁剪起始列
    img0_c = img0[top : top + crop_h, left : left + crop_w, :]  # 裁剪第一帧图像
    img1_c = img1[top : top + crop_h, left : left + crop_w, :]  # 裁剪第二帧图像
    flow_c = flow[top : top + crop_h, left : left + crop_w, :]  # 裁剪光流图像
    return img0_c, img1_c, flow_c  # 返回裁剪后的三元组


def _synthetic_triplet(crop_h: int, crop_w: int, rng: random.Random) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # 定义合成样本函数
    """生成用于降级容错的合成样本。"""  # 说明函数用途
    img0 = (rng.random() * np.ones([crop_h, crop_w, 3], dtype=np.float32) * 255.0).astype(np.float32)  # 生成合成第一帧图像
    img1 = (rng.random() * np.ones([crop_h, crop_w, 3], dtype=np.float32) * 255.0).astype(np.float32)  # 生成合成第二帧图像
    flow = np.zeros([crop_h, crop_w, 2], dtype=np.float32)  # 生成合成光流标签
    return img0, img1, flow  # 返回合成样本三元组


class FC2BatchProvider:  # 定义FC2批采样类
    """提供FC2训练与验证批数据。"""  # 说明类用途

    def __init__(  # 定义初始化函数
        self,  # 定义实例引用
        samples: List[str],  # 定义样本路径列表
        crop_h: int,  # 定义裁剪高度
        crop_w: int,  # 定义裁剪宽度
        seed: int = 42,  # 定义随机种子
        allow_synthetic: bool = True,  # 定义是否允许合成样本兜底
    ):  # 定义采样器初始化结束
        self.samples = list(samples)  # 保存样本路径列表副本
        self.crop_h = int(crop_h)  # 保存裁剪高度
        self.crop_w = int(crop_w)  # 保存裁剪宽度
        self.allow_synthetic = bool(allow_synthetic)  # 保存合成兜底开关
        self.rng = random.Random(int(seed))  # 初始化可复现随机数生成器

    def _load_one(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # 定义单样本加载函数
        """加载单个样本并完成随机裁剪。"""  # 说明函数用途
        if not self.samples:  # 判断样本列表是否为空
            if self.allow_synthetic:  # 判断是否允许合成兜底
                return _synthetic_triplet(crop_h=self.crop_h, crop_w=self.crop_w, rng=self.rng)  # 返回合成样本
            raise RuntimeError("FC2 sample list is empty and synthetic fallback disabled")  # 抛出样本为空错误
        if cv2 is None:  # 判断是否缺失OpenCV依赖
            if self.allow_synthetic:  # 判断是否允许合成兜底
                return _synthetic_triplet(crop_h=self.crop_h, crop_w=self.crop_w, rng=self.rng)  # 返回合成样本
            raise RuntimeError("OpenCV not available and synthetic fallback disabled")  # 抛出OpenCV缺失错误
        for _ in range(32):  # 设置最多尝试次数避免死循环
            img0_path = self.samples[self.rng.randint(0, len(self.samples) - 1)]  # 随机采样第一帧路径
            img0_path, img1_path, flow_path = _build_fc2_triplet(img0_path=img0_path)  # 推导样本三元组路径
            img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)  # 读取第一帧图像
            img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)  # 读取第二帧图像
            if img0 is None or img1 is None:  # 判断图像是否读取失败
                continue  # 读取失败时跳过当前尝试
            if not os.path.exists(flow_path):  # 检查光流路径是否存在
                continue  # 光流缺失时跳过当前尝试
            try:  # 尝试读取光流文件
                flow = _read_flow_file(flow_path)  # 读取光流张量
            except Exception:  # 捕获光流读取异常
                continue  # 读取异常时跳过当前尝试
            img0 = img0.astype(np.float32)  # 转换第一帧为float32
            img1 = img1.astype(np.float32)  # 转换第二帧为float32
            flow = np.clip(flow, a_min=-50.0, a_max=50.0).astype(np.float32)  # 对光流执行范围裁剪
            try:  # 尝试执行同步随机裁剪
                return _random_crop_triplet(img0=img0, img1=img1, flow=flow, crop_h=self.crop_h, crop_w=self.crop_w, rng=self.rng)  # 返回裁剪结果
            except Exception:  # 捕获裁剪异常
                continue  # 裁剪异常时跳过当前尝试
        if self.allow_synthetic:  # 判断是否允许合成兜底
            return _synthetic_triplet(crop_h=self.crop_h, crop_w=self.crop_w, rng=self.rng)  # 返回合成样本
        raise RuntimeError("failed to load valid FC2 sample after retries")  # 抛出加载失败错误

    def next_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # 定义批数据加载函数
        """加载一个训练批次。"""  # 说明函数用途
        p1_batch = []  # 初始化第一帧批列表
        p2_batch = []  # 初始化第二帧批列表
        flow_batch = []  # 初始化光流批列表
        for _ in range(int(batch_size)):  # 按批大小循环采样
            img0, img1, flow = self._load_one()  # 加载单个样本三元组
            p1_batch.append(img0)  # 添加第一帧到批列表
            p2_batch.append(img1)  # 添加第二帧到批列表
            flow_batch.append(flow)  # 添加光流到批列表
        p1 = np.asarray(p1_batch, dtype=np.float32)  # 堆叠第一帧批张量
        p2 = np.asarray(p2_batch, dtype=np.float32)  # 堆叠第二帧批张量
        label = np.asarray(flow_batch, dtype=np.float32)  # 堆叠光流标签批张量
        input_pair = np.concatenate([p1, p2], axis=3).astype(np.float32)  # 拼接双帧输入张量
        return input_pair, p1, p2, label  # 返回训练批次张量
