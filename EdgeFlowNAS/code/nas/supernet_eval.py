"""超网评估脚本。"""  # 定义模块用途
import argparse  # 导入参数解析模块
import json  # 导入JSON模块
from pathlib import Path  # 导入路径工具
from typing import Any, Dict, List, Optional  # 导入类型注解

import numpy as np  # 导入NumPy模块
import tensorflow as tf  # 导入TensorFlow模块

from code.data.dataloader_builder import build_fc2_provider  # 导入FC2数据加载器构建函数
from code.data.transforms_180x240 import standardize_image_tensor  # 导入输入标准化函数
from code.engine.eval_step import build_epe_metric  # 导入EPE指标构建函数
from code.nas.eval_pool_builder import build_eval_pool, check_eval_pool_coverage  # 导入验证池工具
from code.network.MultiScaleResNet_supernet import MultiScaleResNetSupernet  # 导入超网模型
from code.utils.json_io import read_json, write_json  # 导入JSON读写工具
from code.utils.path_utils import ensure_directory, project_root  # 导入路径工具

try:  # 尝试导入YAML模块
    import yaml  # 导入YAML模块
except Exception:  # 捕获YAML导入异常
    yaml = None  # 回退为空解析器占位


def _parse_scalar(value_text: str) -> Any:  # 定义标量解析函数
    """将YAML标量文本解析为Python值。"""  # 说明函数用途
    lowered = value_text.lower()  # 生成小写文本副本
    if lowered == "true":  # 判断是否为布尔真
        return True  # 返回布尔真
    if lowered == "false":  # 判断是否为布尔假
        return False  # 返回布尔假
    if value_text.startswith('"') and value_text.endswith('"'):  # 判断是否为双引号字符串
        return value_text[1:-1]  # 返回去引号字符串
    if value_text.startswith("'") and value_text.endswith("'"):  # 判断是否为单引号字符串
        return value_text[1:-1]  # 返回去引号字符串
    try:  # 尝试解析整数
        return int(value_text)  # 返回整数值
    except Exception:  # 捕获整数解析异常
        pass  # 跳过到下一个解析分支
    try:  # 尝试解析浮点数
        return float(value_text)  # 返回浮点值
    except Exception:  # 捕获浮点解析异常
        pass  # 跳过到字符串分支
    return value_text  # 返回原始字符串值


def _load_simple_yaml(path: Path) -> Dict[str, Any]:  # 定义轻量YAML解析函数
    """在无PyYAML环境下解析简化YAML配置。"""  # 说明函数用途
    root: Dict[str, Any] = {}  # 初始化根字典
    stack = [(-1, root)]  # 初始化缩进栈
    with path.open("r", encoding="utf-8") as handle:  # 以UTF-8打开配置文件
        for raw in handle:  # 遍历配置文件每行
            content = raw.split("#", 1)[0].rstrip("\n")  # 去除注释与换行
            if not content.strip():  # 判断是否为空内容行
                continue  # 空内容时跳过当前行
            indent = len(content) - len(content.lstrip(" "))  # 计算当前行缩进
            stripped = content.strip()  # 提取去空白后的内容
            while stack and indent <= stack[-1][0]:  # 弹出不匹配缩进层
                stack.pop()  # 弹出当前栈顶层
            parent = stack[-1][1]  # 获取当前父级字典
            if stripped.endswith(":"):  # 判断是否为字典起始行
                key = stripped[:-1].strip()  # 提取字典键名
                node: Dict[str, Any] = {}  # 初始化子字典
                parent[key] = node  # 将子字典挂载到父级
                stack.append((indent, node))  # 将子层压入缩进栈
                continue  # 进入下一行解析
            key, value_text = stripped.split(":", 1)  # 拆分键和值文本
            parent[key.strip()] = _parse_scalar(value_text.strip())  # 解析并写入标量值
    return root  # 返回解析后的配置字典


def _load_config(path_like: str) -> Dict[str, Any]:  # 定义配置加载函数
    """读取YAML配置文件。"""  # 说明函数用途
    config_path = Path(path_like)  # 构造配置路径对象
    if not config_path.is_absolute():  # 判断配置路径是否为绝对路径
        config_path = project_root() / config_path  # 将相对路径转换为项目内绝对路径
    if yaml is None:  # 判断是否缺失PyYAML依赖
        payload = _load_simple_yaml(config_path)  # 使用轻量解析器读取配置
    else:  # 使用标准PyYAML解析器
        with config_path.open("r", encoding="utf-8") as handle:  # 以UTF-8打开配置文件
            payload = yaml.safe_load(handle)  # 安全解析YAML内容
    if not isinstance(payload, dict):  # 检查解析结果是否为字典
        raise ValueError("配置文件顶层必须是字典结构。")  # 抛出结构错误
    return payload  # 返回配置字典


def _resolve_output_dir(config: Dict[str, Any]) -> Path:  # 定义输出目录解析函数
    """解析实验输出目录并确保目录存在。"""  # 说明函数用途
    runtime_cfg = config.get("runtime", {})  # 读取运行配置字典
    output_root = runtime_cfg.get("output_root", "outputs/supernet")  # 读取输出根目录配置
    experiment_name = runtime_cfg.get("experiment_name", "default_experiment")  # 读取实验名称配置
    return ensure_directory(str(project_root() / output_root / experiment_name))  # 创建并返回输出目录


def _build_checkpoint_paths(experiment_dir: Path) -> Dict[str, Path]:  # 定义checkpoint路径构建函数
    """构建评估阶段使用的checkpoint路径。"""  # 说明函数用途
    ckpt_root = ensure_directory(str(experiment_dir / "checkpoints"))  # 计算并创建checkpoint目录
    return {"best": ckpt_root / "supernet_best.ckpt", "last": ckpt_root / "supernet_last.ckpt"}  # 返回路径字典


def _checkpoint_exists(path_prefix: Path) -> bool:  # 定义checkpoint存在检查函数
    """检查checkpoint索引文件是否存在。"""  # 说明函数用途
    return Path(str(path_prefix) + ".index").exists()  # 返回存在状态


def _find_existing_checkpoint(path_prefix: Path) -> Optional[Path]:  # 定义checkpoint查找函数
    """优先查找指定前缀，失败时回退目录最新checkpoint。"""  # 说明函数用途
    if _checkpoint_exists(path_prefix=path_prefix):  # 判断指定前缀是否存在
        return path_prefix  # 返回指定前缀路径
    latest = tf.train.latest_checkpoint(str(path_prefix.parent))  # 查找目录内最新checkpoint
    if latest:  # 判断是否找到最新checkpoint
        return Path(latest)  # 返回最新checkpoint路径
    return None  # 未找到时返回空值


def _load_checkpoint_meta(path_prefix: Path) -> Dict[str, Any]:  # 定义checkpoint元信息读取函数
    """读取checkpoint同名前缀meta信息。"""  # 说明函数用途
    meta_path = Path(str(path_prefix) + ".meta.json")  # 计算meta路径
    if not meta_path.exists():  # 判断meta文件是否存在
        return {}  # 文件缺失时返回空字典
    payload = read_json(str(meta_path))  # 读取meta内容
    if isinstance(payload, dict):  # 判断读取结果是否为字典
        return payload  # 返回字典结果
    return {}  # 非字典时返回空字典


def _build_eval_graph(config: Dict[str, Any], batch_size: int) -> Dict[str, Any]:  # 定义评估图构建函数
    """构建超网评估阶段计算图。"""  # 说明函数用途
    data_cfg = config.get("data", {})  # 读取数据配置字典
    input_h = int(data_cfg.get("input_height", 180))  # 读取输入高度配置
    input_w = int(data_cfg.get("input_width", 240))  # 读取输入宽度配置
    flow_channels = int(data_cfg.get("flow_channels", 2))  # 读取光流通道配置
    pred_channels = int(flow_channels * 2)  # 计算不确定性版本预测通道数
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, input_h, input_w, 6], name="Input")  # 创建输入占位符
    label_ph = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, input_h, input_w, flow_channels], name="Label")  # 创建标签占位符
    arch_code_ph = tf.compat.v1.placeholder(tf.int32, shape=[9], name="ArchCode")  # 创建架构编码占位符
    is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="IsTraining")  # 创建训练标志占位符
    model = MultiScaleResNetSupernet(  # 创建超网模型实例
        input_ph=input_ph,  # 传入输入占位符
        arch_code_ph=arch_code_ph,  # 传入架构编码占位符
        is_training_ph=is_training_ph,  # 传入训练标志占位符
        num_out=pred_channels,  # 传入输出通道数
        init_neurons=32,  # 传入初始通道数
        expansion_factor=2.0,  # 传入通道扩展倍率
    )
    preds = model.build()  # 构建前向输出
    epe_tensor = build_epe_metric(pred_tensor=preds[-1], label_ph=label_ph, num_out=flow_channels)  # 构建EPE指标
    saver = tf.compat.v1.train.Saver(max_to_keep=5)  # 创建Saver对象
    return {  # 返回评估图对象
        "input_ph": input_ph,  # 返回输入占位符
        "label_ph": label_ph,  # 返回标签占位符
        "arch_code_ph": arch_code_ph,  # 返回架构编码占位符
        "is_training_ph": is_training_ph,  # 返回训练标志占位符
        "pred_tensor": preds[-1],  # 返回最终尺度预测
        "epe": epe_tensor,  # 返回EPE指标张量
        "saver": saver,  # 返回Saver对象
    }


def _run_eval_pool(  # 定义验证池评估函数
    sess,  # 定义TensorFlow会话参数
    graph_obj: Dict[str, Any],  # 定义图对象参数
    train_provider,  # 定义训练采样器参数
    val_provider,  # 定义验证采样器参数
    eval_pool: List[List[int]],  # 定义验证池参数
    bn_recal_batches: int,  # 定义BN重估批次数参数
    batch_size: int,  # 定义批大小参数
) -> Dict[str, Any]:  # 定义评估结果返回类型
    """在固定验证池上执行BN重估+EPE评估。"""  # 说明函数用途
    epe_values = []  # 初始化EPE列表
    for arch_code in eval_pool:  # 遍历每个子网架构
        for _ in range(int(bn_recal_batches)):  # 按配置执行BN重估批次
            train_input, _, _, train_label = train_provider.next_batch(batch_size=batch_size)  # 采样训练批数据
            train_input = standardize_image_tensor(train_input)  # 标准化训练输入
            sess.run(  # 执行前向触发BN统计更新
                graph_obj["pred_tensor"],  # 指定前向输出张量
                feed_dict={  # 传入前向数据
                    graph_obj["input_ph"]: train_input,  # 传入训练输入
                    graph_obj["label_ph"]: train_label,  # 传入训练标签
                    graph_obj["arch_code_ph"]: arch_code,  # 传入架构编码
                    graph_obj["is_training_ph"]: True,  # 指定训练模式
                },
            )
        val_input, _, _, val_label = val_provider.next_batch(batch_size=batch_size)  # 采样验证批数据
        val_input = standardize_image_tensor(val_input)  # 标准化验证输入
        epe_val = sess.run(  # 执行EPE评估
            graph_obj["epe"],  # 指定EPE张量
            feed_dict={  # 传入评估数据
                graph_obj["input_ph"]: val_input,  # 传入验证输入
                graph_obj["label_ph"]: val_label,  # 传入验证标签
                graph_obj["arch_code_ph"]: arch_code,  # 传入架构编码
                graph_obj["is_training_ph"]: False,  # 指定推理模式
            },
        )
        epe_values.append(float(epe_val))  # 记录当前子网EPE
    mean_epe = float(np.mean(epe_values)) if epe_values else 0.0  # 计算平均EPE
    std_epe = float(np.std(epe_values)) if epe_values else 0.0  # 计算EPE标准差
    return {"mean_epe_12": mean_epe, "std_epe_12": std_epe, "per_arch_epe": epe_values}  # 返回评估结果


def _load_or_build_eval_pool(output_dir: Path, seed: int, pool_size: int) -> Dict[str, Any]:  # 定义验证池加载函数
    """优先加载已落盘验证池，缺失时按配置重建。"""  # 说明函数用途
    pool_path = output_dir / f"eval_pool_{pool_size}.json"  # 计算验证池落盘路径
    if pool_path.exists():  # 判断验证池文件是否存在
        payload = read_json(str(pool_path))  # 读取验证池文件
        if isinstance(payload, dict) and isinstance(payload.get("pool", None), list):  # 判断文件结构是否合法
            pool = payload["pool"]  # 读取验证池列表
            coverage = payload.get("coverage", check_eval_pool_coverage(pool=pool, num_blocks=9))  # 读取或重算覆盖信息
            return {"pool": pool, "coverage": coverage, "pool_path": pool_path}  # 返回验证池信息
    pool = build_eval_pool(seed=seed, size=pool_size, num_blocks=9)  # 构建新的验证池
    coverage = check_eval_pool_coverage(pool=pool, num_blocks=9)  # 执行覆盖检查
    write_json(str(pool_path), {"pool": pool, "coverage": coverage})  # 写入验证池文件
    return {"pool": pool, "coverage": coverage, "pool_path": pool_path}  # 返回验证池信息


def _build_parser() -> argparse.ArgumentParser:  # 定义参数解析器构建函数
    """创建命令行参数解析器。"""  # 说明函数用途
    parser = argparse.ArgumentParser(description="evaluate trained supernet checkpoint")  # 创建解析器
    parser.add_argument("--config", required=True, help="path to supernet config yaml")  # 添加配置路径参数
    parser.add_argument("--eval_only", action="store_true", help="run eval only flow")  # 添加仅评估开关
    parser.add_argument("--bn_recal_batches", type=int, default=None, help="override bn recalibration batches")  # 添加BN重估批次数参数
    parser.add_argument("--checkpoint_type", choices=["best", "last"], default="best", help="checkpoint type to evaluate")  # 添加checkpoint类型参数
    parser.add_argument("--batch_size", type=int, default=None, help="override eval batch size")  # 添加评估批大小参数
    return parser  # 返回解析器


def main() -> int:  # 定义主函数
    """执行超网checkpoint评估流程。"""  # 说明函数用途
    parser = _build_parser()  # 构建命令行解析器
    args = parser.parse_args()  # 解析命令行参数
    if not args.eval_only:  # 判断是否启用eval_only
        parser.error("supernet_eval requires --eval_only")  # 提示必须启用eval_only
    config = _load_config(args.config)  # 加载配置字典
    runtime_cfg = config.get("runtime", {})  # 读取运行配置字典
    train_cfg = config.get("train", {})  # 读取训练配置字典
    eval_cfg = config.get("eval", {})  # 读取评估配置字典
    data_cfg = config.get("data", {})  # 读取数据配置字典
    seed = int(runtime_cfg.get("seed", 42))  # 读取随机种子配置
    pool_size = int(eval_cfg.get("eval_pool_size", 12))  # 读取验证池大小配置
    bn_recal_batches = int(args.bn_recal_batches) if args.bn_recal_batches is not None else int(eval_cfg.get("bn_recal_batches", 8))  # 读取BN重估批次数配置
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 32))  # 读取评估批大小配置
    output_dir = _resolve_output_dir(config=config)  # 解析输出目录
    pool_info = _load_or_build_eval_pool(output_dir=output_dir, seed=seed, pool_size=pool_size)  # 加载或构建验证池
    eval_pool = pool_info["pool"]  # 读取验证池列表
    coverage = pool_info["coverage"]  # 读取覆盖检查结果
    train_provider = build_fc2_provider(  # 构建训练采样器
        config=config,  # 传入配置字典
        split_file_name=str(data_cfg.get("train_list_name", "FC2_train.txt")),  # 传入训练列表文件名
        seed_offset=0,  # 传入训练采样随机偏移
    )
    val_provider = build_fc2_provider(  # 构建验证采样器
        config=config,  # 传入配置字典
        split_file_name=str(data_cfg.get("val_list_name", "FC2_test.txt")),  # 传入验证列表文件名
        seed_offset=999,  # 传入验证采样随机偏移
    )
    tf.compat.v1.disable_eager_execution()  # 关闭Eager执行模式
    tf.compat.v1.reset_default_graph()  # 重置默认计算图
    graph_obj = _build_eval_graph(config=config, batch_size=batch_size)  # 构建评估图对象
    checkpoint_paths = _build_checkpoint_paths(experiment_dir=output_dir)  # 构建checkpoint路径
    chosen_prefix = checkpoint_paths[args.checkpoint_type]  # 读取目标checkpoint前缀
    checkpoint_prefix = _find_existing_checkpoint(path_prefix=chosen_prefix)  # 查找目标checkpoint
    if checkpoint_prefix is None:  # 判断是否找到目标checkpoint
        raise RuntimeError(f"checkpoint not found: {chosen_prefix}")  # 抛出checkpoint缺失异常
    with tf.compat.v1.Session() as sess:  # 创建TensorFlow会话
        sess.run(tf.compat.v1.global_variables_initializer())  # 初始化全局变量
        graph_obj["saver"].restore(sess, str(checkpoint_prefix))  # 恢复checkpoint权重
        metrics = _run_eval_pool(  # 执行验证池评估
            sess=sess,  # 传入会话对象
            graph_obj=graph_obj,  # 传入图对象
            train_provider=train_provider,  # 传入训练采样器
            val_provider=val_provider,  # 传入验证采样器
            eval_pool=eval_pool,  # 传入评估池
            bn_recal_batches=bn_recal_batches,  # 传入BN重估次数
            batch_size=batch_size,  # 传入批大小
        )
    checkpoint_meta = _load_checkpoint_meta(path_prefix=checkpoint_prefix)  # 读取checkpoint元信息
    result = {  # 组装评估摘要结果
        "status": "ok",  # 标记评估状态
        "checkpoint_type": args.checkpoint_type,  # 记录checkpoint类型
        "checkpoint_path": str(checkpoint_prefix),  # 记录checkpoint路径
        "checkpoint_meta": checkpoint_meta,  # 记录checkpoint元信息
        "eval_pool_path": str(pool_info["pool_path"]),  # 记录验证池路径
        "eval_pool_coverage_ok": bool(coverage.get("ok", False)),  # 记录验证池覆盖状态
        "mean_epe_12": float(metrics["mean_epe_12"]),  # 记录平均EPE
        "std_epe_12": float(metrics["std_epe_12"]),  # 记录EPE标准差
        "bn_recal_batches": int(bn_recal_batches),  # 记录BN重估批次数
        "batch_size": int(batch_size),  # 记录评估批大小
    }
    result_path = output_dir / f"supernet_eval_result_{args.checkpoint_type}.json"  # 计算评估结果落盘路径
    write_json(str(result_path), {"summary": result, "per_arch_epe": metrics.get("per_arch_epe", []), "coverage": coverage})  # 写入评估结果文件
    print(  # 打印执行摘要
        json.dumps(  # 序列化输出摘要
            {  # 构建摘要字典
                "status": "ok",  # 输出状态
                "result_path": str(result_path),  # 输出结果路径
                "mean_epe_12": result["mean_epe_12"],  # 输出平均EPE
                "std_epe_12": result["std_epe_12"],  # 输出标准差
                "coverage_ok": result["eval_pool_coverage_ok"],  # 输出覆盖状态
            },
            ensure_ascii=False,  # 保留中文字符
            indent=2,  # 指定缩进
        )
    )
    return 0  # 返回成功状态码


if __name__ == "__main__":  # 判断是否为脚本直运行
    raise SystemExit(main())  # 以主函数返回码退出
