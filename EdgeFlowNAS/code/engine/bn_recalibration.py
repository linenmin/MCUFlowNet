"""BN 重估执行工具。"""  # 定义模块用途

from typing import Dict, List  # 导入类型注解


def run_bn_recalibration_session(  # 定义BN重估会话函数
    sess,  # 定义TensorFlow会话参数
    forward_fetch,  # 定义前向张量参数
    input_ph,  # 定义输入占位符参数
    label_ph,  # 定义标签占位符参数
    arch_code_ph,  # 定义架构编码占位符参数
    is_training_ph,  # 定义训练标志占位符参数
    batch_provider,  # 定义批采样器参数
    arch_code: List[int],  # 定义目标架构编码参数
    batch_size: int,  # 定义批大小参数
    recal_batches: int,  # 定义重估批次数参数
) -> Dict[str, float]:  # 定义BN重估返回类型
    """在会话中执行BN统计重估。"""  # 说明函数用途
    for _ in range(int(recal_batches)):  # 按重估批次数循环
        input_batch, _, _, label_batch = batch_provider.next_batch(batch_size=batch_size)  # 采样重估批数据
        sess.run(  # 执行前向以触发BN统计更新
            forward_fetch,  # 指定前向张量抓取
            feed_dict={  # 传入重估喂入字典
                input_ph: input_batch,  # 传入输入批数据
                label_ph: label_batch,  # 传入标签批数据
                arch_code_ph: arch_code,  # 传入目标架构编码
                is_training_ph: True,  # 开启训练模式以更新BN统计
            },
        )
    return {  # 返回重估摘要字典
        "bn_recal_batches": float(recal_batches),  # 写入重估批次数
        "bn_mean_shift": 0.0,  # 占位返回均值变化
        "bn_var_shift": 0.0,  # 占位返回方差变化
    }
