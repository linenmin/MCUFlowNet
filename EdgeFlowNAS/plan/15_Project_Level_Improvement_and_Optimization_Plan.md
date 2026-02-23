# Supernet 训练问题排查与优化计划 (Project-Level Improvement & Optimization Plan)

**文档编号**: 15
**更新时间**: 2026-02-23
**目标**: 针对 `mean_epe_12` 评估指标长时间停靠 10.9 的异常现象，进行头脑风暴、根因分析，并给出针对性改进和优化清单。

---

## 1. 核心问题诊断 (Bug Report & Root Cause Analysis)

**观察到的异常现象：**
在 `train.log` 中，尽管主训练损失 (Loss) 和特征蒸馏损失 (Distill Loss) 都在稳定且显著地下降（Loss从第一周期的 18 降低到现在的 7.5），但验证集上的光流特定评价指标 `mean_epe_12` 始终停滞在 **10.9** 左右（范围在 10.91 ~ 10.96 之间无规律波动）。

**为什么是 10.9？**
- 在 FlyingChairs2 (FC2) 数据集中，验证集光流的平均绝对位移幅度 (magnitude) 大约就在 10~15 像素之间。
- 如果模型的验证集端点误差 (EPE) 稳定在 10.9 且对网络收敛完全不敏感，意味着模型的最终预测输出在验证阶段**几乎退化成了全零 (All-Zeros) 也就是完全没有输出位移响应**。EPE 则仅仅计算了与全零预测对比产生的 Ground Truth 的自身范数。
- 由于 `train.log` 已经表明 `epoch_mode=full_pass`且 Loss 大幅下降，模型在训练前向时是可以正常拟合位移场的。这指向了一个经典结论：**特定子网的评估前向推理出现了因为模式切换而导致的数值崩溃**。在权值共享的 NAS 训练中，最常见的元凶是：**Batch Normalization (BN) 运行时统计量污染 (Running Statistics Corruption)**。

### 1.1 致命缺陷分析：BN 重估 (Recalibration) 失效

在权值共享的 Supernet 中，各个子架构（Architecture）共享同一套 BN 层的参数（权重/偏置，以及滑动均值和滑动方差）。但由于每次前向网络连接通道数、深度都会随机改变，网络自动追踪保留的全局 `moving_mean` 和 `moving_variance` 变成了所有子网激活统计量的**杂糅平均值**。因此直接在 `model.eval() ` 模式下做测试，会彻底破坏某一特定抽原子网的特征规范化。

原计划通过调用 `code/engine/bn_recalibration.py` 尝试进行 BN Calibration 规避这一问题，但在代码实现上存在**两个致命级别的硬伤**：

**致命 Bug 1: UPDATE_OPS 遗漏执行**
```python
# bn_recalibration.py 代码摘录
sess.run(forward_fetch, feed_dict={... is_training_ph: True})
```
TensorFlow 1.x 中，仅传入 `is_training=True` 足以让前向计算采用当前 batch 的真实均值与方差，**但并不会去刷新底层存放的 `moving_mean / variance` 变量！！！** 
在 TensorFlow 中，滑动平均的衰减更新运算都被注册在 `tf.GraphKeys.UPDATE_OPS`。而在我们的 `train_supernet_app.py` 中，它们被通过控制依赖 `tf.control_dependencies` 绑定在了反向传播的 `strict_accum_done` 算子上，**根本没有跟 `preds[-1]` 这个获取预测的推理算子强绑定。**
结果就是：重估模块只是跑了几个徒劳的前向，全局存留的 BN 依然完全属于脏状态，一丝未改变。

**致命 Bug 2: 指数移动平均动量 (Momentum) 对刷新速度的限制**
即便我们修正了 Bug 1 传去了 `UPDATE_OPS`，目前的 `bn_recal_batches=8` 也远远不足以洗刷过去的统计。
如果使用默认的 momentum (如 0.9 或 0.999)，更新 8 个批次后，新统计数据的占比大约仅有 $1 - 0.9^8 \approx 56.9\%$，被高度污染的历史均值仍然占据将近一半的影响力，足以导致验证效果崩坏。

---

## 2. 改进方案与修复策略 (Actionable Fixes)

基于上述诊断，提供以下验证评估策略方案：

### 方案 A：采取 Train-Mode Evaluation 评估机制 (非常推荐)

在超网验证领域 (特别是 SPOS，BigNAS)，由于精准修正 BN 代价巨大甚至有时无解，经常通过 **Train Batch Batch-Norm** 作为评估妥协方案。由于模型评测时只需要衡量各子网能力并获得排名（Ranking），这非常适配当前。
只要保证验证时的 Batch Size 不至于太小，即可使用当前 Batch 真实计算的均值方差，彻底屏蔽 `moving_XXX` 脏数据。

**实施方法：**
修改 `code/engine/supernet_trainer.py` 的 `_run_eval_epoch`：
1. **直接移除/注释** 对 `run_bn_recalibration_session` 的任何调用，省掉无谓计算负担。
2. 强制将评估时向计算图传送的标志位改写：
```python
epe_val = sess.run(
    graph_obj["epe"],
    feed_dict={
        graph_obj["input_ph"]: input_batch,
        graph_obj["label_ph"]: label_batch,
        graph_obj["arch_code_ph"]: arch_code,
        # 强制改变：使用训练模式强制让 BN 执行针对单一 batch 大小 (size=32) 的实时均值方差规范化。
        graph_obj["is_training_ph"]: True,
    },
)
```
**此方案优势：** 稳定性极强，且节省评测耗时。直接可让验证 EPE 恢复到预期正常区间 (大概均值能从 10.9 下降到 2.0-4.0 左右范围)。

### 方案 B：严格修复 BN Recalibration (学术标准打法)
如果必须保留 `is_training_ph=False`，则需要做全面整顿。
**实施方法：**
1. 图构建暴露更新 OP：
`"bn_update_ops": tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope="^((?!teacher/).)*")`
2. 重估函数中执行：
`sess.run([forward_fetch, bn_update_ops], feed_dict={...})`
3. 倍级扩大重估数据量：将 YAML 配置文件中的 `eval.bn_recal_batches` 调整至起码 50 或更高。

---

## 3. 其他相关优化观察

### 3.1 教师网络稳定无虞！
蒸馏机制中的教师被固定为推理状态 `is_training_ph=tf.constant(False)`，这意味着教师网络在使用它 Checkpoint 中的高质量 moving statistics，完美避开了超网训练时的全局方差崩溃株连。这是一个正确的实现策略。（唯一提醒的点是蒸馏特征图通道压缩比对采用了绝对值通道最大化，这是兼容错位 NAS 设计最好的 loss 定义，这里无需修改。）

### 3.2 预测量 Scale 的确认 & **真正的终极元凶：缺失的残差累加 (AccumPreds)**
经过对 `MultiScaleResNet_supernet.py` 返回最后一层输出的深入复查：
1. 模型架构在不同尺度输出光流时，其预测目标实际上是**多尺度残差**（正如 `train_step.py` 中的 `_resize_like` 与 `tf.add(flow_accum, flow_pred)` 逻辑所示）。
2. 在原先的测试与超网验证代码中，无论是 `_build_eval_graph` 还是 `_run_eval_epoch`，都**错误地直接提取了 `preds[-1]` 作为最终输出** (`build_epe_metric(pred_tensor=preds[-1], ...)` )。
3. `preds[-1]` 仅仅是最高分辨率 (out_1_1) 分支预测的相对前一个尺度的**微小残差**。因为其数值本身就趋近于全 0，计算 EPE 时相当于全零预测对比 Ground Truth Magnitude，于是完美得到了 10.9 这个固定的背景误差保底值！

**总结性修补**: 
在 `code/engine/eval_step.py` 中引入了 `accumulate_predictions` 通用函数，并在验证图的抽取前，将前三个尺度的多尺度残差真正累加成完整的预测特征图，从而彻底修复了验证阶段无法真实反映网络性能的致命 Bug。

---

## 4. 后续任务分配清单 (Action Items Tracker)

- [x] 本地定位打开 `supernet_trainer.py`，决定采取方案 A 或方案 B。优先应用方案 A 做降配评估。
- [x] 在 `supernet_eval.py` 以及所有的线上离线评估 Wrapper 中追加部署 Train-Mode Validation 防止污染。
- [x] 定位并彻底修复 10.9 保底值的核心元凶：在 EPE 结算前增加完整的 `accumulate_predictions` 操作，确保验证取到的不是单纯的高频残差。
- [x] 应用修改后重启任务，如果能在前 5 个 Epoch 见到 EPE 回落到个位数，则代表问题彻底修复，项目可直接顺利回归既定跑道。

---

## 5. 修复验证与近期结果分析 (Fix Verification & Analysis)

**最新日志观察 (基于包含残差累加的 Epoch 59 ~ 67 结果)**:
1. **EPE 指标破局回落**: 在 Epoch 58 之前不论如何跑都在 `10.94` 的死水一潭，在 Epoch 59 补充了正确的预测累加后，`mean_epe_12` 瞬间猛跌到了 **`5.14`**，并在随后的 Epoch 中稳定探及 **`4.95`**。这代表测试图流场与真实位移场开始对齐！
2. **离散度 (std_epe) 扩大**: `std_epe_12` 从之前的不足 `0.01` 被激活到了 `0.25 ~ 0.31`。这代表在不同的子网络抽出进行验证时，它们因为卷积核大小与深度的差异，产生了**真实的性能分化**，这正是 NAS 评估器应该具备的最重要的辨别力。
3. **排名 (Rank) 逻辑稳定凸显**: 
   - 观察评估池的排名，`Arch 2` 等较重的模型常年稳居头部（EPE 可达 `4.58`）。
   - 被设计为评估池绝对基线的极小模型 `Arch 0`（全 3x3，极浅分支）常年雷打不动垫底（EPE 徘徊在 `5.5` 左右）。
   - 这标志着体系的“判别器”已完全走上正轨。

**下一步操作与调整建议 (Next Steps)**:
1. **调低在线评估频率，换取计算时间 (Restore Eval Frequency)**: 
   因为为了定位 Debug 才把参数设成了 `--eval_every_epoch 1`。现在一切已经拨云见日，每个 Epoch 都去算评价非常拖慢主干进程。强烈建议断开当前训练，将指令中的该参数恢复成 `--eval_every_epoch 5`，顺位接续恢复进度。
2. **让 Cosine 衰减与算力自我接管 (Wait for Convergence)**: 
   在当前的中盘阶段（Epoch 60 附近），由于学习率 (LR) 仍处于相当饱满的状态 (7.4e-5)，损失与 EPE 会显得在 `5.0` 左右平稳震荡。在余弦退火算法引导下，实质性的“断崖式”收敛和子网精细对齐通常发生在 Epoch 150 以后的学习率快速缩口期。
3. **备战最终子网搜索 (Prepare for Post-Training Search)**: 
   所有的基建障碍已全部移除，不需要再做任何代码干预。请静候最后 200 个 Epoch 跑完，之后即可直接调用 `run_supernet_subnet_distribution.py` 进行 512+ 子网样本的离线 Pareto 前沿搜寻。
