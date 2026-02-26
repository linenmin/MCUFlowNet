# 16 搜索架构单模型重训计划 (Standalone Retrain Plan)

**文档编号**: 16
**更新时间**: 2026-02-26
**目标**: 基于 NAS 搜索到的优选架构码，从零开始独立训练固定架构的单模型，使其达到最佳性能后导出为 int8 量化模型部署到板上。

---

## 0. 已确认决策清单

| 决策项         | 选择                     | 说明                              |
| -------------- | ------------------------ | --------------------------------- |
| 目标架构码     | `0,2,1,1,0,0,1,0,1`      | NAS 搜索最优                      |
| 基准对比架构码 | `0,0,0,0,2,1,2,2,2`      | 最接近原作 EdgeFlowNet 基准的架构 |
| 量化策略       | 浮点训练 + PTQ           | 标准流程，如 PTQ 损失过大再加 QAT |
| 蒸馏           | 先不蒸馏                 | 400 epoch 足够，结果更干净        |
| 输入分辨率     | 180×240                  | 与超网搜索/Vela 评估对齐          |
| LR 调度        | Cosine Decay             | 1e-4 → ~1e-6                      |
| Batch Size     | 32                       | 与超网一致                        |
| 评估频率       | 每 5 epoch               | 训练中定期评估 val EPE            |
| 训练阶段       | 第一阶段: FC2 400 epochs | 后续可接 FlyingThings 50 epochs   |
| Checkpoint     | 仅每 epoch 保存          | 不做 step 级保存                  |

---

## 1. 原作基准模型训练细节

从 [run_train.py](file:///d:/Dataset/MCUFlowNet/EdgeFlowNet/wrappers/run_train.py) 和 [train.py](file:///d:/Dataset/MCUFlowNet/EdgeFlowNet/code/train.py) 提取：

| 参数            | 原作值                          | 本次选择                   | 是否改动     |
| --------------- | ------------------------------- | -------------------------- | ------------ |
| 优化器          | Adam (β1=0.9, β2=0.999, ε=1e-8) | **相同**                   | ✅ 照搬       |
| 学习率          | 常数 1e-4                       | **Cosine Decay 1e-4→1e-6** | ❌ 改进       |
| Batch Size      | 16                              | **32**                     | ❌ 改大       |
| Epochs          | 400                             | **400**                    | ✅ 照搬       |
| 输入尺寸        | 352×480                         | **180×240**                | ❌ 与超网对齐 |
| 输入归一化      | 原始 float32                    | **[-1,1] 归一化**          | ❌ 与超网对齐 |
| Loss 函数       | MultiscaleUncertainty           | **相同**                   | ✅ 照搬       |
| 数据增强        | 随机裁剪                        | **随机裁剪**               | ✅ 照搬       |
| Weight Decay    | 无                              | **无**                     | ✅ 照搬       |
| 梯度裁剪        | 无                              | **无**                     | ✅ 照搬       |
| 训练中评估      | **无**                          | **每 5 epoch val EPE**     | ❌ 新增       |
| Checkpoint 策略 | 每 epoch + 每 N step            | **仅每 epoch + best**      | ❌ 简化       |

> [!NOTE]
> 原作 400 epoch 是第一阶段（FC2），后续还有 FlyingThings 50 epochs 的第二阶段微调。因此原作训练循环中没有 val 评估——但我们在第一阶段可以加评估来选 best checkpoint。

---

## 2. 多模型并行训练方案 ⭐

### 2.1 目标

在单张 V100 上**同时训练两个固定架构模型**，共享完全相同的数据流：
- **模型 A（目标）**: `0,2,1,1,0,0,1,0,1`
- **模型 B（基准）**: `0,0,0,0,2,1,2,2,2`

这样你在训练过程中就能实时对比两个架构的 loss 和 val EPE，直观看出目标架构是否优于基准。

### 2.2 方案对比

| 方案                | 做法                                                                      | 优点                               | 缺点                                               |
| ------------------- | ------------------------------------------------------------------------- | ---------------------------------- | -------------------------------------------------- |
| **A: 单进程双图** ⭐ | 同一 TF Session 里构建两个模型图（不同 `variable_scope`），共享输入 batch | 完全相同数据、完美可比、单进程管理 | 略多工程量                                         |
| B: 两个独立进程     | 分别启动两个训练进程                                                      | 代码最简单                         | 数据顺序不同、GPU 显存分配可能冲突、不方便实时对比 |
| C: 交替训练         | 奇数 step 训练 A，偶数 step 训练 B                                        | 数据对齐、显存可控                 | 2× 训练时间、epoch 内逻辑复杂                      |

**推荐方案 A：单进程双图**

### 2.3 方案 A 技术细节

```python
# 伪代码示意
with tf.variable_scope("model_a"):
    model_a = MultiScaleResNetSupernet(input_ph, arch_code_a, ...)
    preds_a = model_a.build()
    loss_a = build_multiscale_uncertainty_loss(preds_a, label_ph, ...)
    optimizer_a = Adam(lr_ph).minimize(loss_a)

with tf.variable_scope("model_b"):
    model_b = MultiScaleResNetSupernet(input_ph, arch_code_b, ...)
    preds_b = model_b.build()
    loss_b = build_multiscale_uncertainty_loss(preds_b, label_ph, ...)
    optimizer_b = Adam(lr_ph).minimize(loss_b)
```

核心特征：
1. **共享输入 placeholder**: `input_ph` 和 `label_ph` 只有一份，两模型看到完全相同的样本
2. **独立变量作用域**: `model_a/` 和 `model_b/` 互不干扰
3. **独立优化器**: 各自有独立的 Adam 状态（momentum/variance）
4. **独立 BN**: 各自维护独立的 moving_mean / moving_variance（因为 `variable_scope` 不同）
5. **独立 Saver**: 分别保存/加载各自的 checkpoint

### 2.4 显存估算

| 项目                            | 单模型      | 双模型       |
| ------------------------------- | ----------- | ------------ |
| 模型参数 (~100K params)         | ~0.4 MB     | ~0.8 MB      |
| Adam 优化器状态 (2×参数)        | ~0.8 MB     | ~1.6 MB      |
| 输入 batch (32×180×240×6, FP32) | ~32 MB      | 32 MB (共享) |
| 中间激活 (前向+后向)            | ~200 MB     | ~400 MB      |
| **合计**                        | **~233 MB** | **~434 MB**  |

> [!TIP]
> V100 32GB 显存，双模型总占用不到 500 MB，完全没有显存压力。即使 V100 16GB 也绰绰有余。

### 2.5 训练循环（双模型版）

```
for epoch in range(400):
    for step in range(steps_per_epoch):
        input_batch, label_batch = data_provider.next_batch(32)
        input_batch = standardize(input_batch)  # [-1,1]

        # 两个模型同时前向+后向
        loss_a, loss_b, _, _ = sess.run(
            [loss_a_tensor, loss_b_tensor, train_op_a, train_op_b],
            feed_dict={input_ph: input_batch, label_ph: label_batch, lr_ph: current_lr}
        )
        # 记录两个模型的 loss

    if epoch % eval_every_epoch == 0:
        epe_a = evaluate(sess, model_a, val_provider)
        epe_b = evaluate(sess, model_b, val_provider)
        log(f"epoch={epoch} epe_a={epe_a:.4f} epe_b={epe_b:.4f} delta={epe_a-epe_b:.4f}")
```

### 2.6 日志输出格式

```
epoch=1   lr=1.00e-04  loss_a=18.52  loss_b=19.31  eval=skipped
epoch=5   lr=9.98e-05  loss_a=12.34  loss_b=13.01  epe_a=5.12  epe_b=5.89  Δepe=-0.77 ✓
epoch=10  lr=9.94e-05  loss_a=9.87   loss_b=10.52  epe_a=4.56  epe_b=5.23  Δepe=-0.67 ✓
...
epoch=400 lr=1.00e-06  loss_a=3.21   loss_b=3.89   epe_a=2.45  epe_b=3.12  Δepe=-0.67 ✓
```

`Δepe < 0` 表示目标架构优于基准架构。

---

## 3. 代码实现路线

### 3.1 新增文件

| 操作      | 文件路径                                          | 说明                              |
| --------- | ------------------------------------------------- | --------------------------------- |
| **[NEW]** | `EdgeFlowNAS/wrappers/run_standalone_train.py`    | CLI 入口 wrapper                  |
| **[NEW]** | `EdgeFlowNAS/code/engine/standalone_trainer.py`   | 核心训练引擎（支持单/双模型模式） |
| **[NEW]** | `EdgeFlowNAS/configs/standalone_fc2_180x240.yaml` | 默认配置                          |

### 3.2 复用模块

| 模块                                        | 说明                                                    |
| ------------------------------------------- | ------------------------------------------------------- |
| `code/network/MultiScaleResNet_supernet.py` | 传入固定 arch_code 即为单模型                           |
| `code/data/fc2_dataset.py`                  | 数据加载、shuffle_no_replacement                        |
| `code/engine/train_step.py`                 | `build_multiscale_uncertainty_loss`, `add_weight_decay` |
| `code/engine/eval_step.py`                  | `accumulate_predictions`, `build_epe_metric`            |
| `code/optim/lr_scheduler.py`                | `cosine_lr`                                             |
| `code/utils/checkpoint.py`                  | checkpoint 工具                                         |

### 3.3 CLI 参数设计

```bash
python wrappers/run_standalone_train.py \
  --config configs/standalone_fc2_180x240.yaml \
  --arch_codes "0,2,1,1,0,0,1,0,1" \        # 必选，逗号分隔 9 位
  --arch_names "target" \                     # 可选，给架构起名
  --experiment_name retrain_run1 \
  --base_path /data/.../MCUFlowNet \
  --train_dir Datasets/FlyingChairs2/train \
  --val_dir Datasets/FlyingChairs2/val \
  --gpu_device 0 \
  --num_epochs 400 \
  --batch_size 32 \
  --lr 1e-4 \
  --eval_every_epoch 5
```

**双模型模式**：用 `+` 分隔多个架构码：

```bash
python wrappers/run_standalone_train.py \
  --config configs/standalone_fc2_180x240.yaml \
  --arch_codes "0,2,1,1,0,0,1,0,1+0,0,0,0,2,1,2,2,2" \
  --arch_names "target+baseline" \
  --experiment_name retrain_dual_run1 \
  --base_path /data/.../MCUFlowNet \
  --train_dir Datasets/FlyingChairs2/train \
  --val_dir Datasets/FlyingChairs2/val \
  --gpu_device 0 \
  --num_epochs 400 \
  --batch_size 32 \
  --lr 1e-4 \
  --eval_every_epoch 5
```

### 3.4 输出目录结构

```
outputs/standalone/retrain_dual_run1/
├── train.log                     # 统一训练日志
├── run_manifest.json
├── model_target/                 # 目标架构
│   ├── checkpoints/
│   │   ├── last.ckpt.*
│   │   └── best.ckpt.*
│   └── eval_history.csv
├── model_baseline/               # 基准架构
│   ├── checkpoints/
│   │   ├── last.ckpt.*
│   │   └── best.ckpt.*
│   └── eval_history.csv
└── comparison.csv                # 并排对比指标
```

---

## 4. `standalone_trainer.py` 核心伪代码

```python
def train_standalone(config):
    # 1. 解析配置
    arch_codes = parse_arch_codes(config)  # 1 个或多个架构码
    num_models = len(arch_codes)

    # 2. 构建 TF 图
    tf.reset_default_graph()
    input_ph = tf.placeholder(...)  # 共享输入
    label_ph = tf.placeholder(...)
    lr_ph = tf.placeholder(...)
    is_training_ph = tf.placeholder(...)

    models = {}
    for i, (arch_code, arch_name) in enumerate(zip(arch_codes, arch_names)):
        with tf.variable_scope(f"model_{arch_name}"):
            model = MultiScaleResNetSupernet(input_ph, fixed_arch_code, is_training_ph, ...)
            preds = model.build()
            loss = build_multiscale_uncertainty_loss(preds, label_ph, ...)
            train_vars = tf.trainable_variables(scope=f"model_{arch_name}")
            optimizer = Adam(lr_ph).minimize(loss, var_list=train_vars)
            bn_updates = [op for op in UPDATE_OPS if op.name.startswith(f"model_{arch_name}/")]
            epe = build_epe_metric(accumulate_predictions(preds), label_ph, ...)
            saver = tf.train.Saver(var_list=scope_vars)
            models[arch_name] = {loss, optimizer, epe, saver, bn_updates, ...}

    # 3. 训练循环
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            data_provider.start_epoch(shuffle=True)
            for step in range(steps_per_epoch):
                batch = data_provider.next_batch(batch_size)
                lr = cosine_lr(base_lr, global_step, total_steps)

                # 所有模型共享同一 batch
                feed = {input_ph: batch.input, label_ph: batch.label, lr_ph: lr, is_training_ph: True}
                fetch = {}
                for name, m in models.items():
                    fetch[f"loss_{name}"] = m["loss"]
                    fetch[f"train_{name}"] = [m["optimizer"]] + m["bn_updates"]
                results = sess.run(fetch, feed_dict=feed)
                # 记录各模型 loss

            # 评估
            if epoch % eval_every_epoch == 0:
                for name, m in models.items():
                    epe = evaluate(sess, m, val_provider)

            # 保存 checkpoint (每个模型独立)
            for name, m in models.items():
                m["saver"].save(sess, last_ckpt_path)
                if epe < best_epe[name]:
                    m["saver"].save(sess, best_ckpt_path)
```

---

## 5. 验证计划

### 5.0 测试环境说明

| 环境           | Conda 环境                     | 数据集              | 用途                     |
| -------------- | ------------------------------ | ------------------- | ------------------------ |
| **本地主机**   | `conda activate tf_work_hpc`   | **无** (仅 dry run) | 代码逻辑验证、图构建测试 |
| **HPC 服务器** | `tf_work_hpc` (或同等 TF 环境) | FC2 完整数据集      | 正式 400 epoch 训练      |

> [!IMPORTANT]
> 本地主机上没有数据集，只能做 dry run（验证代码能正确解析参数、构建计算图、不报错）。真正的含数据集训练需要在 HPC 上执行。代码调试完成后会提供完整的 HPC 运行指令。

### 5.1 本地 Dry Run（代码写完后立即验证）

```bash
conda activate tf_work_hpc

# 验证 --help 正常输出
python wrappers/run_standalone_train.py --help

# dry run: 验证图构建 (会因为找不到数据集而在数据加载阶段报错，这是预期行为)
python wrappers/run_standalone_train.py \
  --config configs/standalone_fc2_180x240.yaml \
  --arch_codes "0,2,1,1,0,0,1,0,1" \
  --experiment_name dry_run_test \
  --num_epochs 2 --eval_every_epoch 1
```

**Dry Run 验收标准**:
1. `--help` 正常打印参数列表
2. 参数解析和配置合并无报错
3. TF 图构建成功（如果能到达 `Session` 创建阶段即可）

### 5.2 HPC 正式训练（含数据集）

```bash
# 双模型对比训练 (400 epochs)
python wrappers/run_standalone_train.py \
  --config configs/standalone_fc2_180x240.yaml \
  --arch_codes "0,2,1,1,0,0,1,0,1+0,0,0,0,2,1,2,2,2" \
  --arch_names "target+baseline" \
  --experiment_name retrain_dual_run1 \
  --base_path /data/leuven/379/vsc37996/test/MCUFlowNet \
  --train_dir Datasets/FlyingChairs2/train \
  --val_dir Datasets/FlyingChairs2/val \
  --gpu_device 0 \
  --num_epochs 400 \
  --batch_size 32 \
  --lr 1e-4 \
  --eval_every_epoch 5
```

**HPC 验收标准**:
1. 启动无报错，日志正常输出
2. Loss 两个模型都在下降
3. EPE 能计算出有意义的数值（< 15）
4. Checkpoint 正确保存（last + best）
5. 恢复训练能正常继续

### 5.3 训练正确性验证

1. 前 10 epoch loss 曲线合理（快速下降）
2. val EPE 与超网中同架构子网表现接近
3. 400 epoch 完成后 best val EPE 应优于超网中该子网的评估 EPE

### 5.4 导出验证（训练完成后）

```
best.ckpt → Frozen PB → TFLite FP32 → TFLite INT8 (PTQ) → Vela
```

---

## 6. 执行顺序

1. ✅ Plan 审核通过
2. 编写 `standalone_trainer.py`（核心引擎，支持多模型）
3. 编写 `run_standalone_train.py`（CLI wrapper）
4. 编写 `standalone_fc2_180x240.yaml`（配置文件）
5. 本地 dry run（`conda activate tf_work_hpc`，无数据集）
6. 提供 HPC 正式训练指令 → 用户在 HPC 上执行
