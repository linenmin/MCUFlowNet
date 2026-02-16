# 05 工程级文件管理与代码注释规范（仅 Supernet）

更新时间: 2026-02-16

## 1. 目标

把后续实现从“能跑”升级到“工程可维护”。  
本规范直接约束 `EdgeFlowNAS` 内 Supernet 训练代码的目录结构、函数分层、命名方式、注释风格和验收标准。

## 2. 硬性要求（本轮锁定）

1. 函数按类别拆分到不同脚本、不同文件夹，禁止把多类职责堆在单文件。
2. 代码保持简洁易读，单函数只做一件事。
3. 新增代码采用简单中文注释，默认“每行代码都有中文注释”（导入行、空行、纯括号换行除外）。
4. 当前只覆盖 Supernet 训练链路，不扩展到搜索与重训。

## 3. 目录分层（目标结构）

```text
MCUFlowNet/EdgeFlowNAS/
  wrappers/
    run_supernet_train.py
  configs/
    supernet_fc2_180x240.yaml
    layering_rules.yaml
  code/
    app/
      train_supernet_app.py
    network/
      MultiScaleResNet_supernet.py
      backbone_blocks.py
      head_blocks.py
    nas/
      arch_codec.py
      fair_sampler.py
      eval_pool_builder.py
      supernet_eval.py
    data/
      fc2_dataset.py
      transforms_180x240.py
      dataloader_builder.py
    engine/
      supernet_trainer.py
      train_step.py
      eval_step.py
      bn_recalibration.py
      early_stop.py
      checkpoint_manager.py
    optim/
      optimizer_builder.py
      lr_scheduler.py
      grad_clip.py
    utils/
      seed.py
      logger.py
      manifest.py
      json_io.py
      path_utils.py
      check_comments.py
      check_layering.py
  tests/
    unit/
    integration/
    smoke/
  outputs/
    supernet/
  scripts/
    hpc/
```

## 4. 分层职责与导入边界

### 4.1 层职责

1. `wrappers/`：CLI 参数接入、命令转发，不写训练逻辑。
2. `code/app/`：应用编排层，负责把配置装配成训练流程。
3. `code/network/`：模型结构层，只放网络与 block 实现。
4. `code/nas/`：架构编码、公平采样、验证子网池、超网评估。
5. `code/data/`：数据集读取、裁剪/增强、dataloader 构建。
6. `code/engine/`：训练/验证执行层（step 逻辑、早停、ckpt、BN 重估）。
7. `code/optim/`：优化器、学习率调度、梯度裁剪。
8. `code/utils/`：通用工具（日志、随机种子、清单、静态检查脚本）。
9. `tests/`：单测、集成、smoke。

### 4.2 导入规则（硬约束）

1. `wrappers -> code/app`（允许）。
2. `code/app -> network/nas/data/engine/optim/utils`（允许）。
3. `engine -> network/nas/data/optim/utils`（允许）。
4. `network` 禁止导入 `engine/app/wrappers`。
5. `nas` 禁止导入 `engine/app/wrappers`。
6. `utils` 禁止导入业务层（`network/nas/engine/data/app`）。

## 5. 函数级管理规范

1. 每个文件聚焦单类别职责，建议文件行数 `<= 300`。
2. 每个函数建议行数 `<= 60`，复杂逻辑拆为私有辅助函数。
3. 函数命名使用动宾结构，如 `build_eval_pool`、`run_train_step`。
4. 禁止“万能函数”与跨层直接调用（例如在 `network` 里写 checkpoint）。
5. 共享常量集中到同层 `constants` 区域，避免散落硬编码。

## 6. 注释规范（中文，按逐行可读）

### 6.1 基本规则

1. 可执行语句默认逐行中文注释（导入行、空行、纯括号换行除外）。
2. 控制流分支、张量形状变化、损失聚合、优化器更新必须注释。
3. 注释只写“这行做什么”，不写空话。
4. 优先行尾短注释；必要时用上一行注释说明意图。

### 6.2 函数注释模板

1. 每个函数顶部使用简短中文 docstring，包含：输入、输出、副作用。
2. 示例：

```python
def build_eval_pool(seed, size):  # 定义构建验证子网池函数
    """根据随机种子生成固定验证子网池。"""  # 说明函数用途
    rng = random.Random(seed)  # 创建可复现随机数发生器
    pool = []  # 初始化子网池列表
    for _ in range(size):  # 按目标数量循环生成
        code = [rng.randint(0, 2) for _ in range(9)]  # 生成9维架构编码
        pool.append(code)  # 将编码加入子网池
    return pool  # 返回固定子网池
```

## 7. 执行任务拆分（新增 E 系列）

### E01 目录骨架重构

目标：

1. 建立第 3 节目标目录。
2. 把已有文件归位到对应层。

验收：

1. `rg --files MCUFlowNet/EdgeFlowNAS/code` 可见分层目录。
2. 不再出现“训练主逻辑 + 模型定义”同文件混放。

### E02 训练入口与编排解耦

目标：

1. `wrappers/run_supernet_train.py` 只保留参数解析和入口调用。
2. 训练编排下沉到 `code/app/train_supernet_app.py`。

验收：

1. `python wrappers/run_supernet_train.py --help` 正常。
2. wrapper 文件内无训练循环代码。

### E03 训练执行层拆分

目标：

1. 训练 step、评估 step、BN 重估、早停、ckpt 分别独立文件。

验收：

1. `code/engine/` 下至少包含 `train_step.py`、`eval_step.py`、`bn_recalibration.py`、`early_stop.py`、`checkpoint_manager.py`。
2. 任一文件删除后应出现明确功能缺失，说明职责边界清晰。

### E04 NAS 层拆分

目标：

1. `arch_codec`、`fair_sampler`、`eval_pool_builder`、`supernet_eval` 各自独立。

验收：

1. `python -m code.nas.fair_sampler --cycles 30 --seed 42 --check` 正常。
2. `python -m code.nas.arch_codec --self_test` 正常。

### E05 注释一致性检查

目标：

1. 增加注释检查脚本，检查新增代码是否满足中文可读注释要求。

验收：

1. `python -m code.utils.check_comments --root code --strict` 通过。
2. 报告中无“可执行语句缺中文注释”条目。

### E06 分层导入检查

目标：

1. 增加分层检查脚本，防止越层导入。

验收：

1. `python -m code.utils.check_layering --rules configs/layering_rules.yaml` 通过。
2. 报告中无 `network -> engine`、`utils -> business` 违规导入。

## 8. 门禁对齐（与 04 的关系）

1. `05` 是工程规范门禁，优先级高于 `04` 的实现门禁。
2. 执行顺序建议：`E01~E06` 完成后，再进入 `04` 的 Gate-1。
3. 若 `05` 与 `04` 冲突，以“分层清晰、可维护性优先”原则调整 `04` 中文件落点。

## 9. 完成定义（本文件）

1. 目录分层落地。
2. 函数分类落地。
3. 注释规范落地。
4. 注释检查与分层检查脚本可运行。
5. 再进入 Supernet 训练功能实现与调试。
