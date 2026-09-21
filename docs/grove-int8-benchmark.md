# Grove分辨率与量化精度：GROVE-INT8-01

目标是先核对转换与算子兼容性，再找到固定宽高比和内存预算下的编译上限，最后比较相同分辨率的浮点/INT8 Sintel EPE。编译上限不直接称为板上最大部署能力。

## 当前已验证的入口

`tools/baselines/export_grove.py`恢复公开权重，使用固定64对FC2训练图像校准，保存浮点和严格INT8 TFLite，再运行Vela4.5.0/U55-64的Size与Performance。TF模型沿用已有WSL容器；输出目录必须不存在。脚本参数包括模型、权重、上游目录、校准目录、输入高宽、Vela配置/安装目录及输出目录。只删除推理不用的不确定度通道或Nano辅助输出，验证浮点结果，不改变光流网络结构。

2026-09-21预检：Edge156×208的Size峰值1430KiB，与论文峰值相符；旧S/L172×224均1386KiB；Nano112×160为210KiB。四项均严格INT8，CPU回退0；这是四个单点，不是最大尺寸。分别输出160×208、176×224和28×40，后续评分必须明确处理输出尺寸与向量单位。

预检使用现有arena_cache_size1730150配置。论文绘图用1.4×1024KiB，旧Edge配置还存在2MiB；真实固件精确arena未找到。正式扫描统一宽÷高1.3–1.4、宽高为16的倍数，预算1.4 MiB（向下取整为1,468,006字节）；不能把芯片总SRAM当可用arena。保留每个尺寸/调度的原始结果和上下边界，量化EPE不用于倒推或挑选标签单位。

## 独立转换环境

用户已批准安装。Windows环境为`C:/00Work/Envs/grove-convert`，从sintel-torch克隆，在副本安装requirements-grove-convert.txt。Conda克隆使用Windows反斜杠绝对路径，并指定`--override-channels -c conda-forge --offline`，避免默认渠道条款交互。未修改原EPE/FPS环境；原环境NumPy2.4.6与Torch2.11.0+cu128核对未变。

新环境pip check通过，已验证ONNX小型卷积转换到浮点及完整INT8 TFLite。还未证明真实PyTorch光流网络转换成功；GridSample、相关性、动态索引等需要逐模型核验，不能把转换工具支持的算子等同于NPU支持。

## 结果边界与记录

原始输出、环境冻结、安装/失败日志及环境小模型测试均在`C:/00Work/Runs/MCUFlowNet/GROVE-INT8-01`。wiki实验总表记录进度，Benchmark总表只收验收后的比较结果。扫描入口为scan_grove.py（--exhaustive检查全部更大候选）；evaluate_grove.py在相同部署尺寸成对评测浮点与INT8。未刷写板卡。

后续评测使用Sintel Final全部1041对、共同416×1024原始GT坐标，同时报告低分辨率浮点和INT8 EPE。先恢复实际输出网格，再按已核实的向量单位转换；Nano单位疑问仍然存在。普通CPU TFLite解释器可评测编译前INT8模型，不能运行Vela Ethos-U命令流，也不代表实际板上输出已验证。
