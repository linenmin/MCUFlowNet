# Grove分辨率与量化精度：GROVE-INT8-01

目标是先核对转换与算子兼容性，再找到固定宽高比和内存预算下的编译上限，最后比较相同分辨率的浮点/INT8 Sintel EPE。编译上限不直接称为板上最大部署能力。

## 当前已验证的入口

`tools/baselines/export_grove.py`恢复公开权重，使用固定64对FC2训练图像校准，保存浮点和严格INT8 TFLite，再运行Vela4.5.0/U55-64的Size与Performance。TF模型沿用已有WSL容器；输出目录必须不存在。脚本参数包括模型、权重、上游目录、校准目录、输入高宽、Vela配置/安装目录及输出目录。只删除推理不用的不确定度通道或Nano辅助输出，验证浮点结果，不改变光流网络结构。

2026-09-21预检：Edge156×208的Size峰值1430KiB，与论文峰值相符；旧S/L172×224均1386KiB；Nano112×160为210KiB。四项均严格INT8，CPU回退0；这是四个单点，不是最大尺寸。分别输出160×208、176×224和28×40，后续评分必须明确处理输出尺寸与向量单位。

预检使用现有arena_cache_size1730150配置。论文绘图用1.4×1024KiB，旧Edge配置还存在2MiB；真实固件精确arena未找到。正式扫描统一宽÷高1.3–1.4、宽高为16的倍数，预算1.4 MiB（向下取整为1,468,006字节）；不能把芯片总SRAM当可用arena。保留每个尺寸/调度的原始结果和上下边界，量化EPE不用于倒推或挑选标签单位。

## 独立转换环境

用户已批准安装。Windows环境为`C:/00Work/Envs/grove-convert`，从sintel-torch克隆，在副本安装requirements-grove-convert.txt。Conda克隆使用Windows反斜杠绝对路径，并指定`--override-channels -c conda-forge --offline`，避免默认渠道条款交互。未修改原EPE/FPS环境；原环境NumPy2.4.6与Torch2.11.0+cu128核对未变。

新环境pip check通过，已验证ONNX小型卷积转换到浮点及完整INT8 TFLite。真实模型用export_grove_torch.py转换，再用audit_grove_torch.py独立检查保存的产物。检查浮点预测一致性、浮点回退和CPU算子；不能把生成TFLite文件当作部署成功。详细结果保留在实验总表。

## 结果边界与记录

原始输出、环境冻结、安装/失败日志及环境小模型测试均在`C:/00Work/Runs/MCUFlowNet/GROVE-INT8-01`。wiki实验总表记录进度，Benchmark总表只收验收后的比较结果。先用scan_grove.py找经过64对校准的候选边界，再用scan_grove_frozen.py冻结该模型的权重和量化参数、逐一编译全部更大候选。resize_grove_tflite.py只调整静态空间尺寸及对应常量；四模型均与重新导出的浮点图比对，抽样最大差为0。若扫描出现更大的通过尺寸，须重新用64对校准并验收，不能直接使用筛查图报告精度。evaluate_grove.py在选定尺寸成对评测浮点与INT8。未刷写板卡。

评测协议使用Sintel Final全部1041对、共同416×1024原始GT坐标，同时报告低分辨率浮点和INT8 EPE。先恢复实际输出网格，再按已核实的向量单位转换；Nano单位疑问仍然存在。普通CPU TFLite解释器可评测编译前INT8模型，不能运行Vela Ethos-U命令流，也不代表实际板上输出已验证。


## 范围与表格含义

本轮优先检查EdgeFlowNet Full、旧MCUFlowNet S/L、NanoFlowNet，以及SPyNet、FastFlowNet、RAFT-Small、NeuFlow v2、RAPIDFlow。EdgeFlowNet Chunking的整帧缓存和分块调度需要固件实现，不能把单块上限简单乘四当作整帧可部署尺寸。没有公开权重的Ajna不填写虚构结果。

宽高比约束针对输入，Sintel共同评分区域仍是416×1024。输入缩放保留整个区域，会分别压缩横向和纵向；恢复光流时对应乘回1024/W与416/H。INT8使用训练后量化，未做量化感知训练。不同模型各自最大尺寸用于回答部署能力，不是同尺寸结构消融。

运行输出保留在Runs，wiki只保存精简结果、失败原因和证据指纹。完整INT8、零CPU回退、预算内编译是本轮可完成的筛查条件；未完成Grove固件适配、板上SRAM分配或板上输出核验。CPU回退或转换失败不能推导出模型在所有实现中都不可部署。
