# EdgeFlowNAS 璁″垝绱㈠紩

鏇存柊鏃堕棿: 2026-02-16

## 鐩爣

鍩轰簬 `EdgeFlowNet` 鐨?Bilinear 楠ㄦ灦锛屽湪 `MCUFlowNet/EdgeFlowNAS` 鍐呮瀯寤?FairNAS 椋庢牸 NAS 鏂规銆傚綋鍓嶉樁娈靛彧鎺ㄨ繘 **Supernet 璁粌**锛屼笉鎺ㄨ繘鎼滅储涓庨噸璁€?
## 宸插畬鎴愰槄璇讳笌瀵归綈

宸查槄璇绘枃妗ｏ細

1. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/00_README.md`
2. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/01_Architecture_Overview.md`
3. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/02_Bottleneck_and_33Decoder.md`
4. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/03_Backbone_Block_Comparison.md`
5. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/04_Head_Upsample_NAS.md`
6. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/05_Block_Type_NAS_Brainstorm.md`
7. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/06_Final_NAS_Strategy.md`
8. `MCUFlowNet/EdgeFlowNet/sramTest/analysis/07_Appendix.md`

宸查槄璇绘簮鐮侊細

1. `MCUFlowNet/EdgeFlowNet/sramTest/network/MultiScaleResNet_bilinear.py`
2. `MCUFlowNet/EdgeFlowNet/sramTest/network/MultiScaleResNet_cell.py`
3. `MCUFlowNet/EdgeFlowNet/sramTest/network/MultiScaleResNet_cell_33decoder.py`
4. `MCUFlowNet/EdgeFlowNet/sramTest/network/MultiScaleResNet_cell_upsample_search.py`
5. `MCUFlowNet/EdgeFlowNet/sramTest/auto_benchmark_all.py`
6. `MCUFlowNet/EdgeFlowNet/wrappers/run_train.py`
7. `MCUFlowNet/EdgeFlowNet/code/train.py`
8. `MCUFlowNet/EdgeFlowNet/code/misc/DataHandling.py`
9. `MCUFlowNet/EdgeFlowNet/code/network/MultiScaleResNet.py`
10. `MCUFlowNet/EdgeFlowNet/code/network/BaseLayers.py`
11. `FairNAS/README.md`
12. `FairNAS/models/FairNAS_A.py`
13. `FairNAS/models/FairNAS_B.py`

## 璁″垝鏂囨。搴忓垪

1. `00_Plan_Index.md`锛氱储寮曚笌杩涘害鎬昏锛堟湰鏂囦欢锛?2. `01_Framework_Options.md`锛氬ぇ妗嗘灦鏂规涓庨杞喅绛栭」锛堝凡瀹屾垚锛?3. `02_Supernet_Training_Spec.md`锛氳秴缃戣缁冧笓椤硅鏍硷紙宸插畬鎴愶紝涓昏鏍兼枃妗ｏ級
4. `03_Supernet_Task_Breakdown.md`锛氫换鍔℃媶鍒嗕笌楠屾敹鏍囧噯锛堝凡瀹屾垚锛?5. `04_Supernet_Execution_Gates.md`锛氭寜鎵ц椤哄簭鐨勯棬绂佹竻鍗曚笌娴嬭瘯妯℃澘锛堝凡瀹屾垚锛?6. `05_Engineering_File_Management_and_Code_Style.md`锛氬伐绋嬬骇鏂囦欢绠＄悊涓庝唬鐮佹敞閲婅鑼冿紙宸插畬鎴愶級

## 褰撳墠閿佸畾鍐崇瓥

1. 瓒呯綉鏂规硶锛歋trict Fairness One-Shot锛?-path same-batch 绱Н鍚庡崟娆℃洿鏂帮級銆?2. 璁粌鏍堬細TF1 鍏煎閾捐矾锛屼唬鐮佽惤鍦板湪 `EdgeFlowNAS`銆?3. 鏁版嵁涓庡垎杈ㄧ巼锛欶C2锛岄殢鏈鸿鍓埌 `180x240`銆?4. 楠岃瘉鏉ユ簮锛歚FC2_test`锛堜粎鐢ㄤ簬 supernet 鎺掑悕闃舵锛夈€?5. 浼樺寲鍣細`Adam + cosine decay`銆?6. 姝ｅ垯涓庣ǔ瀹氬寲锛歚weight_decay=4e-5`锛宍grad_clip_global_norm=5.0`銆?7. BN锛氬叡浜?BN锛岃瘎浼板墠閲嶄及 `8 batches`銆?8. 楠岃瘉姹狅細鍥哄畾 `12` 瀛愮綉锛屽垎灞傝鐩栭噰鏍枫€?9. 璇勪及涓庡仠姝細姣?epoch 璇勪及锛宍patience=15`锛宍min_delta=0.002`銆?10. 棰勭畻涓庡鐜帮細`200 epochs`锛屽崟绉嶅瓙鍥哄畾銆?
## 褰撳墠鐘舵€?
1. 宸插畬鎴愶細璁″垝妗嗘灦銆佽秴缃戣鏍笺€佷换鍔℃媶鍒嗐€侀獙鏀堕棬绂佹ā鏉裤€佸伐绋嬬骇鏂囦欢绠＄悊瑙勮寖銆?2. 涓嬩竴姝ワ細鍏堟寜 `05` 瀹屾垚鐩綍涓庡嚱鏁版媶鍒嗭紝鍐嶆寜 `04` 鐨?Gate-1 鍒?Gate-4 浠ｇ爜钀藉湴涓庢湰鏈?smoke銆?
## 补充文档（新增）

1. `06_Implementation_Log.md`：分阶段实现与验收日志。
2. `07_User_Manual.md`：Supernet 训练与评估详细使用手册。

## 当前状态（更新）

1. Gate-1 ~ Gate-4 已完成代码实现与本机验收。
2. 已补齐产物校验工具与 HPC 模板脚本。
3. 下一阶段可进入 NAS 搜索与候选重训规划（不在当前范围内）。