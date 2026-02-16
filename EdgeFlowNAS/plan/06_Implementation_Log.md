# 06 瀹炴柦鏃ュ織

鏇存柊鏃堕棿: 2026-02-16

## Part-01锛圗01 + E02 + Gate-1 鍩虹鍏ュ彛锛?
### 鏈儴鍒嗙洰鏍?
1. 寤虹珛宸ョ▼绾х洰褰曞垎灞傞鏋躲€?2. 瀹屾垚 supernet 璁粌鍏ュ彛 `wrapper -> app -> engine` 鏈€灏忛棴鐜€?3. 瀹屾垚 `arch_codec` 鑷祴鑳藉姏銆?4. 鍥哄寲閰嶇疆鏂囦欢涓庤鍒欐枃浠躲€?
### 浠ｇ爜鏀瑰姩娓呭崟

1. 鏂板鐩綍锛歚wrappers/`, `configs/`, `code/*`, `tests/*`, `outputs/supernet`, `scripts/hpc`銆?2. 鏂板鍏ュ彛锛歚wrappers/run_supernet_train.py`銆?3. 鏂板缂栨帓锛歚code/app/train_supernet_app.py`銆?4. 鏂板鎵ц灞傞鏋讹細`code/engine/supernet_trainer.py` 涓庣浉鍏冲瓙妯″潡銆?5. 鏂板 NAS 鏍稿績锛歚code/nas/arch_codec.py`銆乣code/nas/fair_sampler.py`銆?6. 鏂板宸ュ叿锛歚code/utils/*.py`锛堟棩蹇椼€佽矾寰勩€乯son銆乻eed銆乵anifest銆佹鏌ヨ剼鏈級銆?7. 鏂板閰嶇疆锛歚configs/supernet_fc2_180x240.yaml`銆乣configs/layering_rules.yaml`銆?
### 楠屾敹鍛戒护

1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.nas.arch_codec --self_test`

### 缁撴灉璁板綍

1. 鐘舵€侊細宸插畬鎴愬苟閫氳繃鑷銆?2. 鍛戒护缁撴灉锛?- `python wrappers/run_supernet_train.py --help` 閫氳繃銆?- `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run` 閫氳繃銆?- `python -m code.nas.arch_codec --self_test` 閫氳繃銆?- `python -m code.nas.model_shape_check --h 180 --w 240 --batch 2 --samples 5` 閫氳繃銆?- `python -m code.nas.fair_sampler --cycles 300 --seed 42 --check` 閫氳繃銆?- `python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 8` 閫氳繃銆?- `python -m code.utils.check_comments --root code --strict` 閫氳繃銆?- `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml` 閫氳繃銆?3. 澶囨敞锛氬綋鍓嶈缁冪幆鑺備负鍙繍琛屽崰浣嶉鏋讹紝涓嬩竴閮ㄥ垎浼氶€愭鏇挎崲涓虹湡瀹炶缁冮€昏緫銆?
## Part-02锛圗04 澧炲己 + 璁粌鍏钩璁℃暟瀹炶锛?
### 鏈儴鍒嗙洰鏍?
1. 灏嗛獙璇佹睜浠庨殢鏈哄崰浣嶅崌绾т负鍒嗗眰瑕嗙洊鏋勫缓銆?2. 灏嗗叕骞宠鏁颁粠闈欐€佸崰浣嶅崌绾т负鎸夊叕骞冲懆鏈熺湡瀹炵疮璁°€?3. 璁╄缁冩姤鍛婁笌璇勪及浜х墿鍖呭惈瑕嗙洊鐘舵€佷笌鍏钩缁熻銆?
### 浠ｇ爜鏀瑰姩娓呭崟

1. 鏇存柊 `code/nas/eval_pool_builder.py`锛?- 澧炲姞鍩虹瑕嗙洊缂栫爜绛栫暐銆?- 澧炲姞 `check_eval_pool_coverage` 瑕嗙洊妫€鏌ュ嚱鏁般€?- 澧炲姞鍙洿鎺ュ懡浠よ妫€鏌ョ殑 `--check` 鍏ュ彛銆?2. 鏇存柊 `code/nas/supernet_eval.py`锛?- 璇勪及杈撳嚭鍐欏叆瑕嗙洊妫€鏌ョ粨鏋溿€?3. 鏇存柊 `code/engine/supernet_trainer.py`锛?- 鎺ュ叆 `generate_fair_cycle` 鍛ㄦ湡閲囨牱骞剁疮绉叕骞宠鏁般€?- 鎺ュ叆鍒嗗眰楠岃瘉姹犳瀯寤轰笌瑕嗙洊妫€鏌ュ啓鐩樸€?- 鍦ㄨ缁冩姤鍛婇噷璁板綍 `final_fairness_gap` 涓?`eval_pool_coverage_ok`銆?4. 鏇存柊 `code/engine/train_step.py`锛?- 鎺ユ敹 `cycle_codes` 杈撳叆锛屾樉寮忚〃杈?3-path 鍛ㄦ湡璇箟銆?
### 楠屾敹鍛戒护

1. `python -m code.utils.check_comments --root code --strict`
2. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
3. `python -m code.nas.eval_pool_builder --seed 42 --size 12 --check`
4. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 3 --fast_mode`
5. `python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 8`

### 缁撴灉璁板綍

1. 鐘舵€侊細宸插畬鎴愬苟閫氳繃鑷銆?2. 瑕嗙洊妫€鏌ワ細`coverage.ok = true`銆?3. 鍏钩璁℃暟锛氳缁冪煭璺戜腑 `fairness_gap=0`銆?
## Part-03锛堢湡瀹?TF1 Supernet 鍥句笌涓ユ牸鍏钩璁粌姝ワ級

### 鏈儴鍒嗙洰鏍?
1. 灏嗗崰浣嶇綉缁滄浛鎹负鐪熷疄 TF1 Bilinear supernet 鍥俱€?2. 灏嗗崰浣嶈缁冨惊鐜浛鎹负鐪熷疄 `3-path same-batch` 姊害绱Н鏇存柊銆?3. 灏嗘暟鎹眰鏇挎崲涓?FC2 閲囨牱鍣紙甯﹀悎鎴愭暟鎹厹搴曪級銆?4. 鍦?`tf_work_hpc` 鐜瀹屾垚鍙繍琛岀殑 1 epoch 鐭窇楠岃瘉銆?
### 浠ｇ爜鏀瑰姩娓呭崟

1. 缃戠粶灞傦細
- 鏂板 `code/network/decorators.py` 涓?`code/network/base_layers.py`銆?- 閲嶅啓 `code/network/MultiScaleResNet_supernet.py`锛?  - 4 涓?backbone 娣卞害閫夋嫨鍧楋紙Deep1/2/3锛夈€?  - 5 涓?head 鍗风Н鏍搁€夋嫨鍧楋紙7/5/3锛夈€?  - 杈撳嚭灏哄害鍥哄畾涓?`45x60 / 90x120 / 180x240`銆?2. 鏁版嵁灞傦細
- 閲嶅啓 `code/data/fc2_dataset.py`锛氭敮鎸?`FC2_dirnames + split index` 瑙ｆ瀽銆乣.flo` 璇诲彇銆侀殢鏈鸿鍓€佺己渚濊禆鏃跺悎鎴愭牱鏈厹搴曘€?- 閲嶅啓 `code/data/dataloader_builder.py`锛氭瀯寤?train/val provider銆?- 閲嶅啓 `code/data/transforms_180x240.py`锛氳緭鍏ユ爣鍑嗗寲鍒?`[-1,1]`銆?3. 璁粌鎵ц灞傦細
- 閲嶅啓 `code/engine/train_step.py`锛氬灏哄害 L1 鎹熷け涓庢潈閲嶈“鍑忔嫾鎺ャ€?- 閲嶅啓 `code/engine/eval_step.py`锛欵PE 鎸囨爣鏋勫缓銆?- 閲嶅啓 `code/engine/bn_recalibration.py`锛氫細璇濈骇 BN 閲嶄及娴佺▼銆?- 閲嶅啓 `code/engine/supernet_trainer.py`锛氱湡瀹?TF1 鍥炬瀯寤恒€佷弗鏍煎叕骞崇疮绉洿鏂般€佸浐瀹?eval pool 璇勪及銆佹棭鍋?鏃ュ織/artifact 杈撳嚭銆?4. 鍏ュ彛涓庨厤缃細
- 鏇存柊 `code/app/train_supernet_app.py`锛氬欢杩熷鍏ヨ缁冩ā鍧楋紝鏀寔鏃?`PyYAML` 鐜鐨勮交閲?YAML 瑙ｆ瀽銆?- 鏇存柊 `wrappers/run_supernet_train.py`锛氭柊澧?`--steps_per_epoch`銆乣--base_path`銆?- 鏇存柊 `configs/supernet_fc2_180x240.yaml`锛氳ˉ鍏?`steps_per_epoch/base_path/train_list_name/allow_synthetic_fallback`銆?5. 楠岃瘉宸ュ叿锛?- 閲嶅啓 `code/nas/model_shape_check.py`锛屾敼涓虹湡瀹?TF1 鍥炬帹鐞嗘鏌ャ€?
### 楠屾敹鍛戒护

1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.utils.check_comments --root code --strict`
4. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
5. `conda run -n tf_work_hpc python -m code.nas.model_shape_check --h 180 --w 240 --batch 2 --samples 2`
6. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 1 --steps_per_epoch 1 --batch_size 2 --fast_mode`

### 缁撴灉璁板綍

1. 鐘舵€侊細宸插畬鎴愬苟閫氳繃鑷銆?2. 褰㈢姸妫€鏌ワ細`out_1_4=45x60`锛宍out_1_2=90x120`锛宍out_1_1=180x240`銆?3. 璁粌鐭窇锛歚tf_work_hpc` 鐜瀹屾垚 `1 epoch`锛屾棩蹇楁樉绀轰弗鏍煎叕骞冲惊鐜笌璇勪及杈撳嚭姝ｅ父銆?
## Part-03B（Checkpoint真实化 + Supernet评估真实化）

### 本部分目标
1. 将 `checkpoint_manager` 从占位JSON改为真实 TF1 Saver 保存/恢复。
2. 打通 `--load_checkpoint` 与 `--resume_experiment_name` 的恢复训练链路。
3. 将 `code/nas/supernet_eval.py` 从占位脚本改为真实 checkpoint 评估。
4. 补齐 BN 重估阶段输入标准化，保证与训练输入一致。

### 代码改动清单
1. 重写 `code/engine/checkpoint_manager.py`：
- 新增 `checkpoint_exists`、`find_existing_checkpoint`、`save_checkpoint`、`restore_checkpoint`、`load_checkpoint_meta`。
- 保留 `write_checkpoint_placeholder` 兼容旧接口。
2. 重写 `code/engine/supernet_trainer.py`：
- 接入真实 checkpoint 保存（best/last）与 meta 状态写盘。
- 接入恢复逻辑：`load_checkpoint + resume_experiment_name`。
- 恢复状态包含 `start_epoch/global_step/fairness_counts/best_metric/bad_epochs`。
- 评估池落盘文件名改为 `eval_pool_{size}.json`。
3. 更新 `code/engine/bn_recalibration.py`：
- BN 重估前加入 `standardize_image_tensor`。
4. 重写 `code/nas/supernet_eval.py`：
- 真实 TF1 图构建与 checkpoint 加载。
- 逐子网执行 BN 重估 + EPE 评估。
- 输出 `supernet_eval_result_{checkpoint_type}.json`。
- 支持参数：`--checkpoint_type`、`--batch_size`、`--bn_recal_batches`。

### 验收命令
1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.utils.check_comments --root code --strict`
4. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
5. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 1 --steps_per_epoch 1 --batch_size 2 --fast_mode`
6. `conda run -n tf_work_hpc python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 2 --checkpoint_type last --batch_size 2`
7. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 2 --steps_per_epoch 1 --batch_size 2 --load_checkpoint --resume_experiment_name edgeflownas_supernet_fc2_180x240 --fast_mode`

### 结果记录
1. 状态：已完成并通过自检。
2. `supernet_eval` 已输出真实评估文件：`outputs/supernet/edgeflownas_supernet_fc2_180x240/supernet_eval_result_last.json`。
3. 断点恢复验证通过：日志显示从 `start_epoch=2` 恢复并继续训练。