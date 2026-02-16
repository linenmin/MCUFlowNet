# 06 鐎圭偞鏌﹂弮銉ョ箶

閺囧瓨鏌婇弮鍫曟？: 2026-02-16

## Part-01閿涘湕01 + E02 + Gate-1 閸╄櫣顢呴崗銉ュ經閿?
### 閺堫剟鍎撮崚鍡欐窗閺?
1. 瀵よ櫣鐝涘銉р柤缁狙呮窗瑜版洖鍨庣仦鍌烆€囬弸韬测偓?2. 鐎瑰本鍨?supernet 鐠侇厾绮岄崗銉ュ經 `wrapper -> app -> engine` 閺堚偓鐏忓繘妫撮悳顖樷偓?3. 鐎瑰本鍨?`arch_codec` 閼奉亝绁撮懗钘夊閵?4. 閸ュ搫瀵查柊宥囩枂閺傚洣娆㈡稉搴ゎ潐閸掓瑦鏋冩禒韬测偓?
### 娴狅絿鐖滈弨鐟板З濞撳懎宕?
1. 閺傛澘顤冮惄顔肩秿閿涙瓪wrappers/`, `configs/`, `code/*`, `tests/*`, `outputs/supernet`, `scripts/hpc`閵?2. 閺傛澘顤冮崗銉ュ經閿涙瓪wrappers/run_supernet_train.py`閵?3. 閺傛澘顤冪紓鏍ㄥ笓閿涙瓪code/app/train_supernet_app.py`閵?4. 閺傛澘顤冮幍褑顢戠仦鍌烆€囬弸璁圭窗`code/engine/supernet_trainer.py` 娑撳海娴夐崗鍐茬摍濡€虫健閵?5. 閺傛澘顤?NAS 閺嶇绺鹃敍姝歝ode/nas/arch_codec.py`閵嗕梗code/nas/fair_sampler.py`閵?6. 閺傛澘顤冨銉ュ徔閿涙瓪code/utils/*.py`閿涘牊妫╄箛妞尖偓浣界熅瀵板嫨鈧汞son閵嗕够eed閵嗕沟anifest閵嗕焦顥呴弻銉ㄥ壖閺堫剨绱氶妴?7. 閺傛澘顤冮柊宥囩枂閿涙瓪configs/supernet_fc2_180x240.yaml`閵嗕梗configs/layering_rules.yaml`閵?
### 妤犲本鏁归崨鎴掓姢

1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.nas.arch_codec --self_test`

### 缂佹挻鐏夌拋鏉跨秿

1. 閻樿埖鈧緤绱板鎻掔暚閹存劕鑻熼柅姘崇箖閼奉亝顥呴妴?2. 閸涙垝鎶ょ紒鎾寸亯閿?- `python wrappers/run_supernet_train.py --help` 闁俺绻冮妴?- `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run` 闁俺绻冮妴?- `python -m code.nas.arch_codec --self_test` 闁俺绻冮妴?- `python -m code.nas.model_shape_check --h 180 --w 240 --batch 2 --samples 5` 闁俺绻冮妴?- `python -m code.nas.fair_sampler --cycles 300 --seed 42 --check` 闁俺绻冮妴?- `python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 8` 闁俺绻冮妴?- `python -m code.utils.check_comments --root code --strict` 闁俺绻冮妴?- `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml` 闁俺绻冮妴?3. 婢跺洦鏁為敍姘秼閸撳秷顔勭紒鍐箚閼哄倷璐熼崣顖濈箥鐞涘苯宕版担宥夘€囬弸璁圭礉娑撳绔撮柈銊ュ瀻娴兼岸鈧劖顒為弴鎸庡床娑撹櫣婀＄€圭偠顔勭紒鍐偓鏄忕帆閵?
## Part-02閿涘湕04 婢х偛宸?+ 鐠侇厾绮岄崗顒€閽╃拋鈩冩殶鐎圭偠顥婇敍?
### 閺堫剟鍎撮崚鍡欐窗閺?
1. 鐏忓棝鐛欑拠浣圭潨娴犲酣娈㈤張鍝勫窗娴ｅ秴宕岀痪褌璐熼崚鍡楃湴鐟曞棛娲婇弸鍕紦閵?2. 鐏忓棗鍙曢獮瀹狀吀閺侀绮犻棃娆愨偓浣稿窗娴ｅ秴宕岀痪褌璐熼幐澶婂彆楠炲啿鎳嗛張鐔烘埂鐎圭偟鐤拋掳鈧?3. 鐠佲晞顔勭紒鍐╁Г閸涘﹣绗岀拠鍕強娴溠呭⒖閸栧懎鎯堢憰鍡欐磰閻樿埖鈧椒绗岄崗顒€閽╃紒鐔活吀閵?
### 娴狅絿鐖滈弨鐟板З濞撳懎宕?
1. 閺囧瓨鏌?`code/nas/eval_pool_builder.py`閿?- 婢х偛濮為崺铏诡攨鐟曞棛娲婄紓鏍垳缁涙牜鏆愰妴?- 婢х偛濮?`check_eval_pool_coverage` 鐟曞棛娲婂Λ鈧弻銉ュ毐閺佽埇鈧?- 婢х偛濮為崣顖滄纯閹恒儱鎳℃禒銈堫攽濡偓閺屻儳娈?`--check` 閸忋儱褰涢妴?2. 閺囧瓨鏌?`code/nas/supernet_eval.py`閿?- 鐠囧嫪鍙婃潏鎾冲毉閸愭瑥鍙嗙憰鍡欐磰濡偓閺屻儳绮ㄩ弸婧库偓?3. 閺囧瓨鏌?`code/engine/supernet_trainer.py`閿?- 閹恒儱鍙?`generate_fair_cycle` 閸涖劍婀￠柌鍥ㄧ壉楠炲墎鐤粔顖氬彆楠炲疇顓搁弫鑸偓?- 閹恒儱鍙嗛崚鍡楃湴妤犲矁鐦夊Ч鐘崇€杞扮瑢鐟曞棛娲婂Λ鈧弻銉ュ晸閻╂ǜ鈧?- 閸︺劏顔勭紒鍐╁Г閸涘﹪鍣风拋鏉跨秿 `final_fairness_gap` 娑?`eval_pool_coverage_ok`閵?4. 閺囧瓨鏌?`code/engine/train_step.py`閿?- 閹恒儲鏁?`cycle_codes` 鏉堟挸鍙嗛敍灞炬▔瀵繗銆冩潏?3-path 閸涖劍婀＄拠顓濈疅閵?
### 妤犲本鏁归崨鎴掓姢

1. `python -m code.utils.check_comments --root code --strict`
2. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
3. `python -m code.nas.eval_pool_builder --seed 42 --size 12 --check`
4. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 3 --fast_mode`
5. `python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 8`

### 缂佹挻鐏夌拋鏉跨秿

1. 閻樿埖鈧緤绱板鎻掔暚閹存劕鑻熼柅姘崇箖閼奉亝顥呴妴?2. 鐟曞棛娲婂Λ鈧弻銉窗`coverage.ok = true`閵?3. 閸忣剙閽╃拋鈩冩殶閿涙俺顔勭紒鍐叚鐠烘垳鑵?`fairness_gap=0`閵?
## Part-03閿涘牏婀＄€?TF1 Supernet 閸ュ彞绗屾稉銉︾壐閸忣剙閽╃拋顓犵矊濮濄儻绱?
### 閺堫剟鍎撮崚鍡欐窗閺?
1. 鐏忓棗宕版担宥囩秹缂佹粍娴涢幑顫礋閻喎鐤?TF1 Bilinear supernet 閸ヤ勘鈧?2. 鐏忓棗宕版担宥堫唲缂佸啫鎯婇悳顖涙禌閹诡澀璐熼惇鐔风杽 `3-path same-batch` 濮婎垰瀹崇槐顖溞濋弴瀛樻煀閵?3. 鐏忓棙鏆熼幑顔肩湴閺囨寧宕叉稉?FC2 闁插洦鐗遍崳顭掔礄鐢箑鎮庨幋鎰殶閹诡喖鍘规惔鏇礆閵?4. 閸?`tf_work_hpc` 閻滎垰顣ㄧ€瑰本鍨氶崣顖濈箥鐞涘瞼娈?1 epoch 閻叀绐囨宀冪槈閵?
### 娴狅絿鐖滈弨鐟板З濞撳懎宕?
1. 缂冩垹绮剁仦鍌︾窗
- 閺傛澘顤?`code/network/decorators.py` 娑?`code/network/base_layers.py`閵?- 闁插秴鍟?`code/network/MultiScaleResNet_supernet.py`閿?  - 4 娑?backbone 濞ｅ崬瀹抽柅澶嬪閸ф绱橠eep1/2/3閿涘鈧?  - 5 娑?head 閸楅袧閺嶆悂鈧瀚ㄩ崸妤嬬礄7/5/3閿涘鈧?  - 鏉堟挸鍤亸鍝勫閸ュ搫鐣炬稉?`45x60 / 90x120 / 180x240`閵?2. 閺佺増宓佺仦鍌︾窗
- 闁插秴鍟?`code/data/fc2_dataset.py`閿涙碍鏁幐?`FC2_dirnames + split index` 鐟欙絾鐎介妴涔?flo` 鐠囪褰囬妴渚€娈㈤張楦款梿閸擃亗鈧胶宸辨笟婵婄閺冭泛鎮庨幋鎰壉閺堫剙鍘规惔鏇樷偓?- 闁插秴鍟?`code/data/dataloader_builder.py`閿涙碍鐎?train/val provider閵?- 闁插秴鍟?`code/data/transforms_180x240.py`閿涙俺绶崗銉︾垼閸戝棗瀵查崚?`[-1,1]`閵?3. 鐠侇厾绮岄幍褑顢戠仦鍌︾窗
- 闁插秴鍟?`code/engine/train_step.py`閿涙艾顦跨亸鍝勫 L1 閹圭喎銇戞稉搴㈡綀闁插秷鈥滈崙蹇斿閹恒儯鈧?- 闁插秴鍟?`code/engine/eval_step.py`閿涙PE 閹稿洦鐖ｉ弸鍕紦閵?- 闁插秴鍟?`code/engine/bn_recalibration.py`閿涙矮绱扮拠婵堥獓 BN 闁插秳鍙婂ù浣衡柤閵?- 闁插秴鍟?`code/engine/supernet_trainer.py`閿涙氨婀＄€?TF1 閸ョ偓鐎鎭掆偓浣峰紬閺嶇厧鍙曢獮宕囩柈缁夘垱娲块弬鑸偓浣告祼鐎?eval pool 鐠囧嫪鍙婇妴浣规－閸?閺冦儱绻?artifact 鏉堟挸鍤妴?4. 閸忋儱褰涙稉搴ㄥ帳缂冾噯绱?- 閺囧瓨鏌?`code/app/train_supernet_app.py`閿涙艾娆㈡潻鐔奉嚤閸忋儴顔勭紒鍐┠侀崸妤嬬礉閺€顖涘瘮閺?`PyYAML` 閻滎垰顣ㄩ惃鍕氦闁?YAML 鐟欙絾鐎介妴?- 閺囧瓨鏌?`wrappers/run_supernet_train.py`閿涙碍鏌婃晶?`--steps_per_epoch`閵嗕梗--base_path`閵?- 閺囧瓨鏌?`configs/supernet_fc2_180x240.yaml`閿涙俺藟閸?`steps_per_epoch/base_path/train_list_name/allow_synthetic_fallback`閵?5. 妤犲矁鐦夊銉ュ徔閿?- 闁插秴鍟?`code/nas/model_shape_check.py`閿涘本鏁兼稉铏规埂鐎?TF1 閸ョ偓甯归悶鍡橆梾閺屻儯鈧?
### 妤犲本鏁归崨鎴掓姢

1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.utils.check_comments --root code --strict`
4. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
5. `conda run -n tf_work_hpc python -m code.nas.model_shape_check --h 180 --w 240 --batch 2 --samples 2`
6. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 1 --steps_per_epoch 1 --batch_size 2 --fast_mode`

### 缂佹挻鐏夌拋鏉跨秿

1. 閻樿埖鈧緤绱板鎻掔暚閹存劕鑻熼柅姘崇箖閼奉亝顥呴妴?2. 瑜般垻濮稿Λ鈧弻銉窗`out_1_4=45x60`閿涘畭out_1_2=90x120`閿涘畭out_1_1=180x240`閵?3. 鐠侇厾绮岄惌顓＄獓閿涙瓪tf_work_hpc` 閻滎垰顣ㄧ€瑰本鍨?`1 epoch`閿涘本妫╄箛妤佹▔缁€杞板紬閺嶇厧鍙曢獮鍐叉儕閻滎垯绗岀拠鍕強鏉堟挸鍤锝呯埗閵?
## Part-03B锛圕heckpoint鐪熷疄鍖?+ Supernet璇勪及鐪熷疄鍖栵級

### 鏈儴鍒嗙洰鏍?1. 灏?`checkpoint_manager` 浠庡崰浣岼SON鏀逛负鐪熷疄 TF1 Saver 淇濆瓨/鎭㈠銆?2. 鎵撻€?`--load_checkpoint` 涓?`--resume_experiment_name` 鐨勬仮澶嶈缁冮摼璺€?3. 灏?`code/nas/supernet_eval.py` 浠庡崰浣嶈剼鏈敼涓虹湡瀹?checkpoint 璇勪及銆?4. 琛ラ綈 BN 閲嶄及闃舵杈撳叆鏍囧噯鍖栵紝淇濊瘉涓庤缁冭緭鍏ヤ竴鑷淬€?
### 浠ｇ爜鏀瑰姩娓呭崟
1. 閲嶅啓 `code/engine/checkpoint_manager.py`锛?- 鏂板 `checkpoint_exists`銆乣find_existing_checkpoint`銆乣save_checkpoint`銆乣restore_checkpoint`銆乣load_checkpoint_meta`銆?- 淇濈暀 `write_checkpoint_placeholder` 鍏煎鏃ф帴鍙ｃ€?2. 閲嶅啓 `code/engine/supernet_trainer.py`锛?- 鎺ュ叆鐪熷疄 checkpoint 淇濆瓨锛坆est/last锛変笌 meta 鐘舵€佸啓鐩樸€?- 鎺ュ叆鎭㈠閫昏緫锛歚load_checkpoint + resume_experiment_name`銆?- 鎭㈠鐘舵€佸寘鍚?`start_epoch/global_step/fairness_counts/best_metric/bad_epochs`銆?- 璇勪及姹犺惤鐩樻枃浠跺悕鏀逛负 `eval_pool_{size}.json`銆?3. 鏇存柊 `code/engine/bn_recalibration.py`锛?- BN 閲嶄及鍓嶅姞鍏?`standardize_image_tensor`銆?4. 閲嶅啓 `code/nas/supernet_eval.py`锛?- 鐪熷疄 TF1 鍥炬瀯寤轰笌 checkpoint 鍔犺浇銆?- 閫愬瓙缃戞墽琛?BN 閲嶄及 + EPE 璇勪及銆?- 杈撳嚭 `supernet_eval_result_{checkpoint_type}.json`銆?- 鏀寔鍙傛暟锛歚--checkpoint_type`銆乣--batch_size`銆乣--bn_recal_batches`銆?
### 楠屾敹鍛戒护
1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.utils.check_comments --root code --strict`
4. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
5. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 1 --steps_per_epoch 1 --batch_size 2 --fast_mode`
6. `conda run -n tf_work_hpc python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 2 --checkpoint_type last --batch_size 2`
7. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 2 --steps_per_epoch 1 --batch_size 2 --load_checkpoint --resume_experiment_name edgeflownas_supernet_fc2_180x240 --fast_mode`

### 缁撴灉璁板綍
1. 鐘舵€侊細宸插畬鎴愬苟閫氳繃鑷銆?2. `supernet_eval` 宸茶緭鍑虹湡瀹炶瘎浼版枃浠讹細`outputs/supernet/edgeflownas_supernet_fc2_180x240/supernet_eval_result_last.json`銆?3. 鏂偣鎭㈠楠岃瘉閫氳繃锛氭棩蹇楁樉绀轰粠 `start_epoch=2` 鎭㈠骞剁户缁缁冦€
## Part-04A（Gate-4 交付校验工具 + HPC脚本修复）

### 本部分目标
1. 新增训练产物校验命令 `python -m code.nas.check_manifest --path ...`。
2. 修复 HPC 模板脚本 `scripts/hpc/run_supernet_train.sh` 的续行与参数化问题。
3. 保证本机与 HPC 命令参数风格一致，并支持环境变量覆盖。

### 代码改动清单
1. 新增 `code/nas/check_manifest.py`：
- 校验 `train_manifest.json` 必填字段：`seed/config_hash/git_commit/input_shape/optimizer/lr_schedule/wd/grad_clip`。
- 校验 `fairness_counts.json` 九个 block 的三选项计数是否严格相等。
- 校验 `eval_epe_history.csv` 是否包含核心列且至少有一行数据。
- 支持 `--strict` 失败返回非零退出码。
2. 重写 `scripts/hpc/run_supernet_train.sh`：
- 改为命令数组构建，避免 `\ + 注释` 导致的 bash 断行失效。
- 支持环境变量覆盖：`CONFIG_PATH/GPU_DEVICE/NUM_EPOCHS/STEPS_PER_EPOCH/BATCH_SIZE/LEARNING_RATE/EXPERIMENT_NAME/RESUME_EXPERIMENT_NAME/LOAD_CHECKPOINT/FAST_MODE`。
- 输出实际执行命令，便于HPC日志定位。

### 验收命令
1. `python -m code.utils.check_comments --root code --strict`
2. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
3. `python -m code.nas.check_manifest --path outputs/supernet/edgeflownas_supernet_fc2_180x240/train_manifest.json --strict`
4. `bash -n scripts/hpc/run_supernet_train.sh`

### 结果记录
1. 状态：已完成并通过自检。
2. Gate-4 中的 `check_manifest` 命令已具备可执行实现。
3. HPC 训练模板脚本已修复为可执行且可参数化。