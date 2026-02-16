# 06 閻庡湱鍋為弻锕傚籍閵夈儳绠?
闁哄洤鐡ㄩ弻濠囧籍閸洘锛? 2026-02-16

## Part-01闁挎稑婀?1 + E02 + Gate-1 闁糕晞娅ｉ、鍛村礂閵夈儱缍撻柨?
### 闁哄牜鍓熼崕鎾礆閸℃瑦绐楅柡?
1. 鐎点倛娅ｉ悵娑橆啅閵壯€鏌ょ紒鐙欏懏绐楃憸鐗堟礀閸ㄥ海浠﹂崒鐑嗏偓鍥几闊祴鍋?2. 閻庣懓鏈崹?supernet 閻犱緡鍘剧划宀勫礂閵夈儱缍?`wrapper -> app -> engine` 闁哄牃鍋撻悘蹇撶箻濡挳鎮抽妯峰亾?3. 閻庣懓鏈崹?`arch_codec` 闁煎浜濈粊鎾嚄閽樺顫旈柕?4. 闁搞儱鎼€垫煡鏌婂鍥╂瀭闁哄倸娲ｅ▎銏＄▔鎼淬値娼愰柛鎺撶懄閺嬪啯绂掗煬娴嬪亾?
### 濞寸媴绲块悥婊堝绩閻熸澘袟婵炴挸鎳庡畷?
1. 闁哄倹婢橀·鍐儎椤旇偐绉块柨娑欑摢wrappers/`, `configs/`, `code/*`, `tests/*`, `outputs/supernet`, `scripts/hpc`闁?2. 闁哄倹婢橀·鍐礂閵夈儱缍撻柨娑欑摢wrappers/run_supernet_train.py`闁?3. 闁哄倹婢橀·鍐磽閺嶃劌绗撻柨娑欑摢code/app/train_supernet_app.py`闁?4. 闁哄倹婢橀·鍐箥瑜戦、鎴犱沪閸岀儐鈧洭寮哥拋鍦獥`code/engine/supernet_trainer.py` 濞戞挸娴峰ù澶愬礂閸愯尙鎽嶆俊顖椻偓铏仴闁?5. 闁哄倹婢橀·?NAS 闁哄秶顭堢缓楣冩晬濮濇瓭ode/nas/arch_codec.py`闁靛棔姊梒ode/nas/fair_sampler.py`闁?6. 闁哄倹婢橀·鍐啅閵夈儱寰旈柨娑欑摢code/utils/*.py`闁挎稑鐗婂Λ鈺勭疀濡炲皷鍋撴担鐣岀唴鐎垫澘瀚ㄩ埀顑挎睘son闁靛棔澶焑ed闁靛棔娌焌nifest闁靛棔鐒﹂ˉ鍛村蓟閵夈劌澹栭柡鍫墾缁辨岸濡?7. 闁哄倹婢橀·鍐煀瀹ュ洨鏋傞柨娑欑摢configs/supernet_fc2_180x240.yaml`闁靛棔姊梒onfigs/layering_rules.yaml`闁?
### 濡ょ姴鏈弫褰掑川閹存帗濮?
1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.nas.arch_codec --self_test`

### 缂備焦鎸婚悘澶屾媼閺夎法绉?
1. 闁绘鍩栭埀顑跨筏缁辨澘顔忛幓鎺旀殮闁瑰瓨鍔曢懟鐔兼焻濮樺磭绠栭柤濂変簼椤ュ懘濡?2. 闁告稒鍨濋幎銈囩磼閹惧浜柨?- `python wrappers/run_supernet_train.py --help` 闂侇偅淇虹换鍐Υ?- `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run` 闂侇偅淇虹换鍐Υ?- `python -m code.nas.arch_codec --self_test` 闂侇偅淇虹换鍐Υ?- `python -m code.nas.model_shape_check --h 180 --w 240 --batch 2 --samples 5` 闂侇偅淇虹换鍐Υ?- `python -m code.nas.fair_sampler --cycles 300 --seed 42 --check` 闂侇偅淇虹换鍐Υ?- `python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 8` 闂侇偅淇虹换鍐Υ?- `python -m code.utils.check_comments --root code --strict` 闂侇偅淇虹换鍐Υ?- `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml` 闂侇偅淇虹换鍐Υ?3. 濠㈣泛娲﹂弫鐐烘晬濮橆剛绉奸柛鎾崇Х椤斿嫮绱掗崘顏勭畾闁煎搫鍊风拹鐔煎矗椤栨繄绠ラ悶娑樿嫰瀹曠増鎷呭澶樷偓鍥几鐠佸湱绀夊☉鎾愁儎缁旀挳鏌堥妸銉ョ€诲ù鍏煎哺閳ь剚鍔栭鐐哄即閹稿骸搴婂☉鎾规濠€锛勨偓鍦仩椤斿嫮绱掗崘顔瑰亾閺勫繒甯嗛柕?
## Part-02闁挎稑婀?4 濠⒀呭仜瀹?+ 閻犱緡鍘剧划宀勫礂椤掆偓闁解晝鎷嬮埄鍐╂閻庡湱鍋犻ˉ濠囨晬?
### 闁哄牜鍓熼崕鎾礆閸℃瑦绐楅柡?
1. 閻忓繐妫濋悰娆戞嫚娴ｅ湱娼ㄥù鐘查叄濞堛垽寮甸崫鍕獥濞达絽绉村畷宀€鐥鐠愮喖宕氶崱妤冩勾閻熸洖妫涘ú濠囧几閸曨偆绱﹂柕?2. 閻忓繐妫楅崣鏇㈢嵁鐎圭媭鍚€闁轰線顣︾划鐘绘濞嗘劏鍋撴担绋跨獥濞达絽绉村畷宀€鐥鐠愮喖骞愭径濠傚絾妤犵偛鍟块幊鍡涘嫉閻旂儤鍩傞悗鍦仧閻ゎ喚鎷嬫幊閳?3. 閻犱讲鏅為鍕磼閸愨晛袚闁告稑锕ｇ粭宀€鎷犻崟顏勫挤濞存籂鍛挅闁告牕鎳庨幆鍫㈡啺閸℃瑦纾伴柣妯垮煐閳ь兛妞掔粭宀勫礂椤掆偓闁解晝绱掗悢娲诲悁闁?
### 濞寸媴绲块悥婊堝绩閻熸澘袟婵炴挸鎳庡畷?
1. 闁哄洤鐡ㄩ弻?`code/nas/eval_pool_builder.py`闁?- 濠⒀呭仜婵偤宕洪搹璇℃敤閻熸洖妫涘ú濠勭磽閺嶎偆鍨崇紒娑欑墱閺嗘劙濡?- 濠⒀呭仜婵?`check_eval_pool_coverage` 閻熸洖妫涘ú濠偽涢埀顒勫蓟閵夈儱姣愰柡浣藉焽閳?- 濠⒀呭仜婵偤宕ｉ婊勭函闁规亽鍎遍幊鈩冪閵堝牜鏀芥俊顐熷亾闁哄被鍎冲▓?`--check` 闁稿繈鍎辫ぐ娑㈠Υ?2. 闁哄洤鐡ㄩ弻?`code/nas/supernet_eval.py`闁?- 閻犲洤瀚崣濠冩綇閹惧啿姣夐柛鎰懃閸欏棛鎲伴崱娆愮０婵☆偀鍋撻柡灞诲劤缁劑寮稿┃搴撳亾?3. 闁哄洤鐡ㄩ弻?`code/engine/supernet_trainer.py`闁?- 闁规亽鍎遍崣?`generate_fair_cycle` 闁告稏鍔嶅﹢锟犳煂閸ャ劎澹夋鐐插閻ゎ喚绮旈姘絾妤犵偛鐤囬鎼佸极閼割兘鍋?- 闁规亽鍎遍崣鍡涘礆閸℃婀村Δ鐘茬焷閻﹀效閻樺磭鈧垰顕欐潪鎵憿閻熸洖妫涘ú濠偽涢埀顒勫蓟閵夈儱鏅搁柣鈺偳滈埀?- 闁革负鍔忛鍕磼閸愨晛袚闁告稑锕崳椋庢媼閺夎法绉?`final_fairness_gap` 濞?`eval_pool_coverage_ok`闁?4. 闁哄洤鐡ㄩ弻?`code/engine/train_step.py`闁?- 闁规亽鍎查弫?`cycle_codes` 閺夊牊鎸搁崣鍡涙晬鐏炵偓鈻旂€殿喖绻楅妴鍐╂綇?3-path 闁告稏鍔嶅﹢锛勬嫚椤撴繄鐤呴柕?
### 濡ょ姴鏈弫褰掑川閹存帗濮?
1. `python -m code.utils.check_comments --root code --strict`
2. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
3. `python -m code.nas.eval_pool_builder --seed 42 --size 12 --check`
4. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 3 --fast_mode`
5. `python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 8`

### 缂備焦鎸婚悘澶屾媼閺夎法绉?
1. 闁绘鍩栭埀顑跨筏缁辨澘顔忛幓鎺旀殮闁瑰瓨鍔曢懟鐔兼焻濮樺磭绠栭柤濂変簼椤ュ懘濡?2. 閻熸洖妫涘ú濠偽涢埀顒勫蓟閵夘垳绐梎coverage.ok = true`闁?3. 闁稿浚鍓欓柦鈺冩媼閳╁啯娈堕柨娑欎亢椤斿嫮绱掗崘顏嗗彋閻犵儤鍨抽懙?`fairness_gap=0`闁?
## Part-03闁挎稑鐗忓﹢锛勨偓?TF1 Supernet 闁搞儱褰炵粭灞剧▔閵夛妇澹愰柛蹇ｅ墮闁解晝鎷嬮鐘电煀婵縿鍎荤槐?
### 闁哄牜鍓熼崕鎾礆閸℃瑦绐楅柡?
1. 閻忓繐妫楀畷鐗堟媴瀹ュ洨绉圭紓浣圭矋濞存盯骞戦～顓＄闁活亞鍠庨悿?TF1 Bilinear supernet 闁搞儰鍕橀埀?2. 閻忓繐妫楀畷鐗堟媴瀹ュ牜鍞茬紓浣稿暙閹﹪鎮抽娑欑闁硅婢€鐠愮喖鎯囬悢椋庢澖 `3-path same-batch` 婵鍨扮€瑰磭妲愰婧炴繈寮寸€涙ɑ鐓€闁?3. 閻忓繐妫欓弳鐔煎箲椤旇偐婀撮柡鍥ㄥ瀹曞弶绋?FC2 闂佹彃娲﹂悧閬嶅闯椤帞绀勯悽顖ょ畱閹酣骞嬮幇顓熸闁硅鍠栭崢瑙勬償閺囶亞绀嗛柕?4. 闁?`tf_work_hpc` 闁绘粠鍨伴。銊р偓鐟版湰閸ㄦ岸宕ｉ婵堢閻炴稑鐬煎▓?1 epoch 闁活収鍙€缁愬洦顨ュ畝鍐闁?
### 濞寸媴绲块悥婊堝绩閻熸澘袟婵炴挸鎳庡畷?
1. 缂傚啯鍨圭划鍓佷沪閸岋妇绐?- 闁哄倹婢橀·?`code/network/decorators.py` 濞?`code/network/base_layers.py`闁?- 闂佹彃绉撮崯?`code/network/MultiScaleResNet_supernet.py`闁?  - 4 濞?backbone 婵烇絽宕€规娊鏌呮径瀣仴闁秆勵殣缁辨eep1/2/3闁挎稑顦埀?  - 5 濞?head 闁告顥撹ⅶ闁哄秵鎮傞埀顒€顦扮€氥劑宕稿Δ瀣7/5/3闁挎稑顦埀?  - 閺夊牊鎸搁崵顓犱焊閸濆嫬顔婇柛銉ユ惈閻ｇ偓绋?`45x60 / 90x120 / 180x240`闁?2. 闁轰胶澧楀畵浣轰沪閸岋妇绐?- 闂佹彃绉撮崯?`code/data/fc2_dataset.py`闁挎稒纰嶉弫顕€骞?`FC2_dirnames + split index` 閻熸瑱绲鹃悗浠嬪Υ娑?flo` 閻犲洩顕цぐ鍥Υ娓氣偓濞堛垽寮垫ウ娆炬⒖闁告搩浜楅埀顑胯兌瀹歌鲸绗熷┑濠勵洬闁哄啳娉涢幃搴ㄥ箣閹邦厾澹夐柡鍫墮閸樿鎯旈弴妯峰亾?- 闂佹彃绉撮崯?`code/data/dataloader_builder.py`闁挎稒纰嶉悗顖氼嚈?train/val provider闁?- 闂佹彃绉撮崯?`code/data/transforms_180x240.py`闁挎稒淇虹欢顓㈠礂閵夛妇鍨奸柛鎴濇鐎垫煡宕?`[-1,1]`闁?3. 閻犱緡鍘剧划宀勫箥瑜戦、鎴犱沪閸岋妇绐?- 闂佹彃绉撮崯?`code/engine/train_step.py`闁挎稒鑹鹃ˇ璺ㄤ焊閸濆嫬顔?L1 闁瑰湱鍠庨妵鎴炵▔鎼淬垺缍€闂佹彃绉烽垾婊堝礄韫囨柨顏婚柟鎭掑劘閳?- 闂佹彃绉撮崯?`code/engine/eval_step.py`闁挎稒顑怭E 闁圭娲﹂悥锝夊几閸曨偆绱﹂柕?- 闂佹彃绉撮崯?`code/engine/bn_recalibration.py`闁挎稒鐭槐鎵嫚濠靛牓鐛?BN 闂佹彃绉抽崣濠偯规担琛℃煠闁?- 闂佹彃绉撮崯?`code/engine/supernet_trainer.py`闁挎稒姘ㄥ﹢锛勨偓?TF1 闁搞儳鍋撻悗顖氼嚈閹巻鍋撴担宄扮船闁哄秶鍘ч崣鏇㈢嵁瀹曞洨鏌堢紒澶樺灡濞插潡寮懜顑藉亾娴ｅ憡绁奸悗?eval pool 閻犲洤瀚崣濠囧Υ娴ｈ锛嶉柛?闁哄啨鍎辩换?artifact 閺夊牊鎸搁崵顓㈠Υ?4. 闁稿繈鍎辫ぐ娑欑▔鎼淬劌甯崇紓鍐惧櫙缁?- 闁哄洤鐡ㄩ弻?`code/app/train_supernet_app.py`闁挎稒鑹惧▎銏℃交閻斿鍤ら柛蹇嬪劥椤斿嫮绱掗崘鈹犱線宕稿Δ瀣闁衡偓椤栨稑鐦柡?`PyYAML` 闁绘粠鍨伴。銊╂儍閸曨喕姘﹂梺?YAML 閻熸瑱绲鹃悗浠嬪Υ?- 闁哄洤鐡ㄩ弻?`wrappers/run_supernet_train.py`闁挎稒纰嶉弻濠冩櫠?`--steps_per_epoch`闁靛棔姊?-base_path`闁?- 闁哄洤鐡ㄩ弻?`configs/supernet_fc2_180x240.yaml`闁挎稒淇鸿棢闁?`steps_per_epoch/base_path/train_list_name/allow_synthetic_fallback`闁?5. 濡ょ姴鐭侀惁澶婎啅閵夈儱寰旈柨?- 闂佹彃绉撮崯?`code/nas/model_shape_check.py`闁挎稑鏈弫鍏肩▔閾忚鍩傞悗?TF1 闁搞儳鍋撶敮褰掓偠閸℃﹩姊鹃柡灞诲劘閳?
### 濡ょ姴鏈弫褰掑川閹存帗濮?
1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.utils.check_comments --root code --strict`
4. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
5. `conda run -n tf_work_hpc python -m code.nas.model_shape_check --h 180 --w 240 --batch 2 --samples 2`
6. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 1 --steps_per_epoch 1 --batch_size 2 --fast_mode`

### 缂備焦鎸婚悘澶屾媼閺夎法绉?
1. 闁绘鍩栭埀顑跨筏缁辨澘顔忛幓鎺旀殮闁瑰瓨鍔曢懟鐔兼焻濮樺磭绠栭柤濂変簼椤ュ懘濡?2. 鐟滆埇鍨绘慨绋课涢埀顒勫蓟閵夘垳绐梎out_1_4=45x60`闁挎稑鐣璷ut_1_2=90x120`闁挎稑鐣璷ut_1_1=180x240`闁?3. 閻犱緡鍘剧划宀勬儗椤擄紕鐛撻柨娑欑摢tf_work_hpc` 闁绘粠鍨伴。銊р偓鐟版湰閸?`1 epoch`闁挎稑鏈Λ鈺勭疀濡や焦鈻旂紒鈧潪鏉跨船闁哄秶鍘ч崣鏇㈢嵁閸愬弶鍎曢柣婊庡灟缁楀瞼鎷犻崟顏勫挤閺夊牊鎸搁崵顓烆潰閿濆懐鍩楅柕?
## Part-03B閿涘湑heckpoint閻喎鐤勯崠?+ Supernet鐠囧嫪鍙婇惇鐔风杽閸栨牭绱?
### 閺堫剟鍎撮崚鍡欐窗閺?1. 鐏?`checkpoint_manager` 娴犲骸宕版担宀糞ON閺€閫涜礋閻喎鐤?TF1 Saver 娣囨繂鐡?閹垹顦查妴?2. 閹垫捇鈧?`--load_checkpoint` 娑?`--resume_experiment_name` 閻ㄥ嫭浠径宥堫唲缂佸啴鎽肩捄顖樷偓?3. 鐏?`code/nas/supernet_eval.py` 娴犲骸宕版担宥堝壖閺堫剚鏁兼稉铏规埂鐎?checkpoint 鐠囧嫪鍙婇妴?4. 鐞涖儵缍?BN 闁插秳鍙婇梼鑸殿唽鏉堟挸鍙嗛弽鍥у櫙閸栨牭绱濇穱婵婄槈娑撳氦顔勭紒鍐翻閸忋儰绔撮懛娣偓?
### 娴狅絿鐖滈弨鐟板З濞撳懎宕?1. 闁插秴鍟?`code/engine/checkpoint_manager.py`閿?- 閺傛澘顤?`checkpoint_exists`閵嗕梗find_existing_checkpoint`閵嗕梗save_checkpoint`閵嗕梗restore_checkpoint`閵嗕梗load_checkpoint_meta`閵?- 娣囨繄鏆€ `write_checkpoint_placeholder` 閸忕厧顔愰弮褎甯撮崣锝冣偓?2. 闁插秴鍟?`code/engine/supernet_trainer.py`閿?- 閹恒儱鍙嗛惇鐔风杽 checkpoint 娣囨繂鐡ㄩ敍鍧唀st/last閿涘绗?meta 閻樿埖鈧礁鍟撻惄妯糕偓?- 閹恒儱鍙嗛幁銏狀槻闁槒绶敍姝歭oad_checkpoint + resume_experiment_name`閵?- 閹垹顦查悩鑸碘偓浣稿瘶閸?`start_epoch/global_step/fairness_counts/best_metric/bad_epochs`閵?- 鐠囧嫪鍙婂Ч鐘烘儰閻╂ɑ鏋冩禒璺烘倳閺€閫涜礋 `eval_pool_{size}.json`閵?3. 閺囧瓨鏌?`code/engine/bn_recalibration.py`閿?- BN 闁插秳鍙婇崜宥呭閸?`standardize_image_tensor`閵?4. 闁插秴鍟?`code/nas/supernet_eval.py`閿?- 閻喎鐤?TF1 閸ョ偓鐎杞扮瑢 checkpoint 閸旂姾娴囬妴?- 闁劕鐡欑純鎴炲⒔鐞?BN 闁插秳鍙?+ EPE 鐠囧嫪鍙婇妴?- 鏉堟挸鍤?`supernet_eval_result_{checkpoint_type}.json`閵?- 閺€顖涘瘮閸欏倹鏆熼敍姝?-checkpoint_type`閵嗕梗--batch_size`閵嗕梗--bn_recal_batches`閵?
### 妤犲本鏁归崨鎴掓姢
1. `python wrappers/run_supernet_train.py --help`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `python -m code.utils.check_comments --root code --strict`
4. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
5. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 1 --steps_per_epoch 1 --batch_size 2 --fast_mode`
6. `conda run -n tf_work_hpc python -m code.nas.supernet_eval --config configs/supernet_fc2_180x240.yaml --eval_only --bn_recal_batches 2 --checkpoint_type last --batch_size 2`
7. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 2 --steps_per_epoch 1 --batch_size 2 --load_checkpoint --resume_experiment_name edgeflownas_supernet_fc2_180x240 --fast_mode`

### 缂佹挻鐏夌拋鏉跨秿
1. 閻樿埖鈧緤绱板鎻掔暚閹存劕鑻熼柅姘崇箖閼奉亝顥呴妴?2. `supernet_eval` 瀹歌尪绶崙铏规埂鐎圭偠鐦庢导鐗堟瀮娴犺绱癭outputs/supernet/edgeflownas_supernet_fc2_180x240/supernet_eval_result_last.json`閵?3. 閺傤厾鍋ｉ幁銏狀槻妤犲矁鐦夐柅姘崇箖閿涙碍妫╄箛妤佹▔缁€杞扮矤 `start_epoch=2` 閹垹顦查獮鍓佹埛缂侇叀顔勭紒鍐︹偓
## Part-04A锛圙ate-4 浜や粯鏍￠獙宸ュ叿 + HPC鑴氭湰淇锛?
### 鏈儴鍒嗙洰鏍?1. 鏂板璁粌浜х墿鏍￠獙鍛戒护 `python -m code.nas.check_manifest --path ...`銆?2. 淇 HPC 妯℃澘鑴氭湰 `scripts/hpc/run_supernet_train.sh` 鐨勭画琛屼笌鍙傛暟鍖栭棶棰樸€?3. 淇濊瘉鏈満涓?HPC 鍛戒护鍙傛暟椋庢牸涓€鑷达紝骞舵敮鎸佺幆澧冨彉閲忚鐩栥€?
### 浠ｇ爜鏀瑰姩娓呭崟
1. 鏂板 `code/nas/check_manifest.py`锛?- 鏍￠獙 `train_manifest.json` 蹇呭～瀛楁锛歚seed/config_hash/git_commit/input_shape/optimizer/lr_schedule/wd/grad_clip`銆?- 鏍￠獙 `fairness_counts.json` 涔濅釜 block 鐨勪笁閫夐」璁℃暟鏄惁涓ユ牸鐩哥瓑銆?- 鏍￠獙 `eval_epe_history.csv` 鏄惁鍖呭惈鏍稿績鍒椾笖鑷冲皯鏈変竴琛屾暟鎹€?- 鏀寔 `--strict` 澶辫触杩斿洖闈為浂閫€鍑虹爜銆?2. 閲嶅啓 `scripts/hpc/run_supernet_train.sh`锛?- 鏀逛负鍛戒护鏁扮粍鏋勫缓锛岄伩鍏?`\ + 娉ㄩ噴` 瀵艰嚧鐨?bash 鏂澶辨晥銆?- 鏀寔鐜鍙橀噺瑕嗙洊锛歚CONFIG_PATH/GPU_DEVICE/NUM_EPOCHS/STEPS_PER_EPOCH/BATCH_SIZE/LEARNING_RATE/EXPERIMENT_NAME/RESUME_EXPERIMENT_NAME/LOAD_CHECKPOINT/FAST_MODE`銆?- 杈撳嚭瀹為檯鎵ц鍛戒护锛屼究浜嶩PC鏃ュ織瀹氫綅銆?
### 楠屾敹鍛戒护
1. `python -m code.utils.check_comments --root code --strict`
2. `python -m code.utils.check_layering --root code --rules configs/layering_rules.yaml`
3. `python -m code.nas.check_manifest --path outputs/supernet/edgeflownas_supernet_fc2_180x240/train_manifest.json --strict`
4. `bash -n scripts/hpc/run_supernet_train.sh`

### 缁撴灉璁板綍
1. 鐘舵€侊細宸插畬鎴愬苟閫氳繃鑷銆?2. Gate-4 涓殑 `check_manifest` 鍛戒护宸插叿澶囧彲鎵ц瀹炵幇銆?3. HPC 璁粌妯℃澘鑴氭湰宸蹭慨澶嶄负鍙墽琛屼笖鍙弬鏁板寲銆
## Part-04B（Gate-4 实机化验收补跑）

### 本部分目标
1. 在 `tf_work_hpc` 环境完成 2-epoch 真实 smoke。
2. 用 `check_manifest --strict` 对新实验产物做严格校验。
3. 完成 HPC 脚本参数覆盖执行验证。

### 验收命令
1. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --gpu_device 0 --num_epochs 2 --steps_per_epoch 1 --batch_size 2 --lr 1e-4 --config configs/supernet_fc2_180x240.yaml --fast_mode --experiment_name edgeflownas_supernet_fc2_180x240_gate4`
2. `python -m code.nas.check_manifest --path outputs/supernet/edgeflownas_supernet_fc2_180x240_gate4/train_manifest.json --strict`
3. `bash -lc "cd /mnt/d/Dataset/MCUFlowNet/EdgeFlowNAS && NUM_EPOCHS=1 STEPS_PER_EPOCH=1 BATCH_SIZE=2 LEARNING_RATE=1e-4 FAST_MODE=1 DRY_RUN=1 EXPERIMENT_NAME=edgeflownas_supernet_fc2_180x240_hpc_smoke bash scripts/hpc/run_supernet_train.sh"`

### 结果记录
1. 2-epoch smoke 通过，日志输出 `epoch=1` 与 `epoch=2` 指标。
2. `check_manifest --strict` 返回 `ok=true`，`row_count=2`。
3. HPC 脚本已完成参数覆盖执行验证（dry-run 路径）。

## Part-05（详细使用手册交付）

### 本部分目标
1. 在全部任务验收通过后，交付详细使用手册 Markdown。

### 代码/文档改动
1. 新增 `plan/07_User_Manual.md`。
2. 更新 `plan/00_Plan_Index.md` 增加手册与日志索引。

### 结果记录
1. 使用手册已覆盖: 环境准备、训练/恢复/评估命令、产物验收、HPC 使用、常见问题与推荐流程。

## Part-08（Supernet Uncertainty 对齐）

### 本部分目标
1. 将超网训练从纯 L1 版本切换为与单模型一致的 uncertainty 版本（LinearSoftplus 语义）。
2. 保持评估指标 EPE 仅基于光流前 2 通道计算。
3. 保持 Strict Fairness 训练主流程不变，仅替换输出语义与损失定义。

### 代码改动
1. 更新 `code/engine/train_step.py`：
- 新增 `build_multiscale_uncertainty_loss(...)`，实现累计光流 + 累计不确定性的多尺度损失。
- 保留 `build_multiscale_l1_loss(...)` 兼容函数。
2. 更新 `code/engine/eval_step.py`：
- 新增 `extract_flow_prediction(...)`，明确切片前 `num_out` 通道作为流场。
- `build_epe_metric(...)` 改为只对流场分支计算 EPE。
3. 更新 `code/engine/supernet_trainer.py`：
- 读取 `data.flow_channels`。
- 将 `num_out` 改为 `flow_channels * 2`。
- 训练损失改为 `build_multiscale_uncertainty_loss(...)`。
- EPE 调用改为 `build_epe_metric(..., num_out=flow_channels)`。
4. 重写 `code/nas/supernet_eval.py`：
- 清理原文件行粘连问题并重构为可维护版本。
- 评估图改为 uncertainty 输出通道（`flow_channels * 2`）。
- EPE 统一走 `code.engine.eval_step.build_epe_metric`。
5. 更新 `code/nas/model_shape_check.py`：
- 形状检查默认 `num_out=4`。
6. 更新 `code/network/MultiScaleResNet_supernet.py`：
- 默认 `num_out` 调整为 `4`。
7. 更新 `configs/supernet_fc2_180x240.yaml`：
- 新增 `train.uncertainty_type: LinearSoftplus`。
- 新增 `data.flow_channels: 2`。
8. 更新 `code/utils/manifest.py`：
- manifest 新增 `flow_channels` 与 `uncertainty_type` 字段。

### 验收命令
1. `python -m py_compile code/engine/train_step.py code/engine/eval_step.py code/engine/supernet_trainer.py code/nas/supernet_eval.py code/nas/model_shape_check.py code/network/MultiScaleResNet_supernet.py`
2. `python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --dry_run`
3. `conda run -n tf_work_hpc python -m code.nas.model_shape_check --h 180 --w 240 --batch 2 --samples 1`

### 通过标准
1. 编译检查全部通过。
2. dry-run 输出配置中包含 `uncertainty_type` 和 `flow_channels`。
3. 形状检查命令成功返回，且三尺度空间尺寸为 `45x60 / 90x120 / 180x240`。

### 补充验收（实际执行）
1. `conda run -n tf_work_hpc python wrappers/run_supernet_train.py --config configs/supernet_fc2_180x240.yaml --num_epochs 1 --steps_per_epoch 1 --batch_size 2 --fast_mode --experiment_name edgeflownas_supernet_fc2_180x240_uncert_smoke`
2. `python -m code.nas.check_manifest --path outputs/supernet/edgeflownas_supernet_fc2_180x240_uncert_smoke/train_manifest.json --strict`

### 补充结果
1. 训练命令返回 `exit_code=0`，日志包含 `epoch=1` 并正常收敛输出。
2. `check_manifest --strict` 返回 `ok=true`。
