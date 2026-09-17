# 本机开发准备：Windows + RTX 5060 Ti

状态（2026-09-17）：代码已克隆并完成静态核查；经用户同意，已安装Git for Windows 2.55.0.windows.3及WSL 2.7.13.0，并启用VirtualMachinePlatform。Windows明确要求重启，Ubuntu发行版列表仍为空。Docker、TensorFlow和GPU/旧权重验收尚未完成。

下一步：保存工作并重启Windows；随后检查`wsl --list --verbose`，必要时继续`wsl --install -d Ubuntu-24.04 --no-launch --web-download`。确认Ubuntu可以启动和识别GPU后，再安装Docker及NVIDIA Container Toolkit，执行下文验收。不要把安装命令返回成功当作模型运行成功。

## 先看目录

| 位置 | 放什么 |
| --- | --- |
| `C:/00Work/Code/MCUFlowNet` | 日常开发，基于发布版；当前本地分支`setup/local-validation` |
| `C:/00Work/Code/MCUFlowNet-dev` | 历史参照，保持原样；不要同时在这里维护另一套新代码 |
| `C:/00Work/Datasets` | 已有FC2、Sintel等数据，不放进Git |
| `C:/00Work/Runs/MCUFlowNet/<实验编号>` | 后续生成的大权重、完整日志等；实际运行时再创建 |
| `C:/00Work/Lem_brain/wiki/项目/MCUFlowNet` | 中文计划、实验目录、结论；小型结果和环境记录放附件 |

WSL中上述代码目录对应`/mnt/c/00Work/Code/MCUFlowNet`。继续从Windows维护Git，Linux环境负责运行。小测试先直接读取`/mnt/c`数据；如果数据读取成为瓶颈，再单独准备WSL本地数据缓存，不复制第二份开发代码。

## 代码版本与名称

- 发布版起点：`f960ff41ff6f0f7093c610b06b2a8c7991415f50`。
- dev当前头：`68f8565ba0c0bbc8d4ef58ab77b620d0261a2002`。
- 旧训练记录引用的提交已在dev找到：`75bea8028275040a4039ca48a2d34b3bdceeedfe`。
- 核查19个训练/评测相关文件：14个文本一致；另5个在只忽略说明文字、明确的导入模块改名后，语法树一致。这不是完整仓库或运行行为等价证明。
- 发布版EdgeFlowNAS的100个Python文件通过语法解析；静态发现的`from efnas...`模块路径均存在。这不检查外部依赖和动态导入。

| 用途 | dev名称 | 发布版名称 |
| --- | --- | --- |
| FC2入口 | `run_retrain_v3_fc2.py` | `run_retrain_fc2.py` |
| FT3D入口 | `run_retrain_v3_ft3d.py` | `run_retrain_ft3d.py` |
| 主训练程序 | `retrain_v3_trainer.py` | `retrain_trainer.py` |
| S/L固定模型 | `fixed_arch_models_v3.py` | `fixed_arch_models.py` |
| 模型S内部名称 | `v3_light` | 仍为`v3_light` |
| 模型L内部名称 | `v3_efn_fps` | 仍为`v3_efn_fps` |

论文名称S/L可以用于报告，但先保留内部名称。评测器从模型目录名生成变量作用域，改目录名可能使旧权重无法恢复；元信息里的`checkpoint_path`也可能仍指向HPC。旧权重加载检查要比较变量名称和形状，不能只看文件能否找到。

复跑静态核查（只需Python标准库和Git）：

```powershell
python tools/setup/audit_repositories.py --dev C:/00Work/Code/MCUFlowNet-dev --output C:/00Work/Lem_brain/wiki/项目/MCUFlowNet/附件/本机环境核查-20260917/repository-audit.json
```

## 环境分两件事，不强求完全复制HPC

HPC的`module load`是在加载管理员已经编译好的软件，再叠加`~/tf_work`里的包。它并不等于简单的`pip install tensorflow==2.15.1`；还需要导出实际Python版本、包清单和TensorFlow编译信息才能精确追溯。原来的`PYTHONPATH`命令依赖Python3.11目录，不能原样放到本机其他版本环境。

1. **本机GPU环境，优先准备：** WSL2 + Ubuntu 24.04，随后在WSL中安装Docker Engine和NVIDIA Container Toolkit，先试官方`nvcr.io/nvidia/tensorflow:25.02-tf2-py3`。它是TF2.17.0、CUDA12.8的候选环境，不是HPC环境副本，也尚未在这张5060 Ti实测。容器已停止后续月度发布，因此这里只将它作为有官方Blackwell支持依据的兼容起点；验收成功后记录镜像digest。
2. **旧版本参照环境，按需要增加：** 在已有Miniconda中新建Python3.11 + TensorFlow2.15.1的CPU环境，检查旧接口、权重和少量预测。Windows CPU结果用于排查差异，不当作Linux HPC完全复现。不要改现有base环境。

不推荐Windows原生TF2.10作为GPU绕行方案，也不直接将TF2.15.1/CUDA12.1当作5060 Ti已支持组合。新版普通TensorFlow安装包也要经过实际算子验证，不能仅看版本号。Windows驱动已存在，WSL复用它，不在WSL另装Linux显卡驱动。

本项目依赖`tf.compat.v1.layers`。较新TensorFlow默认Keras3时需要检查旧Keras兼容方式；若采用匹配的`tf-keras`，须在导入TensorFlow前设置`TF_USE_LEGACY_KERAS=1`。先检查容器已装内容，再补依赖，避免安装命令把NVIDIA版TensorFlow替换成普通wheel。这里没有提供未经验证的“最终锁定依赖”。

首轮只安装训练/评测所需包：TensorFlow、兼容Keras、NumPy、OpenCV、SciPy、scikit-image、Pillow、Matplotlib、PyYAML、tqdm等。根据实际导入补齐。Vela、LLM API客户端、NAS搜索附加依赖在需要对应功能时再装。

参考：[TensorFlow安装限制](https://www.tensorflow.org/install/pip)、[Keras2兼容方式](https://blog.tensorflow.org/2024/03/whats-new-in-tensorflow-216.html)、[NVIDIA 25.02说明](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-25-02.html)、[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。访问2026-09-17。

## 安装后，按顺序验收

1. WSL能执行`nvidia-smi`。
2. 容器内TensorFlow能真正执行GPU卷积和类型转换，而不只是列出显卡。
3. S/L的真实网络和损失能完成前向、反向及参数更新；先用很小的随机输入排除环境问题。
4. FC2读一小批真实图片，Sintel读少量图片；检查输入颜色、范围、位移单位。
5. 恢复旧权重，检查路径、变量名、形状和预测；必要时与dev历史代码在相同环境中对照。
6. 通过后再接入双列EPE并做100步左右短跑。批次先小，实测显存再增加；小批次的BN行为与历史batch32不同，不直接以短跑分数判断训练优劣。

第2–3项已有验收脚本，但本次仅检查语法和`--help`，没有运行TensorFlow：

```bash
python tools/setup/check_tensorflow.py --device gpu --output /mnt/c/00Work/Lem_brain/wiki/项目/MCUFlowNet/附件/本机环境核查-20260917/gpu-check.json
```

它使用随机数据对S/L各更新两步，不加载旧权重、不启动正式训练。成功后还需完成第4–6项。CPU参照环境可用同一脚本的`--device cpu`，输出到另一个文件。

## Git与SSH：先本地提交，再连接远端

目前两个仓库通过公开HTTPS下载；SSH只用于后续远端认证。Git for Windows已安装到`C:/Program Files/Git`，新开终端后可使用；WSL组件已装，等待重启后继续Ubuntu与容器安装。

两个仓库已按用户指定设置本地提交署名，以下命令仅供日后核对或重新配置：

```powershell
git -C C:/00Work/Code/MCUFlowNet config user.name "Enmin Lin"
git -C C:/00Work/Code/MCUFlowNet config user.email "1780474486@qq.com"
```

本机初查没有`.ssh`目录。生成专用钥匙时在自己的终端运行，下列操作需在检查目标文件不存在后执行；为钥匙设置口令，不把口令发到聊天中：

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE/.ssh"
ssh-keygen -t ed25519 -C "MCUFlowNet-5060Ti" -f "$env:USERPROFILE/.ssh/id_ed25519_github_mcuflownet"
Get-Content "$env:USERPROFILE/.ssh/id_ed25519_github_mcuflownet.pub"
```

将**`.pub`公钥**添加到GitHub Settings → SSH and GPG keys → New SSH key。私钥留在本机。然后先测试：

```powershell
ssh -i "$env:USERPROFILE/.ssh/id_ed25519_github_mcuflownet" -o IdentitiesOnly=yes -T git@github.com
```

第一次连接按[GitHub公布的主机指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)核对。出现已认证但不提供shell的消息属于正常情况。测试通过后再在本仓库设置专用钥匙及SSH remote：

```powershell
git -C C:/00Work/Code/MCUFlowNet config core.sshCommand "C:/Windows/System32/OpenSSH/ssh.exe -i C:/Users/Administrator/.ssh/id_ed25519_github_mcuflownet -o IdentitiesOnly=yes"
git -C C:/00Work/Code/MCUFlowNet remote set-url origin git@github.com:linenmin/MCUFlowNet.git
```

此设置供Windows Git使用；WSL负责运行，不混用这条Windows路径执行Linux Git。以后需要免重复输入口令时，再按[GitHub Windows ssh-agent说明](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=windows)配置agent。

每次实验记录当前提交和配置；有未提交改动时先保存补丁或提交，不能只记录HEAD。大型权重不提交Git。新实验使用独立输出目录，不覆盖仓库已附的旧论文结果。当前没有push，也没有生成SSH私钥。
