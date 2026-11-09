# NSCO causal transfer v1 — 冻结完整模型上的配对因果实验

## 范围及边界

这是**真实训练 checkpoint 上的一步局部反事实实验**，不是新网络训练，也不是独立最终测试。使用已有 formal 原始冻结快照、原模型、PairDataset、CR1 466/52 去重清单和原有 best/last checkpoint。源数据、checkpoint、快照不改；输出不含图像和权重。它不尝试模拟历史 AdamW 更新，而明确采用相同全局 L2 范数的纯参数方向扰动。原始 validation 曾参与 best checkpoint 选择，因此 audit 子集只相对于 pilot 的策略选取独立，不是外部测试集。

A/B 输入来自同一张 cloudy/clear 配对图像，`x_alpha = clamp(y + alpha*(x-y),0,1)`，alpha A=0.5，B=1.0。这是相同数据、相同目标的退化强度条件，**不是两个独立去云/去雨任务**。

每个 checkpoint 进行：训练子集计算 A/B 梯度；测量梯度夹角；对留出 val 子集施加 A 梯度 / 同范数随机方向 / 反方向；在独立 pilot val 图上选择最佳单层组，与梯度余弦选择组及随机组在 audit val 上比较。每次干预后严格恢复参数。统计是单次冻结检查点的局部响应；不能据此声称完整训练收益。分层候选按参数数量选至多 8 个，必须记录这一筛选规则。

## 0. 验证服务器原有工程

需要在服务器已经存在的目录：`/data/bohan_zhang/nsco_workspace/nsco-lab`，其中 `formal/p0_gain_sweep.py` 和原始正式训练输出存在，激活原有 bench 环境即可；不自动更新 git 或安装依赖。

服务器已有实验路径可从：

```bash
cd /data/bohan_zhang/nsco_workspace/nsco-lab
source scripts/activate_bench.sh
cat local/p0_source_run.txt
find outputs -maxdepth 3 -name frozen_config.json -print
nvidia-smi -L
```

必须确认 `EXPERIMENT` 指向 `frozen_config.json` 和 `snapshot_hash.json` 所在目录，且至少有 `jobs/N0_nafnet_seed0/best.pt`、`jobs/N4_cross_extra_seed0/best.pt`。这里不假设 GPU2/3 是物理编号还是掩码内编号：**服务器 `nvidia-smi -L` 确认后显式填写 GPU_A/GPU_B 的物理索引或 UUID**。每个进程仅暴露自己一张卡，因此内部都是 `cuda:0`。两张卡绝不混用同一个进程。

## 1. 本地 Windows PowerShell 上传（替换三项连接变量）

```powershell
$Server = 'YOUR_USER@YOUR_HOST'
$Port = 22
$LocalPackage = "$HOME\Downloads\causal_transfer_v1.tar.gz"
$RemoteFolder = '/data/bohan_zhang/nsco_workspace'
scp -P $Port $LocalPackage "${Server}:${RemoteFolder}/"
if ($LASTEXITCODE -ne 0) { throw '上传失败' }
```

下载本聊天提供的压缩包放入本地 Downloads 文件夹。`RemoteFolder` 必须存在且有写权限，否则改为远程用户的家目录，并同步调整下一步路径。

## 2. 服务器解包、固定路径、启动

```bash
set -euo pipefail
cd /data/bohan_zhang/nsco_workspace
mkdir -p causal_transfer_v1
# 不覆盖现有代码；对本次压缩包独立验 hash
sha256sum causal_transfer_v1.tar.gz
# 可与本聊天提供的 .sha256 文件比对；若不符停止
tar -xzf causal_transfer_v1.tar.gz -C causal_transfer_v1
cd /data/bohan_zhang/nsco_workspace/nsco-lab
source scripts/activate_bench.sh
export REPO="$PWD"
export EXPERIMENT="$(cat local/p0_source_run.txt)"
# 如没有 local/p0_source_run.txt，用 find 输出中真实路径手动 export EXPERIMENT=/absolute/path
export GPU_A=2
export GPU_B=3
nvidia-smi -i "$GPU_A" --query-gpu=index,uuid,name,memory.free --format=csv
nvidia-smi -i "$GPU_B" --query-gpu=index,uuid,name,memory.free --format=csv
nohup bash /data/bohan_zhang/nsco_workspace/causal_transfer_v1/start_server.sh \
 > "$REPO/causal_v1_launcher.log" 2>&1 < /dev/null &
echo $! > "$REPO/causal_v1_launcher.pid"
```

其中 GPU_A=2 和 GPU_B=3 **仅表示用户确认物理 GPU 2 和 3 的情况**。若你的掩码使用 UUID，请设成对应两个授权 UUID；不要硬运行编号。若 GPU 忙碌不要与已有训练抢卡。

默认 N0 与 N4，单实验 train6/pilot6/audit6、128×128 中央裁剪。减轻负载：`export TRAIN_IMAGES=3 PILOT_IMAGES=3 AUDIT_IMAGES=3`，默认建议不变。

查看：

```bash
cd /data/bohan_zhang/nsco_workspace/nsco-lab
cat local/causal_v1_last_run.txt
RUN="$(cat local/causal_v1_last_run.txt)"
tail -n 60 causal_v1_launcher.log
tail -n 40 "$RUN/N0_nafnet.log"
tail -n 40 "$RUN/N4_cross_extra.log"
cat "$RUN/RUN_INDEX.json"
```

若失败：`RUN_INDEX.json` status=failed，检查两个 log；不要覆盖已有输出。修复环境后换 `RUN_ID` 重跑；程序每次输出均为独立目录。

## 3. 服务器打包结果（仅全部成功后）

```bash
cd /data/bohan_zhang/nsco_workspace/nsco-lab
export REPO="$PWD"
bash /data/bohan_zhang/nsco_workspace/causal_transfer_v1/collect_server.sh
cat "$(cat local/causal_v1_last_run.txt)/RUN_INDEX.json"
ls -lh "outputs/$(basename "$(cat local/causal_v1_last_run.txt)")_transfer.tar.gz"
```

## 4. Windows 下载结果（连接变量同上）

```powershell
$Server = 'YOUR_USER@YOUR_HOST'
$Port = 22
$RemoteFolder = '/data/bohan_zhang/nsco_workspace/nsco-lab'
# 把下面文件名替换为服务器 collect_server.sh 显示 PACKAGE= 的确切文件名
$ResultFile = 'causal_v1_YYYYMMDDTHHMMSSZ_transfer.tar.gz'
$SaveDir = "$HOME\Downloads\causal_results"
New-Item -ItemType Directory -Force $SaveDir | Out-Null
scp -P $Port "${Server}:${RemoteFolder}/outputs/$ResultFile" $SaveDir
if ($LASTEXITCODE -ne 0) { throw '下载失败' }
scp -P $Port "${Server}:${RemoteFolder}/outputs/$ResultFile.sha256" $SaveDir
if ($LASTEXITCODE -ne 0) { throw '下载 hash 失败' }
```

## 输出分析

`result.json`：A/B 梯度范数及夹角、参数等范数对照、pilot 单组效果、audit 单组效果、恢复精度、源 sha、实际采样索引。`group_pilot.csv` 和 `group_audit.csv`：直接比较三类选层策略。`RUN_INDEX.json` 状态为 completed 才收集结果。

若 `pilot_selected` 在 audit 子集不优于随机组，暂不声称发现层级控制规则；若提高，也只是选择条件下的局部一步验证。下一轮多种子与完整训练需要独立计划，不能用本包结果冒充。