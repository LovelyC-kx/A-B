# v2 答辩准备 · Part C：25+ 预设问答

> 每题格式：**短答（30 秒口头可用）** / **深答（追问时展开）** / **退路（被逼到墙角）**。

---

## A 类：导师级「大方向 / 新颖性 / 投稿价值」 · 5 问

### A1. 这不就是把 FLSNN 换个数据集再跑一遍吗？
- **短答**：不是。我们有 3 项原创：(1) Proposition 1（闭式 ζ²）；(2) Claim C16（FedBN 条件冗余）；(3) 反转结果的理论解释为 FLSNN 自己 Theorem 2 的边界情况。
- **深答**：FLSNN 没给 ζ² 的显式表达式，也从未对比 FedBN vs FedAvg 或 TDBN vs BN2d。我们补了这两个消融矩阵，并提供机制层证据（Table III 的 Var(γ) = 0.58 比率）。
- **退路**：即使 reviewer 说 "novelty 不够"，我们也仍是**首个在卫星 SNN-FL 上做像素回归 + 控制消融**的工作（§II.D/II.E positioning 已这样定位）。

### A2. 这能发什么刊物？
- **短答**：目标 **IEEE Network** 或 **IEEE TGRS**。
- **深答**：IEEE Network 因为 Matthiesen 2023 "FL in Satellite Constellations" 就在同刊；TGRS 因为 Sui 2024 的 CUHK-CR benchmark 在同刊，能同 benchmark 比较。
- **退路**：退而求其次 IEEE Communications Letters（FL-卫星 letter 有先例：razmi2022wcl）。

### A3. 主卖点一句话是什么？
- **短答**：**在源级非 IID + 像素回归下，FedBN 条件冗余，AllReduce Pareto 最优。**
- **深答**：两个表达式：
  - 理论：Proposition 1 + Corollary 1 给 boundary-case 解释
  - 实验：Claim C16（FedBN Δ < 0.05 dB）+ AllReduce 用 60% 通信达同 PSNR

### A4. 接下来怎么走？v3 做什么？
- **短答**：三 tier。Tier-1 multi-seed（228 GPU hr）；Tier-2 α-sweep + 能耗测量；Tier-3 多时序云去除。
- **深答**：v3 关键要把单 seed 的 "近似 vs 真 null" 辨清。α-sweep（0.01/1/10）能实证追出 c_α 曲线从 0.204 降到 0.012，对应 Corollary 1 的 scheme-collapse 到 scheme-separate 的过渡。
- **退路**：如果资源有限，v3 最小版只做 multi-seed + energy measurement 就够回应 reviewer。

### A5. 最容易被挂的点是什么？你怎么答？
- **短答**：**单 seed**。回答：(1) 已在 §VI-H L1 显式披露；(2) Corollary 1 的理论预测让单 seed 不是孤立事实；(3) v3 multi-seed 已规划。
- **深答**：其次易挂 **"ANN 在 GPU 更快" 和 SNN 的价值矛盾**。回答：我们不 claim SNN 在 GPU 上更快；SNN 的价值在**部署时的神经形态硬件能耗**，v2 未测但 v3 补齐（§VI-E.4.3）。

---

## B 类：师兄级「技术 / 推导 / 实现」 · 12 问

### B1. Proposition 1 真的原创吗？FLSNN 难道没给过？
- **短答**：FLSNN 只把 ζ² 当抽象常数。Proposition 1 在 2-源 Beta(α,α) 下给闭式 $\tfrac{N-1}{4N(2\alpha+1)}$。
- **深答**：FLSNN §IV 的 C、C₁、z 吸收了 ζ²；未暴露 α-依赖。Proposition 1 让 α → ζ² 可计算，使 Corollary 1 成为 decidable claim。
- **退路**：如 reviewer 指出 "这是 Beta 方差的标准结果"，我们承认**推导步骤**经典，但**连接到 FLSNN ζ²** 是新的。

### B2. 为什么选 Dirichlet 而不是 label-shift？
- **短答**：卫星数据天然有源级异构（不同传感器 / 云厚），没有 label。
- **深答**：像素回归任务没有 per-sample class label，label-shift 无意义。CUHK-CR1 vs CR2 是天然双源。
- **退路**：若 reviewer 要求 label-shift 对比，v3 可做 patch-level partition 近似 label-shift。

### B3. 为什么 α=0.1 不是更极端的 0.01？
- **短答**：FLSNN Fig 5 用 ς=0.02 over 10 labels；我们 α=0.1 over 2 sources 的 effective K 约等价。
- **深答**：Proposition 1 给 α-sensitivity 表（§D1.8）：α=0.01 → c_α=0.22, α=0.1 → 0.20, α=1 → 0.08。0.1 与 0.01 差不到 10%，但 0.1 的 Dirichlet 采样更稳定（0.01 下许多客户端会低于 min_per_client=5 被迫 clipping）。
- **退路**：v3 α-sweep 会包含 0.01 / 0.5 / 5.0 三个对比点。

### B4. MultiSpike-4 和 FLSNN 的 binary 不一致，能耗比较还有意义吗？
- **短答**：**v2 不报 SNN/ANN 能耗比**。§VI-E.4.3 显式声明。
- **深答**：MultiSpike-4 是 {0, ¼, ½, ¾, 1} 5 级。Horowitz 0.9 pJ/AC 是 **binary-spike 下界**，对 5 级低估。v3 给两版估：aggressive（sparsity × 0.9 pJ）+ conservative（sparsity × 4.6 pJ）。
- **退路**：为保稳妥，v2 只 claim ANN 在 GPU 上的测得速度，不涉能耗。

### B5. TDBN 的 γ_init=α·V_th 是你们原创吗？
- **短答**：不是。是 Zheng 2021 AAAI 的 spikingjelly 实现。我们直接用。
- **深答**：我们的贡献是**在 FL 下识别到 TDBN 的 γ_init 共享效应使 plane 间 BN 漂移减半**。这是 Claim C16 机制层（Var(γ) 比率 0.58）的根源。
- **退路**：若 reviewer 说 "TDBN 的 alignment 效应在文献早有报告"，我们指出 **FL 场景下 plane 间 Var(γ) 测量**是我们的新贡献。

### B6. ANN 比 SNN 快 1.61× 和 SNN 文献说更快矛盾吗？
- **短答**：不矛盾。SNN 的能耗优势在**部署时的神经形态硬件**（Loihi / Truenorth）。GPU 上 SNN 仍然是 ANN 的 4× 外循环。
- **深答**：我们 ANN backbone 未优化（保留 T=4 outer loop），所以 1.61× 是**下界**；T=1 重实现预期 ~4×。即便如此，ANN 训练快是符合 FLSNN Fig 6 方向（~2% 精度差，2× 速度）的一致观察。
- **退路**：若被逼到墙角，我们说 SNN 部署的 deployment-time value 是 FLSNN 原文的立论，v2 专注于训练行为。

### B7. 51 BN layers 这个数你怎么确定的？旧脚本是 41 / 54？
- **短答**：架构数学推导 = 13 SRB × 3 BN + 3 SUNet × 2 内联 + 3 边缘 = 51。旧脚本 substring 法漏 13 个（SRB.shortcut[1] 路径无 "bn" 子串）。
- **深答**：`analyze_bn_drift_posthoc.py`（commit 17cd881）改用 BN structural signature（`.weight` + `.bias` + `running_mean` + `running_var` 同前缀），跟 class naming 无关，捕到全部 51 层。
- **退路**：若 reviewer 要求 `isinstance` 验证：我们的脚本已经有这个逻辑。

### B8. 如果 reviewer 说你的反转是 lr_scale=2.093 造成的而不是任务类型？
- **短答**：**我们不单一归因**。§VI-D 列 3 个 candidate cause（D1/D6/D9），未单独消融。
- **深答**：D1 任务（回归 vs 分类）+ D6 优化器（AdamW vs SGD）+ D9 lr_scale 都可能贡献。单 seed 信噪比不足以拆分。v3 的 D9-ablation（lr_scale=1.0 重跑 4 cell，~24 GPU hr）可拆出这一项。
- **退路**：Corollary 1 的 ζ² → 0 极限预测scheme collapse，这独立于 lr_scale 的具体值；所以即便 lr_scale 选错，scheme 差距小这件事仍成立。

### B9. FedBN 的 +0.008 dB 有统计意义吗？
- **短答**：单 seed 不能判定"统计意义"。我们声明其在 noise floor 内。
- **深答**：§VI-D.1 实测 per-scheme 6-cell PSNR std ~0.05 dB；0.008 和 0.044 都 < 这个地板。机制层（Table III Var(γ)=0.58 比率）独立支持 TDBN alignment 效应。
- **退路**：claim 从 "FedBN 普遍冗余" 收紧为 "在此 regime 下条件冗余"（§VI-C.5 已这样写）。

### B10. 定性图（§VI-F）和 test set 排名反了怎么解释？
- **短答**：6 样本的 PSNR std ~2-3 dB，远大于 cell 间 spread ~0.3 dB。样本选择偏差。
- **深答**：`visualize.py` 用 seed=42 固定选 6 个 test sample（indices [24, 6, 153, 212, 199, 177]），恰好是 A 强 / C 弱的样本组合。**6-sample mean 的标准误 ≈ 1 dB > Table I 所有 cell 间 spread**，所以 qualitative rank 不能下论断。
- **退路**：§VI-F 已显式说"qualitative panel 不做 ranking claim"，只支撑"easy / moderate / thick-cloud failure"三档描述。

### B11. per-plane PSNR 是 ensemble per-image，这会影响什么？
- **短答**：FedAvg+AllReduce 下所有 plane 相同，per-plane == mean-of-plane-means 等价。FedBN 下不等价。
- **深答**：我们用 ensemble-per-image（先对每图取 5 plane 平均，再对 245 图取均）。FLSNN 原文用类似语义。Reviewer 若要 per-plane mean，我们有 `per_plane_psnr` 数组可重算。
- **退路**：§VI-A.6 已明确披露 metric 定义。

### B12. 你们的 §V MDST 其实没跑实验，是不是掺水？
- **短答**：不。§V 明确声明继承 FLSNN §V 公式，**不 claim 实证验证**；我们的 5 plane chain 的 MDST 是 trivially chain 本身，无优化空间。
- **深答**：大星座（42/7/1 Walker-Delta）才是 MDST 非平凡。§VI-H.3 列 MDST 为 v3 item，~168 GPU hr。
- **退路**：如要求删 §V，我们可把它压缩成 §III 最后一段 "topology optimisation out of scope for v2"。

---

## C 类：方法论 / 严谨度 · 6 问

### C1. 3 run 的配置严格相同吗？
- **短答**：是。每 run 的 summary.json["config"] 逐字段比过：A↔B 只差 bn_variant，A↔C 只差 backbone。B↔C 差 2 项，**不做直接对比**（Table IV 明文标明）。
- **深答**：配置表含 16 字段（seed, partition_seed, α, planes, sats, T, dim, en/de_blocks, lr, warmup, epochs, intra/local iters, Charbonnier eps, SSIM weight, eval_mode）。

### C2. 你们的 51 BN drift 数字来自哪个脚本？是否交叉验证？
- **短答**：`analyze_bn_drift_posthoc.py`（commit 17cd881），用 BN signature 而非 key substring。
- **深答**：AutoDL 上跑出 `Outputs_v2/v2_drift_report.md`（截图确认存在）。Sanity check：FedAvg+AllReduce Var(γ) = 0.000e+00（两 run 都），证明 AllReduce 把所有 plane 搞成 bit-identical，aggregation 正确性通过。

### C3. 为什么单 seed 信心够发？
- **短答**：三理由：(1) 11 headline numbers 跨节一致（MASTER_INDEX §5）；(2) Corollary 1 理论解释让 0.3 dB spread 不意外；(3) §VI-H L1 显式披露。
- **退路**：若 reviewer 要求，v3 multi-seed 已规划成 Tier-1 优先级。

### C4. npz 里 bn_drift 和 cos_sim 字段没存是不是 bug？
- **短答**：**代码 gap，非 bug**。`inline_logging.py` 记录到 `history` dict，但 `run_smoke.py:602-612` 的 `_atomic_savez` 漏 persist。§VI-G.5 + §VI-H.2 L6/L7 显式披露。
- **深答**：因此 §VI-G 只能给 end-of-training 截面（由 ckpt post-hoc 算），不给 per-epoch 轨迹。v3 1 行修复即可。

### C5. Partition 是否真的复现 72% pure single source?
- **短答**：是。`plot_partition_heatmap.py` 用 seed=0 固定，`Outputs_v2/v2a_v2a_80ep_partition_summary.txt` 记录 72%。
- **深答**：跨机器 bit-stable，因为 `np.random.RandomState(seed=0)` 在 `dirichlet_source_partition()` 内部。

### C6. 你们 13 cell 主结果的 wall-time 数字对吗？
- **短答**：A 36.31 h + B 36.66 h + C 3.86 h = 76.83 h（§VI-A.7 精确数）。
- **深答**：每 cell 的 `total_wall_seconds` 来自 summary.json["final"][cell]；6 cell × 2 run + 1 cell C 求和，与 Table I 的 h 列全位一致。

---

## D 类：未来工作 / 投稿技巧 · 2 问

### D1. 若 reviewer 要求 centralised baseline？
- **短答**：v3 ledger V8，~12 GPU hr，单客户端 VLIFNet 训 full 982 图。预计给出 "federation loss" 上界。
- **深答**：centralised 应比我们最好 SNN cell (21.79) 高 ~1-2 dB；ANN centralised 可能接近 centralised cloud-removal SOTA (DE-MemoryNet 26.18 dB 在 CR1)。

### D2. 如果评审给大改意见？
- **短答**：接受的方向——补 multi-seed + α-sweep + energy measurement 是我们已规划的 v3 3 项核心。**拒绝的方向**——重做 classification / 换模型架构 / 改变 FL 算法，因为这些会偏离"对 FLSNN 自然扩展"的 core positioning。
- **退路**：如被要求重做主实验，最坏情况是 v3 → 全新 submission（走 venue 更匹配）。

---

## 结语：答辩 talking points

记住 **3 个最强数字 + 3 个最强 slogan**：

| 数字 | 对应 slogan |
|:----:|:-----------|
| **c_α = 0.204** | "Small ζ² 解释了 scheme collapse" |
| **Var(γ) ratio 0.58** | "TDBN 有 alignment 机制，但效应层微小" |
| **+0.75 dB ANN** | "GPU 上 ANN 全胜，SNN 价值在部署" |

**3 次重申 non-claim**（§II.G / §VI-H.2 / §VII.A）让对方抓不到"过度推销"把柄。
