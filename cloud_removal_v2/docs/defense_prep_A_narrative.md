# v2 答辩准备 · Part A：执行摘要与研究叙事

> 面向导师汇报与师兄考较。Part A 建立全局图景，Part B 深入技术细节，
> Part C 预设问答。此文件约 170 行，控制在单页内读完。

---

## 1. 一句话摘要（汇报开场）

> **"我们把 FLSNN 从 EuroSAT 分类扩展到 CUHK-CR 云去除的像素回归，
> 发现：在源级 Dirichlet(α=0.1) 非 IID 下，Theorem 2 的 ζ² 项收缩
> 足够小，使得 AllReduce / Gossip / RelaySum 的收敛阶相等——这正是我
> 们实测排名反转（AllReduce Pareto 最优）的理论解释。**
>
> **同时的控制消融发现：FedBN 的 PSNR 增益在 TDBN / BN2d 下都处于
> 单 seed 噪声带内（+0.008 / +0.044 dB），而 TDBN 确实把 plane 间
> Var(γ) 压到 BN2d 的 58%——机制存在但效应消失。"**

---

## 2. 三段式研究动机（讲给导师）

### 2.1 问题：Space-CPN 的能耗与通信瓶颈
LEO 卫星星座是近未来的"空中边缘计算平台"，但满足对地观测 / 灾害监测
等应用的 **分钟级时延** 需要把推理 / 训练搬上卫星。然而星上运行 ANN 会
直接和通信、姿控、载荷争抢有限电力。每一焦耳用于矩阵乘法的能量，就是
一焦耳不用于下行或电池寿命维持的能量。

### 2.2 解：FLSNN 开创的 "SNN + 去中心化 FL"
FLSNN（Yang et al., arXiv:2501.15995, 2025）首次把 SNN 与 50/5/1
Walker-Star 星座的去中心化 FL 结合，在 EuroSAT 10 类分类任务上：
- SNN 相对 ANN **精度损失 < 3%**（即达 ANN ~97%）
- **每层能耗 ~10× 下降**（Horowitz 2014，45 nm CMOS：4.6 pJ/MAC → 0.9 pJ/AC）

### 2.3 Gap：真实卫星任务是像素回归，FLSNN 没覆盖
FLSNN 只做分类。真实卫星任务（云去除、去噪、超分辨）是 **像素回归** ——
损失结构、梯度方差、BN 作用都不同。**我们的 v2 就是去填这个 gap**，
并在这个新设置下重新审视 FLSNN 的三个核心 claim。

---

## 3. 三个 gap（§I.D 原文版）

| Gap | 内容 | 为什么重要 |
|:---:|:-----|:----------|
| **G1 任务类型** | FLSNN 只评估分类，不知道 RelaySum 排名在回归下是否仍成立 | Charbonnier + SSIM 损失的梯度方差结构与 CE 完全不同 |
| **G2 非 IID 类型** | FLSNN 用标签级 Dirichlet（10 类 × ς=0.02）| 真实卫星数据更自然是**源级**异构（不同传感器 / 不同云厚）|
| **G3 归一化消融** | FLSNN 把 TDBN 当"砌块"未和 BN2d / FedBN 对比 | TDBN 的 α·V_th 共享缩放是否让 FedBN 的 BN-local 机制变冗余，是开放问题 |

---

## 4. 我们的 5 个贡献（§I.E 骨架）

### C1（理论）Proposition 1 — Dirichlet 到 ζ² 的闭式
$$ \mathbb{E}_{\mathbf{p}}\Bigl[\tfrac{1}{N}\sum_i \|\nabla f_i - \nabla f\|^2\Bigr] = \tfrac{N-1}{4N(2\alpha+1)}\,\|\nabla f_1 - \nabla f_2\|^2 $$
在 N=50, α=0.1（我们的 setting）代入：**c_α = 0.204**。

### C2（理论）Corollary 1 — Scheme 层级崩塌
在小 ζ² 下 Theorem 2 的 T_4 项消失在 T_1 之前，三种 scheme 渐近阶相等。

### C3（实验·主）Claim C16 — FedBN 在此任务下条件冗余
| run | mean Δ(FedBN − FedAvg) |
|:---:|:----------------------:|
| A (TDBN/SNN) | **+0.008 dB** |
| B (BN2d/SNN) | **+0.044 dB** |
两者都 < 0.05 dB 的 single-seed 噪声地板。**机制层** 存在：TDBN plane 间
Var(γ) 只有 BN2d 的 **58%**；但即使 BN2d 的漂移也太小不足以让 FedBN
产生可观增益。

### C4（实验）ANN 在 GPU 上全面胜 SNN
同 fedbn+AllReduce cell（run A vs C）：
- **+0.751 dB PSNR**
- **+0.0266 SSIM**
- **1.61× wall-clock 快**（此为下界；若 ANN 重实现为 T=1 会再快 ~2.5×）

### C5（基础设施）诚实可复现
3 个 summary.json + 60 个 ckpt → 所有表格 1 条 shell 命令能重算；
51-layer drift script + 4 tables + 4 audit commits。

---

## 5. 最关键 3 个数字 + 每个数字的"故事"

### 数字 ① — **c_α = 0.204**（Proposition 1 N=50, α=0.1）
- 意义：我们的源级 Dirichlet 比 FLSNN 的 ς=0.02/10 label 弱很多
- 推导：(N-1)/(4N(2α+1)) 在 N=50, α=0.1 代入 = 49/240 ≈ 0.204
- 为什么小：2 源 vs 10 label；源间视觉相似（同卫星不同云厚）；
  Dirichlet 支撑集更小
- 闭环：小 c_α → 小 ζ² → Theorem 2 的 ζ²/σ² 比小 → T_4 dominate by T_1
  → 三 scheme 渐近阶相等 → **§VI-D 实测 ≤0.3 dB 差距得到理论解释**

### 数字 ② — **FedBN Δ = +0.008 / +0.044 dB**
- 意义：主实验 null result —— FedBN 在此任务下几乎无效
- 来源：A、B 两 run 的 6 cell PSNR_final 算 pairwise 差 + mean
- 机制：Table III 的 Var(γ) 比 0.58 说明 TDBN 确实对齐 plane 间统计；
  但绝对值 ~10⁻⁴ 已经"太小"对 FedBN 无所谓
- 对比文献：FedBN 原文在分类 + 域迁移下有 **+7.8 pp**；我们的 +0.008
  dB 把 FedBN 推到了其 regime 的 **末端**

### 数字 ③ — **+0.75 dB ANN − SNN（同 cell）**
- 意义：FLSNN 说 SNN 精度差 ~3%，我们说差 ~0.75 dB——同方向，不矛盾
- 诚实披露：**1.61× 墙钟加速是 lower bound**（ANN 仍保留 T=4 外循环，冗余 4×）
- **SNN 能耗优势 v2 不报**：MultiSpike-4 是 5 级量化，不是 binary，
  0.9 pJ/AC 的 Horowitz 公式不适用。v3 会补精确测量。

---

## 6. 叙事闭环一句话版

```
小 ζ²    ——Proposition 1──▶   Corollary 1 (scheme collapse)
   ▲                                    │
   │                                    ▼
源级 2-src                          §VI-D 实测
Dirichlet 天然                    AllReduce Pareto 占优
α=0.1                              spread ≤ 0.3 dB
   │                                    │
   └────§VI-B 非IID 诊断──▶ §VI-C FedBN null result ←──Table III Var(γ)=0.58×
                                                       (机制在，效应消)
```

---

## 7. 导师最可能问的 3 句话 & 我们的回答

**Q1（大方向）："这不就是把 FLSNN 换个数据集再跑一遍？"**

> 答：不。我们贡献 3 点：(a) **Proposition 1** 是新的理论结果，FLSNN 没
> 给闭式 ζ²。(b) **Claim C16** 是新的消融：FLSNN 从未在 SNN 上对比
> FedBN vs FedAvg × TDBN vs BN2d。(c) 我们的 scheme-ranking reversal
> 不是"复现失败"，而是 **FLSNN 自己 Theorem 2 的边界情况的实证**。

**Q2（价值）："能发什么刊物？"**

> 答：目标 IEEE Network（Matthiesen 2023 同刊）或 TGRS（Sui 2024 云去
> 除 benchmark 同刊）。主卖点包装为 "First systematic BN / scheme /
> backbone ablation on satellite SNN-FL for pixel regression"。

**Q3（风险）："单 seed，reviewer 会不会挂？"**

> 答：会被要求多 seed，但我们已 (a) 把 §VI-H L-ledger 和 §VII.A v3 列
> 表显式披露；(b) 所有 claim 都加了 "within single-seed noise floor"
> 措辞；(c) Corollary 1 提供了理论解释使得单 seed 结果不是孤立事实。
> 如 reviewer 坚持，v3 multi-seed 已规划（228 GPU hr）。

---

## 8. Part A 到此为止

Part B（理论+系统+实验**深度细节**）、Part C（**25+ 问答**）在后续文件。
