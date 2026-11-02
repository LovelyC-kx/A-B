# v2 答辩准备 · Part B：技术深度细节

> 师兄级"考较"用。每一步推导 + 每一处代码实现 + 每一个实验决策的
> 来源都列在此。Part A 讲"what"，Part B 讲"how and why"。

---

## 1. Proposition 1 全步推导（§IV.C / §D1.1-D1.5）

### 1.1 Setup（§D1.1）
两数据源 $\mathcal{D}_1$（CR1）与 $\mathcal{D}_2$（CR2），定义纯源风险
$f_s(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}_s}[\ell(\theta;x,y)]$。

客户端 $i$（$i=1,\dots,N$）的混合比 $p_i \sim \mathrm{Beta}(\alpha,\alpha)$，
其条件风险 $f_i(\theta; p_i) = p_i f_1(\theta) + (1-p_i) f_2(\theta)$。
全局风险 $f = \tfrac{1}{N}\sum_i f_i = \bar p_N f_1 + (1-\bar p_N) f_2$。

### 1.2 Gradient decomposition（§D1.2）
对 $\theta$ 求梯度（$p_i$ 与 $\theta$ 独立）：
$$ \nabla f_i - \nabla f = (p_i - \bar p_N)(\nabla f_1 - \nabla f_2) $$
**核心秩-1 结构**：客户端维度标量 $(p_i - \bar p_N)$ 乘以 $\theta$ 空间
定向量 $(\nabla f_1 - \nabla f_2)$。平方：
$$ \|\nabla f_i - \nabla f\|^2 = (p_i - \bar p_N)^2 \cdot \|\nabla f_1 - \nabla f_2\|^2 $$

### 1.3 Beta moments（§D1.3）
Johnson-Kotz-Balakrishnan §25.2：
- $\mathbb{E}[p_i] = 1/2$（对称）
- $\mathrm{Var}(p_i) = \alpha^2/((2\alpha)^2(2\alpha+1)) = 1/(4(2\alpha+1))$

### 1.4 $\mathbb{E}[\sum_i (p_i - \bar p_N)^2]$（§D1.4）
用标准身份 $\sum_i(p_i - \bar p_N)^2 = \sum_i p_i^2 - N\bar p_N^2$，
取期望 + 独立性 + $\mathrm{Var}(\bar p_N) = \mathrm{Var}(p_i)/N$：
$$ \mathbb{E}\!\left[\sum_i (p_i - \bar p_N)^2\right] = (N-1)\,\mathrm{Var}(p_i) = \frac{N-1}{4(2\alpha+1)} $$
除以 $N$ 得平均：$\tfrac{N-1}{4N(2\alpha+1)}$。

### 1.5 合并（§D1.5）
代入 1.2 的 pointwise 等式：
$$ \boxed{\mathbb{E}_{\mathbf{p}}\!\left[\tfrac{1}{N}\sum_i \|\nabla f_i - \nabla f\|^2\right] = \frac{N-1}{4N(2\alpha+1)}\|\nabla f_1 - \nabla f_2\|^2} $$

**在 v2 设置 N=50, α=0.1**：系数 = 49/(4·50·1.2) = 49/240 = **0.2042**。

---

## 2. Corollary 1 详细论证（§IV.D）

FLSNN Theorem 2 四项：$T_1 = O(\sigma r_0/\sqrt{NT})$；$T_2 = O((\sqrt{\tilde\tau}\sigma/\rho)^{2/3}/T^{2/3})$；
$T_3 = O(\sqrt{\tilde\tau} L /\rho T)$；$T_4 = O(z \cdot F(E,R,\rho,\tilde\tau)/T^{2/3})$ 其中 $z^2 = \sigma^2+\delta^2+\zeta^2$。

**关键**：只有 $T_4$ 显式含 $\zeta^2$（经 $z^2$）；$T_2$/$T_3$ 只含 $\sigma$；
$T_1$ 完全不含 topology。

令 $c_\alpha = (N-1)/(4N(2\alpha+1))$，则 $\zeta^2 \le c_\alpha G^2$ 其中
$G=\|\nabla f_1-\nabla f_2\|$。随 $c_\alpha \to 0$：
$T_4 \le O(\sqrt{c_\alpha}\cdot G/T^{2/3}) \to 0$ 在 $T_1$ 之前。

**含义**：三 scheme（AllReduce / Gossip / RelaySum）唯一差别在 $(\rho,\tilde\tau)$，
而 $(\rho,\tilde\tau)$ 只入 $T_2/T_3/T_4$。当 $T_1$ 主导时，scheme 差消失。

**我们的 v2 数字**：$c_\alpha \!=\! 0.204$ → $\zeta^2 \!\le\! 0.204 G^2$。
实测 $\sigma^2$（Charbonnier + SSIM 像素回归）量级远 > $G^2 \!\cdot\! c_\alpha$
→ $T_1$ dominate → 三 scheme 实测差 **0.1-0.3 dB** ≈ single-seed noise ✓。

---

## 3. VLIFNet 架构内部（§III.C + `vlifnet.py`）

### 3.1 总体：2,308,856 参数 U-Net
```
Input [T=4, B, 3, 64, 64]
  ↓ patch_embed (Conv2d 3→24)
  ↓ encoder_level1 (SUNet_Level1_Block × 1)
  ↓ down1_2 (DownSampling) + encoder_level2 (SRB × 2)
  ↓ down2_3 + encoder_level3 (SRB × 4)
  ↓ decoder_level3 (SRB × 2)
  ↓ up3_2 + skip_fusion_level2 + decoder_level2 (SRB × 2)
  ↓ up2_1 + skip_fusion_level1 + decoder_level1 (SUNet × 1)
  ↓ additional_sunet_level1 (SUNet × 1)
  ↓ output (Conv2d 24→3) + global residual
Output [B, 3, 64, 64]
```

### 3.2 两个可切换轴
**BN variant**（`_make_bn()` factory at `vlifnet.py:113-127`）
- `tdbn` → `spikingjelly.ThresholdDependentBatchNorm2d(num_features, α, V_th)`
  - γ 初始化 = α·V_th ≈ 0.106（α=1/√2, V_th=0.15）
- `bn2d` → `StandardBN2dWrapper(nn.BatchNorm2d)` 包 [T,B,C,H,W]→[TB,C,H,W]
  - γ 初始化 = 1

**Backbone variant**（`_make_lif_or_relu()` at `vlifnet.py:170-183`）
- `snn` → `spikingjelly.LIFNode` + 自定义 `mem_update`(内联 LIF 积分 + MultiSpike4)
- `ann` → `nn.ReLU(inplace=False)` 替换每个 LIF 神经元

**关键：总参数量两个轴完全一致**（A 和 B 均为 2,308,856；用户用
state_dict size 实证过）。

---

## 4. 51 BN layers 怎么来的（审计发现之旅）

### 4.1 架构数学推导
- 13 × `Spiking_Residual_Block` (SRB)，每个 3 BN（bn1, bn2, shortcut.1）= **39**
- 3 × `SUNet_Level1_Block`，每个另加 2 内联 BN（bn_1, bn_2）= **6**
  - 内含 1 SRB 已算在 13
  - 内含 FreMLPBlock 的 GroupNorm **不算 BN**（没有 running stats）
- 2 × `DownSampling` × 1 BN = **2**
- 2 × `UpSampling` × 1 BN = **2**
- 2 × `GatedSkipFusion` × 1 BN = **2**
- **总计 39 + 6 + 2 + 2 + 2 = 51** ✓

### 4.2 为什么旧脚本报 41 vs 54
旧 substring filter `"bn" in key.lower() or "norm" in key.lower()`：
- TDBN 的 `SRB.shortcut[1]` 是 `nn.Sequential`，BN 键路径 `shortcut.1.weight`
  **不含 "bn" 子串** → **漏 13 层** → 41
- BN2dWrapper 包一层后键变 `shortcut.1.bn.weight` 含 "bn" → 54
- 54 = 51 BN + 3 GroupNorm（FreMLPBlock 的 `self.norm`，被 "norm" 子串捕）

新脚本用 BN signature（必须有 `running_mean/var`）正确识别 **51 层**。

---

## 5. RelaySum 实现正确性（§III.D + `constellation.py:280-367`）

### 5.1 原文 FLSNN Algorithm 2 / Equation 8
- 每 plane 维护 per-neighbour relay buffer
- 每 round 发送 tailored message = 自己 + 收到的 relays（不含目的邻居）
- 聚合：$\hat x_p = \tfrac{1}{N}(\sum_q b_{q,p} + (N - n_p^{\rm rec}) x_p)$

### 5.2 我们的实现核心（line 328-357）
```python
agg = zeros_like_state(...)
received_count = 0
for q in neighbours:
    agg += self.received_relay_weights[p][q]
    received_count += self.received_relay_counts[p][q]
# fill (N - received_count) with self
agg.add_(self_w, alpha=float(N - received_count))
state_div(agg, float(N))
```

### 5.3 历史 bug 披露（§III.D 尾 + commit 11f10f3）
v1 版曾用 `agg / received_count` 而非 `/ N` → 相当于 Gossip 在前 N-1 round。
**commit `11f10f3`（2026-04-18）修复为 `/N`**，**早于 v2 任何数据收集**。

### 5.4 Lr_scale 2.093 披露（constellation.py:190）
```python
lr_scale = 2.093 if aggregation_scheme == RELAYSUM else 1.0
```
- 来源：FLSNN 作者 `revised_constellation.py:204-205` 硬编码
- 针对 EuroSAT + SGD + 分类的经验补偿
- 我们未针对 Charbonnier + AdamW + 回归重新验证
- §VI-D 把它列为反转的 candidate cause **D9**（不单一归因）

---

## 6. 实验设计方法论（§VI-A + Table III ckpt 布局）

### 6.1 Run 矩阵
| Run | bn_variant | backbone | cells | 用途 |
|:---:|:----------:|:--------:|:-----:|:-----|
| A | tdbn | snn | 6 (2 bn_mode × 3 scheme) | 主实验、机制层 |
| B | bn2d | snn | 6 | A vs B → BN 消融 |
| C | tdbn | ann | 1（fedbn+AllReduce） | A vs C → backbone 消融 |

**配置奇偶校验**（3 份 summary.json["config"] 按位比较）：
seed=1234, partition_seed=0, α=0.1, 5×10=50 sat, T=4, en=[2,2,4,4], de=[2,2,2,2],
lr=1e-3, warmup=3, 80 epoch, intra=2, local=2, Charbonnier+0.1·SSIM。
- A vs B 唯一差：`bn_variant`
- A vs C 唯一差：`backbone`
- **B vs C 差 2 项**（bn_variant + backbone）→ 论文不做直接对比

### 6.2 数据依据链
```
60 plane ckpts (5 plane × 6 cell × 2 run) + 5 C ckpt (1 cell × 5 plane)
        ↓
3 summary.json  (13 cell × {PSNR, SSIM, Comm, Wall, train_loss})
6×2 npz        (per-cell epochs/train_loss/eval_psnr/eval_ssim/...)
6×2 per_plane.txt (5-plane PSNR std)
        ↓
Table I (13 cell 主结果)
Table II (FedBN Δ 6 cell)
Table III (cross-plane Var(γ) 13 cell × isinstance script)
Table IV (A vs C 1 cell × 5 metric)
```

---

## 7. 能耗诚实披露（§VI-E.4 详述）

### 7.1 FLSNN 的能耗方法（我们可继承）
- 每 MAC（ANN）= 4.6 pJ（Horowitz 2014, 45 nm CMOS）
- 每 AC（binary SNN）= 0.9 pJ
- 公式：E = ops × spike_rate × 0.9 + ops_ann × 4.6
- 实测 spike rate 通过 `model.named_modules()` hook 于 `LIFLayer.avg_spike_rate`

### 7.2 我们为什么在 v2 不给出 SNN/ANN 能耗比
1. 我们 `MultiSpike4` 输出 5 级 {0, ¼, ½, ¾, 1}，**不是 binary**
   → 0.9 pJ/AC 的 Horowitz 公式 **是 binary-spike 下界**，对 5 级是 underestimate
2. 我们的 `mem_update`（`vlifnet.py:225-246`）是独立 LIF，不暴露 `avg_spike_rate`
3. VLIFNet FLOPs 硬 code 的 FLSNN 针对 200K 参数 ResNet；我们 2.3M U-Net 要重算

### 7.3 v3 补齐（§VI-H.3 item E1）
从零写 `cloud_removal_v2/energy_estimation.py`：
- 对 `mem_update` 加 forward hook 记 per-layer non-zero rate + 5-level histogram
- 用 `fvcore` 算 VLIFNet per-layer FLOPs
- 给两版估算：aggressive 0.9 pJ/AC × sparsity；conservative 4.6 pJ/MAC × sparsity

---

## 8. Part B 结束。继续 Part C 会提供 25+ 预设问答。
