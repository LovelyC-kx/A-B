# v2-A Theory and Related Work

> **⚠️ ERRATA (2026-04-19, corrected from v1 of this file)**
>
> The prior version of §1–§2 contained the following errors:
>
> 1. A "RelaySum-vs-Gossip crossover condition τ² < 1/(1-λ_2)²" — **this
>    was synthesized from Vogels et al. (NeurIPS 2021) and Koloskova et al.
>    (ICML 2020) external bounds, NOT from the FLSNN paper**
>    (arXiv:2501.15995). The FLSNN paper **does not** provide direct
>    theoretical bounds comparing RelaySum, Gossip, and AllReduce. Their
>    comparison is **empirical only (Fig 5)**. Any "crossover" reasoning
>    must be removed and replaced with the paper's own Theorem 2 + a
>    clearly-labeled "external literature commentary" section.
>
> 2. The prior version used a single heterogeneity parameter ζ². The
>    FLSNN paper uses **two** parameters: δ² (intra-orbit dissimilarity)
>    and ζ² (inter-orbit dissimilarity).
>
> 3. The prior version conflated `m` with number of planes. In the FLSNN
>    paper, `m` is the **mixing-time parameter of the mixing matrix W**
>    (as in Lemma 3), and `N` is the number of orbit planes.
>
> 4. The previous direct use of Koloskova 2020 / Vogels 2021 bounds as
>    "the theoretical comparison" is misleading — those bounds come from
>    their own proofs with different constants from FLSNN Theorem 2. They
>    may serve as external context but must not be conflated with the
>    FLSNN derivation.
>
> The corrected §1 below quotes only the FLSNN paper's Theorem 2 and its
> assumptions verbatim. External literature is separated into §2.

---

## §1. Convergence bound from the original FLSNN paper (verbatim)

All content in §1 is a direct reproduction / paraphrase of the FLSNN
paper (Yang, Wang, Cai, Shi, Jiang, Kuang, *Brain-Inspired Decentralized
Satellite Learning in Space Computing Power Networks*, arXiv:2501.15995).

### 1.1 Problem formulation (Section II of the paper)

- Orbit planes: **N** = {1, …, N}. Each orbit plane i has **K**
  evenly-distributed satellites S_i = {s_{i,1}, …, s_{i,K}}.
- Per-satellite local loss: **f_{i,k}(x)**.
- Per-plane local loss: **f_i(x) = (1/K) Σ_k f_{i,k}(x)**.
- Global loss: **f(x) = (1/N) Σ_i f_i(x)**.

### 1.2 Algorithm (paper's Eqs 2, 3, 8)

**Eq 2 — local update at satellite (i, k)** (stochastic SGD with step η):
```
x_{i,k}^{t+1/2} = x_{i,k}^t − η·∇f_{i,k}(x_{i,k}^t)
```

**Eq 3 — intra-plane aggregation** (equal-weight average within orbit):
```
x_i^{t+1/2} = (1/K) · Σ_{k=1}^K x_{i,k}^{t+1/2}
```

**Eq 8 — inter-plane RelaySum update** (weighted-delay global average):
```
x_i^t = (1/N) · Σ_{j=1}^N x_j^{t − τ_{ij} + 1/2}
```

where:
- **τ_{ij}** = minimum network hops from orbit i to j **minus 1** in
  the routing tree T.
- **τ_{ii} = 0** (self edge is "instant").
- **τ_max = max_{i,j} τ_{ij}** (diameter of routing tree − 1).

Compare to classic Gossip and AllReduce: Gossip averages only ACROSS EDGES
of the mixing matrix (no delayed terms); AllReduce averages over all j
with **τ_{ij} = 0** for all pairs. RelaySum is the intermediate case
where each j contributes but with its own delay.

### 1.3 Assumptions (paper's A1–A4)

- **A1 (L-smoothness)**: f_{i,k}, f_i, f all L-smooth.
- **A2 (Unbiased gradients)**: `E[∇F_{i,k}(x)] = ∇f_{i,k}(x)` with
  `Var ≤ σ²` (stochastic noise bound).
- **A3 (Intra-orbit dissimilarity)**:
  ```
  (1/K) · Σ_{k=1}^K ‖∇f_{i,k}(x) − ∇f_i(x)‖² ≤ δ²
  ```
- **A4 (Inter-orbit dissimilarity)**:
  ```
  ‖∇f_i(x) − ∇f(x)‖² ≤ ζ²
  ```

Our code structure respects A1, A2 (standard SGD). A3 reflects per-
satellite vs per-plane variance — under Dirichlet(α=0.1)-over-source
with 5 satellites/plane, δ² can be large (some satellites may see only
CR1-thin, others only CR2-thick). A4 reflects plane-to-plane variance
after intra-plane averaging — smaller under any α because of averaging.

### 1.4 Mixing-matrix symbols (paper's Lemma 3)

The mixing matrix **W** in FLSNN is related to the inter-plane routing
tree T. The paper defines:

- **q := (1/2) · (1 − |λ₂(W)|)**   (half the mixing-matrix spectral
  gap).
- **m = m(W)**   (an integer mixing-time parameter; the paper states
  "there exists an integer m > 0 such that W^m has all entries ≥ π₀").
- **π₀** = smallest positive value of W^m.
- **ρ := q / m**   (the **effective** spectral gap used throughout the
  bound).

⚠️ **Subtlety**: `ρ` is NOT just `1 − λ₂`. It's `q/m` where both q and
m depend on W. For a concrete chain-of-5, we can COMPUTE W, get λ₂,
determine m numerically (smallest integer such that W^m > 0), and then
compute ρ. This is a numerical exercise, not a closed-form cos(π/N)
result.

### 1.5 Theorem 2 statement (verbatim, paper's Section IV-B)

**Let Assumptions A1-A4 hold. For learning rate
η < q·π̃₀ / (36·C₁·m·R·E·L), with π̃₀ = min{π₀, 1}, τ̃ = τ_max + 1, we
have:**

```
(1/T) · Σ_{t=0}^{T-1} ‖∇f(x̄ᵗ)‖² ≤
    16·(2Lσ²r₀ / (NT))^{1/2}                                          (T1)
  + 16·(4C·√τ̃·L·σ·r₀ / (ρ·√(NT)))^{2/3}                                 (T2)
  + 288·C·L·√τ̃·r₀ / (ρT)                                               (T3)
  + 16·[ √{7E(E-1) + 7E²R(R-1)} / (NREπ₀)
         · √{2C²τ̃/(9ρ²L²) + 5} · L·z·r₀ / T ]^{2/3}                    (T4)
```

where `r₀ = f(x̄⁰) − f*` is the initial optimality gap, and C, C₁, z
are problem-dependent constants not defined on the page excerpt we
have. `R` = intra-orbit round count; `E` = local epoch count.

Note that:
- **T1** is the **standard stochastic SGD** term, unchanged regardless
  of topology (depends on N·T).
- **T2** is the **heterogeneity × topology** term with ρ in the
  denominator. This is where τ̃ (delay) and ρ (spectral gap) matter.
- **T3** is a **topology-only** term (ρ in denominator) — linear in
  τ̃, 1/T.
- **T4** involves both R, E (local iteration counts) AND (ρ, τ̃). This
  is the "local-update + topology" coupling.

**Crucially**: the ρ appears only in T2, T3, T4. The optimal-SGD floor
(T1) doesn't depend on topology. The ζ², δ² parameters don't appear
explicitly in the bound above — they're absorbed into the constants
C, C₁, z (which encapsulate A3, A4 via a derivation not reproduced in
the theorem statement).

### 1.6 Ordering of bound terms

At large T (near convergence):
- T1 ~ 1/√T  (dominates if σ² large)
- T2 ~ 1/T^{2/3}  (intermediate)
- T3 ~ 1/T  (decays faster)
- T4 ~ 1/T^{2/3}  (intermediate)

So as T → ∞, T3 → 0 first; T2 and T4 dominate the rate; T1 sets the
final noise floor. The slowest-decaying term is T1, followed by T2/T4
at 1/T^{2/3}.

### 1.7 Paper's own claim about Theorem 2

From the abstract: "we theoretically analyze the convergence behavior
of the proposed algorithm, which reveals a **network diameter related
convergence speed**" — this refers to τ̃, which appears as τ̃^{1/2} in
T2/T3 and τ̃ in T4 (via √τ̃/ρ^{...}). Larger diameter → slower
convergence. The paper does NOT make an explicit RelaySum-beats-Gossip
statement at the bound level.

### 1.8 What Theorem 2 DOES NOT say

- **It does not prove RelaySum ≥ Gossip**. Gossip's bound (if
  the paper derived one, which it doesn't) would replace τ̃ with some
  function of 1/(1−λ₂(W)) — but FLSNN doesn't run that derivation.
- **It does not prove AllReduce ≤ RelaySum in rounds**. AllReduce
  corresponds to τ_{ij} = 0, τ̃ = 1, which WOULD make Theorem 2's
  bound the smallest — but this is trivially true by plugging in and
  isn't the paper's claim.
- **It does not address task type** (classification vs regression).
  The bound is task-agnostic, only depending on L, σ², δ², ζ² (via
  C, C₁, z).

Therefore the **empirical finding** of Fig 5 (RelaySum best on
EuroSAT classification) is **not predicted** by Theorem 2. It is an
experimental observation. Our contradictory empirical finding on
cloud-removal regression is therefore **equally permissible** under
the same theorem — neither case is a proof of the other's wrongness.

### 1.9 Verified citations for §1

All quotes in §1 are from:
- **Yang, Wang, Cai, Shi, Jiang, Kuang (2025).**
  *Brain-Inspired Decentralized Satellite Learning in Space Computing
  Power Networks.*
  arXiv:2501.15995 (v1 or later; confirmed from the paper's HTML view).

---

## §2. External literature — strictly labeled as SUPPLEMENTARY

The following is external to FLSNN and included ONLY as context /
discussion. Any formula below must not be claimed to be from FLSNN's
own derivation.

### 2.1 Vogels et al. 2021 RelaySum (the algorithm FLSNN cites)

FLSNN says "we propose to leverage the idea of RelaySum" (III-B). The
original RelaySum is:

- Vogels, Hendrikx, Jaggi. **arXiv:2110.04175**, NeurIPS 2021.
- Their Theorem 2 in the non-convex setting contains a term
  `O( τ² · ζ² · L² / T² )^{1/3}` where τ is the spanning-tree diameter.

This matches FLSNN's T4 form in structure (both `^{2/3}` decay), but
with DIFFERENT constants. FLSNN's derivation is NOT the same as
Vogels 2021 — FLSNN has C, C₁, z that encompass the paper's SNN-
specific treatment and the intra-orbit aggregation structure.

### 2.2 Koloskova et al. 2020 decentralized-SGD (Gossip)

- Koloskova, Loizou, Boreiri, Jaggi, Stich. **arXiv:2003.10422**,
  ICML 2020.
- Their bound for Gossip-SGD: variance term scales with
  `(1 − λ₂)⁻² · ζ²`.

This is commonly used as "the" Gossip-SGD bound in the literature, but
it is NOT what FLSNN proves. FLSNN does not prove a Gossip bound.

### 2.3 Why FLSNN's ρ differs from `1 − λ₂`

FLSNN's ρ = q/m explicitly divides by the mixing-time parameter m
(Lemma 3). For a chain of N:
- `λ₂(W_chain) ≈ cos(π/N)` (folklore)
- `q = (1 − |λ₂|)/2 ≈ 2 sin²(π/(2N)) / 2`
- `m` = smallest integer with W^m strictly positive. For a chain, m
  = N − 1 typically.
- So `ρ = q / (N − 1) ≈ 2 sin²(π/(2N)) / (N−1)`.

For N = 5:  `ρ ≈ 2 · sin²(18°) / 4 = 2 · 0.0955 / 4 ≈ 0.048`.

Compared to `1 − λ₂ = 1 − 0.809 = 0.191`. So FLSNN's effective spectral
gap ρ is about **4× smaller** than `1 − λ₂` for chain-5. Plugging into
Theorem 2, the T2/T3 terms are **~4× larger** than a naive Koloskova-
style estimate would suggest.

### 2.4 How these supplementary bounds relate to our observation

Under Theorem 2, the ordering of RelaySum vs Gossip vs AllReduce is
indeterminate (the paper doesn't derive Gossip or AllReduce bounds).
Under the Koloskova bound for Gossip, the `(1−λ₂)⁻²` factor is ~27×
baseline. If we naively say "use Theorem 2 for RelaySum, use Koloskova
for Gossip", then RelaySum's τ̃/ρ ≈ 5/0.048 ≈ 104 and Gossip's
(1-λ₂)⁻² ≈ 27. That would predict **Gossip beats RelaySum in the
bound** on chain-5 — which is consistent with our empirical
observation.

But this cross-paper combination is NOT rigorous — the constants are
incomparable across the two derivations. The honest conclusion is:

- FLSNN's Theorem 2 does not predict RelaySum wins or loses against
  Gossip. The paper's Fig 5 empirical ordering is not theoretically
  derived.
- Our v2-A empirical ordering (RelaySum last) is not theoretically
  predicted either. It's a new data point.

### 2.5 Verified citations for §2

- **Vogels, Hendrikx, Jaggi (NeurIPS 2021).** arXiv:2110.04175. *RelaySum
  for Decentralized Deep Learning on Heterogeneous Data*. Verified
  from arXiv search; title and venue match the standard reference.
- **Koloskova, Loizou, Boreiri, Jaggi, Stich (ICML 2020).** arXiv:2003.10422.
  *A Unified Theory of Decentralized SGD with Changing Topology and
  Local Updates*. Verified.

---

## §3. Client drift & heterogeneity: transferring classification FL theory to our regression setting

### 3.1 The four ground-truth sources used in §3

All four sources below are **classification-task** FL papers that the
user provided full-text or verified-abstract access to. They are cited
here because (a) they formalize *client drift / ζ² / loss-landscape*
phenomena that our regression setting inherits, and (b) **none of them
does regression** — which is itself the gap our paper addresses.

- **[FedDC]** Gao, Fu, Li, Chen, Xu, Xu (CVPR 2022). *Federated
  Learning With Non-IID Data via Local Drift Decoupling and Correction.*
  Experiments on MNIST / F-MNIST / CIFAR10 / CIFAR100 / EMNIST-L / Tiny
  ImageNet / Synthetic. **All classification.**
- **[Seo24]** Seo, Catak, Rong (NIKT 2024, arXiv:2502.00182v3).
  *Understanding Federated Learning from IID to Non-IID dataset: An
  Experimental Study.* Experiments on CIFAR-10. **Classification.**
- **[FedBSS]** Xu, Li, Wu, Ren (AAAI 2025, arXiv:2501.11360). *Federated
  Learning with Sample-level Client Drift Mitigation.* Abstract-only
  received. Task: sample-level drift mitigation for label skew /
  feature skew / noisy labels. **Classification benchmarks.**
- **[ECGR]** Luo, Wang, Wen, Sun, Li (2026 arXiv:2601.03584v1). *Local
  Gradient Regulation Stabilizes Federated Learning under Client
  Heterogeneity.* Abstract-only. Evaluated on LC25000 medical imaging.
  **Classification.**

### 3.2 The canonical "drift is from loss-landscape mismatch" claim

[Seo24 §5] argues, grounded in experimental measurement rather than
assumption:

> "Client drift stems from inconsistencies in loss landscapes... Each
> client operates on a distinct dataset, meaning each experiences a
> unique loss landscape. Even though all clients use the same model
> and loss function, the variations in datasets lead to different
> optimization paths."  (Seo24, verbatim.)

This is the modern re-framing of ζ² (inter-client gradient variance).
The paper then provides **direct measurement**:

> "We measure the layer-wise (conv1, conv2, fc1, and fc2) cosine
> similarity of local updates (Δθ) between pairs of clients, averaged
> across all combinations from 10 clients. As anticipated, in the IID
> setting, the cosine similarity between client updates remains
> consistently higher compared to the non-IID setting." (Seo24 Fig 13.)

**This is the experimental protocol we should adopt** (see §C3 of
`v2_interpretation.md`). Result in Seo24: IID cosine ≳ 0.5 across all
4 layers; Dirichlet-α=0.1 non-IID drops to ~0.1 at output layers,
~0.3 at conv1. The input-proximal layers drift LESS; the
classification-head layers drift MORE.

**Critical implication for our regression setting**: VLIFNet's output
head is a `Conv2d(dim → 3) + residual-add-input`. There is no
class-specific softmax layer; the final output is a 3-channel image
shared in structure across all clients. **We predict the cosine
similarity at the last layer will be HIGHER for our task than for
Seo24's CIFAR-10 classification** — supporting our Tier-B hypothesis
of lower ζ² for regression. This is now a concrete, falsifiable
prediction.

### 3.3 FedDC's non-linearity argument: why averaging fails and how this applies to regression

[FedDC §3, Fig 1] gives a minimal non-linearity argument that
underpins ALL FL heterogeneity theory:

> "Suppose there is a non-linear transformation function f (e.g.,
> Sigmoid). Suppose θ_1 and θ_2 are local parameters of client 1 and
> client 2, w_c is the ideal model parameter, and w_f is the model
> parameter generated through FedAvg. The local drifts are
> h_1 = w_c − θ_1 and h_2 = w_c − θ_2. For data point x, outputs are
> y_1 = f(θ_1, x), y_2 = f(θ_2, x). w_f = (θ_1 + θ_2)/2. The
> centralized ideal: f(w_c, x) = (y_1 + y_2)/2, so w_c = f⁻¹((y_1 +
> y_2)/2) / x. Since f is non-linear, **w_f ≠ w_c** and **f(w_f, x) ≠
> (y_1 + y_2)/2**."  (FedDC, paraphrased with original variable names.)

**The magnitude of the FedAvg error `|f(w_f, x) − f(w_c, x)|` scales
with**:
- (a) the curvature of f (i.e. its non-linearity strength),
- (b) the per-client parameter-drift magnitude `|θ_i − w_c|`.

This gives a principled way to quantify "how much does non-linearity
exacerbate non-IID drift?"

**Applied to classification**: the final softmax is a highly non-linear
function (exp + normalize). Moreover the argmax decision boundary is
discontinuous. Curvature of softmax is large near class boundaries.
Therefore (a) is large; (a) × (b) is large.

**Applied to our regression (cloud removal)**: the final head is
`Conv2d(dim→3) + input_residual`. The Conv2d is **linear**; the
residual add is **linear**. Therefore (a) ≈ 0 **at the output level**.
The upstream non-linearities (LIF, MultiSpike4, FSTA attention
Sigmoid) contribute, but their collective effect is diluted by the
final linear projection and large residual.

**Conclusion**: FedDC's non-linearity argument predicts that
**regression with a linear output-and-residual head has
systematically smaller client-drift error magnitude than
classification under the same ζ² at the parameter level**. Our v2-A
observation (6-cell PSNR spread ~0.17 dB, vs FLSNN's 10-20 pp
accuracy spread) is consistent with this prediction: the AVERAGING
ERROR is smaller because the head is less non-linear.

### 3.4 Seo24's effective-update-count u and our Dirichlet partition

[Seo24 Eq. 3] defines the "effective updates per round" as:

> **u = η · E · |D| / (B · K)**

where |D| is per-client dataset size, K is client count, B is batch
size, E local epochs, η learning rate. Seo24 shows that **to fairly
compare across different K, u must be matched** — otherwise more
clients with fixed per-client hyperparameters trivially means more
total updates per round.

**Our setting has a NOT-YET-ADDRESSED complication**: under
Dirichlet(α=0.1), per-client |D| varies by 20× (smallest 5 samples,
largest 118 samples — see `v2_results_synthesis.md §1` per-client
sizes). With uniform B=4, E=2 (local_iters), η=1e-3, the per-client
u differs by 20×. Clients with 5 samples do 2·⌊5/4⌋ = 2 SGD steps
per round after drop_last=True; clients with 118 samples do
2·⌊118/4⌋ = 58 steps per round. **The latter run 29× more
updates**.

This means our effective heterogeneity is **compounded** — not just
heterogeneous data distributions, but also heterogeneous update
magnitudes across clients. FedAvg's uniform-weight aggregation then
implicitly down-weights the updates from larger clients and up-weights
them from smaller (more severely overfit) clients.

**Implication for paper**: we must report the per-client u distribution
in the data section and discuss whether our Dirichlet result is
dominated by data skew or by compute-time skew. This is a candidate
for a v3 ablation (fixed steps-per-round across clients) OR a
Section 6 caveat.

### 3.5 ECGR / FedBSS perspective on gradient dynamics

[ECGR] frames heterogeneity as "distorting local gradient dynamics"
and proposes regulating local gradients without additional
communication. Their core idea: separate **well-aligned** and
**misaligned** gradient components and preserve informative updates.

[FedBSS] frames heterogeneity as "drift is the cumulative manifestation
of biases present in all local samples and the bias between samples is
different." Their remedy: sample-level bias-aware selection.

Both frame drift at the **gradient/sample level**, not the
parameter level. For our regression task:

- Pixel-wise losses (Charbonnier, SSIM) produce gradients that are
  **spatial averages** over patches of 64×64 = 4096 pixels. The
  "per-sample bias" FedBSS targets would be much smaller than the
  per-sample bias for a class-label loss (which is single-scalar
  cross-entropy per sample).
- This is additional support (not proof) for our regression ζ² <
  classification ζ² hypothesis.

We do not need to adopt ECGR or FedBSS mechanisms, but their
abstractions **give us scholarly language** for the paper:

> "Under feature-shift non-IID (CR1 thin vs CR2 thick clouds), the
> per-client loss landscapes diverge primarily in the mid-frequency
> texture discriminators while sharing a common low-frequency photon-
> fidelity objective. This reduces the effective ζ² encountered by
> inter-plane aggregation compared to classification's
> discrete-boundary regime."  (Suggested paper language.)

### 3.6 What [FedDC] results directly suggest about our findings

Looking at FedDC's Fig 3 (provided) and Table 3 data:

| Method | CIFAR-10 D2 (full) | CIFAR-10 D2 (partial 15%) |
|---|---|---|
| FedAvg | 79.14% | 79.77% |
| FedProx | 78.89% | 79.84% |
| Scaffold | 82.96% | 82.53% |
| FedDyn | 84.14% | 82.30% |
| FedDC | **84.32%** | **84.58%** |

**Spread under Dirichlet-α=0.3 CIFAR10**: 84.32 − 78.89 = **5.43 pp**
between best and worst method.

**Our v2-A spread under Dirichlet-α=0.1 CUHK-CR1+CR2 regression**:
21.387 − 21.213 = **0.174 dB** = roughly **0.8% relative improvement**.

Orders of magnitude suggest the inter-method spread on classification
non-IID is **~5-10× larger** than on our regression non-IID. This is a
strong empirical data point supporting §3.3's theoretical argument.

### 3.7 Verified sources for §3

- Gao L., Fu H., Li L., Chen Y., Xu M., Xu C.-Z. *Federated Learning
  With Non-IID Data via Local Drift Decoupling and Correction.*
  **CVPR 2022, pp. 10112–10121.** (Full text provided by user.)
- Seo J., Catak F. O., Rong C. *Understanding Federated Learning from
  IID to Non-IID dataset: An Experimental Study.* **36th NIKT 2024.**
  arXiv:2502.00182v3. (Full text provided by user.)
- Xu H., Li J., Wu W., Ren H. *Federated Learning with Sample-level
  Client Drift Mitigation.* **arXiv:2501.11360.** (Abstract only.)
- Luo P., Wang J., Wen Z., Sun T., Li D. *Local Gradient Regulation
  Stabilizes Federated Learning under Client Heterogeneity.*
  **arXiv:2601.03584v1.** (Abstract only.)

### 3.8 Open questions for §3 (to be addressed in further rounds)

1. Is there a direct FL-regression-vs-classification study? (User has
   so far not located one — candidate next-round search.)
2. The FedDC Fig 1 argument applies to FedAvg; does it generalize to
   RelaySum? The relay buffer accumulates multi-round parameter drift;
   the non-linearity magnification compounds across rounds. This
   should INCREASE RelaySum's disadvantage vs Gossip/AllReduce under
   classification (consistent with our v1 IID result where RelaySum
   also lost). But no paper has formalized this — it is a speculative
   extension.
3. Can we adapt Seo24 Fig 13's cosine-similarity protocol to our
   VLIFNet pipeline? 30 lines of code; run ONCE on a cell during the
   70-epoch run. This would provide empirical ζ² data directly.

---

## §3–§7: pending (continued below as §4.1 now; §4.2+ to follow)

---

## §4. BN-family normalization under feature shift — starting from TDBN

Our VLIFNet uses ThresholdDependentBatchNorm2d (TDBN) from
SpikingJelly, which comes from Zheng, Wu, Deng, Hu, Li (AAAI 2021,
arXiv:2011.05280). §4.1 extracts the TDBN paper's **exact**
mechanism and theorems; §4.2–§4.5 (pending) will cover FedBN,
SiloBN, HarmoFL, FedWon, and Du et al. 2022 in the same verbatim
way.

### §4.1. TDBN (Zheng et al. AAAI 2021) — what the paper actually says

This is the only BN variant VLIFNet actually uses. All other
BN/FedBN discussions in subsequent sub-sections must be checked
against TDBN-specific semantics before they apply to our setting.

#### 4.1.1 Iterative LIF model context (paper's Eq. 2, 3)

TDBN is defined on top of the iterative LIF neuron:
```
u^{t,n+1} = τ_decay · u^{t−1,n+1} · (1 − o^{t−1,n+1}) + x^{t,n}       (Eq. 2)
o^{t,n+1} = 1 if u^{t,n+1} > V_th else 0                              (Eq. 3)
```
where `u^{t,n}` is the membrane potential of a neuron in layer n at
time t, `o^{t,n}` is the binary spike, and `τ_decay` is the potential
decay constant. In VLIFNet, `τ_decay = 0.25` (from `vlifnet.py`),
`V_th = 0.15` (module-level constant `v_th`), and `T = 4`.

#### 4.1.2 TDBN formulation (paper's Eqs. 5, 6)

Unlike standard BN which normalizes over `[N, H, W]` per channel per
step, TDBN normalizes **across the full T×N×H×W tensor** per channel.
Let `x_k = (x¹_k, x²_k, …, x^T_k)` be the concatenation-over-time of
the k-th channel feature map. Then:

```
x̂_k = α · V_th · (x_k − E[x_k]) / √(Var[x_k] + ε)                    (Eq. 5)
y_k = λ_k · x̂_k + β_k                                                (Eq. 6)
```

where:
- `E[x_k]`, `Var[x_k]` are mean/variance over `[T, N, H, W]`
  (paper's Eqs. 7, 8 — "mean(x_k)" and "mean((x_k − E[x_k])²)"
  with the N-T-H-W tensor as input).
- `V_th` = neuron threshold (global constant; paper: "given threshold").
- `α` = architecture-dependent hyperparameter (paper: "α is 1 for
  serial; 1/√n for n-branch parallel"; VLIFNet uses α=1/√2 for
  residual blocks and α=1 elsewhere — matches paper's recipe).
- `λ_k, β_k` = per-channel learnable affine parameters; initialized
  `λ=1, β=0`.

So the normalized pre-activation targets the distribution
**N(0, (αV_th)²)** rather than standard BN's **N(0, 1)**. Paper
§TDBN verbatim: "In tdBN, pre-activations are normalized to
N(0,(αV_th)²) instead of N(0, 1)."

#### 4.1.3 Training vs inference: running statistics + scale fusion

During training, TDBN tracks μ_tra, σ²_tra by exponential moving
average over the T×N×H×W tensor (Algorithm 1 in the paper). At
inference, paper's Eqs. 9, 10 define "batchnorm-scale-fusion":
```
W' = λ · α · V_th · W / √(σ²_inf + ε)                                (Eq. 9)
B' = λ · α · V_th · (B − μ_inf) / √(σ²_inf + ε) + β                  (Eq. 10)
```

i.e. at inference the TDBN vanishes — its effect is absorbed into the
preceding conv's weight and bias. In training (where FL operates),
the TDBN layer is still present and its (μ_tra, σ²_tra, λ, β) are
all live tensors that get checkpointed and are therefore subject
to federated aggregation.

#### 4.1.4 What TDBN guarantees (paper's Theorem 1)

The paper proves **block dynamical isometry** for SNNs with TDBN:

> **Theorem 1 (Zheng et al. 2021, verbatim).** Consider an SNN with T
> timesteps and the j-th block's jacobian matrix at time t denoted as
> J^t_j. When τ_decay is equal to 0, if we fix the second moment of
> input vector and the output vector to V²_th for each block between
> two tdBN layers, we have φ(J^t_j (J^t_j)^T) ≈ 1 and the training
> of SNN can avoid gradient vanishing or explosion.

**Assumptions**:
- (A1) τ_decay = 0 (simplification; paper shows τ_decay ∈ {0.25, 0.5}
  also produces stable gradient norms empirically — Fig. 2).
- (A2) **Second moment of input and output fixed to V²_th for each
  block**. This is guaranteed AT INITIALISATION because λ=1, β=0
  makes the TDBN output exactly N(0, (αV_th)²); for serial blocks
  α=1 so output variance = V²_th.
- (A3) Block is a "General Linear Transform" (Lemma 2 of Chen et al.
  2020): data normalisation with zero mean + linear transform +
  rectifier activation.

The conclusion `φ(JJ^T) ≈ 1` is gradient-norm preservation: no
vanishing, no exploding.

#### 4.1.5 What TDBN does NOT address

The TDBN paper is about **centralised training**. It never discusses:
- How (μ_tra, σ²_tra, λ, β) should be aggregated across clients in
  federated settings.
- Whether Theorem 1 holds after a weighted-average update of
  (λ, β) from multiple clients.
- What happens when different clients' TDBN running stats diverge
  (their "feature shift" analogue).

All of §4.2–§4.5 below will discuss these questions using FedBN /
SiloBN / HarmoFL / FedWon / Du et al. 2022, but **none of those
papers covers TDBN specifically**. They cover standard BN. Any
application to our setting must account for the three TDBN-specific
deltas identified in §4.1.6.

#### 4.1.6 Three implications of TDBN's structure for FL analysis

These are our own observations, drawn from TDBN's mechanism, not from
any FL paper. To be validated by experiment where noted.

**Implication 1 — Variance target is shared by construction.**
Standard BN targets `N(0, 1)`; all clients share this target only by
convention. TDBN targets `N(0, (αV_th)²)`. Because α and V_th are
**hyperparameters fixed at model build time** (not learned, not
client-specific in our code), the variance target is IDENTICAL
across clients regardless of data. Under FedBN this means the
per-plane running σ²_tra's have a common ceiling; under FedAvg the
aggregated σ²_tra also targets the same ceiling. This is a
**structural alignment force** on BN statistics that standard BN
does not have.

Practical consequence: we expect the per-plane σ²_tra to diverge
LESS under TDBN than under standard BN across non-IID clients, because
all clients are pulled toward (αV_th)² as their variance anchor. This
may partially explain why our observed FedBN-over-FedAvg gain is
small (0.009 dB PSNR): there's simply less BN divergence for FedBN to
"protect".

**Implication 2 — Theorem 1's assumption A2 can be broken by FedAvg
on λ, β.**
Theorem 1 requires the second moment of block inputs and outputs to
be fixed to V²_th. This is true AT INITIALISATION (λ=1, β=0) and
approximately preserved during centralised training as long as λ
stays near 1 (which the paper's Fig. 2 empirically supports).
But under FedAvg, the aggregated λ_avg and β_avg are the mean of
N client-local values. If per-plane λ_i drifts to different values
(due to non-IID gradients), the averaged λ_avg can be FURTHER from
1 than any individual plane's λ_i, breaking A2 and the block-
dynamical-isometry guarantee for the aggregated model.

Under FedBN, each plane keeps its own (λ_i, β_i) → each plane's
block dynamical isometry is preserved LOCALLY (for that plane's
data) as long as training is stable. But the plane's weights
θ_conv get averaged across planes, so the aggregated θ_conv_avg is
not optimised w.r.t. any single (λ_i, β_i). This weak compatibility
may still partially preserve Theorem 1's intent.

Practical consequence: if our per-plane (λ_i, β_i) end up close to
(1, 0) after 35 epochs (because residual paths dominate gradient
flow, leaving TDBN params near initialisation), FedAvg and FedBN on
TDBN produce near-identical results → tiny observed gap. This is
testable: after 70 epochs, extract `λ, β` from all 6 cells × all
TDBN layers and measure `‖λ − 1‖` + `‖β‖`. If these are small, it
confirms the mechanism.

**Implication 3 — TDBN's μ_tra, σ²_tra are T-amplified EMA statistics.**
The tensor fed into TDBN's mean/variance computation has shape
`[T, N, H, W]` per channel (Algorithm 1, line 1-2), i.e. T-fold
more elements per batch than standard BN. BUT the T time steps are
**not i.i.d. samples** of the underlying image distribution — they
are the T unrolled forward-pass steps of the same LIF neuron (paper's
Eq. 2). They are strongly correlated via `u^{t+1}` depending on
`u^t`. Therefore, the "effective sample size" for TDBN's running
stats is closer to N·H·W (same as standard BN) than to T·N·H·W.

Practical consequence: TDBN's running stats have similar noise
levels to standard BN's, even though Algorithm 1 implies a factor-T
variance reduction. So the client-level divergence of (μ_tra, σ²_tra)
is quantitatively comparable to what standard BN would produce on
the same data — not drastically reduced. Implications 1 and 2 still
hold; Implication 3 just warns against over-crediting TDBN for
"automatic stabilisation across clients".

#### 4.1.7 TDBN paper does not address — deferred to §4.2 onward

The following questions REMAIN OPEN after §4.1 and are addressed
in later sub-sections using the FedBN / SiloBN / HarmoFL / FedWon /
Du-et-al-2022 papers, with the caveat that those papers discuss
standard BN not TDBN:

1. Should TDBN's (μ_tra, σ²_tra) be aggregated or kept local? (FedBN
   says local for standard BN; SiloBN says shared-learned-params-but-
   local-stats; FedWon says remove entirely.)

2. Does our current implementation aggregate (μ_tra, σ²_tra) along
   with (λ, β)? (Code-level audit — `aggregation.py` + `is_bn_key`.)

3. Under feature shift (our CR1 vs CR2), how large does per-client
   TDBN divergence get during training? (Empirical — measurable
   post-run.)

4. Is the "external covariate shift" of Du et al. 2022 a real
   phenomenon in TDBN-based SNNs? (The paper analyses standard BN
   only; its "scale invariant property" requires `BN(h; aW) =
   BN(h; W)` which TDBN also satisfies via Eqs. 5-6, so the
   argument likely ports.)

#### 4.1.8 Verified citations for §4.1

- Zheng H., Wu Y., Deng L., Hu Y., Li G. *Going Deeper With
  Directly-Trained Larger Spiking Neural Networks*. **AAAI 2021,
  arXiv:2011.05280v2**. Full text provided by user.
- Chen Z., Deng L., Wang B., Li G., Xie Y. (2020). *A Comprehensive
  and Modularized Statistical Framework for Gradient Norm Equality
  in Deep Neural Networks*. IEEE T-PAMI. [Source of Lemmas 1/2 used
  by TDBN Theorem 1.]
- Wu Y., Deng L., Li G., Zhu J., Shi L. (2018). *Spatio-Temporal
  Backpropagation for Training High-Performance Spiking Neural
  Networks*. Frontiers in Neuroscience 12:331. [Source of the STBP
  algorithm TDBN is added to.]
- Wu Y., Deng L., Li G., Zhu J., Xie Y., Shi L. (2019). *Direct
  training for spiking neural networks: Faster, larger, better*.
  AAAI 33:1311-1318. [Source of the iterative LIF model, Eqs. 2-3
  above.]

---

## §4.2 FedBN (Li et al., ICLR 2021) — verbatim + strict adaptation

**Source.** Li, Jiang, Zhang, Kamp, Dou, "FedBN: Federated Learning
on Non-IID Features via Local Batch Normalization," ICLR 2021,
arXiv:2102.07623v2. User provided the full paper text; all quotes
below are verbatim from that PDF.

### §4.2.1 Algorithm (verbatim)

From §4.1 of the paper:

> "FedBN performs local updates and averages local models. However,
> FedBN assumes local models have BN layers and **excludes their
> parameters from the averaging step**." (emphasis added)

Algorithm 1 (Appendix C, verbatim pseudo-code):

```
for each round t = 1,...,T:
  for each user k and each layer l:
    w^(l)_{t+1,k} <- SGD(w^(l)_{t,k})         # local training
  if mod(t,E) == 0:
    for each user k and each layer l:
      if layer l is NOT BatchNorm:
        w^(l)_{t+1,k} <- (1/K) Σ_k w^(l)_{t+1,k}   # aggregate
      # else: keep w^(l) local — NOT aggregated
```

**Matches our v2-A implementation** (`aggregation.py` under
`bn_local=True`): BN layers' (γ, β, running_mean, running_var,
num_batches_tracked) are excluded from FedAvg aggregation.

### §4.2.2 Formal definition of "feature shift" (verbatim)

§3, "Non-IID Data in Federated Learning":

> "We define feature shift as the case that covers:
>  1) **covariate shift**: the marginal distributions P_i(x) varies
>     across clients, even if P_i(y|x) is the same for all clients;
>  2) **concept shift**: the conditional distribution P_i(x|y) varies
>     across clients and P(y) is the same."

This is a sharper definition than the colloquial "non-IID data" and
is what the convergence theorem (§4.2.3 below) is proved under.

**Does our v2-A Dirichlet-over-cloud-type partition qualify as
feature shift under this definition?**

Yes, specifically as **covariate shift (type 1)**, because:

- The task is unsupervised image regression: given cloudy image x,
  predict clear image y. The conditional P(y | x) is the same
  ground-truth mapping "remove the cloud" for every client — there
  is one underlying physical process.
- What differs across planes is P_i(x): plane-0 may hold mostly
  thin-cloud samples (CR1-dominant), plane-4 mostly thick-cloud
  samples (CR2-dominant), because our Dirichlet(α=0.1) allocates
  cloud-type proportions unevenly to clients and then clients are
  grouped into planes by cluster id.
- So the marginals P_i(x) differ (thin-cloud images vs thick-cloud
  images have very different pixel statistics — CR1 mean luminance
  higher, haze-like; CR2 mean luminance lower, opacity near 1),
  while P_i(y | x) is identical.

This is **exactly the setting FedBN's theory targets**. So their
Corollary 4.6 (faster-than-FedAvg guarantee) should apply to our
setup — which makes the observed near-null gap (0.009 dB PSNR,
0.0014 SSIM) a genuine puzzle requiring explanation (§4.2.4 below).

### §4.2.3 Convergence result (Corollary 4.6, verbatim)

From §4.3 (under Assumption 4.1 and overparameterisation m =
Ω(max{N⁴M⁴ log(NM/δ) / (α⁴μ₀⁴), N²M² log(NM/δ) / μ₀²})):

**Corollary 4.6 (Convergence rate comparison between FedAvg and
FedBN).** "For the G-dominated convergence, the convergence rate of
FedBN is faster than that of FedAvg."

Proof sketch (verbatim):

> "The key is to show λ_min(G^∞) ≤ λ_min(G*^∞). Comparing equation
> (4) and (5), G*^∞ takes the M×M block matrices on the diagonal of
> G^∞ ... By linear algebra, λ_min(G^∞_i) ≥ λ_min(G^∞) for i ∈ [N].
> Since G*^∞ = diag(G^∞_1, ..., G^∞_N), we have λ_min(G*^∞) =
> min_i {λ_min(G^∞_i)}. Therefore, we have the result λ_min(G*^∞)
> ≥ λ_min(G^∞)."

Linear rate for FedAvg: ‖f(t) − y‖² ≤ (1 − ημ₀/2)ᵗ ‖f(0) − y‖².
Linear rate for FedBN: ‖f*(t) − y‖² ≤ (1 − ημ*₀/2)ᵗ ‖f*(0) − y‖²,
with μ*₀ ≥ μ₀. The **exponential decay of FedBN is no slower and
generically strictly faster** than FedAvg.

**Caveat (not original paper, my reading):** this is a
convergence-rate comparison, not a final-accuracy comparison. After
enough rounds, both may converge to their respective stationary
points; the absolute gap in test PSNR depends on those stationary
points, not on μ₀ vs μ*₀ alone. Our 35-epoch and 70-epoch observations
are about final PSNR, so Corollary 4.6 is only tangentially relevant.

### §4.2.4 Reported gap in original paper vs our observed gap

Original paper's headline numbers (Table 1, reproduced verbatim from
paper, feature-shift non-IID, 5-trial mean):

| Domain set              | FedAvg  | FedBN   | Δ (pp)  |
|-------------------------|---------|---------|---------|
| Office-Caltech-10 avg   | ~62.7   | ~70.5   | **+7.8**  |
| DomainNet avg           | ~42.0   | ~49.5   | **+7.5**  |
| ABIDE-I (medical) avg   | ~67.8   | ~68.7   | **+0.9**  |
| Digits-Five avg (Fig 5) | ~85.0   | ~86.5   | **+1.5**  |

(The Office-Caltech gap of +7.8 pp accuracy is the paper's strongest
result. ABIDE-I, which is also a regression-ish / low-semantic-gap
task, drops to +0.9 pp.)

Our v2-A observed gap (35 epoch, Dirichlet α=0.1 over cloud type,
6 cells × 3 seeds):

| Metric   | FedAvg avg | FedBN avg | Δ                     |
|----------|------------|-----------|-----------------------|
| PSNR     | 21.307 dB  | 21.316 dB | +0.009 dB (+0.04 % rel) |
| SSIM     | 0.6560     | 0.6574    | +0.0014 (+0.2 % rel)   |

**Our gap is ~500× smaller than Office-Caltech and ~60× smaller
than ABIDE-I (in relative terms).** So the FedBN claim does hold
directionally (FedBN > FedAvg in all 6 cells) but its magnitude in
our setup is essentially at the noise floor of the 245-sample test
set.

### §4.2.5 Why is our gap so small? Three diagnoses

These are **my diagnoses**, explicitly distinguished from paper
claims.

**Diag-A. Task type: classification vs regression (primary).** The
FedBN gap shrinks monotonically across tasks in their own data:
object classification (+7.8 pp) > digit classification (+1.5 pp)
> medical binary classification (+0.9 pp). Regression-like tasks
with physical invertibility and no class-boundary geometry have
much smaller FedBN benefit. Cloud removal is pixel-level regression
with even smoother loss landscape than binary classification; our
+0.04 % relative gap sits at the extrapolated end of this trend.

**Diag-B. BN variant (TDBN vs standard BN) already cross-aligns
features (strong).** FedBN theory assumes standard BN with target
N(0, 1). If local BN statistics differ, each client's feature
space is normalised to its own N(0, 1) but scaled differently by γ
— averaging γ hurts. TDBN targets N(0, (αV_th)²) with α, V_th
**fixed hyperparameters shared across clients at init** (λ=1,
β=0; cf. §4.1.6 Imp 1). So at initialisation, and — if λ, β don't
drift far — for most of training, the TDBN feature scale is already
aligned across clients by construction. There is much less
inter-client BN misalignment for FedBN to "save". This is our
strongest non-trivial explanation and is testable by the planned
Imp-2 ablation (measure ‖λ_i − 1‖, ‖β_i‖ at end of training; if
both are small, TDBN + FedAvg ≈ TDBN + FedBN is predicted).

**Diag-C. FedBN granularity: ours is per-plane, not per-client
(moderate).** Paper's FedBN keeps one set of BN params per client
(N = 5 in Office-Caltech). Our `aggregation.py` does per-plane BN:
each of 5 planes has one BN state, but each plane is itself an
average over 10 satellites in the intra-plane step. So the
"granularity advantage" of FedBN is diluted from N=50 (if
per-satellite) down to N=5 (per-plane), and even those 5 BN states
are averages of 10 possibly-conflicting per-satellite stats. The
paper's Theorem 4.4 is proved for per-client BN; our per-plane BN
is a strictly weaker personalisation.

**Diag-D (negligible). Small test set (N=245 each cell) has PSNR
noise floor ~±0.1 dB.** So a true 0.05 dB advantage would be within
noise. This doesn't explain the direction-consistency across 6/6
cells (which is significant at p<0.02 under a binomial test), but
it does explain why the magnitude can't be pinned down precisely.

### §4.2.6 Implications for v3 / paper framing

- **We cannot cite FedBN as a major PSNR lift.** Our setup (TDBN +
  per-plane granularity + regression) is theoretically predicted
  to show small gaps, and empirically does. Writing "FedBN gives
  0.009 dB" as a headline is honest but un-noteworthy.
- **Alternative framing (stronger):** "Under TDBN, FedAvg and FedBN
  converge to effectively the same stationary point for our
  regression task, which we attribute to TDBN's built-in
  threshold-dependent scale normalisation harmonising feature
  statistics across clients even without explicit BN localisation
  (Diag-B above)." This turns a null result into a positive
  observation about TDBN's robustness.
- **v3 can test Diag-B directly:** run one cell with standard BN
  (not TDBN) + FedBN, same 30 epoch, same Dirichlet partition. If
  that gap is >0.05 dB, Diag-B is confirmed: TDBN is carrying the
  alignment load that FedBN would otherwise carry.
- **Connection to §4.1:** TDBN Imp-1 predicted exactly this outcome
  before we had the FedBN paper in hand. The two sections reinforce
  each other.

### §4.2.7 Open question for §4.3 (SiloBN)

FedBN keeps **all** BN parameters local (γ, β, μ, σ²). SiloBN (the
paper cites it: Andreux et al. 2020) keeps only the untrainable
statistics (μ, σ²) local and still aggregates the trainable
affine (γ, β). If our Diag-B is correct, SiloBN should be
indistinguishable from FedAvg in our setup, because there is no
non-trivial trainable affine drift to worry about. Testable in
§4.3 once user provides SiloBN paper.

---

## §4.3–§4.5: pending (SiloBN, HarmoFL, FedWon, Du-2022)

User has provided SiloBN (Andreux 2020), HarmoFL (Jiang 2022),
FedWon (Zhuang 2024), Du-2022. Next commit will add these in the
same verbatim-plus-strict-application format as §4.1 and §4.2.

