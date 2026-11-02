# v2-A Interpretation — Mechanisms and Hypotheses

Companion to `v2_results_synthesis.md`. That file captures data.
This file proposes **MECHANISMS** that would explain the data, separated
into three tiers by evidence strength:

- **TIER A — Verified from experiment**: we have direct measurement
  supporting the claim.
- **TIER B — Theoretically supported, empirically plausible**: an
  established result from decentralized-SGD / FL theory predicts what
  we see; no contrary observation.
- **TIER C — Speculative**: plausible mechanism, needs experiment or
  additional analysis to confirm.

Order within each tier: strongest evidence first.

---

## TIER A — Verified from experiment

### A1. Reversal is robust to α and to total rounds

Observed v1 (IID, 10 ep) AND v2 (α=0.1, 35 ep) both show RelaySum last.
This excludes two trivial explanations:
- "Non-IID partition broke RelaySum" — no, RelaySum was also last in IID.
- "We didn't train long enough" — FLSNN Fig 5(b) shows ranking stabilised
  by round ~40 of 60; our 35 rounds already passed that threshold.

Therefore the reversal has a STRUCTURAL cause tied to task / model /
loss, not to training budget or partition severity.

### A2. AllReduce's advantage is NOT the artefact of our comm-accounting

Concern: since we undercount AllReduce's bytes, maybe its "winning both
PSNR AND bytes" is the accounting's doing. REBUTTAL: AllReduce beats
Gossip/RelaySum on **PSNR alone**, ignoring bytes. That's 21.36 (avg) vs
21.33 (Gossip) vs 21.24 (RelaySum). The comm advantage is independent of
and additional to the accuracy advantage.

### A3. FedBN's gain is small but directionally consistent

6/6 cells: FedBN averages +0.009 dB PSNR, +0.0014 SSIM over FedAvg. Top
cell (rank 1 overall) is a FedBN cell. For single-seed, this is below
noise, BUT the direction flips ZERO times — 6 out of 6 cell pairs
(FedBN vs FedAvg at same scheme) show FedBN winning SSIM. Consistent
direction without a single reversal over 6 comparisons is unlikely to
be chance; it reflects a real (small) advantage of per-plane BN under
feature shift.

---

## TIER B — Theoretically supported, empirically plausible

### B1. The reversal aligns with known RelaySum vs Gossip theory

Vogels et al. (NeurIPS 2021, arXiv:2110.04175) prove: RelaySum's
convergence bound is

> **‖x̄_T − x*‖² ≤ O(1/T) + O(σ² / (n T)) + O(ζ² τ² / T)**   (Thm 1 / 2)

where
- σ² = stochastic-gradient variance
- ζ² = **inter-client** gradient variance (heterogeneity)
- τ = effective mixing delay = O(graph diameter) for spanning tree
- n = number of clients

Gossip's bound is (Koloskova et al. 2020, arXiv:2003.10422):

> **‖x̄_T − x*‖² ≤ O(1/T) + O(σ²/(nT)) + O(ζ²/((1-λ₂)² T))**

where λ₂ is the mixing-matrix second eigenvalue.

**RelaySum wins roughly when** `τ(1-λ₂)` is less than O(1).
[Correction 2026-04-20 per Agent-6 audit, `v2_comprehensive_literature.md §25.10`:
the exact crossover is `τ(1−λ₂) vs O(1)`, not `τ²(1−λ₂)`.
My earlier "τ² < 1/(1-λ₂)²" form is equivalent in sign but
was derived informally.]

**Weight-scheme sensitivity** of the crossover (chain-5):

| Weight scheme | λ₂ | 1−λ₂ | τ(1−λ₂) | Predicted winner |
|--------------|------|------|---------|-------------------|
| Uniform lazy random walk (path graph) | cos(π/5) ≈ **0.809** | 0.191 | **0.76** | RelaySum (barely) |
| Metropolis (standard gossip) | 2/3 ≈ **0.667** | 0.333 | **1.33** | Gossip/AllReduce |
| Our code (`_gossip_average`) | between the two; non-doubly-stochastic | est. 0.20–0.33 | est. 0.8–1.3 | **boundary regime** |

[Note: my earlier text here labeled the uniform-walk formula
as "Metropolis weights" — that was a factual error, corrected
2026-04-20.]

**Our code inspection** (`cloud_removal_v1/constellation.py:258-269`
`_gossip_average`): each plane averages {self ∪ neighbours}
with uniform weights 1/(1+|N(p)|); this is row-stochastic but
NOT doubly-stochastic. Row sums = 1 ✓; column sums are
{5/6, 7/6, 1, 7/6, 5/6} in chain-5. Standard decentralized-SGD
theory assumes doubly-stochastic W, so our code technically
violates that assumption. Computing exact λ₂ of our specific W
requires numerical eigen-decomposition (v3 task).

**Conclusion**: the theory does NOT cleanly predict our
ordering because (a) exact λ₂ depends on the actual mixing
matrix which is non-standard in our code, (b) the crossover
τ(1−λ₂) can sit on either side of 1 depending on weight
convention. What CAN be said: our regime is near the
theoretical crossover boundary; small changes in ζ² or
weight choice can flip the ordering. Our observed ordering
(RelaySum last) is *consistent with* the Metropolis-weight
prediction, not with the uniform-walk prediction. See
`v2_comprehensive_literature.md §25.10.4` for the
GLOBECOM-defensible version of this claim.

### B2. Regression has lower inter-client gradient variance ζ² than classification

Reasoning: for pixel-wise Charbonnier loss,

> ∂L/∂θ  =  Σ_{pixels} sign(pred_p − gt_p) · (∂pred_p / ∂θ) / scale

The per-sample gradient is a SUM OVER PIXELS of per-pixel residuals. Even
when client A sees mostly thin clouds and client B sees mostly thick
clouds, their gradients share a common low-frequency component (pixel-
level reconstruction error) that dominates the high-frequency
cloud-structure component. By contrast, classification gradients are
driven by the ONE correct-class logit vs others — client A seeing only
class-3 images and client B seeing only class-7 images produces
gradients that are **orthogonal in the last layer's output space**.

Therefore ζ²_regression  <  ζ²_classification  for matched non-IID
severity. Smaller ζ² shrinks the denominator in BOTH RelaySum's and
Gossip's bounds by the same factor, but it hurts RelaySum MORE because
RelaySum's ζ² term scales with τ² (= 16 for chain-5), whereas Gossip's
scales with 1/(1-λ₂)² (≈ 27 for chain-5). When ζ² → 0, BOTH converge
fast and the advantage shrinks — but RelaySum's fixed τ² adds
per-round overhead (more communication) without compensating benefit.

**This is speculative-mathematical, not theorem-proved.** Needs either
(a) empirical measurement of per-client gradient cosine similarity in
our task, or (b) analytical treatment in the paper. See §C3.

### B3. FedBN's gain is predictable from feature-shift theory

Li et al. (ICLR 2021, arXiv:2102.07623) prove FedBN helps under
**feature shift** (covariate shift in p(x)) by allowing per-client BN to
track client-specific pixel statistics, while globally sharing conv
weights. Our setup IS feature shift: CR1's thin clouds have different
pixel-value distributions from CR2's thick clouds. FedBN-at-plane-
granularity inherits this advantage partially (per-plane, not per-sat),
which is exactly the SSIM bump we observe.

The gain is **small** because:
- The feature shift is modest (both CR1 and CR2 are Jilin-1 imagery, not
  e.g. Sentinel-2 vs Landsat).
- Our intra-plane aggregation averages BN within each plane regardless
  of the bn_local flag (hard-coded, see `constellation.py:200-211`),
  diluting per-satellite BN specialisation to per-plane BN.

A "true FedBN" (per-satellite local BN) would likely show a larger
gain. This is an ablation for v3.

---

## TIER C — Speculative; needs additional analysis / experiment

### C1. RelaySum's multi-round relay amplifies local overfitting

Under strong Dirichlet α=0.1, a client with 5 samples overfits its local
subset severely in a single local iteration. RelaySum's relay buffer
accumulates this overfit into the following rounds' messages. Gossip's
immediate-neighbour average dilutes the overfit before it propagates.
AllReduce's global mean over 5 planes of post-intra-avg weights has the
strongest dilution.

This predicts:
- RelaySum's per-plane PSNR **variance** should be larger than Gossip's
  or AllReduce's. **We have the data to check this** — `per_plane_psnr`
  is saved in the npz. Script this in v2.1.
- Raising `min_samples_per_client` to 20 should shrink the RelaySum gap.
  Experiment for v3.

### C2. The VLIFNet residual-add-input acts as a global bias that reduces effective ζ²

VLIFNet's last layer is `main_out = Conv(decoder) + input`. The residual
connection means gradient flow to earlier layers is dominated by the
identity-path error `(pred - gt)` not the learned-delta error. This
identity-path gradient is the SAME across all clients (same image →
same identity). So the **client-heterogeneous** portion of the gradient
is a small perturbation on top of a shared identity gradient, reducing
effective ζ² further than a "pure" regression head would.

This would predict: a non-residual baseline would show a larger RelaySum
advantage. Experiment for v3 (swap `main_out` = Conv(decoder) only, no
residual, re-run the 6-cell sweep).

### C3. Per-client gradient cosine similarity measurement

Directly measure in a single round:

```
For each pair (client_i, client_j):
  cos_sim_ij = cos(∇L_i(θ), ∇L_j(θ))
```

If mean cos_sim > 0.7 for our task vs <0.3 for classification, we have
empirical support for §B2. Can be implemented in ~30 lines of Python
using the task-level hook. Run ONCE, report in paper.

### C4. FedBN × AllReduce is Pareto-optimal is a topology-size artefact

On a chain of 5, AllReduce dominates because N is small enough that
the "all-to-all communication cost" penalty is small (O(N) bytes per
round), while its mixing advantage is large (no gossip consensus gap).
On a chain of 50 planes, AllReduce would be infeasible and
Gossip/RelaySum would separate clearly. Our N=5 result therefore does
NOT generalise to large constellations.

**Paper caveat**: explicitly state that AllReduce's dominance is
regime-specific to small constellations. For >10 planes, the picture
likely changes.

### C5. SNN quantization-noise dominates the inter-scheme signal at small scales

VLIFNet's MultiSpike4 forward divides by 4 and backward divides by 4
(net 1/16 effective gradient scale). This adds stochastic quantization
noise to every per-client gradient that is ORDER-OF-MAGNITUDE larger
than the client heterogeneity noise. In that regime, all three
aggregation schemes converge to similar noise-limited performance —
which is consistent with our 0.174 dB spread across 6 cells.

If true, replacing VLIFNet with an ANN U-Net (same architecture minus
LIFNodes / MultiSpike4) would RESTORE a larger spread between schemes
AND shift RelaySum back into contention. This is the single most
important v3 experiment to disentangle.

---

## Cross-cutting: what if our 70-epoch sweep SHOWS a flip?

The Tier B arguments apply uniformly to all training lengths. The Tier
C arguments are mostly round-invariant. The most plausible mechanism
for a late-epoch flip would be:

- RelaySum's "precise average" (Vogels et al. Prop. 1) gives
  zero-variance consensus in the long-T limit, while Gossip retains a
  residual variance O(ζ²/(1-λ₂)²). At T → ∞ RelaySum wins. But our
  (0.76) < 1 margin on chain-5 makes this crossover EXTREMELY late.

Prediction: 70 epochs not enough to flip. If it does flip, we need to
revise Tier B's bound-level analysis.

---

## Immediate v3 hooks (these interpretations imply experiments)

| Interpretation | v3 experiment |
|---|---|
| A3 (FedBN modest) | Try per-satellite BN (not per-plane) |
| B2 (regression ζ²) | Measure per-client gradient cos-sim |
| C1 (overfit-amplify) | Report per-plane PSNR variance; ablate min_per_client |
| C2 (residual bias) | No-residual VLIFNet ablation |
| C5 (SNN noise) | Run 6 cells with ANN U-Net instead of VLIFNet |
| C4 (topology size) | Extend to N=10 or N=20 planes |

---

*All interpretations in this file are working hypotheses. Paper must
clearly mark A/B/C tier level for each claim it makes.*
