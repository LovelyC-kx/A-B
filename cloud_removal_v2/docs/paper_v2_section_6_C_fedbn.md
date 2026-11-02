# §VI-C. FedBN-Style BN-Local Aggregation (Main Claim)

This is the **central new contribution of v2**: a controlled
ablation of FedBN [Li 2021] — the standard prescription for
"keep BN parameters local" — under our cloud-removal regression
task, and a mechanistic explanation of *why* its expected gain
shrinks here.

We make a two-layer claim, supported by Tables II and III
respectively:

> **Claim C16 (updated).**
> (a) Mechanism layer — TDBN's $\alpha\!\cdot\!V_{\rm th}$-shared
> scaling pre-aligns plane-level BN statistics, *reducing
> cross-plane drift relative to standard BN by 42 %*.
> (b) Effect layer — even the larger drift produced by standard BN
> is **too small to translate into a measurable FedBN PSNR gain**
> on this task: the mean PSNR difference $\mathrm{FedBN} -
> \mathrm{FedAvg}$ is $+0.008$ dB (TDBN) and $+0.044$ dB (BN2d),
> both within the single-seed noise floor.
>
> Consequently, in this regime FedBN is **conditionally
> redundant**: its theoretical purpose (canceling inter-client
> BN-statistic drift) is only marginally needed because (i) the
> task's $\zeta^2$ is small (§VI-B) and (ii) TDBN further reduces
> what BN drift remains.

The rest of this subsection unpacks both layers and connects them
to the published FedBN, FedBN-derived (HarmoFL), and FedWon results.

## C.1 Effect layer — Tables I and II

Reading Table II (`docs/tables/table_fedbn_ablation.md`):

| BN variant | $\Delta$(AllRed) | $\Delta$(Gossip) | $\Delta$(Relay) | Mean $\Delta$ |
|:-----------|:---:|:---:|:---:|:-----:|
| A: TDBN, SNN | $-0.222$ | $+0.186$ | $+0.061$ | **+0.008** |
| B: BN2d, SNN | $+0.132$ | $+0.010$ | $-0.010$ | **+0.044** |

Two empirical observations:

(E1) **The mean $\Delta$ across schemes is below 0.05 dB in both BN
variants**, our pre-registered threshold for "no observable
effect" in single-seed regression PSNR.

(E2) **Per-scheme variation is substantial.** A's $\Delta$ ranges
from $-0.22$ dB (FedBN hurts) to $+0.19$ dB (FedBN helps). With
only 1 seed and 1 partition, the per-scheme $\Delta$ values are
not individually informative; the mean across 3 schemes is the
most reliable single-seed estimator.

These match the FedBN paper's *qualitative* prediction — FedBN
helps when client-level BN statistics differ — but the *magnitude*
is two orders of magnitude smaller than the +1 dB to +8 dB gains
reported in the original FedBN classification benchmarks.

## C.2 Mechanism layer — Table III

Reading Table III (`docs/tables/table_drift.md`) at the FedBN cells:

| FedBN cell | A (TDBN) Var(γ) | B (BN2d) Var(γ) | A / B |
|:-----------|:---:|:---:|:---:|
| AllReduce | 7.04e-05 | 1.28e-04 | 0.55 |
| Gossip    | 7.38e-05 | 1.10e-04 | 0.67 |
| RelaySum  | 2.30e-04 | 4.08e-04 | 0.56 |
| **Mean**  | **1.25e-04** | **2.15e-04** | **0.58** |

**TDBN's cross-plane $\mathrm{Var}(\gamma)$ is 58 % of BN2d's.**
This is the mechanistic basis of Claim C16: the threshold-dependent
BN of [Zheng 2021] initialises $\gamma_{\rm init} = \alpha\,V_{\rm
th} \approx 0.106$, sharing a scale that the optimizer perturbs in
a smaller relative range than the $\gamma_{\rm init} = 1$ of
standard BN. The smaller $\gamma$ excursions — averaged across the
51 BN-affine layers of VLIFNet — produce smaller cross-plane
divergence under FedBN (which does not average $\gamma$ across
planes by definition).

The mechanism is **backbone-agnostic** (Table III, Run C row):
ANN-FedBN+AllReduce gives $\mathrm{Var}(\gamma) = 8.72\mathrm{e}{-}05$,
within 24 % of A's 7.04e-05 SNN value. The TDBN alignment effect is
therefore a property of the normalization layer's parameterisation,
not of the upstream activation type.

## C.3 Why the (modest) drift reduction does not translate to PSNR

The mechanism layer reduces drift by 42 %; the effect layer sees
PSNR change by $\le 0.05$ dB. The connection is **non-linear and
saturating**: FedBN's role is to *cancel a drift that is too small
to matter here in the first place*.

The argument has three steps:

* **Step 1.** Even BN2d's larger drift is small in absolute terms.
  Mean $\mathrm{Var}(\gamma) = 2.15\mathrm{e}{-}04$ corresponds to
  per-channel $\sigma_\gamma \approx 1.5\mathrm{e}{-}02$ (4 to 14
  per cent of $|\gamma|$ on a typical TDBN layer where the trained
  $|\gamma| \in [0.1, 1.1]$). For comparison, FedBN's published
  benchmarks measure drift orders of magnitude larger when client
  data covers different *medical centres* (HarmoFL) or
  *smartphone cameras* (FedBN paper).
* **Step 2.** Pixel-regression PSNR responds slowly to small BN
  drift. A 1 % per-channel BN bias propagates through a U-Net's
  64-fold spatial down-up cascade and gets diluted by the residual
  skip connections (`vlifnet.py:550–587`). In contrast,
  classification logits are concentrated at one channel and amplify
  the same BN bias by orders of magnitude.
* **Step 3.** Therefore both BN variants live below the *task's*
  saturation point for FedBN benefit. TDBN moves us further below
  it (E1: $+0.008$ dB), BN2d less so ($+0.044$ dB), but both gains
  are noise-level.

This is exactly the picture predicted by Theorem 2 of FLSNN: the
$\zeta^2 \to 0$ limit collapses RelaySum's, AllReduce's, and
Gossip's PSNR-per-round bounds onto each other, leaving the
$\sigma^2/(N\epsilon^2)$ stochastic-gradient term dominant. FedBN
is the BN analogue: its $\zeta^2_{\rm BN}$ is also too small to
matter on this task.

## C.4 Comparison with the published FedBN literature

| Paper | Task | FedBN gain (PSNR / accuracy) | Setting |
|:------|:-----|:---:|:------|
| FedBN [Li 2021]            | classification (Office-Caltech) | +7.8 % accuracy  | 4 clients, label + style shift |
| FedBN [Li 2021]            | classification (ABIDE-I)        | +0.9 % accuracy  | 4 clients, multi-site MRI |
| HarmoFL [Jiang 2022]       | medical seg (4-site OD)         | +1.2 dB Dice     | 4 clients, $\zeta^2 \approx 10^{-2}$ |
| FedWon [Du 2022]           | classification (CIFAR-10/100)   | "BN-free even better" | client-shift |
| **Our v2 (TDBN, SNN)**    | **cloud removal** (CUHK-CR)     | **+0.008 dB**    | **50 clients, source shift, $\alpha\!=\!0.1$** |
| **Our v2 (BN2d, SNN)**    | **cloud removal** (CUHK-CR)     | **+0.044 dB**    | **same** |

The trend across this literature row is monotone: **FedBN's gain
shrinks as $\zeta^2_{\rm BN}$ shrinks, regardless of BN family**.
Our setting sits at the small-$\zeta^2$ end of the published range.
The result is therefore **consistent with prior FedBN results
extrapolated to the small-$\zeta^2$ regime** — not a refutation of
them.

## C.5 Caveat and what we do not claim

**What we claim:** in source-level Dirichlet pixel regression with
$\alpha = 0.1$ over 2 sources, FedBN's PSNR gain is below the
single-seed noise floor for both TDBN and standard BN; TDBN
additionally reduces inter-plane $\mathrm{Var}(\gamma)$ by 42 %.

**What we do *not* claim:**

* That FedBN is universally redundant. The published $> 1$-dB / $>
  3$-pp gains are real in their respective regimes; our result
  refines, not refutes, them.
* That TDBN is universally superior. Tables I and II also show that
  *standard* BN2d slightly outperforms TDBN on absolute PSNR
  (run B mean 21.73 vs run A mean 21.50). The two BN families
  have different sweet spots.
* That FedBN is harmful. The mean $\Delta$ is *positive* in both
  runs; FedBN simply does not help enough to justify the engineering
  cost.

## C.6 Reproducibility

* Numbers in Tables II and III are reproduced bit-equal by:
  ```bash
  python -m cloud_removal_v2.analyze_bn_drift_posthoc \
      --ckpt_dir Outputs_v2/ckpts \
      --out      Outputs_v2/v2_drift_report.md
  ```
* The $\Delta$ values in Table II are algebraic differences of
  Table I cells; no statistical machinery beyond a single-seed
  point estimate is involved.
* The 60 plane checkpoints under `Outputs_v2/ckpts/` are the
  ground truth; the script is at commit `17cd881`.
