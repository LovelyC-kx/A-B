# §VI-A. Experimental Setup (v2 paper draft)

This section reports the v2 experimental configuration. Every
hyperparameter listed here is **bit-identical across runs A, B and C**
unless explicitly flagged as the deliberate ablation axis.

## A.1 Dataset

We use the **CUHK-CR** benchmark introduced by Sui *et al.* (TGRS 2024)
[cite Sui24], comprising paired (cloudy, clear) Sentinel-2 / Jilin-1KF01B
tiles at 0.5 m ground resolution:

* **CUHK-CR1** — 668 thin-cloud pairs (534 train / 134 test)
* **CUHK-CR2** — 559 thick-cloud pairs (448 train / 111 test)

Total: **982 train / 245 test**, two-source feature shift (cloud
optical thickness as the natural non-IID dimension). The training
split feeds the federated partition (§A.3); the test split is held
out and identical for every cell.

## A.2 Constellation topology

We adopt the **50/5/1 Walker-Star** configuration of FLSNN Fig. 5
verbatim — 5 orbit planes × 10 satellites per plane × 1 phasing
factor, intra-plane chain (ring all-reduce-friendly), inter-plane
chain. This identical configuration with FLSNN allows a direct
comparison with their classification result.

## A.3 Federation partition (`dirichlet_source`, $\alpha = 0.1$)

Each of the 50 satellites receives a non-IID slice of the 982-image
training set. Concretely, for each satellite $k$ we sample
$\boldsymbol{p}_k \sim \mathrm{Dir}(\alpha\!=\!0.1)$ over the two
source labels (CR1, CR2), then randomly assign images from each
source according to $\boldsymbol{p}_k$, with a **minimum of 5
images per satellite** (enforced post-hoc to avoid empty clients).

Empirical partition statistics (see partition heatmap PDF):

* **72 % of satellites** are pure-single-source (≥ 95 % from one of
  the two sources). Of these, 35/50 sit at the 5-image minimum.
* **15/50 satellites** hold > 30 of all training images. The most
  imbalanced plane (P2) is dominated by a single satellite (P2S8)
  with 118 training images.

This level of source-skew is **strictly weaker** than the FLSNN
Fig. 5 setting ($\varsigma\!=\!0.02$ over 10 EuroSAT labels), which
is relevant when interpreting the §VI-D RelaySum reversal.

## A.4 Model

**Backbone (variable):** `VLIFNet` U-Net [cite v1 system code], with
`dim = 24`, encoder block counts `[2, 2, 4, 4]`, decoder block
counts `[2, 2, 2, 2]`, `T = 4` time steps, `2_308_856` parameters
(verified bit-identical between BN variants via `state_dict` total
size). The VLIFNet architecture inherits two attention / mixing
modules from the recent image-restoration literature: the
**FSTA module** (Temporal-Amplitude Attention + DCT-Spatial
Attention) from FSTA-SNN (Yu et al., AAAI 2025,
arXiv:2501.14744) and the **FreMLPBlock** (frequency-domain MLP
via real FFT) from DarkIR (Feijoo et al., CVPR 2025,
arXiv:2412.13443). Neither module is a contribution of this
paper; both enter the evaluation as fixed upstream components
(see `cloud_removal_v1/models/fsta_module.py`).

The BN variant and backbone are the only ablation axes:

* **Run A:** TDBN ($\gamma_{\text{init}}\!=\!\alpha V_{\text{th}}\!=\!0.1061$, per Zheng 2021) + SNN (`MultiSpike4` quantisation: outputs in $\{0, \tfrac{1}{4}, \tfrac{1}{2}, \tfrac{3}{4}, 1\}$, NOT binary).
* **Run B:** standard `nn.BatchNorm2d` + SNN.
* **Run C:** TDBN + ANN (every LIF neuron replaced by `nn.ReLU`).

The **`MultiSpike4` 5-level encoding** is a deliberate departure
from the binary spikes of the original FLSNN. It improves
representational capacity per time step but means **the energy
model that uses 0.9 pJ/AC for a binary spike (Horowitz 2014) is a
lower bound, not a tight estimate, of `MultiSpike4` energy** —
discussion in §VI-E.

The **ANN backbone** keeps the original `T = 4` outer time loop
(the U-Net is run four times per forward pass, even though all four
copies are deterministic functions of the same input). The
1.61$\times$ wall-time speedup we measure (Table IV) is therefore a
**lower bound** on the architectural ANN/SNN gap; a `T = 1` ANN
re-implementation would gain another ~2.5$\times$.

## A.5 Training protocol

| Parameter | Value | Source |
|:----------|:------|:------|
| Optimizer | AdamW                            | `cloud_removal_v1/task.py:200` (built from `config.py` fields) |
| Learning rate    | $10^{-3}$, cosine decay to $10^{-7}$ | `config.py:81–82` (`lr`, `min_lr`) |
| Warm-up          | 3 epochs (linear to peak)        | `config.py:84` (`warmup_epochs`) |
| Batch size       | 4 / satellite                    | `config.py:45` (`train_batch_size`) |
| Local epochs     | 2 full dataset passes / intra-plane round | `config.py:91` (`local_iters`) |
| Intra-plane rounds / global epoch | 2               | `config.py:90` (`intra_plane_iters`) |
| Global epochs    | 80 (CLI `--num_epoch 80`; default 30) | `config.py:89` (`num_epoch`) |
| Loss             | $\mathcal{L} = \text{Charbonnier}(\epsilon\!=\!10^{-3}) + 0.1\!\cdot\!(1\!-\!\text{SSIM})$ | `config.py:94–95` (`ssim_weight`, `charbonnier_eps`) |
| Grad-norm clip   | 1.0                              | `config.py:85` (`clip_grad`) |
| Augmentation     | $h$-flip ($p\!=\!0.5$), $v$-flip ($p\!=\!0.5$), 90° / 270° rot ($p\!=\!0.25$ each), per-batch | `config.py:49–54` (`aug_*` + `augment`) |

**RelaySum learning-rate scaling.** Following the FLSNN reference
implementation (`revised_constellation.py:204–205`), the RelaySum
scheme multiplies its effective learning rate by `2.093` to
compensate the delayed-aggregation noise. We retain this constant
verbatim (`constellation.py:190`). It is an EuroSAT-classification
empirical value and is **not validated for our regression task**;
this is one of the candidate sources of the §VI-D RelaySum reversal.

**`bn_local` is hardcoded to `False` for intra-plane aggregation**
(`constellation.py:215–218`). FedBN's "BN-local" semantics applies
only to the inter-plane step; within a plane all 10 satellites
average their full state-dicts (BN included). This matches the
FedBN paper's per-client-domain interpretation lifted to our
two-level orbital topology.

## A.6 Evaluation protocol

**Mode:** `center_patch` with a 64 $\times$ 64 crop (matches training
patch size to keep the model in-distribution at eval time).
**Frequency:** every 5 global epochs (`eval_every = 5`).

**Per-plane PSNR is an *ensemble-per-image* metric.** For each test
image, we compute the per-image PSNR using each plane's converged
intra-plane model (after `intra_plane_iters = 2` averaging steps
that make all 10 satellites in a plane share weights). The
**per-image PSNR is then averaged across the 5 planes**, and the
**245 per-image averages are then meaned**. This is the value
reported in Tables I, II, IV. (See `cloud_removal_v1/evaluation.py:220–253`.)

This definition matches the "test the consensus" interpretation
used in most decentralised-FL papers. It is **not** the same as
"average each plane's mean test PSNR across planes" — the two
coincide for FedAvg+AllReduce (where all 5 planes are bit-equal
at the end of a round) and disagree for FedBN cells (where each
plane has a distinct BN affine layer). Per-plane individual PSNRs
(used in the per-plane spread analysis of §VI-G) are reported
separately.

## A.7 Reproducibility

* Single base seed (`seed = 1234`, `partition_seed = 0`).
  Each of the 6 (bn_mode, scheme) cells additionally resets the
  RNG via a deterministic per-cell offset
  $\mathrm{cell\_seed} = 1234 + 10^4 \cdot (k+1)$ with $k$ the
  stable cell index, so runs A and B see an identical
  augmentation / optimiser noise stream per cell
  (`run_smoke.py:279–290, 583`). A `seed = 1235` re-run would
  produce a fully independent trajectory — "single seed" refers
  to the choice of a single base seed value for the reported
  results, not to only one effective RNG state in the sweep. We
  disclose this as a v2 limitation (§VI-H); the multi-seed run
  is deferred to v3.
* Hardware: NVIDIA RTX 4090 (24 GB), CUDA 13.0, PyTorch with the
  default `'torch'` LIFNode backend (no Cython / Triton). Total v2
  training time: **A 36.31 h + B 36.66 h = 72.97 h** on the shared
  4090 sequential session, plus **run C 3.86 h** separately on the
  same machine → **76.83 h total wall time** across all 13 cells.
* Determinism: `cloud_removal_v2/config.py:27` sets
  `deterministic = False` for AdamW + cuDNN performance reasons;
  re-running from the saved seed reproduces the headline numbers
  to ~2 decimal places of PSNR.
* Code: tag this commit on the `claude/setup-new-output-directory-8xvx9`
  branch. The drift script that produces Table III is
  `cloud_removal_v2/analyze_bn_drift_posthoc.py` (commit `17cd881`).
* Data: CUHK-CR1 + CUHK-CR2 from the official Tianyun release
  (Sui *et al.*, TGRS 2024); not redistributed.

## Training-stability disclosure (preview of §VI-A.8 figure)

The per-round PSNR convergence curves
(`v2a_v2a_80ep_test_psnr.pdf` for Run A and the corresponding
`*_stdbn_test_psnr.pdf` for Run B; rendered as **Fig. 2** in
the camera-ready paper) show
**substantial transient instability** in the FedBN cells:
**RelaySum + FedBN drops to 19.69 dB at round 25** before recovering,
and AllReduce + FedBN oscillates in the **20.3–21.0 dB** band until
round ~55. All cells stabilise within a 0.3 dB band only after
round ~65 (AR-FedBN ep 65–80 PSNR span = 21.135–21.420 dB = 0.285 dB).
We therefore report **PSNR at round 80** as the
end-of-training value in all tables, and explicitly note that
single-seed runs cannot distinguish 0.05 dB differences within
the converged band.
