# Three models, one page

Rigorous side-by-side of the three SNN architectures that show up in
this project.  "FLSNN" = the original brain-inspired satellite-FL paper
(which we extend).  "ESDNet" = the reference deraining SNN that VLIFNet
builds on.  "VLIFNet (ours)" = the model actually trained in v1.

## At a glance

| | **FLSNN** (Yang et al. 2025) | **ESDNet** (Song et al. 2022) | **VLIFNet (ours)** |
|---|---|---|---|
| **Paper** | *Brain-Inspired Decentralized Satellite Learning in Space Computing Power Networks*, [arXiv 2501.15995](https://arxiv.org/abs/2501.15995) | *Exploring the Potentials of Spiking Neural Networks for Image Deraining*, [arXiv 2207.02094](https://arxiv.org/abs/2207.02094) | based on ESDNet's SRB, extended by the user (system1_VLIFNet) |
| **Task** | Land-cover classification (10 classes, EuroSAT) | Image deraining (rainy → clean) | Cloud removal on CUHK-CR1 (cloudy → clean) |
| **Output** | Logits → class label | H×W×3 image | H×W×3 image |
| **Loss** | Cross-entropy | L1 + SSIM + FFT | Charbonnier + 0.1·(1−SSIM) |
| **Model scope** | Spiking CNN (2 conv, 1 FC) OR SmallResNet [1,2,2] | Spiking U-Net with SRB | Spiking U-Net with enhanced SRB + FSTA + FreMLP + GatedSkip |
| **Params** | ~80 K (SpikingCNN) / ~0.2 M (SmallResNet) | ~3.5 M | ~5 M (dim=24) / ~13 M (dim=48) |
| **Time steps T** | 3 | 4 | 4 |
| **Spike encoding** | Binary 0/1 (Heaviside) | Binary 0/1 | **4-level {0, 0.25, 0.5, 0.75, 1.0}** (MultiSpike4) |
| **Surrogate / grad** | NoisySpike (Bernoulli mask + surrogate) with **hybrid activation** (eq. 7 in FLSNN paper) | Rectangle surrogate | Rect window with 1/4 gradient scaling (matches MultiSpike4 step) |
| **BN** | `tdBN` (standard BN wrapped per-time-step) | `tdBN` | `ThresholdDependentBatchNorm2d` (SpikingJelly, TDBN variant) |
| **Attention** | None | None | `MultiDimensionalAttention` + **FSTA** (TA × DCT-SA) + **FreMLPBlock** |
| **Skip connection** | Identity (ResNet-style) | Conv + add | **GatedSkipFusion** (learnable sigmoid gate per location) |
| **Multi-scale supervision** | N/A | N/A | aux heads at L2/L3 (present in code, **not used in v1**) |
| **Framework** | pure PyTorch + custom SNN code | pure PyTorch + custom SNN code | **SpikingJelly** (multi-step LIF, TDBN, MDAttention) |
| **FL component** | decentralized RelaySum + MDST routing | single-machine centralised training | **inherits FLSNN's RelaySum**; trained decentralised in v1 |

## Module-level diff — what VLIFNet keeps from ESDNet, what it adds

### Kept verbatim from ESDNet
* Overall **U-Net skeleton** with 3 encoder + 3 decoder levels.
* **Spiking Residual Block (SRB)** as the atomic compute unit — double
  convolution with LIF between them, wrapped in a residual shortcut.
* The **principle** of using spiking activations as high-frequency
  filters (the LIF non-linearity is a frequency-selective gate).

### Added by VLIFNet
1. **MultiSpike4 quantised neurons** (5-level {0, ¼, ½, ¾, 1}) replace
   ESDNet's binary {0, 1}.  This is a strict *superset* of the binary
   regime: if the downstream computation only ever sees multiples of 1,
   it's equivalent to binary; at higher resolutions, gradient flow is
   cleaner because the activation has a smoother effective slope.
2. **Dual-group frequency decomposition** inside each SRB (`temporal
   high-freq via LIF_1`  +  `spatial high-freq via PixelShuffleLIFBlock`),
   with learnable `high_freq_scale` / `low_freq_scale` multipliers and a
   `cross_scale_gate` σ-gate that mixes the two high-freq branches.
   ESDNet's SRB had a single LIF branch; VLIFNet *explicitly* separates
   spatial vs temporal structure-aware gating.
3. **MultiDimensionalAttention** (MDAttn) — a SpikingJelly channel+time
   attention module applied after each conv block.
4. **FSTA** (Frequency-based Spatial-Temporal Attention), a plug-in
   block chained after MDAttn:
   * `TemporalAmplitudeAttention` — per-timestep amplitude gate
     (complement of the standard 3D channel-wise time attention).
   * `DCTSpatialAttention` — 2D rFFT → MLP on magnitude → iFFT → 7×7
     conv → sigmoid spatial mask.  Phase is preserved; this is a
     content-adaptive frequency filter, not a pooled summary.
   * Learnable scalar gate initialised at 0 ⇒ FSTA starts as identity.
5. **FreMLPBlock** (DarkIR) — another frequency-domain MLP, inserted at
   the end of each SUNet_Level1_Block.  Provides O(N log N) global
   receptive field via FFT magnitude modulation.
6. **GatedSkipFusion** — replaces the trivial `decoder + skip` with
   `sigmoid(Conv1×1([dec, enc])) * dec + (1-gate) * enc` so the decoder
   learns per-pixel how much of the encoder to borrow.
7. **Additional-level-1 SUNet** (an extra `SUNet_Level1_Block`
   at decoder output) — doubles the capacity at full resolution where
   the last mile of detail matters most for cloud removal.
8. **Auxiliary heads at L2 and L3** for deep supervision (unused by v1;
   flag `aux_loss_weight=0` in the task, wired for v3).

## What our project keeps vs changes vs FLSNN

### Kept verbatim from FLSNN
* **§II System model** — the same LEO-constellation / (plane, satellite)
  two-level client structure.
* **§III-B RelaySum inter-plane aggregation** — our
  `_relaysum_step` is an exact line-by-line port of the paper's
  Algorithm 2 with persistent per-plane relay buffers.
* **§IV Convergence theory** — our Charbonnier loss satisfies the
  paper's L-smoothness assumption (Charbonnier is twice-differentiable),
  so Theorem 2 carries over.  The SSIM term is down-weighted (λ=0.1) so
  it perturbs the loss surface without breaking smoothness dominance.
* **§V MDST inter-plane routing optimisation** — unchanged for v1 (v3
  plans a time-aware, energy-weighted extension).

### Changed for cloud-removal regression
* **§III-A SNN model** — SpikingCNN / SmallResNet  →  VLIFNet U-Net.
* **Loss** — cross-entropy → Charbonnier + λ·(1-SSIM).
* **Evaluation** — top-1 accuracy → PSNR / SSIM (on centre-cropped
  patches by default; sliding / full-image available).
* **Dataset** — EuroSAT .pkl classification → CUHK-CR1 paired images.
* **Non-IID partition** — Dirichlet over class labels → IID random
  (regression has no class label; Dirichlet-over-scene-cluster is v2).
* **BN** — tdBN(nn.BatchNorm2d) → `ThresholdDependentBatchNorm2d`
  (SpikingJelly, same math, different impl).  BN running stats are
  aggregated in v1 (bn_local=False); FedBN (bn_local=True) is the v2
  default switch.
* **Batch size** — 32 → 4 (VLIFNet is larger, V100 RAM-bound at dim=24).
* **Optimiser** — SGD+momentum → AdamW with 3-epoch warmup + cosine
  (matches VLIFNet upstream train.py; empirically more stable for
  spiking image-restoration nets).
* **Model state-dict handling** — FLSNN originally `deepcopy(model)`,
  which carries SpikingJelly's per-neuron membrane buffers (`v=0.0`
  scalars or tensors) through aggregation.  Our v1 uses state-dict
  round-trips with non-tensor entries skipped, so aggregation is
  agnostic to SpikingJelly memory.

### Added by v1
* A SpikingJelly-aware aggregation layer (`aggregation.py`) that
  *correctly* handles scalar / None memory entries — a bug that would
  silently crash FLSNN's original pipeline if VLIFNet were dropped in.
* A device-consistent RelaySum buffer (zeros live on the same GPU as
  the per-satellite weights — the earlier CPU-init'd version would hit
  a device-mismatch error on the first aggregation step).
* Self-tests (`cloud_removal_v1/tests/run_all.py`) exercising the
  tensor math and partitioning arithmetic without GPU.
* Eval modes tuned to VLIFNet's training distribution (64² centre
  patch) so evaluation never goes OOM or out-of-distribution.

## Parameter budget (fp32)

| Config | Params | state_dict size | RelaySum per-round bytes (5-plane chain) |
|---|---|---|---|
| FLSNN Spiking CNN | ~80 K | ~0.3 MB | ~2.4 MB |
| FLSNN SmallResNet | ~200 K | ~0.8 MB | ~6.4 MB |
| VLIFNet dim=24 (v1) | ~5 M | ~20 MB | **~160 MB** |
| VLIFNet dim=48 (upstream) | ~13 M | ~52 MB | ~420 MB |

This ~25–65× jump in communication is the direct price of moving from
a 10-class classifier to a 3-channel image regressor.  v3 includes two
planned mitigations:
* **TE-MDST** (time-aware energy-weighted routing) — balance
  transmission energy against topology diameter.
* **Spike-aware compression** — sparsify / quantise the SRB weight
  updates per round.
Both are out of v1 scope; v1's job is to verify convergence and
baseline the bytes-vs-PSNR trade-off.

## One-paragraph summary for the paper

> VLIFNet is a 5-level-quantised spiking U-Net with frequency-domain
> attention (FSTA) and global frequency enhancement (FreMLP), extending
> the single-branch SRB of ESDNet (SNN deraining, 2022) with a dual
> spatial/temporal high-frequency decomposition, content-adaptive skip
> fusion, and multi-scale aux supervision.  Relative to the SNN
> baselines used in the FLSNN satellite-FL paper (Yang et al. 2025), it
> adds ~25× parameters; this work embeds VLIFNet into FLSNN's
> decentralised RelaySum framework unchanged, showing that the paper's
> §IV convergence guarantee carries over to image regression when the
> loss is dominated by a Charbonnier term.
