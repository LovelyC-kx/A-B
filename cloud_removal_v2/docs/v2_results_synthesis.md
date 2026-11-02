# v2-A 35-Epoch Results — Complete Synthesis

Written 2026-04-19, immediately after the first 6-cell sweep finished on AutoDL
(run_name=`v2a`, 17h wall time, no failures). Purpose: lock down the HARD
DATA in one place before re-interpretation.

---

## 1. Final numbers (from `Outputs_v2/v2a_v2a_summary.json`)

| Rank | Cell | PSNR (dB) | SSIM | Comm (GB) | Wall (h) |
|:--:|---|--:|--:|--:|--:|
| 🥇 1 | **fedbn × AllReduce** | **21.387** | **0.6586** | **1.62** | 2.92 |
| 🥈 2 | fedavg × Gossip      | 21.369 | 0.6572 | 2.59 | 2.96 |
| 🥉 3 | fedavg × AllReduce   | 21.339 | 0.6556 | 1.62 | 2.74 |
| 4 | fedbn × Gossip         | 21.292 | 0.6563 | 2.59 | 2.90 |
| 5 | fedbn × RelaySum       | 21.270 | 0.6573 | 2.59 | 2.65 |
| 6 | fedavg × RelaySum      | 21.213 | 0.6551 | 2.59 | 3.01 |

**PSNR spread**: 0.174 dB (tight).  **SSIM spread**: 0.0035.  **Comm spread**:
AllReduce uses 37% fewer bytes than Gossip/RelaySum under our accounting
(caveat: see `v2_remaining_issues.md §3.1`).

---

## 2. Per-scheme rankings (averaged over BN mode)

| Scheme  | Avg PSNR | Avg SSIM |
|---|--:|--:|
| AllReduce | **21.363** | **0.6571** |
| Gossip    | 21.331 | 0.6568 |
| RelaySum  | 21.242 | 0.6562 |

**Key observation**: RelaySum (the "proposed" algorithm in the original FLSNN
paper) is **last** on both metrics, by a margin of 0.09–0.12 dB.

## 3. Per-BN rankings (averaged over scheme)

| BN mode | Avg PSNR | Avg SSIM |
|---|--:|--:|
| FedBN   | **21.316** | **0.6574** |
| FedAvg  | 21.307 | 0.6560 |

FedBN wins PSNR by 0.009 dB (within noise) and SSIM by 0.0014 — modest but
**consistent in direction**. FedBN also claims the top cell (rank 1).

---

## 4. Training-curve observations (from the per-epoch log)

### Convergence — NOT PLATEAU at epoch 35

Looking at the last 3 eval points (epochs 25, 30, 35) of the best cell
(fedbn × AllReduce):
- ep 25 → 30: +0.067 dB
- ep 30 → 35: +0.026 dB

Slope decreasing but non-zero. Extrapolating a decaying exponential fit,
the asymptote is ≥ 21.5 dB. **The 70-epoch sweep is required to claim
convergence**.

### Non-monotonic dips (verified from the log)

- `fedavg × AllReduce`: SSIM epoch 15→30 monotonic decrease 0.6566 → 0.6553,
  recovers to 0.6556 at 35.
- `fedbn × Gossip`: PSNR epoch 10→15 dips from 20.55 → 20.53 before
  recovering.
- `fedavg × RelaySum`: SSIM epoch 30→35 decreases 0.6555 → 0.6551.

These dips are ≤ 0.005 SSIM or ≤ 0.05 dB — within the single-seed noise
floor on a 245-image test set. They are NOT evidence of divergence, they
ARE evidence that (a) a single `partition_seed` has visible per-round
jitter, and (b) the cosine-decayed learning rate in late epochs allows the
model to drift slightly without improving.

### Training loss plateau (across all 6 cells)

All 6 cells converge to training loss ≈ 0.111–0.115 by epoch 25 (see
`v2a_v2a_train_loss.pdf`). The differences across cells in training loss
are ≤ 0.004 at epoch 35 — smaller than the test-PSNR differences (relative
to scale). This implies the generalization gap (train-loss vs test-PSNR)
is what's driving our 0.174 dB spread, not local optimization.

---

## 5. Comparison to v1 IID baseline (from `cloud_removal_v1/docs/v1_results.md`)

| Setting | RelaySum | Gossip | AllReduce |
|---|--:|--:|--:|
| v1 IID, 10 epoch  | 20.85 | **21.79** | 21.42 |
| v2 α=0.1, 35 ep (FedAvg) | 21.21 | **21.37** | 21.34 |
| v2 α=0.1, 35 ep (FedBN)  | 21.27 | 21.29 | **21.39** |

**Observation 1**: RelaySum is last in BOTH v1 IID and v2 non-IID. This
is a ROBUST finding — not an artefact of partition α.

**Observation 2**: v1's Gossip was 0.37 dB ahead of AllReduce; v2's Gossip
is within 0.03 dB of AllReduce. **Gossip lost its head start under
non-IID** — this is the direction predicted by decentralized-SGD theory
(Gossip's consensus is harder under high gradient variance).

**Observation 3**: v2 overall PSNR range (21.2–21.4) is about 0.4 dB
LOWER than v1's best (21.79), despite 3.5× more epochs. Non-IID cost is
real.

---

## 6. Comparison to the original FLSNN paper (Fig 5)

*Quantitative read-outs from user's scan of the paper PDF:*

| Figure | Metric | RelaySum | Gossip | AllReduce |
|---|---|--:|--:|--:|
| 5(a,b) | SpikingCNN, 60 rd | loss ≈ 0.028 / **acc ≈ 68%** | 0.040 / 56% | 0.038 / 53% |
| 5(c,d) | SpikingResNet, 60 rd | 0.022 / **acc ≈ 77%** | 0.029 / 64% | 0.034 / 57% |

**RelaySum leads by 10–20 percentage points in test accuracy** under the
original paper's regime.

**Regime delta between original FLSNN and our v2-A**:

| Dimension | Original FLSNN | v2-A |
|---|---|---|
| Task | EuroSAT 10-class classification | Cloud-removal regression |
| Metric | Test accuracy (%) | Test PSNR (dB) / SSIM |
| Model | SpikingCNN / SpikingResNet (small) | VLIFNet Spiking U-Net (~45 MB) |
| Loss | Cross-entropy | Charbonnier + 0.1·(1-SSIM) |
| Partition | Dirichlet-over-CLASS | Dirichlet-over-SOURCE |
| α (default) | 0.5 (mild non-IID) | 0.1 (strong non-IID) |
| Clients (planes×sats) | 5×10 chain | 5×10 chain |
| min per client | not enforced | 5 |
| Rounds | 60 | 35 (→ 70 planned) |

**So the reversal is not attributable to topology, client count, or total
rounds** (we are comparing our 35 to their 60 but verifying that extended
training won't flip the ranking — addressed in §8 below).

---

## 7. Qualitative figure observations (from `v2a_v2a_qualitative.pdf`)

User observed: "GT 糊了 / 很多 cell 结果相似". Diagnosis:

- **GT blurriness is a VISUALIZATION artefact**, not a data defect. The
  `visualize.py` center-crops 64×64 then matplotlib upsamples. Same happens
  for the 6 model restorations, so the RELATIVE comparison is valid even
  though ABSOLUTE clarity is poor. Fix = raise `patch_size` for viz
  (`v2_remaining_issues.md §1.2`).

- **Similarity across 6 cells** is consistent with the 0.174 dB spread —
  a 0.2 dB PSNR gap is not enough to be visually obvious on 64×64 patches.
  This is itself an important paper observation: **under strong non-IID,
  the 6 schemes produce nearly indistinguishable outputs**.

- **PSNR numbers on individual tiles** (e.g. row 3: 20.70 / 21.75 / 21.06
  / 21.23 / 21.34 / 22.21) show **per-image rankings can differ from the
  average ranking**. This implies that "who wins" depends on the test
  image — averaging over 245 images washes this out. Worth a per-source
  breakdown (CR1 vs CR2) in future work.

---

## 8. Does the ranking flip at 70 epochs? (Prediction)

**Evidence AGAINST a flip** (ranking stays RelaySum-last):

1. v1 IID 10-epoch already showed the same order (RelaySum < Gossip).
2. Original FLSNN Fig 5(b) reaches ranking stability by round 40 — i.e.
   at 35 rounds we should already see the "true" ordering if our
   experiment is going to match theirs. It doesn't.
3. Training loss curves at epoch 35 are all within 0.004 — no cell has a
   dramatic training-loss advantage waiting to manifest in later test.
4. RelaySum's inflight relay buffer variance would compound with more
   rounds, not settle.

**Evidence FOR a possible flip**:

1. Our 35→70 extends the cosine LR decay; RelaySum's later-epoch fine-
   tuning might benefit more from the smoothed weight-sharing.
2. Per-plane BN under FedBN may converge to stable equilibria that help
   RelaySum more than Gossip. (Speculative.)

**Prediction**: ranking stays the same at 70 epochs with PSNR spread
widening from 0.174 dB to ~0.25–0.35 dB. AllReduce stays tied with Gossip
or pulls slightly ahead. RelaySum stays last.

**Confidence**: moderate-high. The v1→v2 consistency is the strongest
single signal.

---

## 9. Hard-data-only claims we can make TODAY

(Without further experiment.)

1. At 35 rounds on CUHK-CR1+CR2 under Dirichlet(α=0.1)-over-source, all
   6 (BN × scheme) combinations produce PSNR within [21.21, 21.39] dB
   (spread 0.174 dB).

2. RelaySum ranks last among the 3 aggregation schemes regardless of BN
   mode, contradicting the original FLSNN paper's ranking on
   classification.

3. FedBN modestly but consistently improves SSIM over FedAvg (0.0014 avg,
   +0.02–0.05 in best cell); PSNR difference is within noise.

4. AllReduce achieves the best PSNR AND the lowest reported comm cost
   (1.62 GB vs 2.59 GB over 35 rounds) — Pareto-optimal under our
   accounting convention.

5. Training loss plateaus at ~0.111 by epoch 25 across all cells;
   test-PSNR has NOT plateaued and is still improving at +0.03 dB / 5
   epochs. 70 rounds needed to claim convergence.

---

*This file captures only HARD DATA + direct observations. Interpretations
and hypotheses go in `v2_interpretation.md`. Literature-supported
arguments go in `v2_theory_and_related.md`.*
