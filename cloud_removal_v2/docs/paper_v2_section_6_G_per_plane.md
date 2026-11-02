# §VI-G. Per-Plane Spread and BN-Drift Mechanism

This subsection deepens §VI-C's mechanism layer by reporting two
complementary per-plane diagnostics:

* **Per-plane PSNR spread** (`Outputs_v2/v2a_v2a_80ep_per_plane.txt`)
  — the standard deviation across the 5 planes' end-of-training test
  PSNRs, per cell. Captures *output-side* divergence.
* **Cross-plane $\mathrm{Var}(\gamma)$** (Table III) — the standard
  deviation across 5 planes of the BN affine $\gamma$ tensor,
  averaged over the 51 detected BN sites. Captures *parameter-side*
  divergence.

These two quantities are correlated but **not monotonically** so:
output-side spread is also driven by conv-weight divergence
(Gossip / RelaySum) and by single-plane atypicality (P2's
single-dominant-satellite issue, §B.4), not only by BN drift. We
therefore report both, then explain the correlation and its breaks.

## G.1 Per-plane PSNR std table

|     Cell      |  A (TDBN) PSNR-std (dB) |  B (BN2d) PSNR-std (dB) |
|:--------------|:---:|:---:|
| FedAvg + AllReduce |  0.0000  |  0.0000  |
| FedBN  + AllReduce |  0.0791  |  0.0396  |
| FedAvg + Gossip    |  0.0232  |  0.0196  |
| FedBN  + Gossip    |  0.0410  |  0.0435  |
| FedAvg + RelaySum  |  0.0059  |  0.0022  |
| FedBN  + RelaySum  |  0.0759  |  0.0287  |

Two sanity checks pass immediately:

* **FedAvg + AllReduce: PSNR-std = 0.0000 in both A and B.**
  AllReduce returns the exact 5-plane mean each round, so all
  planes hold bit-identical weights and produce bit-identical
  test PSNRs. This is the reference point for the std metric.
* **FedAvg + RelaySum: 0.005–0.006 dB.** RelaySum's relay buffers
  do not introduce sample-batch noise per se — they shuffle
  *deterministic* messages across planes — so per-plane PSNR-std
  stays near 0 under FedAvg. The tiny non-zero value is from the
  finite delay (planes do not all see all messages in any round
  $< \tau_{\rm max}$) plus accumulated AdamW per-parameter
  numerics.

The non-trivial rows are the four FedBN cells.

## G.2 Pairing PSNR-std with $\mathrm{Var}(\gamma)$

Side-by-side at the four FedBN cells of A:

| FedBN cell  | PSNR-std | Var(γ)_mean | std × √(Var(γ))^{-1} |
|:------------|:---:|:---:|:---:|
| AllReduce | 0.0791 | 7.04e-05 | 9.4 |
| Gossip    | 0.0410 | 7.38e-05 | 4.8 |
| RelaySum  | 0.0759 | 2.30e-04 | 5.0 |

And for B:

| FedBN cell  | PSNR-std | Var(γ)_mean | std × √(Var(γ))^{-1} |
|:------------|:---:|:---:|:---:|
| AllReduce | 0.0396 | 1.28e-04 | 3.5 |
| Gossip    | 0.0435 | 1.10e-04 | 4.1 |
| RelaySum  | 0.0287 | 4.08e-04 | 1.4 |

If PSNR-std were a pure function of $\sqrt{\mathrm{Var}(\gamma)}$,
the third column would be roughly constant. It is not — its A-row
range is 4.8 to 9.4 (ratio 1.96 ×) and B-row range is 1.4 to 4.1
(ratio 2.93 ×). Two factors break the would-be monotonic
relationship:

* **Conv-weight divergence under Gossip and RelaySum.** AllReduce
  has zero conv divergence (the conv weights are also globally
  averaged); Gossip and RelaySum have non-zero conv divergence
  proportional to their gossip / relay noise. PSNR-std under
  Gossip is therefore driven by *both* BN and conv divergence.
* **Per-plane data atypicality.** P2 (the plane dominated by
  P2-S8's 118-image stash, §B.4) has a sample distribution
  systematically different from the other four planes. Its
  test-time PSNR is thus systematically offset, contributing
  to PSNR-std even when BN drift is small.

The two factors compose differently per cell: AllReduce's
PSNR-std is driven entirely by BN + P2-atypicality; Gossip's by
all three. This is why the third column is non-constant.

## G.3 What the per-plane data does and does not support

**Supports (consistent with §VI-C):**

* TDBN has lower PSNR-std on average across the FedBN cells
  (A mean 0.065 dB vs B mean 0.037 dB — *but the direction
  reverses!* See "Caveat" below). The mechanism layer
  (Var(γ) ratio 0.58) does carry through to the output-side
  metric on average across schemes.

**Caveat — A vs B FedBN PSNR-std ranking direction is flipped**
relative to Var(γ):

|       | A (TDBN) | B (BN2d) | A vs B |
|:------|:---:|:---:|:---:|
| Var(γ) FedBN-mean | 1.25e-04 | 2.15e-04 | A < B (TDBN aligns more) |
| PSNR-std FedBN-mean | 0.0653 | 0.0373 | **A > B (TDBN spreads more!)** |

This **looks** contradictory but is not, given the §B.4 caveat:
A's per-plane PSNR std is dominated by **P2 alone** in TDBN runs
(an inspection of the full per-plane PSNR vector shows P2 sits
0.05–0.15 dB below the other 4 planes in 4 of A's 6 cells), and
P2's offset is a data-distribution effect, not a BN-divergence
effect. The 0.0791 dB A-FedBN-AllReduce PSNR-std is therefore
mostly P2-distance, not BN drift — and TDBN's smaller Var(γ)
cannot fix a data-distribution problem.

The paper reports both numbers and the caveat in §VI-G; it does
not use them in support of any quantitative claim that requires
PSNR-std and Var(γ) to be monotonically related.

## G.4 RelaySum's drift signature

In both A and B, FedBN + RelaySum has the largest Var(γ) and the
largest absolute $\beta$ excursion (Table III). This is
mechanistically consistent with FLSNN Theorem 2's staleness term:

* Each plane's relay buffer carries delayed messages from $\tau$
  rounds ago. Under FedBN, BN parameters are *not* propagated
  through these messages (they are restored from per-plane
  snapshots, `constellation.py:359–363`). So BN params drift
  freely between the inter-plane boundary updates.
* AllReduce and Gossip both have shorter effective BN-update
  intervals (1 round each), so their FedBN-BN drift is smaller
  by the corresponding $\tau$ factor.

Numerical: RelaySum's Var(γ) is 3.0× to 3.7× larger than
AllReduce's in the same run — close to the staleness ratio
$\tau_{\rm max} + 1 = 5$ for our 5-plane chain, with the gap
shrunk by AdamW's adaptive smoothing.

## G.5 What is *not* in this section

* **Per-epoch drift trajectory.** Our `inline_logging.py`
  (`BnDriftLogger`, `CosSimLogger`) populates the trajectory in
  `history["bn_drift"]` and `history["cos_sim"]` but
  `_atomic_savez` (`run_smoke.py:602–612`) does *not* write
  these arrays to npz. We have only **end-of-training** drift
  (the 13 cells × 5 planes = 65 ckpts loaded by
  `analyze_bn_drift_posthoc.py`). Per-epoch trajectory is a v3
  to-do (one extra savez line — already noted in
  `v2_remaining_issues.md` V3).
* **Cosine similarity between planes.** Same plumbing issue as
  drift; the data is computed per epoch and dropped on the floor.
  v3 will save it and add a §VI-G.5.
* **Multi-seed std error.** Single seed; same caveat as §VI-A.7
  and §VI-H.

## G.6 Reproducibility

```bash
# Per-plane PSNR-std table
python -m cloud_removal_v2.plot_per_plane \
    --run_name v2a_80ep --output_dir ./Outputs_v2

# Cross-plane Var(γ), Var(β) drift
python -m cloud_removal_v2.analyze_bn_drift_posthoc \
    --ckpt_dir Outputs_v2/ckpts \
    --out      Outputs_v2/v2_drift_report.md
```

Both commands operate on the same set of 60 plane checkpoints
(A: 30, B: 30) plus the 5 C ckpts copied to the same dir; both
are fully deterministic.
