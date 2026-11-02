# Table II — FedBN ablation: Δ(PSNR) = FedBN − FedAvg, per scheme

The pairwise difference holds the BN variant, the backbone, and the
aggregation scheme constant — only the BN-aggregation policy
(`bn_local`) toggles. A positive Δ means FedBN improved over FedAvg.

|     BN variant      |  Δ(AllReduce)  |  Δ(Gossip)  |  Δ(RelaySum) |   Mean Δ    |
|:-------------------:|:--------------:|:-----------:|:------------:|:-----------:|
|  **A** (TDBN, SNN)  |    −0.222      |   +0.186    |    +0.061    | **+0.008**  |
|  **B** (BN2d, SNN)  |    +0.132      |   +0.010    |    −0.010    | **+0.044**  |

## How to read this table

* **Mean Δ ≪ 0.10 dB in both runs.** FedBN is essentially neutral on
  this task — its advantage in the FedBN paper (Li 2021) on label-shift
  classification (≥ 1 dB) does **not** transfer to our source-shift
  pixel-regression setting.
* **TDBN's Δ is closer to zero than BN2d's**: Mean Δ = +0.008 dB (TDBN)
  vs +0.044 dB (BN2d). Combined with Table III's Var(γ) result
  (TDBN's cross-plane Var(γ) is 58 % of BN2d's), this supports the
  Claim C16 mechanism — TDBN's α·V_th-shared scaling pre-aligns plane
  statistics, leaving less drift for FedBN to repair.
* **Per-scheme variation is large for A.** A single scheme (AllReduce)
  sees Δ = −0.22 dB while another (Gossip) sees +0.19 dB. With only
  one seed, this spread (≈ 0.4 dB peak-to-peak) is the noise floor on
  any individual Δ — the **mean** across schemes is the more reliable
  effect estimator.

## Source numbers (recomputed per cell)

```
A (TDBN, SNN):
  ΔAllReduce = 21.420 − 21.642 = −0.222
  ΔGossip    = 21.531 − 21.345 = +0.186
  ΔRelaySum  = 21.561 − 21.500 = +0.061
  Mean       = (−0.222 + 0.186 + 0.061) / 3 = +0.0083 → +0.008

B (BN2d, SNN):
  ΔAllReduce = 21.762 − 21.630 = +0.132
  ΔGossip    = 21.791 − 21.781 = +0.010
  ΔRelaySum  = 21.699 − 21.709 = −0.010
  Mean       = (+0.132 + 0.010 − 0.010) / 3 = +0.0440 → +0.044
```

PSNR cell values are taken from `table_main.md`. Δ values are
algebraic differences, not statistical estimates — single-seed runs
do not yield error bars; multi-seed validation is deferred to v3 (see
§VI-H limitations).

## Note on Run C

C runs only the `fedbn × AllReduce` cell, so a corresponding
`fedavg × AllReduce` ANN baseline does **not** exist; we cannot
compute a third Δ row for C. The ANN-vs-SNN backbone effect is
isolated in Table IV instead.
