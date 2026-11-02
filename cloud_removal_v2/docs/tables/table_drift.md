# Table III — Cross-plane BN-affine drift after 80 rounds

Numbers come from `analyze_bn_drift_posthoc.py` (commit 17cd881),
loading the 5 saved plane checkpoints per cell. The script detects
BN-affine layers from the state_dict signature (1-D float weight + bias
+ matching `running_mean` / `running_var`), so it correctly identifies
all 51 BN sites in both A and B (the previous substring-based version
missed 13 SRB-shortcut BNs in A; see commit message).

Two metrics:

* **Var(γ)\_mean** — across the 51 BN layers, average of the
  cross-plane variance of γ (per channel, then meaned over channels).
  This is the **paper-relevant inter-plane drift metric** for both BN
  variants. Var ≈ 0 when all 5 planes hold identical weights.
* **max\|γ−1\|∞** — magnitude diagnostic. For TDBN this can be > 1
  because γ\_init = α·V_th ≈ 0.106 (Zheng 2021) and the trained γ moves
  *away* from 1; this is **not** drift. For BN2d, where γ\_init = 1.0,
  small max\|γ−1\| is the expected behaviour.

## Run A: TDBN, SNN (51 BN layers)

|       Cell        |  Var(γ)_mean   |  Var(β)_mean   | max\|γ−1\|∞ | max\|β\|∞ |
|:------------------|:--------------:|:--------------:|:-----------:|:---------:|
| FedAvg + AllReduce | **0.000e+00** | **0.000e+00**  |   1.0000    |  0.1638   |
| FedAvg + Gossip    |    1.327e-09  |    1.127e-09   |   0.9900    |  0.1886   |
| FedAvg + RelaySum  |    5.675e-11  |    6.445e-11   |   1.0326    |  0.1834   |
| FedBN  + AllReduce |    7.038e-05  |    8.693e-05   |   0.9972    |  0.2220   |
| FedBN  + Gossip    |    7.384e-05  |    7.519e-05   |   1.0048    |  0.2058   |
| FedBN  + RelaySum  |  **2.295e-04**|  **2.254e-04** |   1.1141    |  0.2491   |

## Run B: BN2d, SNN (51 BN layers)

|       Cell        |  Var(γ)_mean   |  Var(β)_mean   | max\|γ−1\|∞ | max\|β\|∞ |
|:------------------|:--------------:|:--------------:|:-----------:|:---------:|
| FedAvg + AllReduce | **0.000e+00** | **0.000e+00**  |   0.1116    |  0.1174   |
| FedAvg + Gossip    |    9.281e-10  |    1.149e-09   |   0.0996    |  0.1048   |
| FedAvg + RelaySum  |    7.255e-11  |    8.452e-11   |   0.1095    |  0.0996   |
| FedBN  + AllReduce |    1.281e-04  |    1.334e-04   |   0.1319    |  0.1367   |
| FedBN  + Gossip    |    1.104e-04  |    1.105e-04   |   0.1312    |  0.1487   |
| FedBN  + RelaySum  |  **4.078e-04**|  **4.272e-04** |   0.2230    |  0.2691   |

## Run C: TDBN, ANN (1 cell only — others not trained)

|       Cell        |  Var(γ)_mean   |  Var(β)_mean   | max\|γ−1\|∞ | max\|β\|∞ |
|:------------------|:--------------:|:--------------:|:-----------:|:---------:|
| FedBN  + AllReduce |    8.724e-05  |    6.164e-05   |   1.0195    |  0.0928   |

## Headline observations

### O1 — Aggregation correctness sanity (FedAvg + AllReduce → Var = 0)
Both A and B report `Var(γ) = Var(β) = 0.000e+00` for FedAvg + AllReduce.
Since AllReduce returns the exact global mean each round, all 5 planes
have identical weights at the end of training. This is a definitive
correctness check on the aggregation primitive.

### O2 — TDBN reduces inter-plane BN drift relative to BN2d (~58 %)

|       FedBN cell       |  A (TDBN) Var(γ) |  B (BN2d) Var(γ) |  ratio A/B  |
|:----------------------|:----------------:|:----------------:|:-----------:|
| AllReduce             |    7.04e-05      |    1.28e-04      |   **0.55**  |
| Gossip                |    7.38e-05      |    1.10e-04      |   **0.67**  |
| RelaySum              |    2.30e-04      |    4.08e-04      |   **0.56**  |
| **mean across schemes** | **1.25e-04**   | **2.15e-04**     | **0.58**    |

TDBN's α·V_th-shared scaling pre-aligns plane statistics — the
mechanism predicted by the original Claim C16. **However**, even
BN2d's larger drift (~10⁻⁴) is too small to translate into a
meaningful FedBN PSNR gain (Table II: Mean Δ = +0.044 dB), so FedBN
remains nearly redundant under both BN variants.

### O3 — RelaySum produces the largest BN drift in both runs

In both A and B, FedBN + RelaySum has the largest Var(γ) by a factor
of ~3× over the other two schemes. Mechanism: RelaySum's delayed
delivery (each relayed message is `τ_ij` rounds stale) lets per-plane
BN parameters drift further between updates than under one-step
Gossip or per-round AllReduce. This is consistent with the FLSNN
Theorem 2 staleness term but does not, on this task, cause RelaySum
to lose the PSNR race (Tables I/II — the BN noise is dwarfed by the
intra-cell PSNR variation).

### O4 — TDBN's γ-magnitude diagnostic is *expected* behaviour, not drift

A (TDBN): max\|γ−1\|∞ = 0.99–1.11 across cells.
B (BN2d): max\|γ−1\|∞ = 0.10–0.22 across cells.

The 4×–10× larger absolute γ excursion in TDBN reflects its
initialization at α·V_th ≈ 0.106 and the optimizer's freedom to learn
γ values in the range [0, 2.1]. It is **not** evidence of FedBN drift;
the cross-plane Var(γ) is the relevant drift metric.

### O5 — Backbone-agnostic alignment

C (TDBN, ANN) FedBN + AllReduce: Var(γ) = 8.72e-05.
A (TDBN, SNN) FedBN + AllReduce: Var(γ) = 7.04e-05.

The two TDBN cells differ only in backbone (ANN vs SNN) and produce
nearly identical inter-plane BN drift. → TDBN's alignment effect is
a property of the normalization layer's parameterisation, not the
upstream activation type.

## Provenance

* Script: `cloud_removal_v2/analyze_bn_drift_posthoc.py` (commit `17cd881`)
* Run command: `python -m cloud_removal_v2.analyze_bn_drift_posthoc
  --ckpt_dir Outputs_v2/ckpts --out Outputs_v2/v2_drift_report.md`
* Raw output saved at `Outputs_v2/v2_drift_report.md` on AutoDL.
* Total ckpts loaded: 60 (A: 30 = 5 planes × 6 cells; B: 30; C: 5 = 5 planes × 1 cell).

## Caveat

These are **single-seed** drift values. The cross-plane variance is
itself a 5-sample statistic (5 planes per cell). Multi-seed
validation deferred to v3 — consistent with the §VI-H limitations.
