# Table IV — ANN vs SNN backbone (matched FedBN + AllReduce)

Pairwise comparison at the only cell that both A (SNN) and C (ANN)
share — `fedbn × AllReduce`. All other hyperparameters (BN variant,
data partition, optimizer, loss, schedule) are bit-identical between
the two runs (verified via the two `summary.json["config"]` dicts).

|       Metric              |  A (TDBN, SNN)  |  C (TDBN, ANN)  |  Δ = C − A  | Notes |
|:--------------------------|:---------------:|:---------------:|:-----------:|:-----:|
| PSNR (dB)                 |     21.420      |     22.171      |   **+0.751**| C wins |
| SSIM                      |     0.6589      |     0.6855      |   **+0.0266**| C wins |
| Wall (h)                  |     6.23        |     3.86        |   **−2.37** | C is 1.61× faster |
| Comm (MB) — total over 80 rounds | 3694    |     3694        |       0     | identical (same architecture, same #rounds) |
| BN-drift Var(γ)_mean      |   7.04e-05      |   8.72e-05      | +1.68e-05   | within 24 % — TDBN alignment is backbone-agnostic |

## Headline observations

### O1 — On a single GPU, ANN wins on every paper-relevant axis
ANN PSNR is **+0.75 dB** higher (3 % relative), SSIM is +0.027
higher, and the same training schedule completes in **62 % of the
SNN's wall time**. The ANN advantage is not subtle: it is larger than
the entire spread between A's six aggregation × BN cells (which
covers 21.34–21.64 dB).

### O2 — But: 1.61× speedup understates the *architectural* ANN/SNN gap

Our ANN backbone is implemented by replacing every LIF neuron with
`nn.ReLU(inplace=False)` (`vlifnet.py:153–167, 186–193`); the model
keeps the original `T = 4` outer time loop. So the ANN backbone runs
the U-Net **4 times** per forward pass even though the 4 copies are
deterministic functions of the same input — wasting ~75 % of its
conv FLOPs on redundant work. A T = 1 ANN re-implementation would
recover an additional ~2.5× speedup. The measured 1.61× should
therefore be read as **a lower bound** on the true GPU-side ANN
advantage, not a tight estimate.

(Section VI-E reports both: the as-measured 1.61× and the projected
~4× upper bound for a T = 1 ANN.)

### O3 — Headline numbers we are NOT entitled to claim

* **"ANN-FedBN beats SNN-FedAvg" type cross-cell claims.** We have
  not run ANN with FedAvg, nor any SNN cell at C's `fedbn × AllReduce`
  combination with a different backbone. The only fair ANN-vs-SNN
  comparison this dataset permits is the one row pair above.
* **Energy efficiency.** GPU wall time is **not** an energy proxy.
  The eventual SNN-energy story (§VI-E) requires either a separate
  spike-rate measurement script (G1 in inventory) or a 45-nm-CMOS
  estimate via Horowitz 2014's MAC/AC table — neither has been
  produced yet. Until then we report energy as "not measured" and
  note the 5-level `MultiSpike4` quantization as an additional
  conservatism (the original FLSNN Section VI-B's 0.9 pJ/AC formula
  assumes binary spikes; we will report a more conservative bound).

## Provenance

* **A row:** `Outputs_v2/v2a_v2a_80ep_summary.json["final"]["fedbn_AllReduce_Aggregation"]`
* **C row:** `_quanxin/Outputs_v2/v2a_v2a_80ep_ann_fedbnar_summary.json["final"]["fedbn_AllReduce_Aggregation"]`
* **BN-drift row:** Table III's drift report.

Configuration parity: A and C identical except `backbone` (`"snn"`
vs `"ann"`). All 16 other config fields verified bit-equal.

## Why no B row in this table

Run B (BN2d, SNN) differs from C (TDBN, ANN) in two variables
simultaneously (`bn_variant` + `backbone`). Inserting a B vs C row
would conflate two effects, so the paper does not make any direct
B vs C claim.
