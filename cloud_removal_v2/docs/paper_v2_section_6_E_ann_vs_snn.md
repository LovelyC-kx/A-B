# §VI-E. ANN vs SNN Backbone, Pareto, and Energy Disclosure

This subsection isolates the **backbone effect** (LIF + MultiSpike-4
SNN versus plain ReLU ANN) at the only matched cell where both runs
exist — `FedBN + AllReduce`. It then connects the result to the
communication-PSNR Pareto picture from §VI-D and disentangles
"GPU wall time" (an architecture proxy) from "energy" (a deployment
proxy) — the latter we honestly report as **not measured in v2**,
with a recipe and conservative bounds for v3.

## E.1 Single-cell ANN vs SNN comparison

Reading Table IV (`docs/tables/table_ann_vs_snn.md`):

| Metric | A: TDBN, SNN | C: TDBN, ANN | $\Delta$ = C − A |
|:-------|:---:|:---:|:---:|
| PSNR (dB)  | 21.420  | **22.171** | **+0.751** |
| SSIM       | 0.6589  | **0.6855** | **+0.0266** |
| Wall (h)   | 6.23    | **3.86**   | **−2.37** (1.61 × faster) |
| Comm (MB)  | 3694    | 3694       | 0 (identical) |
| Var(γ)     | 7.04e-05 | 8.72e-05  | +1.68e-05 (within 24 %) |

**Headline:** the ANN backbone is unambiguously better on every
paper-relevant axis on a single GPU — higher PSNR, higher SSIM,
1.61× faster, identical communication. The 0.75-dB PSNR gap is
larger than the entire spread of A's six (BN, scheme) cells
(21.34–21.64 dB). On a per-image SSIM basis, C beats every single
A cell by at least 0.017.

This is **directionally consistent with FLSNN Fig. 6** (their ANN
beats SNN by ~2 % accuracy on EuroSAT classification). The
magnitude differs (+0.75 dB regression vs +2 pp classification) but
the sign is identical.

## E.2 Why the 1.61× wall speedup understates the architectural gap

Our ANN backbone (`vlifnet.py:153–167, 186–193`) replaces every
LIF neuron with `nn.ReLU(inplace=False)` *but keeps the original
$T = 4$ outer time loop*. The U-Net is therefore run **four times
per forward pass** even though all four copies are deterministic
functions of the same input (`vlifnet.py:563`:
`inp_img.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)`). The four
identical activations are then averaged at the readout
(`vlifnet.py:588`: `out.mean(0)`).

This wastes ~75 % of the conv FLOPs on redundant work. A
`T = 1` ANN re-implementation — where the time-loop is collapsed
and the batch-norm sees the natural `[B, C, H, W]` instead of `[T*B, C, H, W]`
shape — would recover the missing speedup. We estimate it at
**~2.5×** additional speedup, giving a projected total ANN/SNN
wall ratio of ~4× on this hardware. We stress this is **a
projection**, not a measurement.

The measured 1.61× is therefore a **lower bound** on the
architectural ANN/SNN gap. The paper reports both values:

* **Measured:** 1.61× (Table IV, exact, single seed).
* **Projected ceiling:** ~4× (T = 1 re-implementation, requires
  refactor; deferred to v3).

## E.3 Pareto frontier — combining §VI-D + Run C

Mapping the 13 cells of Table I onto the (Comm, PSNR) plane:

| Cell                         | Comm (MB) | PSNR (dB) | On Pareto? |
|:-----------------------------|:---:|:---:|:---:|
| C: ANN-FedBN-AllReduce       | 3694 | **22.171** | ✓ (sole leader) |
| B: BN2d-FedBN-Gossip         | 5911 | 21.791    | ✗ (dominated by C) |
| B: BN2d-FedBN-AllReduce      | 3694 | 21.762    | ✓ (best SNN at low Comm) |
| ... others ...               | ...  | ...       | ✗ (all dominated by C or B-FedBN-AllReduce) |

The (Comm, PSNR) Pareto frontier — using **only the runs we have**
— is therefore:

* **AllReduce + ANN backbone** (Run C) — `(3694 MB, 22.171 dB)`,
  unique global Pareto leader.
* **AllReduce + BN2d + SNN** (Run B's `fedbn_AllReduce`) —
  `(3694 MB, 21.762 dB)`, the SNN-only Pareto leader.

Every other cell is strictly dominated. Notably **no Gossip /
RelaySum cell is on the Pareto frontier**; their extra 60 %
communication does not buy them PSNR that AllReduce cannot match.

## E.4 Energy: what we know, what we do not, what we will do

### E.4.1 What is published in the FLSNN energy story

FLSNN Section VI-B prices each MAC operation at **4.6 pJ** (ANN,
floating-point multiply-accumulate per Horowitz 2014) and each AC
operation at **0.9 pJ** (SNN, accumulate-only — multiplication is
trivial when the spike is binary $\{0, 1\}$). Multiplied by the
operation count per layer and the *measured* spike rate per layer
(via `model.named_modules()` hooks on their `LIFLayer` class —
see `energy_estimation.py:17–63`), they obtain a per-network
energy estimate that is ~10× lower for SNN than ANN on EuroSAT
classification.

### E.4.2 Why we cannot simply re-use that pipeline

* Their `LIFLayer` is in `Spiking_Models/` and exposes
  `module.avg_spike_rate` after each forward pass. Our model uses
  **both** `spikingjelly.LIFNode` (via `_make_lif_or_relu`,
  `vlifnet.py:170–183`, for the `lif_1` / `lif_2` slots in each
  Spiking_Residual_Block and SUNet_Level1_Block) **and** an
  independent soft-reset LIF implementation called `mem_update`
  (`vlifnet.py:225–246`, used by PixelShuffleLIFBlock and the
  SUNet-level `lif_node`). `mem_update` is **not** a wrapper around
  `LIFNode` — it inlines its own leaky-integration loop with a
  `MultiSpike4` quantized output. Neither class exposes a per-call
  spike rate; a custom forward hook would have to be written for
  both.
* Their SNN is binary; ours is **MultiSpike-4** (5-level outputs
  in $\{0, \tfrac{1}{4}, \tfrac{1}{2}, \tfrac{3}{4}, 1\}$). The
  0.9 pJ/AC formula prices a binary-spike accumulate; a 5-level
  spike requires either (a) a 2-bit scaled accumulate (loose
  bound: still 0.9 pJ/AC, treating the spike as a pre-quantized
  multiplier) or (b) a small lookup-multiply (loose bound: closer
  to 4.6 pJ/MAC, treating it as ANN).
* Their layer-by-layer FLOP table is hardcoded for a small
  EuroSAT ResNet (~200 K params); our VLIFNet is 2.31 M params
  with U-Net structure. FLOPs would have to be recomputed.

### E.4.3 What we report in v2 (and defer to v3)

**v2 reports:**

* Total wall time (Table I).
* The architectural ANN advantage on a GPU (Table IV).
* The structural identification of the SNN's role: *low-power
  inference at deployment time*, not training-time speed.

**v2 does not report a numeric ANN-vs-SNN energy ratio.** A
single binary-spike-bound number (e.g. "SNN is 5× more energy
efficient than ANN") would be misleading because the
MultiSpike-4 conservatism could shift the bound by 5×. We deem
this an unacceptable claim density on a single seed without a
spike-rate measurement.

**v3 will deliver:**

* `cloud_removal_v2/energy_estimation.py` — a from-scratch
  measurement script that hooks every `mem_update` instance,
  records per-layer non-zero rates and 5-level histograms, and
  computes the energy under both bounds (binary at 0.9 pJ/AC;
  conservative at 4.6 pJ/MAC).
* The corresponding row in a v3 Table V with both bounds reported
  side-by-side.

## E.5 Summary of E.1–E.4

* On a single 4090 GPU, the ANN backbone wins on PSNR (+0.75 dB),
  SSIM (+0.027), wall time (1.61× faster), at identical
  communication. The wall-time gap is a lower bound; a T = 1 ANN
  would push it to ~4×.
* The ANN-FedBN-AllReduce cell is the unique Pareto-frontier
  point on (Comm, PSNR). The best SNN cell sits on the Pareto
  frontier only at low communication.
* The FLSNN-style "SNN is much more energy efficient" claim
  **requires spike-rate measurement on our model and conservative
  treatment of MultiSpike-4 quantization**. We disclose that we
  have not performed this measurement in v2 and defer it to v3.

## E.6 Reproducibility

* PSNR / Wall / Comm: as in Tables I and IV (saved
  `summary.json["final"]` dicts).
* Pareto plot: `python -m cloud_removal_v2.plot_comm_efficiency
  --run_name v2a_80ep --output_dir ./Outputs_v2` produces
  `v2a_v2a_80ep_comm_efficiency.pdf`.
* Energy: not reproducible in v2 (no measurement script yet).
