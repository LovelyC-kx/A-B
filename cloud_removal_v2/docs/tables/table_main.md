# Table I — Main results on CUHK-CR (80 epochs)

**Bolded** values mark the best within each (Run, BN) sub-block:
PSNR is "higher is better"; Comm is "lower is better"; Wall is reported
for context (lower is better but it is not a paper claim — A/B differ
only by `bn_variant`, A/C only by `backbone`, and runs were sequential
on a shared 4090).

|  Run  |  BN  | Backbone | Aggregation Scheme |  PSNR (dB)  |   SSIM   | Comm (MB) | Wall (h) |
|:-----:|:----:|:--------:|:------------------:|:-----------:|:--------:|:---------:|:--------:|
|   A   | TDBN |   SNN    | FedAvg + AllReduce |  **21.642** |  0.6636  |  **3694** |  5.87    |
|   A   | TDBN |   SNN    | FedAvg + Gossip    |    21.345   |  0.6598  |    5911   |  5.80    |
|   A   | TDBN |   SNN    | FedAvg + RelaySum  |    21.500   |  0.6628  |    5911   |  6.12    |
|   A   | TDBN |   SNN    | FedBN  + AllReduce |    21.420   |  0.6589  |  **3694** |  6.23    |
|   A   | TDBN |   SNN    | FedBN  + Gossip    |    21.531   |**0.6651**|    5911   |  6.20    |
|   A   | TDBN |   SNN    | FedBN  + RelaySum  |    21.561   |  0.6634  |    5911   |  6.09    |
|   B   | BN2d |   SNN    | FedAvg + AllReduce |    21.630   |  0.6672  |  **3694** |  5.78    |
|   B   | BN2d |   SNN    | FedAvg + Gossip    |    21.781   |**0.6686**|    5911   |  5.61    |
|   B   | BN2d |   SNN    | FedAvg + RelaySum  |    21.709   |  0.6674  |    5911   |  6.16    |
|   B   | BN2d |   SNN    | FedBN  + AllReduce |    21.762   |  0.6681  |  **3694** |  6.44    |
|   B   | BN2d |   SNN    | FedBN  + Gossip    |  **21.791** |  0.6682  |    5911   |  6.37    |
|   B   | BN2d |   SNN    | FedBN  + RelaySum  |    21.699   |  0.6672  |    5911   |  6.30    |
|   C   | TDBN | **ANN**  | FedBN  + AllReduce |  **22.171** |**0.6855**|  **3694** |**3.86**  |

## Headline numbers

* **Best PSNR overall:** `C` = **22.171 dB** (TDBN + ANN backbone) — beats best SNN cell by **+0.380 dB**.
* **Best PSNR among SNN runs:** `B` FedBN + Gossip = **21.791 dB**.
* **Best (PSNR, Comm) Pareto cell among SNN runs:** `B` FedBN + AllReduce = **21.762 dB @ 3694 MB**
  (within 0.029 dB of the SNN best, 60% less communication).
* **ANN speedup over SNN at the same cell:** `A` FedBN + AllReduce = 6.23 h vs `C` = 3.86 h → **1.61×**.
  This understates the architectural ANN/SNN gap because the ANN backbone still runs `T = 4`
  forward passes through the (otherwise identical) U-Net (see §VI-E).

## Provenance

All 13 rows are loaded directly from the runs' `summary.json["final"]`:

* **A** (TDBN, SNN, 6 cells):
  `Outputs_v2/v2a_v2a_80ep_summary.json`
* **B** (BN2d, SNN, 6 cells):
  `Outputs_v2/v2a_v2a_80ep_stdbn_summary.json`
* **C** (TDBN, ANN, 1 cell — only `fedbn × AllReduce` was run):
  `_quanxin/Outputs_v2/v2a_v2a_80ep_ann_fedbnar_summary.json`

`Comm (MB)` = `total_comm_bytes / 1e6`, rounded to integer.
`Wall (h)`  = `total_wall_seconds / 3600`, rounded to 2 decimals.

## Configuration parity check

All three runs share these hyperparameters (verified by comparing the
three `summary.json["config"]` dicts):

```
seed=1234, partition_seed=0, partition_mode=dirichlet_source, partition_alpha=0.1
num_planes=5, sats_per_plane=10  →  50 satellites
T=4, vlif_dim=24, en_blocks=[2,2,4,4], de_blocks=[2,2,2,2]
patch_size=64, train_batch_size=4, augment=True
lr=1e-3, min_lr=1e-7, warmup_epochs=3 (cosine), num_epoch=80
intra_plane_iters=2, local_iters=2
loss = Charbonnier(eps=1e-3) + 0.1 · (1 − SSIM)
eval_mode=center_patch, eval_patch_size=64, eval_every=5
```

Differences:

* **A vs B:** `bn_variant` only (`tdbn` ↔ `bn2d`). All other parameters identical → any
  PSNR / SSIM / Comm / Wall difference is causally attributable to the BN choice.
* **A vs C:** `backbone` only (`snn` ↔ `ann`). Same → any difference is causally
  attributable to the backbone choice.
* **B vs C:** **two parameters differ** (BN + backbone) → not a clean comparison;
  the paper avoids any direct B vs C claim.
