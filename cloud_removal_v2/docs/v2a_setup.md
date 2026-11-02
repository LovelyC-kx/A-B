# v2-A Setup — Path-A runbook

## 1. Prerequisites

Environment is inherited from v1 — nothing new to install:

| Component | Version | Notes |
|---|---|---|
| Python | 3.10 | same as v1 |
| PyTorch | 2.3.1 + cu121 | same as v1 |
| spikingjelly | 0.0.0.0.14 (PyPI) | v1's `_sj_compat.py` still applies |
| numpy | < 2 | v1 constraint |
| Pillow | ≥ 10 | |

If you already ran v1 on this machine, you are ready — just `git pull`.

## 2. Data

v2-A needs **BOTH** CUHK-CR1 and CUHK-CR2 on disk:

```
<your_data_root>/
    CUHK-CR1/
        train/input/  *.png
        train/target/ *.png
        test/input/   *.png
        test/target/  *.png
    CUHK-CR2/
        train/input/  *.png
        ...
```

Either:

* Pass `--data_root <path>` and the runner will auto-find both
  `<path>/CUHK-CR1` and `<path>/CUHK-CR2`; OR
* Pass `--source_root_1 <path-to-CR1>` and `--source_root_2 <path-to-CR2>`
  separately.

Total training pool: 534 + 448 = 982 paired images.
Total test pool: 134 + 111 = 245 paired images (served at full 512².

## 3. Sanity before the long run

```bash
# 3.1 Self-tests (~5 s)
python -m cloud_removal_v2.tests.run_all

# 3.2 Dataset probe — verifies both sources are discovered +
#     shows a per-client thin/thick mix table
python -m cloud_removal_v2.dataset /root/autodl-tmp/C-CUHK

# 3.3 3 × 3 × 3-epoch dryrun (~20 min on 4090) — 6 cells each 3 epoch
python -m cloud_removal_v2.run_smoke \
    --data_root /root/autodl-tmp/C-CUHK \
    --num_epoch 3 --num_planes 3 --sats_per_plane 3 \
    --run_name v2a_dryrun
```

At dryrun end, you should see train loss decreasing and PSNR ticking up
for every combination of (bn_mode, scheme).  If any cell diverges, fix
before the long run.

## 4. Full sweep (30 epoch × 50 sat × 6 cells ≈ 15 h on 4090)

**Use tmux** — SSH disconnects will otherwise kill the run.

```bash
cd /root/autodl-tmp/shiyaunmingFLSNN-main/Decentralized-Satellite-FL-dev-main
tmux new -s v2a
python -m cloud_removal_v2.run_smoke \
    --data_root /root/autodl-tmp/C-CUHK \
    --run_name v2a 2>&1 | tee Outputs_v2/v2a.log
# Ctrl-B D   to detach
# tmux attach -t v2a   to resume
```

Outputs land in `./Outputs_v2/`:

```
Outputs_v2/
    v2a_v2a_fedavg_Relaysum_Aggregation.npz          ← per-cell metrics
    v2a_v2a_fedavg_Gossip_Averaging.npz
    v2a_v2a_fedavg_AllReduce_Aggregation.npz
    v2a_v2a_fedbn_Relaysum_Aggregation.npz
    v2a_v2a_fedbn_Gossip_Averaging.npz
    v2a_v2a_fedbn_AllReduce_Aggregation.npz
    v2a_v2a_summary.json                             ← 6 final numbers
    ckpts/
        v2a_fedavg_{Relaysum,Gossip,AllReduce}_plane0.pt
        v2a_fedbn_{Relaysum,Gossip,AllReduce}_plane{0..4}.pt
    tb/v2a/                                          ← tensorboard events
```

Back up `Outputs_v2/` to `/root/autodl-tmp/` before powering down the
AutoDL instance.

## 5. Generate figures + qualitative grid (~10 min)

```bash
python -m cloud_removal_v2.plot_results --run_name v2a
python -m cloud_removal_v2.visualize --run_name v2a --n_samples 6
```

Products:

* `Outputs_v2/v2a_v2a_train_loss.pdf`
* `Outputs_v2/v2a_v2a_test_psnr.pdf`
* `Outputs_v2/v2a_v2a_test_ssim.pdf`
* `Outputs_v2/v2a_v2a_qualitative.pdf` — 6 rows × 8 cols grid

## 6. Sweep-subset mode

Need to rerun just one cell (for instance FedBN × RelaySum after a
config tweak)?

```bash
python -m cloud_removal_v2.run_smoke \
    --data_root /root/autodl-tmp/C-CUHK \
    --only_bn fedbn --only_scheme Relaysum_Aggregation \
    --run_name v2a
```

The `.npz` / `.pt` / summary files for that cell will be overwritten;
other cells' outputs are left intact.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: Could not find paired input/target folders under '…/CUHK-CR2'` | CR2 not on disk; either download it or switch back to v1 (CR1 only) via `python -m cloud_removal_v1.run_smoke`. |
| `Cannot enforce min_per_client` at partition time | `--partition_alpha 1.0` (less skewed) or lower `--num_planes` / `--sats_per_plane`. |
| `CUDA out of memory` at epoch 1 | `--train_batch_size 2 --patch_size 48`. |
| tmux session dies | check AutoDL instance status; if the container auto-restarted, tmux is gone but outputs already written to disk are safe. |
| Per-epoch wall time > 20 min | spikingjelly compat shim not active; make sure `cloud_removal_v1/models/_sj_compat.py` exists and vlifnet.py imports it. |

## 8. Expected final numbers (hypothesis, will be overwritten by the actual run)

| Cell | PSNR (dB) | SSIM | Notes |
|---|---|---|---|
| FedAvg × RelaySum | ≥ 22.5 | ≥ 0.70 | non-IID brings RelaySum above v1's 20.85 dB |
| FedAvg × Gossip   | ~ 22.0 | ~ 0.69 | Gossip's v1 lead should vanish or invert under non-IID |
| FedAvg × All-Reduce | ~ 22.0 | ~ 0.69 | |
| FedBN × RelaySum  | **≥ 23.0** | **≥ 0.72** | best cell; new contribution |
| FedBN × Gossip    | ~ 22.3 | ~ 0.70 | |
| FedBN × All-Reduce | ~ 22.3 | ~ 0.70 | |
