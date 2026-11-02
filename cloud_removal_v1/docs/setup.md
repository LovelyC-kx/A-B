# Setup — cloud_removal_v1

## 1. Environment

```bash
conda create -n flsnn_cr python=3.9 -y
conda activate flsnn_cr
conda install pytorch==2.1.* torchvision==0.16.* pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y
pip install spikingjelly==0.0.0.0.14 thop tensorboard tqdm \
            Pillow scikit-image numpy matplotlib
```

`cupy` is **optional** (set `--vlif_backend cupy` to enable).  v1
defaults to `backend=torch` for reproducibility; numeric outputs are
identical across backends, cupy is only ~20–30 % faster on V100.

## 2. Data

### CUHK-CR1 / CUHK-CR2
Source: Sui et al., *"Diffusion Enhancement for Cloud Removal in
Ultra-Resolution Remote Sensing Imagery"*, TGRS 2024
([arXiv 2401.15105](https://arxiv.org/abs/2401.15105)).

Download: [Baidu Pan](https://pan.baidu.com/s/1z2SgORYz5_t94kya8CeqiQ), code `bean`.

| Subset | Train / Test | On-disk size | Paper-experiment size | Content |
|---|---|---|---|---|
| CUHK-CR1 | 534 / 134 | **512 × 512** | 256 × 256 | Thin clouds (avg 50.7 % cover) |
| CUHK-CR2 | 448 / 111 | **512 × 512** | 256 × 256 | Thick clouds (avg 42.5 % cover) |

**v1 note:** the code reads whatever size is on disk (PIL decode).
With `patch_size=64` random crop for training and `center_patch` mode
eval at 64², the pipeline is resolution-agnostic.

### Accepted on-disk layouts
The dataset loader auto-detects any of:

```
<root>/train/input/  and  <root>/train/target/
<root>/test/input/   and  <root>/test/target/
```
or
```
<root>/input/        and  <root>/target/
```
or
```
<root>/cloudy/       and  <root>/clear/
<root>/cloud/        and  <root>/label/   (or /gt/)
```

Files may be `.png`, `.jpg`, `.tif` etc.  If no explicit `train/` and
`test/` subfolders exist, the runner derives an 8 : 2 split with a
fixed seed (`--partition_seed`, default 0).

### Probe before running
```bash
python -m cloud_removal_v1.dataset /abs/path/to/CUHK-CR1
```
Prints discovered input/target folder names, total pair count, derived
8 : 2 split sizes, per-client sizes for 5 × 10 = 50 satellites, and one
decoded sample's shape / value range.

## 3. Running

```bash
# Smoke (defaults = agreed v1 config)
python -m cloud_removal_v1.run_smoke --data_root /abs/path/to/CUHK-CR1

# Fast dryrun (~20 min)
python -m cloud_removal_v1.run_smoke \
    --data_root /abs/path/to/CUHK-CR1 \
    --num_epoch 3 --num_planes 3 --sats_per_plane 3 \
    --run_name dryrun

# Sliding-window evaluation (slower, memory-safe)
python -m cloud_removal_v1.run_smoke --eval_mode sliding

# Preview v2 FedBN behaviour
python -m cloud_removal_v1.run_smoke --bn_local
```

### CLI knobs (most-used)

| Flag | Default | Notes |
|---|---|---|
| `--data_root` | `./data/CUHK-CR1` | dataset root (absolute path OK) |
| `--run_name` | `v1_smoke` | tag for output files |
| `--num_epoch` | 10 | global rounds |
| `--num_planes` / `--sats_per_plane` | 5 / 10 | 50/5/1 Walker Star shape |
| `--patch_size` | 64 | training crop |
| `--train_batch_size` | 4 | V100 16 GB-safe at dim=24 |
| `--num_workers` | 0 | total spawn = `50 × num_workers` — raise cautiously |
| `--vlif_dim` | 24 | → ~5 M params; original VLIFNet uses 48 |
| `--vlif_backend` | torch | `cupy` optional if CUDA toolkit compatible |
| `--lr` | 1e-3 | AdamW base LR; cosine → `min_lr=1e-7` after 3-epoch warmup |
| `--bn_local` | off | turns on FedBN — v2 knob, already wired |
| `--eval_mode` | center_patch | `center_patch` / `fullimage` / `sliding` |
| `--eval_patch_size` | 64 | used by `center_patch` |

## 4. Outputs

```
Outputs/
├── v1_smoke_<run>_Relaysum_Aggregation.npz
├── v1_smoke_<run>_Gossip_Averaging.npz
├── v1_smoke_<run>_AllReduce_Aggregation.npz
├── v1_smoke_<run>_summary.json            # final numbers + config
├── v1_<run>_train_loss.pdf                # from plot_results.py
├── v1_<run>_test_psnr.pdf
├── v1_<run>_test_ssim.pdf
└── tb/<run>/                              # tensorboard events
```

`.npz` keys: `epochs, train_loss, eval_psnr, eval_ssim, comm_bytes,
wall_seconds, per_plane_psnr, per_plane_ssim`.

## 5. Success criteria

1. `python -m cloud_removal_v1.tests.run_all` — all tests pass.
2. `run_smoke.py` finishes without exception.
3. Training loss of each scheme monotonically decreases.
4. Final centre-patch PSNR ≥ 20 dB, SSIM ≥ 0.6 (CUHK-CR1 is easy).
5. RelaySum ≥ Gossip (directionally, even if numerically close).

Failing any of these means DO NOT move on to v2.

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Could not find paired input/target ...` | Folder layout doesn't match the accepted ones (§2) | Inspect disk tree; rename / symlink `cloudy→input`, `clear→target`. |
| `CUDA out of memory` during training | `train_batch_size` or `patch_size` too large | Halve batch first, then drop patch_size to 48. |
| `CUDA out of memory` during eval | `--eval_mode fullimage` at 512² on V100 16 GB | Default `center_patch` is already on; don't switch to `fullimage` on 16 GB. |
| spikingjelly import error | Version mismatch | `pip install --force-reinstall spikingjelly==0.0.0.0.14`. |
| cupy wheel fails | CUDA toolkit mismatch | Default `vlif_backend=torch`; keep it. |
| First-epoch NaN loss | AdamW 1e-3 too aggressive at init | `--lr 5e-4`. |
| Training really slow (>30 min/ep) | `num_workers>0` on 50 satellites ⇒ worker storm | Keep `--num_workers 0`. |
