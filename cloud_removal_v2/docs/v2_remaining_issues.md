# v2 Remaining Issues — Must-fix Before the 70-Epoch Sweep

Written 2026-04-19 after the 35-epoch dry-run revealed actionable gaps.
**Updated 2026-04-19 (round-2)** to reflect actual done/undone status
after E-series fixes + §23/§25 audits + 3-agent literature integration.

Prioritised TOP = critical (blocks paper alignment) → BOTTOM = nice-to-have.

**Quick status table (round-2 truth):**

| Item | Status |
|------|--------|
| 1.1 xlabel rename | ✅ **DONE** (commit 4993652, E-1.1) |
| 1.2 `--viz_patch_size` flag | ✅ **DONE** (commit e5e4bf0, E-1.2) |
| 1.3 confirm `num_epoch=70` | ⏸️ **parameter ready, run NOT started** |
| 2.1 `plot_comm_efficiency.py` | ✅ **DONE** (commit c8bec42, E-2.1) |
| 2.2 `plot_partition_heatmap.py` | ✅ **DONE** (commit 2319f84, E-2.2) |
| 2.3 `plot_per_plane.py` | ✅ **DONE** (commit fd8c291, E-2.3) |
| 2.4 multi-seed partition | ❌ **deferred** to v3 (V9 in §25.6) |
| 3.1 comm-accounting caveat in paper | ❌ **paper §VI-H not written yet** (V5) |
| 3.2 intra-plane BN-avg caveat in paper | ❌ **paper §VI-H not written yet** (V5) |
| 3.3/3.4/3.5 | ✅ accepted as-is, documented |
| 4.1/4.2/4.3 | ❌ deferred to v3 |

**Additionally discovered during §23/§25 audits (newly required):**

| New v2 item | Status | Source |
|-------------|--------|--------|
| **V1** Start 70-ep run | ❌ not started | §25.6 |
| **V2** Drift-measurement script (SC-16a/b/c) | ❌ not written | §16.4 |
| **V3** Cosine-similarity logging per epoch (P4-cos) | ❌ not implemented | §4.6(b), §19.4 |
| **V4** Fill §19 ledger post-run | ❌ blocked on V1 | §19.8 |
| **V5** Write paper §VI sub-sections | ❌ only outlines exist | §20 |
| **V6** Create `docs/literature_audit.md` (18-paper verification) | ❌ not committed | §23.6 |
| **V7** Paper §VI draft | ❌ only snippets | §20 |
| **V8** SC-16d standard-BN ablation (clean Claim C16 test) | ❌ 16 GPU-hr + 1 day code | §16.4, §21.1 |
| **V9** Multi-seed partition scan | ❌ | §21.2 |
| **V10** Verify FLSNN m parameter | ❌ re-read PDF | §18.2 |
| **V11** Rigorously re-derive §14.5 | ✅ **DONE** (commit TBD, uses Agent-3) | §25.4 |
| **V12** Dataset cite Zhou 2022 → Sui 2024 | ✅ **DONE** (commit TBD) | §25.3.1 |
| **V13** Novelty claim rewrite (Venkatesha / Wang cite) | ⏸️ outline ready, paper not written | §25.2 |
| **V14** Add §II-D FL image restoration (FedMRI/FedFTN/FedFDD/FedNS) | ⏸️ outline ready, paper not written | §25.2 |
| **V15** Empirical variance of p_i under Hsu 2019 partition (20-line script over 1000 seeds, verifies D1.17 coefficient 0.204 vs actual implementation) | ❌ not written | v2_formal_derivations.md §D1.10 caveat 5 |

---

---

## TIER 1 — Paper-alignment blockers (must fix before 70-epoch sweep)

### 1.1  X-axis label: `Global Epoch` → `Inter-Plane Communication Rounds`

**Why now**: the original FLSNN paper's Fig 5 (verified from the actual PDF the user
showed) explicitly labels the x-axis as **"Inter-Plane Communication Rounds"**, even
though the shipping `main.py:32` in the original repo writes `plt.xlabel('Training
Epochs', …)` — i.e. the paper figures differ from the shipping code. Our
`plot_results.py:111` uses `Global Epoch`, which is numerically equivalent (1 global
epoch ≡ 1 inter-plane communication round under `intra_plane_iters=2`,
`local_iters=2`; verified against `constellation.py:train_one_round`) but must be
renamed for apples-to-apples paper-side comparison.

**Location**: `cloud_removal_v2/plot_results.py:111` and anywhere else the xlabel
appears.

**Fix**: change the literal string; no semantic change.

### 1.2  Qualitative viz uses patch_size=64 → upscaled-to-PDF looks pixelated

**Observation from the 35-epoch run**: `v2a_v2a_qualitative.pdf` showed clearly
pixelated "Clear GT" images in rows with low-structure textures (row 1–2 of
user's 6×8 grid). This is NOT a data problem — CUHK-CR originals are 512×512.
Root cause: `visualize.py` feeds `patch_size=64` into `_center_crop_np`, then
matplotlib upsamples 64×64 into a ~2-inch tile in the PDF.

**Fix**: add `--viz_patch_size` CLI flag defaulting to 256 (4× more pixels per
tile) OR use full-image eval mode for viz. Keep training/eval patch_size=64
unchanged (that governs speed, not aesthetics).

**Location**: `cloud_removal_v2/visualize.py` — add CLI arg, wire through to
`_center_crop_np`.

### 1.3  Decide on 70 epoch vs 60 epoch

User's stated plan is **70** rounds; the original FLSNN paper used **60** rounds.
A 70-round sweep gives us more headroom (10 extra rounds to verify plateau) at
a 17% wall-time premium. No code change; just confirm the `--num_epoch 70` CLI
value.

---

## TIER 2 — Publication-quality plots not yet automated

### 2.1  Communication-efficiency plot (bytes-to-target-PSNR)

**Why it matters**: the paper's narrative will HAVE to address the
counter-intuitive observation that AllReduce uses fewer bytes per round than
Gossip/RelaySum in our accounting (5×state vs 8×state per round on a chain of
5). A "bytes cumulative" x-axis plot would either reinforce or weaken this
claim. Without it, reviewers will ask.

**Data already present**: every `npz` has the `comm_bytes` array, one entry per
round.

**New file**: `cloud_removal_v2/plot_comm_efficiency.py` (~50 lines).
Implementation sketch:
- For each of 6 cells, compute `cumsum(comm_bytes)` → x-axis; `eval_psnr` →
  y-axis (drop NaN).
- Plot 6 curves on a shared axis. Mark target PSNR (e.g. 21 dB) as horizontal
  line → read off "bytes to reach 21 dB" per scheme.
- Output: `Outputs_v2/v2a_<run>_comm_efficiency.pdf`.

### 2.2  Dirichlet partition heat-map (client × source)

**Why it matters**: original FLSNN paper reports `plane_alpha` but does NOT
visualize the partition. For our paper's non-IID claim to be reviewer-proof,
a heat-map of per-client (thin, thick) sample counts under Dirichlet(α=0.1)
makes the "severe non-IID" argument unambiguous.

**Data already present**: `MultiSourceCloudDataset.source_labels()` returns a
per-sample source array. `build_plane_satellite_partitions_v2` returns
`flat_indices: List[List[int]]`.

**New file**: `cloud_removal_v2/plot_partition_heatmap.py` (~60 lines).
50 clients × 2 sources → 2-column heat-map, annotated with sample counts.
Output: `Outputs_v2/v2a_<run>_partition.pdf`.

### 2.3  Per-plane PSNR/SSIM spread

**Why it matters**: we save `per_plane_psnr` / `per_plane_ssim` in the npz
(arrays of length 5 per eval epoch) but never visualise them. For FedBN
cells especially, per-plane variance IS the interesting quantity — it
measures how much the 5 planes diverge under FedBN.

**New file**: `cloud_removal_v2/plot_per_plane.py` (~40 lines). Box-plot or
error-bar per cell at final epoch.

### 2.4  Learning-curve smoothing / variance band

**Observation**: the 35-epoch PSNR curve shows non-monotonic dips (e.g.
fedavg-AllReduce SSIM 15→30 decreases). With `eval_every=5` we get only 7
data points → single-seed noise floor is visible. Two options:

- **a)** decrease `eval_every` from 5 → 2 (more points, +~35 min wall time
  per cell for 7 × 35/2 ≈ 50% more eval). Gives smoother curves.
- **b)** run 2–3 different `partition_seed` values and report mean ± std
  shaded band. More rigorous but 2–3× wall time.

For the 70-epoch run: keep `eval_every=5` (no extra cost), plan (b) as a
v2.1 / v3 follow-up if reviewers ask.

---

## TIER 3 — Known-but-acceptable limitations (document, don't fix)

### 3.1  Communication-cost accounting is one-sided

Our `_state_dict_bytes` counts:
- RelaySum / Gossip: Σ (out-degree × state_bytes) = 8 × state_bytes on a
  chain-of-5 (counts each plane's egress to each neighbour).
- AllReduce: `num_planes × state_bytes` = 5 × state_bytes (counts only
  download of aggregated result to each plane).

These are NOT symmetric accounting. AllReduce's upload phase (each plane
uploading its own state to an aggregator) is MISSING from our count. In a
realistic centralized AllReduce, upload + download = 10 × state_bytes, which
would make AllReduce MORE expensive than Gossip, not less.

**Decision**: keep the current accounting (inherited from v1, consistent with
what the original FLSNN code would produce if they had bothered to count
bytes — `revised_constellation.py` in the original repo has NO byte-counting
logic at all). But **document this caveat explicitly** in the paper's
Section 6: "Communication cost is reported as per-plane egress; a
centralized-AllReduce implementation would incur an additional upload phase
of equal cost."

### 3.2  Intra-plane aggregation always uses `bn_local=False`

Hard-coded at `cloud_removal_v1/constellation.py:200–211` (verified by round-3
audit). This means under FedBN the intra-plane step still averages BN within
a plane, enforcing "FedBN-at-plane-granularity, not per-satellite". Already
documented in the updated inline comment. Paper Section 6 needs a one-line
disclosure.

### 3.3  No resume-from-checkpoint; each run starts fresh from
`build_vlifnet`. Acceptable. For the 70-epoch sweep, this is correct
behaviour (no half-trained baseline to restart from).

### 3.4  Dataloader multi-process unused (`num_workers=0`). Deliberate — 50
sats × N workers would oversubscribe the AutoDL box. A single-process
dataloader is the only robust choice on a shared container.

### 3.5  Single-seed experiment (no multi-partition-seed spread). User
already discussed; deferred to v2.1 / v3.

---

## TIER 4 — Optional but high-leverage

### 4.1  Save optimizer state alongside weights

Current `get_weights` returns only model state_dict. If we later want to do
resume-from-ckpt (for fine-tuning to 90+ rounds if reviewer demands longer
training), we'd need AdamW's `m`, `v` buffers too. Adds ~60 MB per plane, so
× 5 × 6 = 1.8 GB extra disk. Not worth it unless resume path is actually
needed.

**Recommendation**: skip for v2, add in v3 if `--resume` becomes a
requirement.

### 4.2  Include current-round relay buffer snapshot in ckpt

Same rationale as 4.1 but for RelaySum's `received_relay_weights`. Skip for
v2.

### 4.3  Unit test for `_state_dict_bytes` accounting

The ambiguity in 3.1 would be caught by a test that (a) builds a tiny
5-plane chain, (b) runs 1 round with RelaySum and AllReduce, (c) asserts
the relative ratio matches the expected value based on the chosen convention.

Low priority — the accounting is already stable and documented.

---

## Summary table

| # | Issue | Tier | Est. effort | Blocks 70-ep sweep? |
|---|---|---|---|---|
| 1.1 | Rename xlabel to "Inter-Plane Communication Rounds" | 1 | 2 min | Yes |
| 1.2 | `--viz_patch_size` flag (default 256) | 1 | 15 min | Yes (viz step after sweep) |
| 1.3 | Confirm num_epoch=70 | 1 | 0 min | — (just run parameter) |
| 2.1 | `plot_comm_efficiency.py` | 2 | 30 min | No (post-hoc) |
| 2.2 | `plot_partition_heatmap.py` | 2 | 30 min | No (run once, reusable) |
| 2.3 | `plot_per_plane.py` | 2 | 20 min | No (post-hoc) |
| 2.4 | Tune eval_every / add multi-seed | 2 | — | No (accept current) |
| 3.1 | Document comm-accounting asymmetry in paper | 3 | 1 paragraph | No |
| 3.2 | Document intra-plane BN policy in paper | 3 | 1 paragraph | No |

**Before the 70-epoch sweep the user should apply 1.1, 1.2, 1.3.**
**During the sweep the user can add 2.1, 2.2, 2.3 (reusable without GPU).**
**Tier 3 items are paper-writing tasks, not code.**
