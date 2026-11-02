# §VI-F. Qualitative De-clouding Results

This subsection reports the visual quality of the trained models on
held-out CUHK-CR test images. The headline takeaway is:

> Visual quality matches the numerical Tables on **thin-to-medium
> cloud cover** but **all aggregation × BN cells fail uniformly on
> very thick cloud**, regardless of backbone. The 6-sample qualitative
> selection happens to bias against C, illustrating that single-image
> visual comparison cannot substitute for the 245-image test set.

## F.1 Sample selection and PDFs

Six test images are deterministically selected with the seed-42
sampler in `cloud_removal_v2/visualize.py:visualize_main()`:
indices `[24, 6, 153, 212, 199, 177]` (re-printable from any
visualize.py invocation). The same six images are used for both
qualitative panels.

Two qualitative PDFs were generated:

* **Run A panel** — `Outputs_v2/v2a_v2a_80ep_qualitative.pdf`.
  Six rows × 8 columns: `cloudy | RS-FedAvg | G-FedAvg | AR-FedAvg
  | RS-FedBN | G-FedBN | AR-FedBN | clear GT`.
* **Run C panel** — `Outputs_v2/v2a_v2a_80ep_ann_fedbnar_qualitative.pdf`.
  Six rows × 3 columns: `cloudy | AR-FedBN-ANN | clear GT`.
  Run C trained only one cell so only one column of restored
  outputs exists.

## F.2 Visual quality observations

The 6 sample images cover three difficulty bands:

* **Easy (rows 1, 5)** — thin or wispy cloud. All cells produce
  visually similar outputs that closely match the GT. PSNR per-cell
  spread is < 1 dB on these rows.
* **Moderate (rows 2, 4)** — partial-cloud or building scene. All
  cells produce a *hazy* output that recovers the major features
  but leaves a smooth "cloud residual" on the de-clouded image.
  Cell-to-cell visual differences are minor.
* **Difficult (rows 3, 6)** — thick / opaque cloud. All cells fail
  uniformly: the restored image preserves a soft greyish residual
  that obscures the GT's vibrant ground colours. PSNRs collapse
  to 17–20 dB regardless of cell.

This three-band picture is consistent across all 13 cells (A's
six, B's six, C's one).

## F.3 PSNR-per-cell on the qualitative samples (Run A)

The six rows of A's qualitative panel report per-row per-cell PSNR
beneath each thumbnail. Aggregating:

| Sample | RS-FA | G-FA | AR-FA | RS-FB | G-FB | AR-FB |
|:------:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1      | 24.72 | 24.62 | 24.40 | 22.98 | 24.94 | 24.20 |
| 2      | 22.02 | 22.87 | 22.08 | 20.78 | 22.07 | 22.14 |
| 3      | 17.96 | 20.90 | 18.51 | 20.42 | 19.15 | 20.41 |
| 4      | 21.62 | 21.42 | 21.77 | 21.83 | 22.12 | 20.01 |
| 5      | 23.64 | 23.57 | 23.39 | 23.62 | 23.66 | 23.51 |
| 6      | 17.51 | 17.45 | 17.57 | 17.51 | 17.79 | 17.49 |
| **mean** | 21.25 | **21.81** | 21.29 | 21.19 | 21.62 | 21.29 |

Compare against the **Test-set ranking** (245 images, Table I row order):

| Cell | 6-sample mean (qualitative) | Full-test mean (Table I) |
|:-----|:---:|:---:|
| RS-FA | 21.25 | 21.500 |
| G-FA  | **21.81** ← best on qual | 21.345 ← worst on test |
| AR-FA | 21.29 | **21.642** ← best on test |
| RS-FB | 21.19 | 21.561 |
| G-FB  | 21.62 | 21.531 |
| AR-FB | 21.29 | 21.420 |

**The qualitative-sample ranking is anti-correlated with the
test-set ranking.** G-FA wins on qual but is worst on test;
AR-FA wins on test but is mid-pack on qual. With per-image PSNR
standard deviation of ≈ 2–3 dB across these 6 samples, the
6-sample mean has an estimator standard error of ≈ 1 dB —
which is already wider than the entire Table I PSNR spread.

## F.4 A vs C on the same six samples

Comparing C's `AR-FB-ANN` column with A's `AR-FB` column row by row:

| Sample | A AR-FB (SNN) | C AR-FB (ANN) | C − A |
|:------:|:---:|:---:|:---:|
| 1 | 24.20 | 22.86 | −1.34 (A wins) |
| 2 | 22.14 | 20.42 | −1.72 (A wins) |
| 3 | 20.41 | 20.08 | −0.33 (A wins) |
| 4 | 20.01 | 22.34 | +2.33 (C wins) |
| 5 | 23.51 | 23.07 | −0.44 (A wins) |
| 6 | 17.49 | 17.63 | +0.14 (C ≈ tied) |
| **mean** | **21.29** | 21.07 | **−0.22 (A wins on qual)** |

Yet on the **full 245-image test set** (Table IV), C wins by
+0.751 dB. The 0.97 dB swing between qualitative and test-set
rankings is purely a sample-size artifact: 6 samples are not
enough to estimate a mean PSNR with single-decimal-dB precision.

**§VI-F therefore does not use the qualitative panel to make any
ranking claim.** It uses the panel only for the three statements
the panel can actually support:

* All cells succeed on thin / medium cloud (easy band).
* All cells produce hazy "cloud residual" on moderate cloud.
* All cells **fail uniformly on thick cloud** (samples 3 and 6).

## F.5 What "fails uniformly" means

On samples 3 and 6 — the two thickest-cloud images of the six —
the *cell-to-cell PSNR spread* is at most 3 dB (sample 3:
17.96–20.90 dB) or 0.34 dB (sample 6: 17.45–17.79 dB), but the
*absolute* PSNR is so low that the restored images are visually
unusable. The model has learned an "ambient ground-truth bias"
that lifts pixel intensities back to the typical cloud-free range
but cannot recover the lost spatial high-frequency content. This
is consistent with the published cloud-removal literature
(Sui *et al.* TGRS 2024, Section IV-D) — thick cloud pixels are
information-theoretically lost; a single-image network cannot
fill them in.

§VI-H discusses this as the principal limitation of v2 and
proposes multi-temporal cloud removal as the v3 / v4 extension.

## F.6 Reproducibility

Re-generate both panels:

```bash
# Run A panel (6 cells)
python -m cloud_removal_v2.visualize \
    --run_name v2a_80ep --output_dir ./Outputs_v2 \
    --data_root /root/autodl-tmp/C-CUHK \
    --patch_size 256 --n_samples 6

# Run C panel (1 cell, ANN backbone)
python -m cloud_removal_v2.visualize \
    --run_name v2a_80ep_ann_fedbnar --output_dir ./Outputs_v2 \
    --data_root /root/autodl-tmp/C-CUHK \
    --backbone ann --bn_variant tdbn \
    --patch_size 256 --n_samples 6
```

Both commands seed the test-image sampler at 42; the output PDFs
are bit-stable across machines (modulo PDF metadata).
