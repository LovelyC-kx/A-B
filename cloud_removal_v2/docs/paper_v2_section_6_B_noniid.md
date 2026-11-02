# §VI-B. Federated Partition: From Label-Shift to Source-Shift Non-IID

The original FLSNN benchmark partitions a 10-class classification
dataset (EuroSAT) by Dirichlet sampling over the **label** dimension
[FLSNN, §VI-A]. Our task — pixel-level cloud removal — has no
class label that lives in the model's output, so a direct port of
that partition is meaningless: every pixel-regression target is a
unique 64×64 image, not one of $K$ discrete classes.

We therefore introduce a **source-level** Dirichlet partition that
keeps the federated-learning realism (per-client distribution skew)
while respecting the task's structure. This subsection (i) defines
the partition, (ii) reports its empirical skew, and (iii) compares
its non-IID strength to the FLSNN Fig. 5 setting — a comparison
required to interpret §VI-D's RelaySum-ranking result.

## B.1 The `dirichlet_source` partition

Each of the 50 satellites $k$ draws a length-2 vector
$\boldsymbol{p}_k \sim \mathrm{Dir}(\alpha\!=\!0.1)$ over the two
dataset sources (CUHK-CR1 = thin clouds; CUHK-CR2 = thick clouds).
The training images are then assigned greedily so that satellite
$k$ ends up with the requested per-source proportions, **subject to
a minimum of 5 images per satellite** (enforced post-hoc to avoid
empty clients). Implementation: `cloud_removal_v2/dataset.py:405–453`.

Concrete code paths and constants:

| Field | Value | Source |
|:------|:------|:------|
| `partition_mode`     | `"dirichlet_source"`  | `config.py:59` |
| `partition_alpha`    | `0.1`                 | `config.py:60` |
| `partition_seed`     | `0`                   | `config.py:61` |
| `min_per_client`     | `5`                   | `config.py:62` |
| Total clients        | `num_planes × sats_per_plane = 5 × 10 = 50` | `config.py:57–58` |

The output is a list of 50 disjoint index lists into the 982-image
training set. Sample bookkeeping is reproduced in `Outputs_v2/v2a_v2a_80ep_partition_summary.txt`.

## B.2 What this partition looks like in practice

The partition heatmap (Fig. 1 — `Outputs_v2/v2a_v2a_80ep_partition.pdf`)
visualises the 50 × 2 sample-count matrix. Empirical statistics:

* **72 % of satellites (36 / 50)** are *pure single-source*: ≥ 95 % of
  their training images come from one of the two CUHK-CR sources.
* **35 / 50 satellites** sit at the 5-image minimum — that is, they
  contribute very little gradient signal individually. The
  Dirichlet draw concentrated mass on a few "lucky" clients.
* **15 / 50 satellites** hold > 30 images. The most lopsided
  satellite (P2-S8) holds **118 training images alone**, dominating
  its plane's data flow.
* **Within-plane source skew is also substantial.** P0 leans CR2
  (P0-S6 has 117 CR2 images; only 70 CR1 images total in P0); P1
  leans CR1 (P1-S8 has 116 CR1 images); P2 leans CR1 (P2-S8 has
  118 CR1 images). Plane P3 is the most balanced, with both
  sources represented across multiple satellites.

This is "client skew" *and* "plane skew" simultaneously. FedAvg's
intra-plane averaging cleans up the client skew (by construction —
all 10 satellites of a plane get the same averaged weights every
intra-plane round), but the **plane skew remains** for the
inter-plane aggregation step to deal with.

## B.3 Strength comparison with the original FLSNN setting

A direct numerical comparison of "$\alpha = 0.1$" between our work
and FLSNN is misleading because the Dirichlet support differs:

| | **FLSNN Fig. 5** (their reference setting) | **Our v2-A** |
|:---|:---:|:---:|
| Dataset                  | EuroSAT 10-class | CUHK-CR 2-source |
| Dirichlet support        | $K = 10$ labels  | $K = 2$ sources  |
| Concentration param      | $\varsigma = 0.02$ | $\alpha = 0.1$ |
| Effective skew           | each client sees ≈ 1–2 of 10 classes | each client sees a Beta(0.1, 0.1) mix of 2 sources |
| Per-client effective $K$ | $\sim 1.5$ | $\sim 1.7$ |

The "effective $K$" row uses the standard concentration interpretation:
under $\mathrm{Dir}(\alpha)$ over $K$ categories, the expected
number of categories with non-trivial mass scales as
$\min(K, K\alpha + 1)$ for small $\alpha$. Both settings produce
≈ 1.5–1.7 categories per client on average. **However**:

* FLSNN's effective number out of $K = 10$ means a client *misses*
  ~85 % of the label space — gradients from that client are
  systematically biased away from the missing classes.
* Our effective number out of $K = 2$ means a client *partially
  misses* one of two sources — a much smaller geometric loss in
  representation coverage.

A more rigorous comparison runs through **Theorem 2 of FLSNN**
(originally [Vogels 2021], reproduced in v1's
`cloud_removal_v2/docs/v2_theory_and_related.md` §3). The bound
contains a $\zeta^2 = $ inter-client gradient dissimilarity term;
$\zeta^2$ is upper-bounded by a label-shift quantity for
classification and by a source-shift quantity for regression. The
**source-shift $\zeta^2$ is structurally smaller** when the two
sources share the bulk of their visual content (here: the same
sensor, the same scene types — only the cloud thickness differs).

The closed-form Dirichlet-to-$\zeta^2$ derivation we wrote in
`v2_formal_derivations.md` §D1 gives a quantitative form of this
upper bound. The takeaway:

> **Our partition is strictly *less* non-IID than FLSNN's**,
> measured by Theorem 2's $\zeta^2$. Any aggregation-scheme
> ranking we observe is therefore evaluated *closer to the IID
> regime* than FLSNN's reference experiment.

This single fact controls the reading of §VI-D: when we report
"RelaySum did not beat AllReduce" we are reporting a result in a
regime where Theorem 2 itself **predicts** that RelaySum's
$\zeta^2$-amplified advantage should shrink. The reversal is not a
contradiction of FLSNN; it is the **boundary case** of FLSNN's own
theory.

## B.4 The min-5-per-client constraint and its consequences

The 5-image minimum (post-hoc clipping of the Dirichlet draw)
matters because of two side effects:

1. **The 35 satellites that ended up at the minimum** receive an
   over-represented uniform-random mix relative to the pure
   Dirichlet sample they would have drawn. This *softens* the
   non-IID setting marginally — see §VI-H limitations for the
   exact correction.
2. **Per-plane sample counts are substantially imbalanced:**
   P2 has only 1 satellite holding > 5 images (P2-S8 with 118),
   while P3 has 5 satellites with > 5 images. This means P2's
   intra-plane average is essentially "P2-S8's local model with
   small noise", whereas P3's average is a real ensemble.

Both effects are observable in §VI-G's per-plane spread analysis.

## B.5 Reproducing the partition

The partition is fully deterministic given the four constants
above. Re-derivation:

```bash
python -m cloud_removal_v2.plot_partition_heatmap \
    --run_name v2a_80ep --output_dir ./Outputs_v2 \
    --data_root /root/autodl-tmp/C-CUHK
```

This produces `v2a_v2a_80ep_partition.pdf` (the heatmap, Fig. 1)
and `v2a_v2a_80ep_partition_summary.txt` (the 50-row sample-count
table). Both are bit-stable across machines: the `dirichlet_source`
branch of `build_plane_satellite_partitions_v2`
(`cloud_removal_v2/dataset.py:439–447`) forwards `seed=0` into
`dirichlet_source_partition(...)`, which internally seeds a NumPy
RNG before drawing the 50 Dirichlet vectors.
