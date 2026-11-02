# §VI-H. Discussion, Limitations, and Future Work

This subsection closes §VI by (i) stating the core takeaways from
the preceding subsections, (ii) honestly disclosing every
single-seed, single-partition, single-topology assumption the
results rest on, and (iii) laying out the v3 work ledger.

## H.1 Core takeaways

The v2 experimental study on CUHK-CR (80 inter-plane rounds,
Dirichlet-source $\alpha = 0.1$, 50/5/1 Walker-Star, VLIFNet
backbone, T = 4, MultiSpike-4) produces four paper-scope claims,
each tied to the subsection that quantifies it:

1. **Source-shift non-IID is structurally weaker than FLSNN's
   label-shift** (§VI-B). Theorem 2's $\zeta^2$ upper bound is
   smaller here than in FLSNN Fig. 5, for reasons that are
   geometric (2 visually-similar sources vs 10 land-cover
   classes) rather than parametric. This framing controls the
   reading of §VI-D.

2. **FedBN is within the single-seed noise floor** (§VI-C).
   Mean $\Delta(\mathrm{PSNR})_{\rm FedBN - FedAvg}$ is $+0.008$ dB
   with TDBN and $+0.044$ dB with BN2d, both under $0.05$ dB.
   TDBN's mechanism (reducing $\mathrm{Var}(\gamma)$ by 42 % vs
   standard BN, Table III) is *real* but does *not* translate to
   PSNR — the task's $\zeta^2_{\rm BN}$ is below the saturation
   point for FedBN benefit.

3. **AllReduce Pareto-dominates Gossip / RelaySum** in the
   (Comm, PSNR) plane (§VI-D, §VI-E). AllReduce uses 60 % less
   inter-plane communication and matches the best alternative's
   PSNR within 0.03 dB. The FLSNN Fig. 5 ranking (RelaySum >
   Gossip > AllReduce) does not transfer to our setting.

4. **On a GPU, the ANN backbone wins on PSNR (+0.75 dB), SSIM
   (+0.027), and wall time (1.61 ×)** at identical
   communication (§VI-E, Table IV). The SNN's deployment
   advantage — lower inference energy — is **not measured** in
   v2 and is deferred to v3 (§H.3 item E1).

## H.2 Limitations

We flag ten single-seed / single-setting assumptions, each of
which a reviewer can reasonably ask us to test in isolation.
Every item is mapped to the v3 ledger entry that will address it.

| ID | Limitation                                                        | v3 ledger |
|:--:|:------------------------------------------------------------------|:---------:|
| L1 | Single seed (`seed=1234`, `partition_seed=0`). The 0.1 dB       | V9        |
|    | intra-cell spread reported in Tables I & III is a point estimate; |           |
|    | we have no confidence interval.                                   |           |
| L2 | Single $\alpha$ (0.1). Theorem 2's $\zeta^2$ dependence on $\alpha$ | α-sweep |
|    | is theoretical; we do not trace it empirically on this task.       |           |
| L3 | Single topology (50/5/1 chain). AllReduce's Pareto dominance is  | MDST    |
|    | a small-$N$ phenomenon; we do not identify the crossover $N$.      |           |
| L4 | Single optimizer (AdamW). FLSNN Fig. 5's SGD result is a         | SGD-abl |
|    | candidate cause for the scheme-ranking reversal (§VI-D cause D6).  |           |
| L5 | RelaySum's $2.093$ LR scaling is borrowed from FLSNN's EuroSAT    | D9-abl  |
|    | empirical value without re-validation.                              |           |
| L6 | BN-drift is end-of-training only; per-epoch trajectory is logged  | V3      |
|    | by `BnDriftLogger` but not saved to npz (`run_smoke.py:602-612`). |           |
| L7 | Cosine-similarity between planes is computed but not saved.       | V3      |
| L8 | Energy ratio is not measured. MultiSpike-4 is 5-level, not       | E1      |
|    | binary; the 0.9 pJ/AC formula from FLSNN is a loose lower bound.  |           |
| L9 | Centralised (non-FL) upper bound not run. We cannot quote         | V8      |
|    | a "federation loss" against a single-client oracle.                |           |
| L10 | Thick-cloud samples fail uniformly (§VI-F.5). Single-image       | v3/v4   |
|    | cloud removal is information-theoretically bounded; multi-         |          |
|    | temporal input is the principled extension.                       |           |

## H.3 v3 work ledger

The v3 extension takes the v2 setup and adds, in priority order:

* **V9 — Multi-seed sweep.** Re-run A, B, C at 3 additional seeds
  (1235, 1236, 1237). Reports mean ± std for all Table I / II / IV
  cells. Estimated GPU budget: 3 × (36 h + 36 h + 3.9 h) = 228 h.
  Lives at commit tag `v3-multiseed`.

* **α-sweep.** Replace $\alpha = 0.1$ with $\alpha \in \{0.01,
  0.1, 1.0\}$ at matched BN variant + scheme. Traces Theorem 2's
  $\zeta^2$ dependence empirically and probes whether the
  scheme-ranking crossover happens within a reasonable $\alpha$
  range. 4 cells × 3 α values × 80 rounds ≈ 75 h.

* **MDST — Minimum-Diameter Spanning Tree on 42/7/1 Walker Delta.**
  The FLSNN paper's §V-C system-optimization argument. Requires
  an STK-simulated 42/7/1 Delta constellation and the
  $\tau_{\rm max}$-optimizing spanning-tree algorithm. Out-of-scope
  for v2 (the paper does not claim the system-optimization
  result); a natural v3 follow-up.

* **SGD-ablation.** Re-run A's 6 cells with SGD(lr=0.05) to
  isolate §VI-D cause D6. 6 × 6 h = 36 h on a 4090.

* **D9-ablation.** Re-run RelaySum cells of A and B with
  `--relay_lr_scale 1.0` (requires a 5-line CLI-flag addition to
  `run_smoke.py`). Isolates §VI-D cause D9. 4 × 6 h = 24 h.

* **V3 — Save trajectory drift and cos-sim.** One-line fix in
  `run_smoke.py:602-612` plus a `plot_drift_trajectory.py`.
  Unblocks the "§VI-G.5 per-epoch trajectory" TODO.

* **E1 — Energy measurement script.** Write
  `cloud_removal_v2/energy_estimation.py` from scratch (§VI-E.4.3
  already committed the design). Produces both a binary-spike
  bound (0.9 pJ/AC, aggressive) and a per-layer MAC conservative
  bound (4.6 pJ/MAC, trivial). Drops into v3 as Table V.

* **V8 — Centralised oracle.** Single-client VLIFNet trained on
  the full 982-image training set for 80 epochs. Produces the
  "federation loss" upper bound for Table I. ≈ 12 h.

## H.4 Threats to validity not on the v3 ledger

A small number of threats cannot be resolved by any v3 experiment
and require separate discussion:

* **T1 — VLIFNet parameterisation choice.** The T = 4 outer loop
  and MultiSpike-4 encoding are design decisions inherited from
  v1; they are *a priori* orthogonal to any FL claim we make but
  could indirectly affect scheme rankings if a different
  spike-encoding regime has different $\zeta^2_{\rm BN}$.

* **T2 — Dataset idiosyncrasy.** CUHK-CR1+CR2 is the only
  cloud-removal benchmark that meets our "two naturally
  distinct source distributions" criterion at sufficient scale.
  Alternative FL-image-restoration benchmarks (e.g. FedMRI,
  FedFTN) would test whether our §VI-C claim generalises
  outside remote-sensing regression.

* **T3 — Single-hardware reproducibility.** All runs on one 4090.
  Theoretical claims (scheme ranking, FedBN effect) should
  survive a different GPU, but we have not tested this. Bit-level
  reproducibility between 4090s at `deterministic=False` is known
  to break; the point claims are however stable to ~0.05 dB.

## H.5 Separation of v2 from v3 / v4

For the purposes of this submission, the paper reports v2 only.
The following are explicitly **not** part of v2's contribution
and are relegated to v3 / v4:

* System-optimization algorithm (v1's §V-C content). The v2
  paper inherits the theoretical result but does not re-run
  the experiment.
* Multi-seed confidence intervals.
* Energy measurement.
* α-sweep / SGD-ablation / D9-ablation.
* Centralised oracle.
* Multi-temporal cloud removal.

A reader who objects "the paper should include X of the above"
is directed to this subsection's L-ledger and to
`v2_remaining_issues.md` in the accompanying repository, both of
which disclose the deliberate scope boundary.

## H.6 Summary

v2's contribution is a controlled federated-learning study of the
TDBN vs BN2d and ANN vs SNN axes on CUHK-CR, with a RelaySum /
Gossip / AllReduce comparison that recovers a FLSNN-Theorem-2
boundary case. The four headline claims (H.1) are each supported
by a single Table and are each qualified by the L-ledger (H.2).
Single-seed caveats are explicit; v3 will deliver the missing
confidence intervals and ablations.
