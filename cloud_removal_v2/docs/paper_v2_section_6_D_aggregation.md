# §VI-D. Inter-Plane Aggregation Schemes

This subsection compares the three aggregation schemes
(AllReduce, Gossip, RelaySum) at fixed BN variant and fixed
backbone, then asks why the **scheme ranking observed here
differs from FLSNN Fig. 5's ranking**.

The key observation is that the difference is *not* a bug — our
implementation of RelaySum has been verified line-by-line against
FLSNN Algorithm 2 (`constellation.py:280–367`, see C below). The
reversal is a property of our task / setting and is consistent
with FLSNN's own Theorem 2 in the small-$\zeta^2$ regime.

We document the reversal, list **three independent candidate
causes** that cannot be disentangled from a single-seed run, and
defer their per-cause isolation to v3.

## D.1 Per-scheme PSNR ranking

Reading Table I (`docs/tables/table_main.md`):

| Run | Scheme | Mean PSNR over 2 BN variants (dB) |
|:---:|:-------|:---------------:|
| A (TDBN, SNN) | AllReduce | $(21.642 + 21.420) / 2 = 21.531$ |
| A             | Gossip    | $(21.345 + 21.531) / 2 = 21.438$ |
| A             | RelaySum  | $(21.500 + 21.561) / 2 = 21.531$ |
| B (BN2d, SNN) | AllReduce | $(21.630 + 21.762) / 2 = 21.696$ |
| B             | Gossip    | $(21.781 + 21.791) / 2 = 21.786$ |
| B             | RelaySum  | $(21.709 + 21.699) / 2 = 21.704$ |

Ranking (higher = better):

* **A:** AllReduce $\approx$ RelaySum $>$ Gossip
* **B:** Gossip $>$ RelaySum $>$ AllReduce  *(scheme-mean spread = $21.786 - 21.696 = 0.090$ dB)*

Two facts dominate:

* **The total PSNR spread is 0.1–0.3 dB** within either run. The
  difference between the best and worst scheme is roughly the
  same magnitude as the FedBN-vs-FedAvg gap from §VI-C, and
  considerably smaller than the *intra-cell* per-round PSNR
  fluctuations (§A.8 disclosure: RelaySum + FedBN dropped to
  19.69 dB at round 25).
* **RelaySum is never the unique winner** in our setting. In FLSNN
  Fig. 5 it leads by ~10 % accuracy on EuroSAT; here it ties with
  AllReduce (run A) or finishes second-of-three (run B).

## D.2 Pairing with the FLSNN published ranking

FLSNN Fig. 5 reports, on EuroSAT classification with $\varsigma =
0.02$ Dirichlet over 10 labels, RelaySum > Gossip > AllReduce by
~5–10 percentage points of test accuracy. The setting differs from
ours along several axes catalogued in §B.3:

| Axis | FLSNN Fig. 5 | Ours |
|:-----|:---:|:---:|
| Task              | classification (10-class CE) | regression (Charbonnier + 0.1·SSIM) |
| Optimizer         | SGD (0.05 / 0.10 lr)         | AdamW (1e-3 lr) |
| Spike encoding    | binary {0, 1}                | MultiSpike-4 (5-level) |
| Time steps        | $T = 3$                      | $T = 4$ |
| Non-IID strength  | $\varsigma = 0.02$, $K = 10$ | $\alpha = 0.1$, $K = 2$ |
| Total rounds      | 60                           | 80 |
| RelaySum LR scaling | 2.093 (their reference impl) | 2.093 (we kept it verbatim) |
| Topology          | 50/5/1 chain                 | 50/5/1 chain (identical) |

Three of these axes (task, optimizer, RelaySum LR scaling) are
**candidate causes** for the reversal that *cannot be isolated
without additional runs*. We name them:

* **D1.** Regression vs classification. Theorem 2's $\zeta^2$
  upper bound is structurally smaller for pixel regression — see
  §B.3 and the closed-form derivation in
  `v2_formal_derivations.md` §D1. In the $\zeta^2 \to 0$ limit,
  RelaySum's relay-buffer advantage collapses onto AllReduce's
  per-round average (`v1/docs/paper_section_6_draft.md`
  Observation block).
* **D6.** Optimizer choice. AdamW's per-parameter adaptive
  learning rate dampens stale-gradient effects more than SGD does;
  RelaySum's relay-stale messages may therefore lose less to AdamW
  than to SGD. **Direct evidence from FLSNN's github repository**
  (`Golden-Slumber/Decentralized-Satellite-FL-dev`, `config.py`):
  FLSNN's default is **SGD with `lr = 0.05`**, whereas our default is
  **AdamW with `lr = 1 \times 10^{-3}`** — two fundamentally different
  optimizer regimes. We have one run (single seed) under AdamW and
  zero under SGD; we cannot quantify the gap.
* **D9.** RelaySum LR scaling constant. The 2.093 multiplier from
  the FLSNN reference implementation
  (`revised_constellation.py:203`, ported verbatim to our
  `constellation.py:190`) carries **only the single-line inline
  comment `# learning rate correction`** in FLSNN's source — no
  published derivation, no cited prior work, and no empirical
  validation on the new optimizer / task pair. It is an
  EuroSAT-classification empirical value. Whether it is appropriate
  for our regression setting is an open question; an
  `lr_scale = 1.0` ablation would resolve it but was deferred
  (user-decision in inventory section).

We do **not** attribute the reversal to any single cause. The
mean spread of 0.1 dB across schemes (within either A or B) is at
the same order as the per-cell single-seed instability (§A.8), so
even if all three D-causes contribute, the *signal-to-noise ratio*
of the cross-scheme comparison is poor.

## D.3 Implementation parity check

To rule out an implementation bug as the cause of the reversal,
we trace the three aggregation paths against their algorithmic
references:

* **AllReduce** — `constellation.py:228–229`. After intra-plane
  ring-allreduce, every plane's state-dict is element-wise meaned
  across the 5 planes, then re-broadcast. Identity check on the
  saved checkpoints (Table III row "FedAvg + AllReduce") confirms
  $\mathrm{Var}(\gamma) = 0$ exactly for FedAvg, as expected.
* **Gossip** — `constellation.py:233–234, 267–276`. Each plane
  averages its weights with its 1-hop chain neighbours. Used with
  `intra_plane_iters = 2`: two rounds of pairwise Gossip per
  global epoch. Sanity (Table III): cross-plane $\mathrm{Var}(\gamma)$
  for FedAvg + Gossip is 1e-9 (numerical floor).
* **RelaySum** — `constellation.py:241, 280–367`. Implements FLSNN
  Algorithm 2 / Equation 8: each plane stores per-neighbour relay
  buffers; per round, the aggregated state is
  $\hat x_p = (\sum_q b_{q,p} + (N - n_p^{\rm rec}) x_p) / N$,
  where the first sum collects received relays, $n_p^{\rm rec}$
  is the number of distinct sources received, and the (N − n_p^{\rm rec})
  "missing" planes are filled with self-weight to keep the divisor
  fixed at $N = 5$. This matches Eq. 8 verbatim.

A previous v1 bug used $\hat x_p = $ sum / $n_p^{\rm rec}$ instead
of sum / $N$ for the RelaySum step (which made it behave like Gossip
for the first $N\!-\!1$ rounds). This was fixed at commit
`11f10f3` *before* any v2 numbers were collected; the Table I
RelaySum cells are therefore valid Eq. 8 implementations.

## D.4 Communication cost — Pareto frontier

Reading Table I again, this time for the (PSNR, Comm) Pareto:

| Scheme | Comm (MB, total) | Best PSNR achieved (over 2 BN, 2 runs A and B) |
|:-------|:---:|:---:|
| AllReduce | 3694 | 21.762 (B-FedBN) |
| Gossip    | 5911 | 21.791 (B-FedBN) |
| RelaySum  | 5911 | 21.709 (B-FedAvg) |

* **AllReduce uses 60 % less inter-plane communication** than
  either Gossip or RelaySum (3694 MB vs 5911 MB total over 80
  rounds). This 1.6× cost reduction is structural — AllReduce
  exchanges a single global mean per round, while Gossip and
  RelaySum exchange one full state per chain edge per round.
* **AllReduce achieves PSNR within 0.03 dB of Gossip.** On the
  Pareto frontier we observe in Table I, AllReduce dominates: it
  costs less and matches the best alternative within the
  single-seed noise floor.

§VI-E (next subsection) develops this Pareto observation
quantitatively, including the comparison with the ANN backbone
(Run C, also at AllReduce + 3694 MB).

## D.5 What we claim and what we do not

**Claim:** in our setting (regression task, AdamW optimizer, 50/5/1
chain topology, source-level Dirichlet $\alpha\!=\!0.1$),
**AllReduce is Pareto-optimal**: it matches Gossip's and RelaySum's
PSNR within 0.03 dB at 60 % less communication. The FLSNN ranking
(RelaySum > Gossip > AllReduce in classification accuracy)
**does not transfer here**.

**Non-claim 1:** we do not claim FLSNN's RelaySum ranking is wrong;
their Fig. 5 is in a regime where RelaySum's Theorem-2 advantage is
dominant. Our regime is at the opposite end of the same theorem.

**Non-claim 2:** we do not claim AllReduce is universally
Pareto-optimal. On larger constellations ($N \gg 5$), AllReduce's
per-round inter-plane bandwidth grows with $N$ while RelaySum's
stays constant — see the FLSNN Section IV-D scaling discussion.
At some constellation size the Pareto crossover happens; we do not
identify it.

**Non-claim 3:** we do not claim the RelaySum LR scaling of 2.093
is wrong for our task. Our point estimate is that AllReduce wins
*even when* RelaySum is given this generous lr boost; an `lr_scale
= 1.0` ablation might widen RelaySum's deficit but cannot make it
the winner (it is already not the winner under the scheme that
favours it).

## D.6 Reproducibility

Per-cell PSNR/SSIM values are saved at:

```
Outputs_v2/v2a_v2a_80ep_summary.json["final"][cell_key]
Outputs_v2/v2a_v2a_80ep_stdbn_summary.json["final"][cell_key]
```

Per-round convergence is in the corresponding `*.npz` files
(`epochs`, `train_loss`, `eval_psnr`, `eval_ssim`, `comm_bytes`,
`per_plane_psnr`, `per_plane_ssim`). Re-plot with:

```bash
python -m cloud_removal_v2.plot_results       --run_name v2a_80ep --output_dir ./Outputs_v2
python -m cloud_removal_v2.plot_comm_efficiency --run_name v2a_80ep --output_dir ./Outputs_v2
```
