# §VII. Conclusion

This paper extended the FLSNN [Yang 2025] decentralised
satellite-SNN FL framework from classification to
pixel-regression cloud removal, under a source-level Dirichlet
partition and with controlled TDBN-vs-BN2d / SNN-vs-ANN
ablations. Five quantitative findings resulted:

1. **Theoretical (§IV).** Proposition 1 gives the closed-form
   $\zeta^2 \le c_\alpha \|\nabla f_1 - \nabla f_2\|^2$
   with $c_\alpha = (N-1) / (4N(2\alpha + 1))$ for an
   $S$-source Dirichlet partition. At $S = 2, N = 50, \alpha = 0.1$
   we have $c_\alpha \approx 0.204$. Corollary 1 shows that in
   this small-$\zeta^2$ regime the three inter-plane aggregation
   schemes' Theorem-1 bounds collapse to the same asymptotic
   order, formally explaining our empirical near-tie between
   AllReduce, Gossip, and RelaySum.

2. **FedBN redundancy (§VI-C, Claim C16).** Across six
   (BN-variant × aggregation-scheme) cells, the FedBN-vs-FedAvg
   PSNR gain is $+0.008$ dB (TDBN) and $+0.044$ dB (BN2d) on
   average — both below the single-seed noise floor. The
   mechanism (§VI-G, Table III) is that TDBN's
   $\alpha V_{\rm th}$-shared $\gamma_{\rm init}$ reduces
   cross-plane $\mathrm{Var}(\gamma)$ to 58% of BN2d's, but
   even the larger drift is too small to hurt FedAvg on this
   regression task. FedBN is *conditionally redundant* in the
   source-level + pixel-regression + small-$\zeta^2$ regime.

3. **AllReduce Pareto-dominance (§VI-D, §VI-E).** AllReduce
   matches the best Gossip / RelaySum PSNR within 0.03 dB
   while using 60% less inter-plane communication. The
   FLSNN Fig. 5 ranking of RelaySum $>$ Gossip $>$ AllReduce
   (under label-shift classification) does not transfer; this
   is a boundary case of FLSNN's own Theorem 2 in the
   small-$\zeta^2$ regime.

4. **ANN-vs-SNN controlled ablation (§VI-E, Table IV).** On a
   single RTX 4090, the ANN backbone wins unambiguously:
   $+0.75$ dB PSNR, $+0.027$ SSIM, $1.61 \times$ wall-time.
   The 1.61 × is a *lower bound* on the architectural gap
   because our ANN implementation retains the $T = 4$ outer
   loop; a $T = 1$ re-implementation would recover ~2.5×
   more. The SNN's deployment-side energy advantage is
   explicitly *not measured* here — `MultiSpike4` 5-level
   quantisation invalidates the binary-spike 0.9 pJ/AC bound,
   and a measurement-grade estimate is deferred to v3
   (§VI-H.3 item E1).

5. **Honest infrastructure (§VI-A.7, tables/README).** Every
   numerical cell in Tables I–IV and every figure is
   reproducible from three shell commands acting on the 60
   saved plane checkpoints plus the three `summary.json` files.
   Every single-seed / single-$\alpha$ / single-partition
   limitation is enumerated in the ten-item L-ledger of
   §VI-H.2.

## VII.A  Limitations and future work

We have disclosed ten specific v2 limitations (§VI-H.2). The
three most influential ones point directly at v3 work:

* **Multi-seed reproduction (L1, V9).** Single-seed point
  estimates cannot distinguish 0.05-dB PSNR differences
  within the converged band. The v3 3-seed sweep (~228 GPU
  hours) will produce mean ± std for every Table I cell.
* **$\alpha$-sweep (L2).** Proposition 1 predicts a linear
  $c_\alpha$ decrease from $\alpha = 0.1 \to 1 \to 10$
  (0.204 → 0.082 → 0.012). An empirical sweep at three
  $\alpha$ values (~75 GPU hours) would trace the
  scheme-hierarchy transition from "collapsed" to "RelaySum
  leads".
* **SNN energy measurement (L8, E1).** A `MultiSpike4`-aware
  spike-rate hook on `mem_update` plus per-layer FLOPs for
  VLIFNet (2.31 M parameters) would produce the measured
  ANN-vs-SNN energy ratio that the present paper defers.

Beyond v3, the most natural v4 direction is **multi-temporal
cloud removal**: §VI-F demonstrates that single-image
restoration fails uniformly on thick cloud (samples 3 and 6,
17–20 dB), which is information-theoretically bounded. A
multi-temporal FL formulation where satellites fuse past
passes of the same scene would break this ceiling.

## VII.B  Closing remark

The experimental results of this paper fit under a single
sentence:

> In satellite FL on a pixel-regression task with source-level
> Dirichlet non-IID, the inter-plane aggregation scheme,
> BN-aggregation policy, and BN-variant choice each matter at
> most 0.3 dB — far less than the $\sim 0.75$ dB backbone
> effect, and well inside the single-seed noise floor.

This is consistent with — and in one new case (Corollary 1),
predicted by — FLSNN's own convergence theory. The practical
recommendation is therefore simple: for satellite
pixel-regression FL in the small-$\zeta^2$ regime, pick
AllReduce with either BN variant and the ANN backbone; reserve
the SNN backbone for deployment-time inference, where its
energy profile remains the compelling motivation that first
brought us to this problem.
