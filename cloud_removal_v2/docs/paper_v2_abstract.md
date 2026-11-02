# v2 Paper — Abstract

This file contains both a **Markdown working draft** and the
**LaTeX-ready version** of the Abstract. The Markdown version
is the canonical source; the LaTeX version is its literal
translation for `paper_v2_main.tex`.

Word count target: ~200 words (typical IEEE Network / IEEE TSP
abstract budget).

---

## Markdown working draft (197 words)

We extend the FLSNN brain-inspired decentralised satellite
federated-learning framework from on-board image classification
to the more demanding setting of pixel-level cloud removal under
source-level Dirichlet non-IID. Inheriting the 50/5/1 Walker-Star
constellation, intra-plane ring-AllReduce, and the inter-plane
RelaySum / Gossip / AllReduce comparison verbatim, we contribute
two theoretical and three empirical results. Theoretically, we
derive a closed-form upper bound on the inter-plane gradient
dissimilarity $\zeta^2$ for an $S$-source Dirichlet partition
(Proposition 1) and show that, in the resulting small-$\zeta^2$
regime, the three aggregation schemes' Theorem-2 bounds collapse
to the same asymptotic order (Corollary 1). Empirically, on the
CUHK-CR benchmark with $\alpha = 0.1$, AllReduce Pareto-dominates
Gossip and RelaySum at 60 % less inter-plane communication, an
ANN backbone beats SNN by $+0.75$ dB PSNR / $1.61\times$ wall
time on a single GPU, and FedBN-style BN-local aggregation
yields gains within the single-seed noise floor regardless of
BN variant — TDBN reduces cross-plane $\mathrm{Var}(\gamma)$ by
42 % vs standard BN, but neither variant produces drift large
enough for FedBN to repair. The five findings are mutually
consistent with FLSNN's Theorem 2 in the small-$\zeta^2$
boundary regime.

## LaTeX-ready version

```latex
\begin{abstract}
We extend the FLSNN brain-inspired decentralised satellite
federated-learning framework from on-board image classification
to the more demanding setting of pixel-level cloud removal
under source-level Dirichlet non-IID. Inheriting the 50/5/1
Walker-Star constellation, intra-plane ring-AllReduce, and the
inter-plane RelaySum / Gossip / AllReduce comparison verbatim,
we contribute two theoretical and three empirical results.
Theoretically, we derive a closed-form upper bound on the
inter-plane gradient dissimilarity $\zeta^2$ for an $S$-source
Dirichlet partition (Proposition~1) and show that, in the
resulting small-$\zeta^2$ regime, the three aggregation schemes'
Theorem-2 bounds collapse to the same asymptotic order
(Corollary~1). Empirically, on the CUHK-CR benchmark with
$\alpha = 0.1$, AllReduce Pareto-dominates Gossip and RelaySum
at $60\,\%$ less inter-plane communication, an ANN backbone
beats SNN by $+0.75$\,dB PSNR / $1.61\times$ wall time on a
single GPU, and FedBN-style BN-local aggregation yields gains
within the single-seed noise floor regardless of BN variant
--- TDBN reduces cross-plane $\mathrm{Var}(\gamma)$ by
$42\,\%$ vs standard BN, but neither variant produces drift
large enough for FedBN to repair. The five findings are
mutually consistent with FLSNN's Theorem~2 in the
small-$\zeta^2$ boundary regime.
\end{abstract}
```

## Index Terms

```latex
\begin{IEEEkeywords}
Spiking neural networks, satellite federated learning, cloud
removal, Dirichlet non-IID, batch normalisation, decentralised
optimisation.
\end{IEEEkeywords}
```

## Provenance and consistency check

Every numerical claim in the Abstract is verbatim from a §VI
table or §IV proposition:

| Claim                              | Source                |
|:-----------------------------------|:----------------------|
| 50/5/1 Walker-Star constellation   | §III.A, §VI-A.2       |
| $\zeta^2$ closed form for $S$ sources, Proposition 1 | §IV.C |
| Corollary 1 (scheme collapse)      | §IV.D                 |
| CUHK-CR + $\alpha = 0.1$           | §VI-A.1, §VI-B.1      |
| AllReduce Pareto-dominance         | §VI-D.4 + Table I     |
| $60\%$ less communication          | §III.E formula + Table I |
| ANN $+0.75$ dB / $1.61\times$      | §VI-E + Table IV      |
| FedBN gain within noise            | §VI-C + Table II      |
| TDBN reduces $\mathrm{Var}(\gamma)$ by $42\%$ | §VI-C.2 + Table III |
| Five findings consistent with FLSNN Theorem 2 | §IV.D / §VII |

Five findings as enumerated in §VII match the abstract's
"three empirical + two theoretical" decomposition:

* Theory: Proposition 1 + Corollary 1 → Abstract sentence 3
* Empirical 1 (AllReduce Pareto): §VI-D → Abstract sentence 4a
* Empirical 2 (ANN +0.75 dB): §VI-E → Abstract sentence 4b
* Empirical 3 (FedBN no-gain + TDBN drift reduction):
  §VI-C → Abstract sentences 4c + 5
