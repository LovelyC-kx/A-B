# §III. System Model and Algorithm

This section specifies the communication / computation
abstraction (§III.A–B), the federated-learning problem under
source-level Dirichlet non-IID (§III.C), and the
brain-inspired decentralised learning (BIDL) algorithm whose
convergence we analyse in §IV and evaluate in §VI (§III.D–E).
Specific implementation constants — dataset, model dimension,
hyperparameters — are deferred to §VI-A.

## III.A  LEO constellation

We consider a Walker-Star constellation with $N$ orbital planes,
each containing $K$ evenly-spaced satellites. The total satellite
count is $NK$. We write $(i, k)$ for the $k$-th satellite of the
$i$-th plane, $i \in [N], k \in [K]$. Our implementation uses
$N = 5, K = 10$ (the 50/5/1 configuration used in FLSNN Fig. 5
[Yang 2025]); the theoretical results of §IV apply for general
$(N, K)$.

**Communication topology.** Two link classes exist:

* **Intra-plane ISLs.** Satellites in the same plane move with
  near-constant relative spacing, so the $K$ satellites of plane
  $i$ form a stable, symmetric ring. We use ring-AllReduce
  [Patarasuk & Yuan 2009] for intra-plane consensus; it is
  bandwidth-optimal for the ring topology.
* **Inter-plane ISLs.** Relative motion across planes is
  non-negligible (Doppler shift, visibility windows), so we model
  inter-plane connectivity as a chain that links each plane to its
  two orbital neighbours. In a 5-plane chain the diameter is
  $\tau_{\max} = N - 1 = 4$, giving $\tilde\tau = \tau_{\max} + 1 = 5$
  — the delay parameter that will appear in §IV's Theorem 1.
  Extensions to optimised spanning-tree topologies are the
  subject of §V (minimum-diameter spanning tree, MDST).

**Ground-station exclusion.** All weight aggregation is performed
over ISLs; we do not use ground-station passes for aggregation.
This matches FLSNN's on-board-decentralised motivation and isolates
the ISL-only learning dynamics that §IV / §VI study.

## III.B  Federated data and per-satellite risks

Each satellite $(i, k)$ owns a local dataset $\mathcal{D}_{i,k}$
drawn from the union of two image sources (thin-cloud CR1 and
thick-cloud CR2; see §VI-A.1). The assignment is governed by a
source-level Dirichlet process parameterised by concentration
$\alpha > 0$:

$$
\mathbf{p}_{i,k} \;=\; (p_{i,k}, 1 - p_{i,k})
\;\sim\; \mathrm{Dir}(\alpha, \alpha),
\qquad (i, k) \in [N] \times [K],
$$

and $|\mathcal{D}_{i,k}|$ images are sampled with proportions
$\mathbf{p}_{i,k}$. A minimum-size constraint
$|\mathcal{D}_{i,k}| \ge n_{\min}$ prevents empty clients (§VI-B.4
reports the effect of this constraint on our implementation).

This partition scheme is the **source-level** analogue of FLSNN's
label-level Dirichlet partition on EuroSAT; §VI-B establishes that
it yields a strictly weaker non-IID than FLSNN's original setting,
a fact that controls the interpretation of §VI-D / §IV.D.

**Per-satellite empirical risk.**
$$
f_{i,k}(\theta) = \frac{1}{|\mathcal{D}_{i,k}|}
\sum_{(x, y) \in \mathcal{D}_{i,k}} \ell(\theta; x, y),
$$
where $\ell(\cdot)$ is a per-sample loss (Charbonnier + SSIM for
our cloud-removal regression; §VI-A.5).

**Plane and global risks.**
$$
f_i(\theta) = \frac{1}{K} \sum_{k=1}^K f_{i,k}(\theta),
\qquad
f(\theta) = \frac{1}{N} \sum_{i=1}^N f_i(\theta).
$$

**FL objective.** We minimise the *unconditional* global risk
$\min_\theta f(\theta)$ without sharing $\mathcal{D}_{i,k}$ across
satellites.

## III.C  On-board model: VLIFNet with configurable backbone / BN

Each satellite runs the same model architecture, a variant of
the VLIFNet U-Net used in our v1 work. Two design axes are
configurable *without retraining:*

* **BN variant** ∈ {`TDBN` [Zheng 2021], `BN2d` [Ioffe & Szegedy 2015]}
  controls which normalisation is inserted after each
  convolutional block. `TDBN`'s affine scale initialises to
  $\gamma_{\rm init} = \alpha_{\rm TDBN}\, V_{\rm th}$ (roughly
  $0.11$ for typical LIF thresholds; see §VI-A.4); `BN2d` uses
  the standard $\gamma_{\rm init} = 1$.
* **Backbone** ∈ {`SNN`, `ANN`} controls whether each LIF
  neuron is replaced by a stateless ReLU (`ANN`) or kept as the
  `mem_update` / `LIFNode` iterative integrator with
  `MultiSpike4` 5-level quantisation (`SNN`).

Both axes produce bit-identical parameter counts; the state-dict
is structurally equivalent across axes (verified in §VI-A). The
concrete layer counts, channel widths, and time steps
$T$ are listed in §VI-A.4 and are held fixed throughout this
paper's evaluation.

## III.D  Brain-inspired decentralised learning (BIDL) algorithm

Given the constellation and FL objective, one global round
$t \in [T_{\rm glob}]$ executes the following three steps:

**(S1) Local update.** Every satellite performs $E$ local
epochs (full passes over $\mathcal{D}_{i,k}$), each consisting
of $\lceil|\mathcal{D}_{i,k}|/B\rceil$ mini-batch SGD updates
of the form
$$
\theta_{i,k}^{s+1} = \theta_{i,k}^{s} - \eta \nabla F_{i,k}(\theta_{i,k}^{s}),
$$
where $s$ indexes individual SGD steps within the $E$ epochs,
$B$ is the mini-batch size, $\nabla F_{i,k}$ is a stochastic
estimate of $\nabla f_{i,k}$ on the current minibatch, and
$\eta$ is the learning rate. The total SGD-step count per
intra-plane iteration is thus
$E \cdot \lceil|\mathcal{D}_{i,k}|/B\rceil$.

**(S2) Intra-plane ring-AllReduce.** After $R$ intra-plane
iterations (each consisting of $E$ local steps followed by a
ring-AllReduce), all $K$ satellites of plane $i$ share the
plane's mean weights
$$
\theta_i^{t+1/2} = \frac{1}{K} \sum_{k=1}^K \theta_{i,k}^{t, E-1}.
$$

**(S3) Inter-plane aggregation.** An inter-plane aggregation
scheme $\mathsf{A} \in \{\mathrm{AllReduce},\mathrm{Gossip},\mathrm{RelaySum}\}$
produces the next global iterate $\theta_i^{t+1}$ from
$\{\theta_j^{t+1/2}\}_{j=1}^N$. The three choices implement
different speed / accuracy trade-offs:

* **AllReduce.** A global mean
  $\theta_i^{t+1} = \tfrac{1}{N}\sum_j \theta_j^{t+1/2}$ is
  broadcast to every plane each round. Exact consensus in one
  round, but each round costs $N$ copies of the state dict sent
  over the chain topology (bandwidth concerns for large $N$).
* **Gossip.** Each plane averages its state with that of its
  one-hop chain neighbours:
  $\theta_i^{t+1} = \frac{1}{|\mathcal{N}(i) \cup \{i\}|}
  \sum_{j \in \mathcal{N}(i) \cup \{i\}} \theta_j^{t+1/2}$.
  Single-hop per round; consensus takes $O(\mathrm{diam}(G))$
  rounds.
* **RelaySum.** Each plane stores per-neighbour relay buffers;
  per round it forwards every received buffer (minus the target)
  plus its own update. The aggregated state is
  $\theta_i^{t+1} = (\sum_{q \in \mathcal{N}(i)} b_{q,i}
  + (N - n_i^{\rm rec}) \theta_i^{t+1/2}) / N$,
  where the "missing" $(N - n_i^{\rm rec})$ planes are filled
  with $\theta_i^{t+1/2}$ so the divisor stays $N$. This matches
  the original [Vogels 2021] formulation; a previous
  implementation bug in v1 (using $n_i^{\rm rec}$ instead of $N$
  as divisor) was fixed in commit `11f10f3` before v2 numbers
  were collected.

**Algorithm summary (pseudocode).**

```
for t = 0, ..., T_glob - 1:
    for r = 0, ..., R - 1:                          # intra-plane rounds
        for each satellite (i, k):
            run E local epochs on D_{i,k} (S1)
            # each epoch iterates over all minibatches of D_{i,k} once
        for each plane i:
            run ring-AllReduce across K satellites  (S2)
    for each plane i:
        run inter-plane aggregation scheme A        (S3)
```

At the end of $T_{\rm glob}$ rounds, the plane models
$\{\theta_i^{T_{\rm glob}}\}_{i=1}^N$ are used for inference.
§VI-A.6 specifies the ensemble-per-image PSNR metric we use to
evaluate the consensus of the five plane models on the held-out
test set.

## III.E  Communication cost accounting

Let $P_\theta$ denote the parameter byte count (4 bytes per
`float32` parameter). One round of each scheme costs:

| Scheme    | Bytes per round                                             |
|:----------|:-----------------------------------------------------------|
| AllReduce | $N \cdot P_\theta$ (each plane broadcasts once)             |
| Gossip    | $\sum_i \lvert\mathcal{N}(i)\rvert \cdot P_\theta \approx 2(N-1) P_\theta$ (chain topology) |
| RelaySum  | $\sum_i \lvert\mathcal{N}(i)\rvert \cdot P_\theta \approx 2(N-1) P_\theta$ |

For our 5-plane chain, Gossip and RelaySum require exchanging
$2(N-1) = 8$ copies of $P_\theta$ per round (one per chain edge,
two directions); AllReduce requires $N = 5$ copies (one broadcast
per plane). The exact per-round byte counts on our VLIFNet model
are reported in §VI-A and Table I (aggregated over $T_{\rm glob}$
rounds).

We intentionally count bytes rather than rounds: on the chain
topology, all three schemes complete one global round in one
communication cycle, so the "rounds-to-consensus" metric
traditionally used in decentralised-SGD literature is
inappropriate here (see §VI-D.4 for the Pareto-frontier
discussion).

## III.F  Summary of notation

| Symbol          | Meaning                                      |
|:----------------|:---------------------------------------------|
| $N, K$          | Planes and satellites-per-plane              |
| $(i, k)$        | Satellite index                              |
| $\mathcal{D}_{i,k}$ | Local dataset of satellite $(i, k)$      |
| $f_{i,k}, f_i, f$ | Satellite, plane, and global risks         |
| $\theta_{i,k}^t$ | Satellite weight at start of round $t$      |
| $\theta_i^{t+1/2}$ | Plane weight after intra-plane step        |
| $\alpha$        | Dirichlet concentration parameter            |
| $T_{\rm glob}$  | Number of global rounds                      |
| $R$             | Intra-plane iterations per global round      |
| $E$             | Local epochs per intra-plane iteration (each epoch = full pass over $\mathcal{D}_{i,k}$) |
| $B$             | Mini-batch size                              |
| $\eta$          | Learning rate                                |
| $\mathsf{A}$    | Inter-plane aggregation scheme               |
| $\mathcal{N}(i)$ | Chain neighbours of plane $i$               |
| $\tau_{\max}, \tilde\tau$ | Chain diameter and delay parameter  |
| $\rho, q, m$    | Mixing-matrix spectral parameters (§IV.A)    |
| $\delta^2, \zeta^2$ | Intra- / inter-plane gradient dissimilarity |
| $P_\theta$      | Parameter byte count (4 × #params)           |

This notation is used throughout §IV (convergence theorem),
§V (topology optimisation), and §VI (experiments).
