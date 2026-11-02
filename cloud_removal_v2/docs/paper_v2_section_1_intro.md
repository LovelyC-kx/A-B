# §I. Introduction

## I.A  Motivation — Space Computing Power Networks and the energy bottleneck

Satellite constellations are the canonical edge-computing
platform of the decade: Earth observation, environmental
monitoring, and disaster-response applications all want
sub-minute end-to-end latency on raw-pixel products that, in a
ground-processing architecture, are bottlenecked by the
narrow-band satellite-to-ground downlink. Space Computing Power
Networks (Space-CPN) push the computation on-board by treating
each orbital plane as a mesh of compute nodes coupled by
inter-satellite laser links (ISLs).

On-board compute is, however, energy-constrained. LEO
satellites run on solar panels charged by sunlit arcs and on
batteries during eclipse; dense artificial-neural-network (ANN)
inference and training on-board compete directly with
communication, attitude control, and payload operation for a
shared power budget. Every joule a satellite spends on matrix
multiplications is a joule not spent on data-taking, downlink,
or battery-cycle lifetime.

## I.B  Why spiking neural networks

Spiking neural networks (SNN) process information as sparse
binary or multi-level *events* rather than dense floating-point
activations. A typical SNN layer fires on only 10–50% of its
neurons per forward pass; every silent neuron draws no dynamic
power. On 45-nm CMOS [Horowitz 2014] the per-operation energy
gap is dramatic — $4.6$ pJ/MAC for ANN vs. $0.9$ pJ/AC for
binary SNN — making SNN a natural fit for satellite inference.
The **FLSNN** work of [Yang 2025] brought SNN into satellite
federated learning, showing that a Spiking-CNN / Spiking-ResNet
backbone trained across planes via decentralised ISL
aggregation reaches accuracy within ~3 % of an ANN baseline
at roughly 10× lower per-layer energy on EuroSAT land-cover
classification (FLSNN Fig. 6 + Fig. 7).

## I.C  Why decentralised federated learning

Training neural networks *on-orbit* without shipping raw data
back to ground is the only path to meeting the short-visibility
satellite-ground window. Satellite FL has matured through three
generations: ground-PS-assisted [Matthiesen 2023], asynchronous
[Elmahallawy 2022], and fully decentralised
[Yang 2025, arXiv 2501.15995]. The last is the architecture
this paper inherits: every satellite trains locally, intra-plane
ring-AllReduce synchronises the 10 satellites within each
orbital plane, and inter-plane aggregation reconciles the $N$
planes via ISLs only.

The inter-plane aggregation step is non-trivial: orbital
dynamics produce a sparse, intermittent inter-plane ISL graph
whose diameter grows with $N$. Three decentralised-SGD schemes
are commonly used — AllReduce, Gossip, and RelaySum — each with
different $(\rho, \tilde\tau)$ topology constants (§III, §IV).
FLSNN [Yang 2025] empirically recommends RelaySum for
classification under strong label-shift non-IID
($\varsigma = 0.02$ Dirichlet over 10 labels), leaning on
RelaySum's delayed-but-complete relay-buffer mechanism
[Vogels 2021].

## I.D  The problem this paper addresses

FLSNN leaves three gaps that limit its applicability to the
most natural satellite FL task — **pixel-level image
regression** (cloud removal, super-resolution, denoising) — on
realistic federated partitions:

* **Task-type gap.** FLSNN only evaluates classification. Its
  RelaySum-leads scheme ranking may or may not hold on dense
  regression losses (Charbonnier + SSIM), which have different
  gradient-variance structure.
* **Non-IID gap.** FLSNN uses label-level Dirichlet. Satellite
  data naturally has *source-level* heterogeneity (e.g.,
  different sensors, different cloud thicknesses), which is
  structurally different from label-shift and produces smaller
  inter-plane gradient dissimilarity $\zeta^2$ (§IV.C).
* **Backbone / normalisation gap.** FLSNN treats TDBN
  [Zheng 2021] as an interchangeable building block and does
  not ablate against standard `BN2d`, or against the FedBN
  [Li 2021] BN-local FL aggregation. Whether TDBN's
  $\alpha V_{\rm th}$-shared scaling makes FedBN redundant in
  decentralised SNN training is an open question.

## I.E  Our approach and contributions

We extend FLSNN to the cloud-removal pixel-regression task on
the CUHK-CR benchmark [Sui 2024], keeping the 50/5/1 Walker-Star
constellation and RelaySum-aware convergence theorem verbatim
(§III, §IV.B) so that our results isolate the task-type,
non-IID, and BN/backbone effects. Concretely, our
contributions are:

1. **Proposition 1 (§IV.C).** A closed-form characterisation of
   the inter-plane gradient dissimilarity $\zeta^2$ under a
   source-level Dirichlet($\alpha, \alpha$) partition over
   $S$ sources. For $S = 2, N = 50, \alpha = 0.1$ (our v2
   setting), $\zeta^2 \le 0.204 \cdot \|\nabla f_1 - \nabla f_2\|^2$.
2. **Corollary 1 (§IV.D).** In the small-$\zeta^2$ regime of
   Proposition 1, the AllReduce / Gossip / RelaySum Theorem-1
   bounds collapse to the same asymptotic order, predicting
   near-identical PSNR performance across schemes. Our empirical
   measurements (§VI-D) place the PSNR-spread at 0.1–0.3 dB —
   consistent with Corollary 1.
3. **Claim C16 (§VI-C, main empirical novelty).** A controlled
   TDBN-vs-BN2d ablation across FedAvg and FedBN inter-plane
   aggregation shows that the FedBN PSNR gain is below the
   single-seed noise floor (mean $\Delta = +0.008$ dB with
   TDBN, $+0.044$ dB with BN2d) for **both** BN variants. The
   mechanism layer (§VI-G) confirms that TDBN's
   $\alpha V_{\rm th}$-shared scaling reduces cross-plane
   $\mathrm{Var}(\gamma)$ by 42% relative to standard BN2d, but
   the resulting drift is already too small for FedBN to
   repair — FedBN is conditionally redundant in this regime.
4. **ANN-vs-SNN controlled ablation (§VI-E).** A matched
   FedBN+AllReduce cell shows ANN beats SNN by $+0.75$ dB PSNR
   and $+0.027$ SSIM at $1.61\times$ wall-clock speedup on a
   single RTX 4090. The SNN energy advantage of [Yang 2025]
   is not rejected; instead, we disclose explicitly that the
   `MultiSpike4` 5-level quantisation used in our VLIFNet
   invalidates the $0.9$ pJ/AC binary-spike cost bound, and
   defer a measurement-grade SNN energy estimate to v3.
5. **Honest infrastructure.** We audit and publish all 4 main
   tables + per-plane drift + qualitative panels + single-seed
   caveat, enabling reviewers to re-derive every numeric cell
   from three shell commands on the 60 plane checkpoints
   (§VI-A.7, `docs/tables/README.md`).

## I.F  Non-claims (explicit scope boundary)

We do *not* refute FLSNN's RelaySum-leads result on
classification. Corollary 1 explains the apparent
"disagreement" as a boundary case of FLSNN's own Theorem 2 in
the small-$\zeta^2$ regime. We do not claim a new RelaySum
algorithm, a new TDBN algorithm, or a new FedBN-style FL
aggregation; each is inherited verbatim. We do not report an
SNN-vs-ANN energy ratio for our VLIFNet (reason as above). We
do not run the MDST system-optimisation experiment on a larger
constellation (§V.D, §VI-H) — the 5-plane chain used
throughout §VI has a trivial MDST (itself). We report a
single-seed / single-$\alpha$ / single-partition result and
explicitly flag this as a v2 scope boundary (§VI-H.2 L1–L10).

## I.G  Organisation of the paper

§II reviews related work along six literature axes:
satellite FL, the BN / FedBN / HarmoFL / FedWon family,
decentralised SGD + SAM, SNN and federated SNN, federated
image restoration, and 45-nm CMOS energy methodology. §III
defines the satellite constellation, the source-level
Dirichlet partition, and the three-step BIDL algorithm, with
communication-cost accounting. §IV states the inherited
Theorem 1 convergence bound, proves Proposition 1
(Dirichlet-to-$\zeta^2$ closed form), and draws
Corollary 1 (scheme-hierarchy collapse). §V documents the
MDST-based inter-plane topology optimisation inherited from
FLSNN §V, with the explicit v3 deferral for the
larger-constellation MDST experiment. §VI reports experiments
on CUHK-CR: setup (VI-A), non-IID partition diagnostics
(VI-B), FedBN ablation (VI-C, main claim), aggregation scheme
comparison (VI-D), ANN-vs-SNN backbone study (VI-E),
qualitative results (VI-F), per-plane drift and mechanism
(VI-G), and an exhaustive limitations list (VI-H). §VII
concludes.

All tables, scripts, plane checkpoints, and the raw `drift_report.md`
are published alongside the paper under the
`cloud_removal_v2/` repository branch listed in §VI-A.7.
