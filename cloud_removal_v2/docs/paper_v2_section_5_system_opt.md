# §V. Inter-Plane Topology Optimisation

Theorem 1 (§IV.B) shows that the convergence bound depends on
the topology through $(\rho, \tilde\tau)$, with $\tilde\tau$
entering $T_2, T_3, T_4$ via $\sqrt{\tilde\tau}$ or $\tilde\tau$.
Given a real constellation whose inter-plane ISL graph is
determined by orbital geometry, one can pose the system-level
question: *which spanning tree on the feasible inter-plane
graph minimises the learning delay $\tilde\tau$?*

This section inherits the problem formulation and algorithmic
solution from FLSNN §V [Yang 2025] with v2-specific adaptations.
As disclosed in §VI-H.5, our v2 experiments use the fixed 5-plane
chain (matching FLSNN Fig. 5 for comparability) and **do not
empirically re-run the MDST topology search**; this subsection
documents the inherited formulation for completeness and flags
the MDST experiment as a v3 work item (see also §VI-H.3).

## V.A  Inter-plane connectivity graph

An inter-plane inter-satellite link between satellites
$u \in \mathrm{Plane}_i$ and $v \in \mathrm{Plane}_j$ is
**eligible** if two physical conditions hold simultaneously:

* **Line-of-sight.** The interplanetary distance satisfies
  $d_{u,v} \le d_{u,v}^{\ast}$, where $d_{u,v}$ is the
  straight-line distance given by the spherical-cosine formula
  in [Yang 2025, Eq. (28)–(29)] and $d_{u,v}^{\ast}$ is the
  maximum slant range before Earth occlusion
  [Yang 2025, Eq. (30)].
* **Doppler tolerance.** The carrier-frequency offset
  $f_{u,v} = \psi_{u,v} f_c / c$ induced by the relative
  velocity $\psi_{u,v}$ does not exceed the maximum tolerable
  offset $f_{\max}$; i.e., $f_{u,v} \le f_{\max}$. Beyond
  $f_{\max}$ no compensation technique keeps the link
  reliable [Yang 2025, Eq. (32)].

**Eligible pair set.** $\mathcal{I}_{i,j}$ denotes the set of
eligible $(u, v)$ pairs between planes $i$ and $j$; two planes
are **connected** in the inter-plane graph iff
$\mathcal{I}_{i,j} \ne \emptyset$ at the relevant timestamp.

**Inter-plane connectivity graph.**
$G = (\mathcal{V}, \mathcal{E})$ with $\mathcal{V} = [N]$ and
$e_{i,j} \in \mathcal{E}$ iff $\mathcal{I}_{i,j} \ne \emptyset$.

## V.B  Problem statement — MDST

Fixing a timestamp and the resulting $G$, we seek a spanning
tree $T = (\mathcal{V}, \tilde{\mathcal{E}})$ with
$\tilde{\mathcal{E}} \subseteq \mathcal{E}$ that **minimises
the tree diameter**:

$$
\mathrm{P1:} \quad
\mathrm{minimise}_{T} \;\; \tilde\tau(T)
\;\;=\;\; \mathrm{diam}(T) + 1.
$$

Since $\tilde\tau$ enters the Theorem-1 bound
multiplicatively, a smaller diameter tightens the convergence
bound directly.

**Tiebreak by link quality.** Multiple trees may attain the
same minimum diameter. Following [Yang 2025 §V-B], we break
ties by favouring trees with higher average SNR on their
chosen edges. Specifically, for an eligible pair $(u, v)$ the
SNR under the free-space path-loss model is
$\mathrm{SNR}_{u,v} = \mathrm{EIRP}_G / (\kappa \varrho B L_{u,v})$,
where $L_{u,v} = (4\pi d_{u,v} f_c / c)^2$ is the path loss,
$\kappa$ is Boltzmann's constant, $\varrho$ the thermal noise
factor, $B$ the system bandwidth, and $\mathrm{EIRP}_G$ the
transmitter-antenna effective isotropic radiated power × gain.
The inter-plane average SNR between planes $i$ and $j$ is
$\xi_{i,j} = |\mathcal{I}_{i,j}|^{-1}
\sum_{(u,v) \in \mathcal{I}_{i,j}} \mathrm{SNR}_{u,v}$.

Assigning edge weights
$w(e_{i,j}) = \tilde\xi + 1/\xi_{i,j}$ with
$\tilde\xi = \sum_{e_{i',j'} \in \mathcal{E}} 1/\xi_{i',j'}$
reformulates the problem as a weighted MDST that returns the
best-SNR tree among all minimum-diameter trees (the
$\tilde\xi$ term dominates any $1/\xi$ term for a single edge,
preserving the diameter primary objective;
[Yang 2025, Proposition 1]).

## V.C  Algorithm — A1CP-based MDST

MDST is NP-hard in general on edge-weighted graphs, but
[Hassin & Tamir 1995] showed it is equivalent to the absolute
1-centre problem (A1CP), which admits a polynomial-time
algorithm. FLSNN's Algorithm 3 adapts A1CP to the inter-plane
graph:

1. **Edge weighting.** For each eligible pair $(e_{i,j})$,
   compute the communication-quality weight per §V.B.
2. **Continuum of centre points.** Construct $\mathcal{A}(G)$,
   the set of all points on the edges of $G$ (including vertex
   midpoints and weighted interior points).
3. **Shortest-path tree per centre candidate.** For each
   candidate $i \in \mathcal{A}(G)$, compute the shortest-path
   tree $T_i$ connecting $i$ to every $j \in \mathcal{V}$ via
   Dijkstra-like expansion. The tree's radius is
   $C(i) = \max_{j \in \mathcal{V}} d_G(i, j)$.
4. **Optimum selection.** Return $T_{i^\ast}$ where
   $i^\ast = \arg\min_{i \in \mathcal{A}(G)} C(i)$.

This runs in $O(|\mathcal{V}|^3 + |\mathcal{E}| \log |\mathcal{E}|)$
time on our 5-plane chain (trivial). Larger constellations
(42/7/1 Walker-Delta used in FLSNN Fig. 9) have non-trivial
$\mathcal{A}(G)$, where the algorithm's polynomial complexity
matters.

## V.D  v2 implementation and deferral to v3

**v2 configuration.** The 5-plane chain has a **unique**
spanning tree (the chain itself), so MDST is trivially the
chain with $\mathrm{diam} = 4, \tilde\tau = 5$. There is no
optimisation gain to report; §VI uses this chain throughout.

**Why we do not run the full MDST experiment in v2.** A
non-trivial MDST benchmark requires a larger constellation
with multiple feasible spanning trees. FLSNN Fig. 10 uses the
42/7/1 Walker-Delta constellation to show MDST's advantage
over a naive chain: MDST reduces diameter from 7 (naive chain)
to 3. Reproducing this requires (i) STK-simulated orbital
geometry for 42/7/1 and (ii) a full 80-round FL training on
the optimised tree, at an estimated 168 GPU hours. §VI-H.3
lists this as v3 work item "MDST".

**What §V of this paper does claim.** We inherit the FLSNN
problem statement, edge-weighting, and algorithm, and
document them for the reader whose implementation may have a
larger constellation. We do not claim a new optimisation
algorithm.

**What §V does *not* claim.** We do not empirically validate
the chain-vs-MDST diameter advantage in v2. Readers wanting
the 42/7/1 Walker-Delta experiment are referred to FLSNN
Fig. 9–11 and its associated discussion.

## V.E  Summary

The inter-plane topology determines the learning-delay
parameter $\tilde\tau$ in Theorem 1. For real constellations
where the set of feasible inter-plane links is non-trivial,
FLSNN's §V provides an A1CP-based MDST algorithm that
minimises $\tilde\tau$ with SNR-weighted tie-breaking. Our
v2 implementation uses the fixed 5-plane chain (in which
MDST is trivially the chain); §VI-H.3 explicitly flags the
larger-constellation MDST experiment as v3 work.

This section exists to preserve v2's self-containedness: a
reader may invoke our paper's theoretical and experimental
results on a different constellation, and §V gives the
topology-optimisation recipe needed to do so without
re-reading FLSNN.
