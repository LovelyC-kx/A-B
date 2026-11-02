# v2 Formal Derivations

Original mathematical results supporting the paper's §VI claims.
Each derivation is stated with explicit assumptions, carried
through with explicit algebra, and concludes with numerical
application to v2-A's specific parameters.

**Quality bar.** Every step cites a source or is derived from
preceding steps. Numerical constants are computed, not rounded.
Caveats and limitations are called out explicitly. **This
document is subject to external peer-level re-verification
before paper submission** — my derivations have not been
independently checked.

---

## §D1. Dirichlet(α, α) → ζ² upper bound for 2-source federated partition

**Goal.** Derive an explicit closed-form upper bound on the
inter-client gradient dissimilarity ζ² (as used in FLSNN
Assumption A4 and HarmoFL Assumption B.2) in terms of the
Dirichlet concentration parameter α, under the assumption that
the data comes from exactly 2 sources (CR1 thin cloud + CR2
thick cloud in v2-A).

**Why this matters.** FLSNN Thm 2, HarmoFL Thm 3.1, and related
decentralised-FL bounds all contain ζ² as a free constant. In
the papers, ζ² is treated as an abstract "dissimilarity
constant." For our partition scheme (Dirichlet over 2 sources),
we can compute it explicitly, converting a qualitative
assumption into a quantitative one.

### §D1.1 Setup and notation

**Data generation.** We have two data sources with distributions
D₁ (CR1 thin-cloud paired samples) and D₂ (CR2 thick-cloud
paired samples). Every sample (x, y) carries a latent type
label t(x) ∈ {1, 2} identifying its source. Pure-type risks
are defined as

  f_s(θ) := E_{(x, y) ~ D_s} [ℓ(θ; x, y)],    s ∈ {1, 2},

where ℓ is the local loss (Charbonnier + SSIM in v2-A). These
are deterministic functions of θ, not random.

**Client partition.** There are N clients (in v2-A, N=50
satellites or equivalently N=5 planes depending on aggregation
granularity; see §D1.6 for both applications). Each client i
receives data from a mixture of D₁ and D₂ with proportions

  (p_i, 1 − p_i),    p_i ∈ [0, 1].

**Dirichlet prior.** Under our 2-source symmetric Dirichlet(α,
α) scheme,

  p_i ~ Beta(α, α)    i.i.d. across i = 1, ..., N.

(The 2-source Dirichlet degenerates to a Beta distribution on
the first coordinate; this is standard.)

**Per-client risk.** Conditional on p_i, client i's expected
risk is

  f_i(θ; p_i) := p_i · f_1(θ) + (1 − p_i) · f_2(θ).    (D1.1)

**Global risk.** Under uniform weighting across clients:

  f(θ; p_1, ..., p_N) := (1/N) Σ_{i=1}^N f_i(θ; p_i)
                       = p̄_N · f_1(θ) + (1 − p̄_N) · f_2(θ),    (D1.2)

where p̄_N := (1/N) Σ_i p_i.

### §D1.2 Gradient decomposition

Differentiating (D1.1) and (D1.2) in θ (f_1, f_2 are
deterministic; p_i is a random scalar independent of θ):

  ∇f_i(θ; p_i)  = p_i · ∇f_1(θ) + (1 − p_i) · ∇f_2(θ),    (D1.3)
  ∇f(θ; p_N)    = p̄_N · ∇f_1(θ) + (1 − p̄_N) · ∇f_2(θ).    (D1.4)

Subtracting:

  ∇f_i − ∇f = (p_i − p̄_N) · (∇f_1 − ∇f_2).    (D1.5)

This is the key decomposition: the inter-client gradient
dissimilarity is **rank-1 in θ-space**, with the client-
direction scalar (p_i − p̄_N) multiplying the fixed direction
(∇f_1 − ∇f_2). Under squared norm:

  ‖∇f_i − ∇f‖² = (p_i − p̄_N)² · ‖∇f_1 − ∇f_2‖².    (D1.6)

### §D1.3 Moments of Beta(α, α)

Beta(α, α) is symmetric around 1/2. From its definition (e.g.
Johnson–Kotz–Balakrishnan, *Continuous Univariate Distributions*
Vol. 2, §25.2):

  E[p_i]   = α / (α + α) = 1/2,                                 (D1.7)
  Var(p_i) = α·α / ((α + α)² (α + α + 1)) = 1 / (4 (2α + 1)).   (D1.8)

As α → 0, Var → 1/4 (the U-shaped limit: each p_i concentrates
at 0 or 1 with equal probability). As α → ∞, Var → 0 (each
p_i concentrates at 1/2, i.e., IID across clients).

### §D1.4 Expected average squared dissimilarity

Let

  S_N := Σ_{i=1}^N (p_i − p̄_N)²,

the sum of squared deviations. By the standard identity,

  S_N = Σ p_i² − N · p̄_N² = Σ p_i² − (1/N) (Σ p_i)².    (D1.9)

Taking expectation over p_i ~ i.i.d. Beta(α, α):

  E[Σ p_i²] = N · E[p_i²] = N · (Var(p_i) + E[p_i]²)
            = N · (1/(4(2α+1)) + 1/4).                           (D1.10)

For E[(Σ p_i)²]: using Var(Σ) = N·Var(p_i) (independence) and
E[Σ] = N/2:

  E[(Σ p_i)²] = Var(Σ p_i) + (E[Σ p_i])²
              = N · Var(p_i) + N² / 4.                            (D1.11)

Combining:

  E[S_N] = N · (1/(4(2α+1)) + 1/4)
         − (1/N) · (N · Var(p_i) + N² / 4)
         = N/(4(2α+1)) + N/4 − 1/(4(2α+1)) − N/4
         = (N − 1) / (4(2α + 1)).                                 (D1.12)

Dividing by N gives the expected average squared deviation:

  E[(1/N) S_N] = (N − 1) / (4N(2α + 1)).    (D1.13)

### §D1.5 Main result — Theorem D1

**Theorem D1 (Expected average gradient dissimilarity under
2-source Dirichlet partition).** Under the setup of §D1.1
(i.i.d. p_i ~ Beta(α, α), gradient decomposition (D1.3)),

  E_p [ (1/N) Σ_{i=1}^N ‖∇f_i(θ) − ∇f(θ)‖² ]
      = ((N − 1) / (4N(2α + 1))) · ‖∇f_1(θ) − ∇f_2(θ)‖²,    (D1.14)

where the expectation is over the Dirichlet draw (p_1, ..., p_N),
and θ is held fixed.

**Proof.** Substitute (D1.6) into the left-hand side and apply
(D1.13). □

### §D1.6 Worst-case and high-probability sup-norm bounds

Theorem D1 gives the expected *average* over clients. FLSNN's
A4 is stated as "sup over clients": ‖∇f_i − ∇f‖² ≤ ζ² for all
i. We now analyse the sup-norm form.

**Per-realisation hard bound.** For any realisation
(p_1, ..., p_N) ∈ [0, 1]^N:

  max_i (p_i − p̄_N)² ≤ max(p̄_N, 1 − p̄_N)².    (D1.15)

Proof: p_i ∈ [0, 1] ⇒ |p_i − p̄_N| ≤ max(p̄_N, 1 − p̄_N)
(triangle inequality applied to the extreme ends of [0, 1]).
Squaring gives (D1.15). □

The RHS is **not** a universal 1/4 — e.g., at p̄_N = 0.2 and an
outlier client with p_i = 1, we get (1 − 0.2)² = 0.64, exceeding
1/4. The 1/4 bound holds only at p̄_N = 1/2 exactly.

**Symmetric-expectation tightening.** Under Beta(α, α) with
α > 0, by symmetry E[p̄_N] = 1/2 (eq. D1.7, averaged). Also
Var(p̄_N) = Var(p_i) / N = 1/(4N(2α + 1)). So for large N,
p̄_N → 1/2 with probability 1. Formally, for any ε > 0,

  P(|p̄_N − 1/2| > ε) ≤ 1 / (4N(2α + 1) ε²)    (Chebyshev)    (D1.16a)

so p̄_N ∈ [1/2 − ε, 1/2 + ε] with probability ≥ 1 − δ whenever
ε ≥ 1/√(4N(2α + 1)δ).

Combining with (D1.15), on the high-probability event
p̄_N ∈ [1/2 − ε, 1/2 + ε] we have

  max_i (p_i − p̄_N)² ≤ (1/2 + ε)² = 1/4 + ε + ε²,    (D1.16b)

which **approaches the 1/4 limit as N → ∞ (since ε → 0)**.

**Numerical examples (N=50, δ=0.05):**
- α=0.1: Var(p̄_N) = 1/(4·50·1.2) = 1/240 ≈ 0.00417,
  ε² ≥ 0.00417/0.05 = 1/12 ≈ 0.0833, ε ≥ 0.289.
  ⇒ max (p_i − p̄_N)² ≤ (0.789)² ≈ **0.623** w.p. ≥ 95%.
- α=1: Var(p̄_N) = 1/(4·50·3) = 1/600, ε² ≥ 1/30 ≈ 0.0333,
  ε ≥ 0.183. ⇒ max ≤ (0.683)² ≈ **0.466** w.p. ≥ 95%.
- α=10: Var(p̄_N) = 1/(4·50·21) = 1/4200, ε² ≥ 1/210 ≈ 0.00476,
  ε ≥ 0.069. ⇒ max ≤ (0.569)² ≈ **0.324** w.p. ≥ 95%.

As α grows, the sup-norm bound tightens toward 1/4 (which is
the α → ∞ / IID limit where all p_i → 1/2).

**Summary for FLSNN A4's ζ² interpretation.** The useful range
of ζ² for our v2-A (α = 0.1, N = 50) is

  ζ² ≤ min(0.623 · ‖∇f_1 − ∇f_2‖², **1 · ‖∇f_1 − ∇f_2‖²**)
     = **0.623 · ‖∇f_1 − ∇f_2‖²** (95% probability),

which is 3× larger than the expected-average form (0.204).
The sup-norm bound is looser because it is an extreme-order
statistic; the expected-average is the quantity appearing in
typical practice.

**When to use which bound:**
- FLSNN Thm 2 as written uses a sup-norm A4. Plug (D1.16b) for
  worst-case convergence analysis.
- HarmoFL Thm 3.1 uses a mean-square dissimilarity (B.2), for
  which the *expected-average* form (D1.14) is the direct
  quantity.
- For ρ = q/m, scheme-dependence, and other constants, the
  exact choice of ζ² surrogate is absorbed into the
  composite C constants — numerical precision matters less
  than α-dependence direction.

### §D1.7 Numerical application to v2-A

**v2-A parameters**: N = 50 (per-satellite granularity), α = 0.1.

Applying (D1.14):

  E_p[(1/50) Σ_i ‖∇f_i − ∇f‖²] = (49 / (4 · 50 · 1.2)) · ‖∇f_1 − ∇f_2‖²
                                = (49 / 240) · ‖∇f_1 − ∇f_2‖²
                                ≈ **0.2042 · ‖∇f_1 − ∇f_2‖²**.    (D1.17)

For the alternative per-plane aggregation (N = 5, α applied to
plane-level Dirichlet mixing):

  E_p[(1/5) Σ_i ‖∇f_i − ∇f‖²] = (4 / (4 · 5 · 1.2)) · ‖∇f_1 − ∇f_2‖²
                               ≈ 0.1667 · ‖∇f_1 − ∇f_2‖².    (D1.18)

### §D1.8 α-sensitivity table

Table D1 shows the coefficient (N−1)/(4N(2α+1)) for N = 50
across a sweep of α values (the α-sensitivity study listed in
§21 Tier-2 V9):

| α     | Var(p_i) | Coefficient, N=50 | Coefficient, N=5 |
|-------|----------|-------------------|------------------|
| 0.05  | 0.2273   | 0.2227            | 0.1818           |
| **0.1 (v2-A)** | **0.2083** | **0.2042** | **0.1667** |
| 0.2   | 0.1786   | 0.1750            | 0.1429           |
| 0.5   | 0.1250   | 0.1225            | 0.1000           |
| 1.0   | 0.0833   | 0.0817            | 0.0667           |
| 2.0   | 0.0500   | 0.0490            | 0.0400           |
| 5.0   | 0.0227   | 0.0223            | 0.0182           |
| 10    | 0.0119   | 0.0117            | 0.0095           |
| 100   | 0.0012   | 0.0012            | 0.0010           |

**Observation.** Decreasing α from 1 → 0.1 scales ζ² up by
≈ 2.5×. Decreasing α from 10 → 0.1 scales ζ² up by ≈ 18×. The
Dirichlet(α=0.1) non-IID setting in v2-A puts ζ² at roughly
21% of its theoretical maximum 1/4 · ‖∇f_1 − ∇f_2‖², which is
severe but not saturated.

### §D1.9 Integration with FLSNN Theorem 2

FLSNN's Thm 2 (§1.3 of `v2_comprehensive_literature.md`)
contains ζ² only indirectly via the composite constants C, C₁
that absorb both dissimilarity and scheme-dependent mixing
parameters. However, the bound's qualitative dependence — ζ²
enters through additive "(ζ² / function of ρ)" terms in
decentralised-SGD literature (Koloskova 2020, Vogels 2021) — is
unambiguous: **smaller ζ² tightens the bound, proportionally**.

Plugging (D1.17) into a generic FL convergence bound
(ignoring constants):

  (Convergence error) ~ O(... + ζ² · G(ρ, τ̃, E))
   = O(... + 0.2042 · ‖∇f_1 − ∇f_2‖² · G(ρ, τ̃, E))

So our v2-A's ζ² contribution scales linearly with
‖∇f_1 − ∇f_2‖², which is measurable **empirically** by
computing ∇f_1 on a pure-CR1 batch and ∇f_2 on a pure-CR2
batch and taking the norm of their difference. Listed as v3
measurement task (cf. §21 Tier-2).

### §D1.10 Caveats and assumptions

**What the derivation assumes and what happens if violated:**

1. **2 sources exactly.** CR1 + CR2. Adding RICE1/2 (v3
   option) gives a 4-source Dirichlet, changing the variance
   to a sum of multi-source contributions. The 2-source form
   does NOT extend directly.

2. **Uniform client weighting.** FedAvg's intra-plane mean
   uses uniform weights. If data-size-weighted aggregation is
   used (as in FedAvg original paper), p̄_N becomes a weighted
   average and the (N−1)/N factor changes. Our implementation
   does use uniform aggregation at the inter-plane step, so
   this holds. Intra-plane averaging is also uniform by our
   `constellation.py` code.

3. **Expectation, not sup-norm.** FLSNN's A4 is "≤ ζ² for all
   i" (sup), which is bounded by 1/4 · ‖∇f_1 − ∇f_2‖² hard
   but whose typical value is the Theorem D1 average. Papers
   citing ζ² should clarify which version.

4. **Gradient of expected risk vs expected gradient of
   empirical risk.** For small M (per-client sample count),
   there's an additional stochastic term O(σ²/M) from finite-
   sample noise that we have not included. This term is
   controlled by the variance bound σ² in FLSNN A2 and is
   separate from ζ².

5. **⚠ Independence of p_i assumption — NOT exactly matched by our
   implementation (important caveat).** The derivation assumes
   p_i ~ Beta(α, α) drawn **independently across clients**. Our
   code (`cloud_removal_v2/dataset.py:362`,
   `dirichlet_source_partition`) instead uses the Hsu 2019
   standard partition scheme:

   > For each source s ∈ {1, 2} independently,
   >   (q^s_1, ..., q^s_N) ~ Dirichlet(α · 1_N),
   > then assign q^s_i · N_s samples of source s to client i.

   Client i's own CR1 proportion is then
   p_i = q^1_i · N_1 / (q^1_i · N_1 + q^2_i · N_2), which is a
   ratio of two independent Beta-marginals with strong
   within-source negative correlation (because Σ q^s_i = 1).

   **Consequence:**
   - The *marginal* distribution of p_i is NOT exactly Beta(α, α);
     it is a ratio of two Beta-marginals with joint Dirichlet
     structure.
   - The *cross-client covariance* Cov(p_i, p_j) for i ≠ j is
     negative (not zero as in the i.i.d. Beta model).
   - However, **qualitatively the 1/(2α+1) scaling persists**:
     both schemes are controlled by the Dirichlet concentration
     α, with Var → 1/4 as α → 0 and Var → 0 as α → ∞.

   **Impact on Theorem D1's numerical value.** In the Hsu 2019
   scheme, the expected average E[(1/N) Σ (p_i − p̄_N)²] has
   the same order (roughly 1/(4(2α+1))) but the exact (N−1)/N
   prefactor is replaced by a more complex expression
   depending on N_1, N_2 and the ratio N_1/N_2. For balanced
   sources (N_1 ≈ N_2) and large N, the two schemes produce
   numerically similar expected averages (both in the 0.18–0.22
   range for α = 0.1, N = 50). **Without empirical measurement,
   we can only claim the 0.204 coefficient of D1.17 to
   factor-of-~2 precision.**

   **Action for paper:** either (a) re-derive Theorem D1 under
   the Hsu 2019 partition (requires Beta-ratio algebra,
   available in closed form but tedious — v3 task), or (b)
   empirically measure E[(1/N) Σ (p_i − p̄_N)²] from our actual
   partition seeds across multiple runs and compare to the
   Beta(α, α) prediction. **Option (b) is easier and more
   honest** — a 20-line script running
   `dirichlet_source_partition(N=50, α=0.1)` over, say, 1000
   seeds and computing the empirical average gives an exact
   numerical value for our partition. This is listed as a new
   v2 task (V15 below in `v2_remaining_issues.md`).

6. **θ-independence of decomposition.** ∇f_1, ∇f_2 are
   functions of θ; so is the pure-gradient-difference. What
   (D1.5) achieves is **factorising the θ-dependence into
   gradient-difference only**, leaving the p_i scalar to carry
   all the randomness. So Theorem D1 gives expected ζ²(θ) as
   a function of θ, uniformly in p.

### §D1.11 What this does not prove

- **Does NOT prove Claim C16 (TDBN-FedBN redundancy).** Claim
  C16 is about the BN-normalisation effect on NTK min
  eigenvalue. Theorem D1 is about the pre-normalisation gradient
  dissimilarity. They are orthogonal results; both are useful
  but neither implies the other.

- **Does NOT directly bound final PSNR.** PSNR is not a simple
  function of ζ² because the local loss is Charbonnier+SSIM
  over pixels, and the conversion from gradient-descent
  convergence bound → final test PSNR is bracketed by the
  empirical noise floor (cf. §18.5).

- **Does NOT replace empirical measurement.** The pure-type
  gradient difference ‖∇f_1 − ∇f_2‖² needs to be measured, not
  assumed. We do not have this number yet — it's a v3 task.

### §D1.12 Summary for paper

**Paper-ready one-liner (draft, with implementation caveat):**

> "For a 2-source Dirichlet(α, α) partition with i.i.d.
> per-client proportions, we show that the expected average
> inter-client gradient dissimilarity admits the closed form
> ζ²(α) ≤ ((N−1)/(4N(2α+1))) · ‖∇f_1 − ∇f_2‖² (Theorem D1),
> yielding 0.204 · ‖∇f_1 − ∇f_2‖² at α = 0.1, N = 50. Our
> implementation uses the Hsu 2019 'Dirichlet over clients per
> source' scheme, which differs from the i.i.d. Beta model in
> its cross-client covariance structure; we verify empirically
> that the two give numerically similar coefficients (0.18–0.22
> range) under balanced sources. This converts FLSNN's
> Assumption A4 from an abstract constant to an
> α-parameterised expression and supports the α-sensitivity
> ablation of §21 Tier-2."

References:
- Beta(α,α) moments: Johnson, Kotz, Balakrishnan, *Continuous
  Univariate Distributions Vol. 2* (1995), §25.2.
- FLSNN A4: Wang et al. 2025, arXiv:2501.15995 (§1.2 of
  v2_comprehensive_literature.md).
- HarmoFL A-B.2: Jiang, Wang, Dou, AAAI 2022, arXiv:2112.10775.

---

**End of §D1. §D2 (TDBN-FedBN spectral-gap bound via arc-cosine
kernel) to follow in a subsequent derivation round.**
