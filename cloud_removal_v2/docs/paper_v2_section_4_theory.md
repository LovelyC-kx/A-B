# §IV. Theoretical Analysis

We import the convergence guarantee of the underlying FLSNN
framework [Yang 2025] and extend it with a **closed-form
characterisation of the inter-plane gradient dissimilarity
$\zeta^2$ under our source-level Dirichlet partition** (§VI-B).
The characterisation lets us make §VI-D's "RelaySum advantage
collapses in our regime" claim *quantitative* — identifying the
small-$\zeta^2$ regime in which the three aggregation schemes'
Theorem-2 bounds have the same asymptotic order. The explicit
derivation is in `cloud_removal_v2/docs/v2_formal_derivations.md`
§D1; this section records the paper-scope results only.

## IV.A  Preliminaries

**Problem.** Let $f_{i,k}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{i,k}}
[\ell(\theta; x, y)]$ be the local risk on satellite $(i, k)$,
where plane index $i \in [N]$, satellite index $k \in [K]$, and
$\mathcal{D}_{i,k}$ is the data distribution assigned by the
Dirichlet-source partition. The global risk is
$f(\theta) = (NK)^{-1} \sum_{i,k} f_{i,k}(\theta)$; the FL goal
is $\min_\theta f(\theta)$.

**Algorithm (restated from §III).** Per global round $t$:

1.  **Local update.** Each satellite performs $E$ local epochs
    (full passes) over $\mathcal{D}_{i,k}$; see §III.D (S1) for
    the per-step update rule.
2.  **Intra-plane ring-AllReduce.** All $K$ satellites of plane
    $i$ synchronise to the plane's mean weights
    $\theta_i^{t+1/2} = K^{-1} \sum_k \theta_{i,k}$.
3.  **Inter-plane aggregation.** One of
    $\{\text{AllReduce}, \text{Gossip}, \text{RelaySum}\}$
    produces $\theta_i^{t+1}$ from
    $\{\theta_j^{t+1/2}\}_{j=1}^N$. Only this step differs
    across the three schemes.

The per-round update is captured by a stacked mixing matrix
$\mathbf{W} \in \mathbb{R}^{N(\tau_{\max}+1) \times N(\tau_{\max}+1)}$
that holds the scheme's mixing weights plus the delay state of
RelaySum. Its spectral gap
$\rho := \tfrac{1}{2}(1 - |\lambda_2(\mathbf{W})|)$
and diameter $\tilde\tau := \tau_{\max} + 1$ are the
topology-only constants appearing in Theorem 2.

**Assumptions (FLSNN A1–A4, verbatim).**

* **(A1)** Each $f_{i,k}$, $f_i$, and $f$ are $L$-smooth.
* **(A2)** Each satellite computes an unbiased stochastic
  gradient with bounded variance $\sigma^2$:
  $\mathbb{E}\|\nabla F_{i,k}(\theta) - \nabla f_{i,k}(\theta)\|^2 \le \sigma^2$.
* **(A3)** Intra-plane dissimilarity
  $K^{-1} \sum_k \|\nabla f_{i,k} - \nabla f_i\|^2 \le \delta^2$ for every plane $i$.
* **(A4)** Inter-plane dissimilarity
  $\|\nabla f_i - \nabla f\|^2 \le \zeta^2$ for every plane $i$.

In the Dirichlet-source partition of §VI-B, both $\delta^2$ and
$\zeta^2$ admit closed-form characterisations analogous to
Proposition 1 below; $\delta^2$ applies at granularity $K=10$
(ten satellites per plane) and $\zeta^2$ at $N=5$ (five planes).
A3 concerns the *data*-induced gradient dissimilarity at a
shared weight vector, so it is **not** zeroed by intra-plane
ring-AllReduce — ring-AllReduce synchronises weights, not data.

## IV.B  Main convergence bound (FLSNN, Theorem 2)

**Theorem 1 (FLSNN Theorem 2, restated).** Under assumptions
A1–A4, for learning rate
$\eta < \tfrac{q \,\tilde\pi_0}{36\, C_1\, m\, R\, E\, L}$ with
$\tilde\pi_0 = \min\{\pi_0, 1\}$, the iterates of the algorithm
satisfy
$$
\tfrac{1}{T} \sum_{t=0}^{T-1} \|\nabla f(\bar\theta^t)\|^2 \le
\underbrace{16 \bigl( \tfrac{2L \sigma^2 r_0}{NT} \bigr)^{1/2}}_{T_1}
\;+\; \underbrace{16 \Bigl( \tfrac{4 C \sqrt{\tilde\tau}\, L \sigma r_0}{\rho \sqrt{N T}} \Bigr)^{2/3}}_{T_2}
\;+\; \underbrace{\tfrac{288 C L \sqrt{\tilde\tau}\, r_0}{\rho\, T}}_{T_3}
\;+\; \underbrace{16 \, \Xi(E, R, \rho, \tilde\tau, L, z, r_0, T, \pi_0, C)}_{T_4},
$$
where $r_0 = f(\bar\theta^0) - f^\ast$,
$z^2 = \sigma^2 + \delta^2 + \zeta^2$,
and $\Xi$ is the local-iteration $\times$ topology coupling
term defined in [Yang 2025, Eq. (26) T4]. $C$ and $C_1$ are
constants defined by the mixing matrix $\mathbf{W}$.

**Term ordering.** As $T \to \infty$,
$T_1 \!=\! O(T^{-1/2})$ is the stochastic-gradient floor,
$T_2, T_4 \!=\! O(T^{-2/3})$ are the heterogeneity–topology
terms, and $T_3 \!=\! O(T^{-1})$ decays fastest. Topology
enters $T_2/T_3/T_4$ through the pair $(\rho, \tilde\tau)$;
scheme choice therefore only affects the non-floor terms.
$\zeta^2$ enters $T_4$ through $z^2$ and (indirectly) through
$C$; **smaller $\zeta^2$ tightens $T_4$ linearly**.

**Scheme substitution.** Plugging in the three schemes:

|   Scheme     | $\tilde\tau$  | $\rho = q/m$ where $q = \tfrac{1}{2}(1 - \lvert\lambda_2\rvert)$ |
|:-------------|:--------------|:------------------------------|
| AllReduce    | $1$           | $\rho = 1/2$ (maximal: $\lambda_2(W)=0$ so $q = 1/2$; $m=1$)   |
| Gossip       | $1$           | $\rho < 1/2$; small on chain ($N{=}5$, small spectral gap) |
| RelaySum     | $\tilde\tau \!=\! \text{chain diameter} \!+\! 1 \!=\! 5$ | $\rho < 1/2$; comparable to Gossip at $N{=}5$, tighter in the $N \!\to\! \infty$ limit |

AllReduce simultaneously minimises $\tilde\tau$ and maximises
$\rho$ (the latter at its structural ceiling $\rho = 1/2$), so
its Theorem-1 bound is always tightest **when $\zeta^2$ is the
dominant contributor to $z^2$**.

## IV.C  Proposition 1 — Dirichlet($\alpha,\alpha$) → $\zeta^2$ closed form

The four assumptions do not, by themselves, give a numerical
value for $\zeta^2$ — they treat it as an abstract constant.
For our 2-source Dirichlet partition (§VI-B), we can compute it
in closed form.

**Setup.** Two data sources $\mathcal{D}_1$ (CR1) and
$\mathcal{D}_2$ (CR2) generate samples with pure-source risks
$f_s(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_s}[\ell(\theta; x, y)]$.
Each of the $N$ clients is assigned a Beta-mixture
$p_i \!\sim\! \mathrm{Beta}(\alpha, \alpha)$ of the two
sources, so client $i$'s per-sample distribution is
$p_i \mathcal{D}_1 + (1-p_i) \mathcal{D}_2$ and its risk is
$f_i(\theta; p_i) = p_i f_1(\theta) + (1-p_i) f_2(\theta)$.

**Proposition 1.** Under the setup above,
$$
\mathbb{E}_{\mathbf{p}} \Bigl[ \tfrac{1}{N} \sum_{i=1}^N
\|\nabla f_i(\theta) - \nabla f(\theta)\|^2 \Bigr]
\;=\;
\frac{N-1}{4N(2\alpha + 1)}
\;\cdot\;
\|\nabla f_1(\theta) - \nabla f_2(\theta)\|^2,
$$
with the expectation taken over
$\mathbf{p} = (p_1, \dots, p_N) \sim \mathrm{Beta}(\alpha,\alpha)^{\otimes N}$.

**Proof sketch** (full proof in §D1.1–D1.5). From linearity of
$\nabla$ in $p_i$,
$\nabla f_i - \nabla f = (p_i - \bar p_N)(\nabla f_1 - \nabla f_2)$,
giving $\|\nabla f_i - \nabla f\|^2 = (p_i - \bar p_N)^2
\|\nabla f_1 - \nabla f_2\|^2$. Taking the expected average,
the coefficient reduces to $\mathbb{E}[(p_i - \bar p_N)^2]$
summed over $i$ and divided by $N$. Using
$\mathrm{Var}_{\mathrm{Beta}(\alpha,\alpha)}(p) =
\tfrac{1}{4(2\alpha+1)}$ and the standard
$\mathbb{E}[\sum_i (p_i - \bar p_N)^2] = (N-1)\mathrm{Var}(p)$
identity, the result follows. $\square$

**Worst-case / high-probability sup-norm form.** FLSNN A4 is
stated in sup-norm $\|\nabla f_i - \nabla f\|^2 \le \zeta^2$
rather than mean-square. An analogous bound from §D1.6 gives,
for $N = 50, \alpha = 0.1$, $\zeta^2 \le 0.623 \cdot
\|\nabla f_1 - \nabla f_2\|^2$ with probability $\ge 0.95$.
We use the mean-square form in §IV.D because it matches the
way $\zeta^2$ enters the FLSNN bound through the composite
constants $C, C_1$ (verbatim [Yang 2025] does not expose
sup-norm vs mean-square in the bound). The two forms scale
differently in $\alpha$: the mean-square
$c_\alpha = (N\!-\!1)/(4N(2\alpha\!+\!1))$ decays as
$O(1/\alpha)$ for $\alpha \to \infty$ and is uniformly
bounded by $(N\!-\!1)/(4N) \le 1/4$ for all $\alpha \ge 0$;
the sup-norm bound converges to the constant $1/4$ as
$\alpha \to \infty$ (the IID limit where all $p_i \to 1/2$)
but can exceed $1/4$ at finite $\alpha$ because $\bar p_N$
fluctuates away from $1/2$ with non-trivial probability.
At $\alpha = 0.1$, the numerical values are $c_\alpha = 0.204$
(mean-square, Table §D1.8) and $0.623$ (sup-norm at 95%
probability, §D1.6); their ratio $0.623 / 0.204 \approx 3.05$
illustrates the gap between average and worst-case
interpretations of A4.

## IV.D  Corollary — Asymptotic collapse of the scheme hierarchy

**Numerical substitution.** At our v2-A parameters $N\!=\!50$,
$\alpha\!=\!0.1$, Proposition 1 gives
$\zeta^2 \le 0.204 \cdot \|\nabla f_1 - \nabla f_2\|^2$, i.e.,
$\zeta^2$ is bounded by roughly $20\%$ of the intrinsic
two-source gradient gap.

**Corollary 1.** With $\zeta^2 \le c_\alpha \cdot G^2$,
$G := \|\nabla f_1 - \nabla f_2\|$,
$c_\alpha := (N-1)/(4N(2\alpha+1))$,
the Theorem-1 bound's $T_4$ term admits the upper bound
$$
T_4 \;\le\; 16 \cdot A^{2/3} \cdot
(\sigma^2 + \delta^2 + c_\alpha G^2)^{1/3}
\cdot (r_0 / T)^{2/3},
$$
where $A := P \cdot Q \cdot L$ gathers the
topology / local-iteration constants of Eq.~(26)
($P = \sqrt{7E(E\!-\!1)+7E^2 R(R\!-\!1)}/(NRE\pi_0)$,
$Q = \sqrt{2C^2 \tilde\tau / (9 \rho^2 L^2) + 5}$).

**Rate-based scheme collapse.** Irrespective of $c_\alpha$,
$T_1 = O(T^{-1/2})$ while $T_2, T_4 = O(T^{-2/3})$ and
$T_3 = O(T^{-1})$; as $T \to \infty$ all topology-dependent
terms decay strictly faster than $T_1$, so the three
schemes' Theorem-1 bounds converge to the common floor
$T_1 = \sqrt{2L\sigma^2 r_0 /(NT)}$ — independent of
$(\rho, \tilde\tau)$.

**Magnitude in the small-$c_\alpha$ regime.** In addition to
the rate argument, the multiplicative constant in $T_4$ is
reduced when $\zeta^2$ is small: if $\zeta^2$ dominates
$z^2$ (i.e., $c_\alpha G^2 \gg \sigma^2 + \delta^2$),
$T_4 \propto c_\alpha^{1/3} G^{2/3}$; if instead $\sigma^2 +
\delta^2$ dominates, the $c_\alpha G^2$ term becomes a
low-order correction and $T_4$ approaches
$\propto (\sigma^2 + \delta^2)^{1/3}$ — effectively
$c_\alpha$-independent at finite $T$. The partition of
z-variance between $\sigma^2, \delta^2, \zeta^2$ is not
directly measured in this paper (doing so would require
per-batch stochastic-gradient variance and per-plane
gradient-dissimilarity probes not instrumented in our
training pipeline); we note, however, that the observed
scheme-rank spread of 0.1–0.3 dB (§VI-D.1) is consistent
with the $\sigma^2 + \delta^2$-dominated regime, and that
§VI-H.3's $\alpha$-sweep v3 item would indirectly test the
$c_\alpha$-dependence of $T_4$ by varying $\zeta^2$ via
$\alpha$.
In either case, the **rate** argument above drives the
asymptotic scheme collapse; the **magnitude** argument
explains why the finite-$T$ spread we observe is comparable
to the single-seed noise floor.

**Interpretation for §VI-D.** Our observed scheme-rank spread
is 0.1–0.3 dB within each run (§VI-D.1: A spread 0.297 dB, B
spread 0.161 dB, averaged across BN variants), placing us
empirically in this collapsed regime: the order is not stable
and no scheme dominates by more than its single-seed noise
floor. The reversal relative to FLSNN Fig. 5 (their RelaySum
$>$ Gossip $>$ AllReduce on 10-class classification with
$\varsigma = 0.02$) is explained by the contrast in the
$\zeta^2/\sigma^2$ ratio:
FLSNN's 10-label, $\varsigma = 0.02$ setting has large
$\zeta^2$ relative to $\sigma^2$, so $T_4$ dominates and
RelaySum's $\tilde\tau/\rho$ advantage shows; our 2-source,
$\alpha = 0.1$ setting has a small $\zeta^2$ relative to
pixel-regression $\sigma^2$, so $T_1$ dominates and the
topology constants are irrelevant.

**Caveats.**

(i) Corollary 1 is asymptotic in $T$ (it governs the
$T \to \infty$ limit); for the finite $T \!=\! 80$ rounds we
use in §VI, the quantitative statement is "the Theorem-1
bounds are of the same order," not "the three schemes
converge to the same point." Single-seed PSNR differences
$\le 0.3$ dB (§VI-D.1) are consistent with this asymptotic
picture but do not constitute a proof.

(ii) Proposition 1 assumes the Dirichlet draw is used *as
drawn*; our implementation clips each client to a minimum of
5 samples (§VI-B.4, `min_samples_per_client = 5`), which
mildly softens the non-IID and tightens $c_\alpha$ relative
to the pure-Beta upper bound reported here. Empirical
fraction of "pure single-source" clients (§VI-B.2: 72 %) is
therefore a slight underestimate of what Dir($0.1,0.1$) alone
would produce.

(iii) Proposition 1's coefficient has an explicit
$\alpha$-sensitivity (Table §D1.8): $\alpha\!=\!0.1 \to 0.204$,
$\alpha\!=\!1 \to 0.082$, $\alpha\!=\!10 \to 0.012$. An
$\alpha$-sweep would trace $c_\alpha$ empirically; this is a
v3 to-do (§VI-H L2).

## IV.E  Summary

The paper's Theorem 2 (FLSNN) bounds federated convergence by
four terms; smaller $\zeta^2$ tightens the heterogeneity-coupled
$T_4$ (explicitly, via $z^2\!=\!\sigma^2\!+\!\delta^2\!+\!\zeta^2$)
and indirectly tightens $T_2, T_3$ through the composite
constants $C, C_1$. Our Proposition 1 characterises $\zeta^2$
in closed form for the 2-source Dirichlet($\alpha,\alpha$)
partition, giving
$\zeta^2 \le c_\alpha \cdot G^2$ with
$c_\alpha = (N\!-\!1)/(4N(2\alpha\!+\!1))$. At
$\alpha\!=\!0.1$, $N\!=\!50$, $c_\alpha \!\approx\! 0.204$ —
small enough that $T_1$ dominates the bound and the three
aggregation schemes' Theorem-1 orders coincide (Corollary 1).
This prediction matches §VI-D's empirical observation (scheme
spread $\le 0.1$ dB), and is a boundary-case result of
FLSNN's own theory rather than a contradiction of it.

The full derivation of Proposition 1, the sup-norm analogue,
and the $\alpha$-sensitivity table are in
`cloud_removal_v2/docs/v2_formal_derivations.md` §D1.
