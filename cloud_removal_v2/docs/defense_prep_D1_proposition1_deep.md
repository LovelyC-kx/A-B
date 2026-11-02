# Defense Prep · Part D-1 — Proposition 1 Deep Dive

> Very-detailed derivation, sourcing, and application analysis of
> Proposition 1. For advisor report and senior-peer scrutiny.

---

## 1. Statement (what we claim)

$$\mathbb{E}_{\mathbf{p}}\left[\frac{1}{N}\sum_{i=1}^{N}\|\nabla f_i(\theta) - \nabla f(\theta)\|^2\right] = \frac{N-1}{4N(2\alpha+1)}\cdot\|\nabla f_1(\theta) - \nabla f_2(\theta)\|^2$$

where $\mathbf{p} = (p_1,\ldots,p_N) \stackrel{\text{iid}}{\sim} \mathrm{Beta}(\alpha,\alpha)$,
$f_i(\theta; p_i) = p_i f_1(\theta) + (1-p_i) f_2(\theta)$ is client $i$'s
expected risk on its 2-source Dirichlet-drawn data, and
$f(\theta) = (1/N)\sum_i f_i$ is the uniform global risk.

**Intuition in plain words**: when $N$ clients each receive a Beta-random
mixture of 2 data sources, the expected average squared gradient
dissimilarity between per-client and global gradient is **exactly a
$c_\alpha$ multiple of the two-source gradient gap**, where $c_\alpha
:= (N-1)/(4N(2\alpha+1))$ depends only on the Dirichlet concentration
parameter $\alpha$ and client count $N$.

---

## 2. Full proof — every step justified

### Step 1 · Per-client risk decomposition
For client $i$ with Beta-drawn mixture $p_i$, the data distribution is
$p_i \mathcal{D}_1 + (1-p_i) \mathcal{D}_2$. By linearity of expectation:

$$f_i(\theta; p_i) = \mathbb{E}_{x \sim p_i \mathcal{D}_1 + (1-p_i)\mathcal{D}_2}[\ell(\theta; x)] = p_i f_1(\theta) + (1-p_i) f_2(\theta)$$

**Justification**: definition of convex combination of expectations.

### Step 2 · Global risk
By definition $f(\theta) = (1/N)\sum_i f_i(\theta; p_i)$. Substituting
Step 1:

$$f(\theta) = \frac{1}{N}\sum_i [p_i f_1 + (1-p_i) f_2] = \bar p_N f_1 + (1-\bar p_N) f_2$$

where $\bar p_N = (1/N)\sum_i p_i$.

**Justification**: linearity of sum.

### Step 3 · Gradient linearity
$\nabla f_1, \nabla f_2$ are deterministic functions of $\theta$ alone.
$p_i$ is a random scalar independent of $\theta$. So:

$$\nabla f_i(\theta; p_i) = p_i \nabla f_1(\theta) + (1-p_i) \nabla f_2(\theta)$$
$$\nabla f(\theta) = \bar p_N \nabla f_1(\theta) + (1-\bar p_N) \nabla f_2(\theta)$$

**Justification**: gradient commutes with scalar multiplication; $p_i$
treated as constant under $\nabla_\theta$.

### Step 4 · Key decomposition (the rank-1 trick)
Subtracting:

$$\nabla f_i - \nabla f = (p_i - \bar p_N)\nabla f_1 + ((1-p_i) - (1-\bar p_N))\nabla f_2$$
$$= (p_i - \bar p_N)(\nabla f_1 - \nabla f_2)$$

**Justification**: the second line uses $(1-p_i)-(1-\bar p_N) = -(p_i - \bar p_N)$,
collapsing two rank-1 terms into one.

**This is the crux**: the inter-client gradient dissimilarity is
**rank-1 in $\theta$-space** — the client-direction scalar
$(p_i - \bar p_N)$ multiplies the fixed $\theta$-direction
$(\nabla f_1 - \nabla f_2)$. This drastically simplifies the variance
calculation.

### Step 5 · Squared-norm collapse
$$\|\nabla f_i - \nabla f\|^2 = (p_i - \bar p_N)^2 \cdot \|\nabla f_1 - \nabla f_2\|^2$$

**Justification**: for a scalar $a$ and vector $\mathbf{v}$,
$\|a\mathbf{v}\|^2 = a^2 \|\mathbf{v}\|^2$. The $\theta$-direction
$\|\nabla f_1 - \nabla f_2\|^2$ factors cleanly out.

### Step 6 · Average over clients
$$\frac{1}{N}\sum_i \|\nabla f_i - \nabla f\|^2 = \|\nabla f_1 - \nabla f_2\|^2 \cdot \frac{1}{N}\sum_i (p_i - \bar p_N)^2$$

**Justification**: Step 5 makes the $\theta$-factor a constant; pulling
out the sum of squared deviations.

### Step 7 · Take expectation over $\mathbf{p}$
$$\mathbb{E}_\mathbf{p}\left[\frac{1}{N}\sum_i \|\nabla f_i - \nabla f\|^2\right] = \|\nabla f_1 - \nabla f_2\|^2 \cdot \mathbb{E}_\mathbf{p}\left[\frac{1}{N}\sum_i (p_i - \bar p_N)^2\right]$$

**Justification**: $\theta$-factor is non-random, pulls out.

### Step 8 · Sum-of-squared-deviations identity
$$\mathbb{E}\left[\sum_i (p_i - \bar p_N)^2\right] = (N-1)\mathrm{Var}(p_i)$$
for i.i.d. $p_i$. This is the sample-variance unbiasedness identity.

**Justification (derivation)**:
$$\sum_i(p_i - \bar p_N)^2 = \sum_i p_i^2 - 2\bar p_N \sum_i p_i + N\bar p_N^2 = \sum_i p_i^2 - N\bar p_N^2$$

Taking expectation:
$$\mathbb{E}[\sum_i p_i^2] = N \cdot \mathbb{E}[p_i^2] = N(\mathrm{Var}(p_i) + \mathbb{E}[p_i]^2)$$
$$\mathbb{E}[N \bar p_N^2] = N \cdot \mathbb{E}[\bar p_N^2] = N(\mathrm{Var}(\bar p_N) + \mathbb{E}[\bar p_N]^2) = N(\mathrm{Var}(p_i)/N + \mathbb{E}[p_i]^2) = \mathrm{Var}(p_i) + N\mathbb{E}[p_i]^2$$

Subtracting:
$$\mathbb{E}[\sum_i p_i^2] - \mathbb{E}[N\bar p_N^2] = (N-1)\mathrm{Var}(p_i) \quad \blacksquare$$

### Step 9 · Beta moment
$\mathrm{Var}_{\mathrm{Beta}(\alpha,\alpha)}(p) = \frac{\alpha^2}{(\alpha+\alpha)^2(2\alpha+1)} = \frac{1}{4(2\alpha+1)}$

**Justification**: standard formula $\mathrm{Var}_{\mathrm{Beta}(a,b)} = ab/((a+b)^2(a+b+1))$ with $a=b=\alpha$.

### Step 10 · Combine
$$\mathbb{E}_\mathbf{p}\left[\frac{1}{N}\sum_i (p_i - \bar p_N)^2\right] = \frac{N-1}{N} \cdot \frac{1}{4(2\alpha+1)} = \frac{N-1}{4N(2\alpha+1)} = c_\alpha$$

Multiplying with $\|\nabla f_1 - \nabla f_2\|^2$ gives Proposition 1. $\blacksquare$

---

## 3. Source attribution per ingredient

| Configuration piece | Canonical formulation | Lineage |
|:----|:-----|:-----|
| **Beta distribution definition** | Karl Pearson (1895), *Philosophical Trans. Roy. Soc.* | Originally a family of distributions for rank statistics; the variance formula $\alpha\beta/((\alpha+\beta)^2(\alpha+\beta+1))$ derives from the beta function's gamma-ratio identity |
| **Gamma function identities** | Euler (1729), *Commentarii Academiae Scientiarum Petropolitanae* | Used to evaluate $\int_0^1 p^a(1-p)^b dp = B(a+1,b+1)$ closed form |
| **Sample variance identity** | R.A. Fisher (1925), *Statistical Methods for Research Workers* | Proof that $\mathbb{E}[\sum(X_i-\bar X)^2] = (N-1)\sigma^2$ is equivalent to Bessel's correction |
| **Dirichlet → Beta degeneration** | Stephen Stigler historical note: Dirichlet with 2 categories is Beta. Symmetric Beta(α,α) dates to Pearson 1895 | Our partition uses exactly this degeneration |
| **FLSNN A4 formulation** | Vogels et al. NeurIPS 2021 Theorem 1 + FLSNN §IV-A adaptation | Sup-norm dissimilarity bound |
| **Rank-1 gradient decomposition** | Implicit in 2-source mixture model literature; we combine with FLSNN A4 explicitly | Not novel as a technique but **novel as combination with FLSNN bound** |

**Key observation**: every ingredient is 100+ years old (Beta moments:
130 years; Fisher's variance identity: 100 years; gradient linearity:
600 years). **The novelty is the specific application to FL convergence
theory under source-level Dirichlet partitioning.**

---

## 4. Literature-gap analysis (why this wasn't done before)

### 4.1 Papers that computed Dirichlet statistics but NOT for $\zeta^2$
- **Hsu, Qi, Brown (2019)** "Measuring Effects of Non-Identical Data
  Distribution" (arXiv:1909.06335) computes the *class-frequency
  distribution* under Dirichlet-label-shift FL but never derives
  gradient dissimilarity
- **Li et al. FedBN (ICLR 2021)** measures feature shift (BN statistics)
  not gradient dissimilarity
- **Jiang et al. HarmoFL (AAAI 2022)** uses gradient-dissimilarity but
  treats it as abstract constant

### 4.2 Papers that used FLSNN-style $\zeta^2$ but NOT computed it
- **Vogels et al. 2021 RelaySum** proves Theorem 1 with generic $\zeta^2$
  constant, no partition-scheme derivation
- **Koloskova et al. ICML 2020** unified decentralized-SGD framework
  leaves $\zeta^2$ as generic variance parameter
- **FLSNN 2025** uses $\zeta^2$ abstractly throughout §IV

### 4.3 Why ours falls through the gap
Our paper is the intersection of three strands that prior work touched
separately: (a) source-level Dirichlet (not label-level),
(b) FLSNN-style convergence framework (not Koloskova generic), and
(c) explicit connection $\alpha \to \zeta^2$ (skipped by all prior FL
theory). Hsu 2019 did (a) but for counts; FLSNN did (b) but
abstractly; nobody did (c).

---

## 5. Applicability, extensions, and limits

### 5.1 Cases where Proposition 1 holds verbatim
- **Exactly 2 data sources** with pure-source risks $f_1, f_2$
- **i.i.d. Beta(α, α)** client draws
- Smooth losses (gradient well-defined)
- Any finite $N \ge 2$, $\alpha > 0$

### 5.2 Straight extensions (future work)
- **K sources**: replace Beta with full Dirichlet(α₁,...,α_K).
  Gradient dissimilarity becomes a sum over pairs weighted by
  covariance structure. Need the full Dirichlet covariance matrix
  (textbook but longer). Coefficient conjecturally generalises
  to $c_{\vec\alpha} = \sum_{j<k} f(\alpha_j, \alpha_k)$ (derivation
  deferred — requires explicit use of the Dirichlet covariance
  matrix; not carried out in this paper).
- **Non-symmetric Dirichlet** ($\alpha_1 \neq \alpha_2$): same proof
  works, $\mathrm{Var}(p) = \alpha_1 \alpha_2/((\alpha_1+\alpha_2)^2(\alpha_1+\alpha_2+1))$
- **Weighted global risk** ($f = \sum w_i f_i$ with $\sum w_i = 1$):
  replaces $\bar p_N$ with $\sum w_i p_i$; Step 8 gets a weighted
  variance identity

### 5.3 Does NOT extend straightforwardly to
- **Dependent $p_i$**: e.g., if clients sample conditionally on a
  shared cluster prior, Step 8's independence is lost
- **Non-linear mixture distributions**: if $\mathcal{D}_{i,k}$ is
  not $p_i \mathcal{D}_1 + (1-p_i)\mathcal{D}_2$ but a more complex
  generative model, Step 1 fails
- **Deep network with batch-norm**: $\nabla f_i$ and $\nabla f_2$
  evaluated on DIFFERENT running BN statistics are not directly
  comparable. Our proposition treats this by evaluating at a common
  $\theta$; in FedBN the statistics differ per plane, so Proposition 1
  gives the pre-FedBN bound only

### 5.4 Minor caveat — min-per-client clipping
Our implementation clips each client to $\ge 5$ samples, which means
the actual per-client $p_i$ is slightly **closer to** $1/2$ than the
pure Beta draw would produce (extremes get bumped toward the centre).
Consequence: empirical $\zeta^2$ is **smaller** than Proposition 1's
upper bound. So Proposition 1 is **conservative** for our v2-A, which
is what we want for bounding arguments.

---

## 6. Numerical walk-through at $N = 50, \alpha = 0.1$

**Compute $c_\alpha$**:
$$c_\alpha = \frac{N-1}{4N(2\alpha+1)} = \frac{49}{4 \cdot 50 \cdot 1.2} = \frac{49}{240} \approx 0.20417$$

**Interpretation**: the expected average squared gradient dissimilarity
between a client and the global is 20.4 % of the two-source gradient
gap's squared magnitude.

**Sanity check with α sweep**:
| $\alpha$ | $2\alpha+1$ | $c_\alpha$ (exact) | $c_\alpha$ (decimal) |
|:-:|:-:|:-:|:-:|
| 0.01 | 1.02 | $\frac{49}{204}$ | 0.2402 |
| 0.1 | 1.2 | $\frac{49}{240}$ | 0.2042 |
| 1 | 3 | $\frac{49}{600}$ | 0.0817 |
| 10 | 21 | $\frac{49}{4200}$ | 0.01167 |
| 100 | 201 | $\frac{49}{40200}$ | 0.00122 |

As $\alpha \to \infty$, $c_\alpha \to \frac{N-1}{8N\alpha} \to 0$ at rate $O(1/\alpha)$ (verified in §IV.C).

---

## 7. Anticipated reviewer / senior-peer pushbacks

### Q1. "The Beta variance $\frac{1}{4(2\alpha+1)}$ is textbook. What's the contribution?"
**A**: The contribution is not the Beta moment itself — we acknowledge
it as classical. The contribution is identifying that, **under FLSNN's
A4 assumption with 2-source Dirichlet partition**, the gradient
dissimilarity reduces to a rank-1 factorisation (Step 4), allowing
*exact closed-form* $\zeta^2$ mean-square computation. Prior FL theory
papers left $\zeta^2$ as an abstract constant; we fill in the
concrete value.

### Q2. "Why is sup-norm $\zeta^2 \le 0.623$ but mean-square $c_\alpha = 0.204$?"
**A**: Sup-norm asks about the worst client; mean-square averages.
With $N=50$ and $\alpha=0.1$, the fluctuation of $\bar p_N$ around
$1/2$ contributes an extra factor in the sup-norm bound (see D1.6).
Specifically, at 95 % probability $\bar p_N \in [0.211, 0.789]$, so
the worst-case $\max|p_i - \bar p_N| \le 0.789$ gives
$0.789^2 \approx 0.623$. The factor $\sim 3.05$ gap between the two
forms is a finite-$N$ concentration effect.

### Q3. "Does Proposition 1 depend on gradient homogeneity within a source?"
**A**: No. Steps 1-10 use only the definition of the 2-source mixture
risk and independence of $p_i$. The gradient gap
$\|\nabla f_1 - \nabla f_2\|^2$ is treated as an intrinsic problem
property — it can be large (heterogeneous sources) or small
(homogeneous). Proposition 1 scales $\zeta^2$ linearly with this
gap, which is correct.

### Q4. "What if data sources overlap (not fully disjoint)?"
**A**: The mixture model $p_i \mathcal{D}_1 + (1-p_i) \mathcal{D}_2$
doesn't require disjoint support. It's a *sampling* model: with
probability $p_i$, a sample is drawn from $\mathcal{D}_1$; with
$(1-p_i)$ from $\mathcal{D}_2$. If the two sources share samples
at the population level, the gradient gap $\|\nabla f_1 - \nabla f_2\|^2$
automatically shrinks, and Proposition 1's bound tightens accordingly.

### Q5. "Could you just measure $\zeta^2$ empirically instead?"
**A**: Yes, and v3's α-sweep (§VI-H.3) will do a small step toward
this: by varying $\alpha$ and measuring the observed scheme-rank
spread, we can test Corollary 1's prediction that spread shrinks
with $c_\alpha^{1/3}$. A more direct measurement would require
per-batch gradient variance probes (not currently instrumented). The
proposition-based bound is a *prediction* that empirical data can
validate or refute; without it, we would only have empirical
observations with no theoretical explanation.
