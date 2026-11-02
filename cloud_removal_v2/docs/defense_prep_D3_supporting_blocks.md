# Defense Prep · Part D-3 — Supporting Theoretical Blocks

> Very-detailed support material for D-1 (Prop. 1) and D-2
> (Corollary 1). Covers (a) sup-norm analogue, (b) A-decomposition,
> (c) scheme substitution table, (d) α-sensitivity, (e) E-semantics
> impact. For advisor report and senior-peer scrutiny.

---

## 1. Sup-norm analogue of Proposition 1 (§D1.6)

### 1.1 Why we need a sup-norm form
FLSNN A4 is written as $\|\nabla f_i - \nabla f\|^2 \le \zeta^2$ for
**every** $i$ — a sup-norm (worst-case) bound, not a mean-square
average. Proposition 1 gives the mean-square, so we need a
high-probability sup-norm bound to plug into A4 verbatim.

### 1.2 Setup
Let $p_1, \ldots, p_N \stackrel{\text{iid}}{\sim} \mathrm{Beta}(\alpha, \alpha)$.
We want, with probability $\ge 1 - \delta$:
$$\max_i (p_i - \bar p_N)^2 \le \zeta_\delta^2 / G^2$$

### 1.3 Derivation (Chebyshev bound, matches §D1.6 of `v2_formal_derivations.md`)

**Step 1 — Concentration of $\bar p_N$ via Chebyshev.** By Chebyshev's inequality,
$$P\!\left(|\bar p_N - 1/2| > \epsilon\right) \le \frac{\mathrm{Var}(\bar p_N)}{\epsilon^2} = \frac{1}{4N(2\alpha+1)\,\epsilon^2}.$$
For probability $\ge 1 - \delta = 0.95$, require $\epsilon^2 \ge \mathrm{Var}(\bar p_N)/\delta$. At $N = 50, \alpha = 0.1$: $\mathrm{Var}(\bar p_N) = 1/240 \approx 0.00417$; with $\delta = 0.05$, $\epsilon^2 \ge 0.00417 / 0.05 = 1/12 \approx 0.0833$, so $\epsilon \ge 0.289$.

$\Rightarrow \bar p_N \in [0.211, 0.789]$ with probability $\ge 0.95$.

**Step 2 — Per-realisation hard bound on $\max_i |p_i - \bar p_N|$.** For any $p_i \in [0, 1]$ and any $\bar p_N \in [0, 1]$:
$$|p_i - \bar p_N| \le \max(\bar p_N,\ 1 - \bar p_N),$$
because the distance from an interior point $\bar p_N$ to anything in $[0, 1]$ is maximised at the endpoints (§D1.6 eq. D1.15). On the Step-1 high-probability event, $\max(\bar p_N, 1 - \bar p_N) \le 0.789$, so
$$\max_i |p_i - \bar p_N| \le 0.789 \quad \text{w.p. } \ge 0.95.$$

**Step 3 — Squaring.**
$$\max_i (p_i - \bar p_N)^2 \le 0.789^2 \approx 0.623 \quad \text{w.p. } \ge 0.95.$$

Hence $\zeta^2 \le 0.623 \cdot G^2$ with probability $\ge 0.95$.

**Remark — why Chebyshev, not CLT?** CLT would give a narrower 95% CI ($[0.374, 0.626]$ using std $0.0645$ and $z_{0.975} = 1.96$), but Chebyshev is distribution-free and yields a non-asymptotic hard bound. Since FLSNN A4 is a *for-all-$i$* sup-norm assumption, the conservative Chebyshev bound is the appropriate one to plug into the convergence theorem — the source file §D1.6 uses Chebyshev for the same reason.

### 1.4 Resulting claim
$$\zeta^2 \le 0.623 \cdot G^2 \quad \text{with probability} \ge 0.95$$

**Comparison**: mean-square gives $c_\alpha = 0.204$ at same $\alpha, N$.
Ratio $\approx 3.05 \times$ — **sup-norm is conservative**.

### 1.5 Which one to use?
- Plugging into FLSNN A4 **verbatim** → sup-norm (0.623)
- For intuition / average-case → mean-square (0.204)
- For Corollary 1's magnitude argument, mean-square is what enters
  $z^2$ through the bound decomposition; sup-norm is a worst-case
  upper bound

We report both in §IV.C (main paper) and note the $3.05 \times$
gap as a finite-$N$ concentration effect.

---

## 2. A-decomposition (the topology-constant bundle)

### 2.1 Recap
Corollary 1 uses $A := P \cdot Q \cdot L$ where
- $P = \sqrt{7E(E-1) + 7E^2 R(R-1)} / (NRE\pi_0)$
- $Q = \sqrt{2C^2 \tilde\tau / (9\rho^2 L^2) + 5}$
- $L$ = smoothness constant

$P$ and $Q$ are FLSNN's own Eq. (26) constants. We do not re-derive
them; we only track which scheme-dependent quantities they contain.

### 2.2 Scheme dependence
- $P$: depends only on $N, R, E, \pi_0$ — **not scheme-dependent**
- $Q$: contains $\rho$ (scheme-dependent) and $\tilde\tau$
  (scheme-dependent)
  - AllReduce: $\rho = 0.5, \tilde\tau = 1 \Rightarrow Q^2 = 2C^2/(2.25 L^2) + 5$
  - Gossip (chain, $N=5$): $\rho \approx 0.05$ (rough lower bound for
    chain graph Laplacian) $\Rightarrow Q^2 = 2 C^2/(0.0225 L^2) + 5
    \approx 89 C^2/L^2 + 5$
  - RelaySum (chain, $N=5$): $\rho$ comparable to Gossip but
    $\tilde\tau = 5$ $\Rightarrow Q^2 \approx 444 C^2/L^2 + 5$

### 2.3 Implication
Gossip's and RelaySum's $Q^2$ bound is much larger than AllReduce's
(say 20–100× larger on chain $N=5$). Cubed root: $(Q_{\rm gossip}/Q_{\rm AR})^{2/3}$
could be ~8×. This means **AllReduce's $T_4$ upper bound is the
tightest of the three by a noticeable factor — but only when $T_4$
dominates**. When $T_1$ (scheme-independent) dominates instead, the
$A$-factor advantage is masked. That's Corollary 1's point.

### 2.4 Source
Eq. (26) constants are **verbatim from FLSNN**. We treat them as a
black box; our contribution is plugging Prop. 1's value of $\zeta^2$
into $z^2$ and observing the regime.

---

## 3. Scheme substitution table (what we adopt from FLSNN)

| Scheme | $\tilde\tau$ | $\rho$ (chain $N=5$) | Asymptotic $T_4$ rate |
|:----|:---:|:---:|:---:|
| AllReduce | 1 | 0.5 (ceiling) | $O(T^{-2/3})$ |
| Gossip    | 1 | $\sim 0.05$ | $O(T^{-2/3})$ |
| RelaySum  | 5 | $\sim 0.05$ | $O(T^{-2/3})$ |

**All three have the same asymptotic $T^{-2/3}$ rate** but different
multiplicative constants (via $\rho, \tilde\tau$).

**Key observation from FLSNN**: this table is entirely in their paper.
We adopt it verbatim.

**What we add**: the rate comparison against $T_1 = O(T^{-1/2})$ is
the rate argument of D-2 §2.1. FLSNN's Eq. (27) also contains $T_1$,
but FLSNN's Fig. 5 interpretation emphasises the topology-dependent
terms; we argue these terms are dominated by $T_1$ in our regime.

---

## 4. α-sensitivity of $c_\alpha$

### 4.1 Closed form
$c_\alpha = (N-1)/(4N(2\alpha+1))$ at $N=50$ simplifies to
$49/(200(2\alpha+1))$.

### 4.2 Sweep table

| $\alpha$ | $2\alpha+1$ | $c_\alpha$ | Regime |
|:-:|:-:|:-:|:-:|
| 0.01 | 1.02 | 0.2402 | Strong non-IID (most clients near 0 or 1) |
| 0.1  | 1.2  | 0.2042 | Our v2-A setting |
| 1    | 3    | 0.0817 | Mild non-IID (Beta(1,1) = Uniform) |
| 10   | 21   | 0.0117 | Near-IID |
| 100  | 201  | 0.0012 | Effectively IID |

### 4.3 Interpretation
- **$\alpha \to 0^+$**: $c_\alpha \to (N-1)/(4N) \le 0.25$, i.e.,
  the bound saturates at 1/4 (U-shaped Beta limit)
- **$\alpha \to \infty$**: $c_\alpha \to 0$ like $O(1/\alpha)$ (IID
  limit, $p_i \to 1/2$ delta)
- **Our $\alpha = 0.1$**: close to saturation (0.204 vs the 0.245 cap)

### 4.4 Why an α-sweep would be a strong v3 experiment
Corollary 1 predicts: if $c_\alpha G^2$ remains small relative to
$\sigma^2$, scheme spread stays at 0.1–0.3 dB. Decreasing $\alpha$
from 0.1 to 0.01 only increases $c_\alpha$ from 0.204 to 0.240 —
**too small a lever** to push into heterogeneity-dominated regime.

To truly flip the regime, one would need either:
- **More sources** (K = 10 instead of K = 2, like FLSNN's EuroSAT): $G^2$ grows with number of pairwise gradient gaps
- **Much smaller $\sigma^2$** (tiny batches, smaller lr): unlikely to help alone

This is why our paper focuses on the 2-source CUHK-CR setting and
does not claim α-sensitivity will reverse the ranking.

---

## 5. E-semantics impact analysis

### 5.1 The issue
Earlier in this project we noticed a potential terminology mismatch:
paper says "E SGD steps" in places where code does "E full dataset
passes" (local_iters = 2 epochs in `config.py:91`). Commit 4757e1d
fixed the paper text to use "local epochs / dataset passes."

### 5.2 Does this affect the theory?
Theorem 1 / Corollary 1 contain $E$ as an abstract local-iteration
counter in the $P$ constant. FLSNN's Eq. (26) treats $E$ as the
number of SGD steps between aggregations, not the number of epochs.

### 5.3 Numerical impact on $P$
With $K = 10$ satellites/plane, batch size 4, plane dataset ~200
images on average:
- **"E epochs" interpretation**: $E = 2$ epochs $\times$ 50 batches /
  epoch = 100 SGD steps
- **"E SGD steps" interpretation**: $E = 2$ steps

So $P \propto \sqrt{7E(E-1) + \ldots}$ with $E = 100$ vs $E = 2$
differs by factor $\sqrt{7 \cdot 9900 / (7 \cdot 2)} \approx \sqrt{4950}
\approx 70.4 \times$. Taking 2/3 power: $\approx 17 \times$ factor in
the magnitude of $T_4$.

### 5.4 Does this affect Corollary 1?
**No, on the rate argument** — $E$ is a constant (not a function of
$T$), so it doesn't change the $O(T^{-2/3})$ rate.

**Yes, on the magnitude** — a 17× multiplier in $T_4$'s constant
brings $T_4$ "closer" to $T_1$'s magnitude at finite $T$. The
"$T_1$-dominated regime" argument becomes more sensitive to the $E$
value.

### 5.5 Resolution for the paper
- §III.D uses "local epoch" consistently (`config.py:91`)
- §IV.A Assumption (A1-A4) preserves FLSNN's abstract $E$ without
  committing to a specific interpretation
- Corollary 1's bound is stated with the FLSNN $E$; we note the
  epoch-vs-step distinction in §III.D and its numerical consequence
  is acknowledged in §VI-H as a "single-seed + terminology-gap"
  caveat

### 5.6 Is there a bug?
**No.** The *code* is consistent (each satellite does 2 full epochs
per round). The *paper text* used to say "E SGD steps" informally;
that text was corrected in 4757e1d. The theoretical bounds use $E$
abstractly, which still works — the bound holds with $E$ interpreted
as the number of local SGD steps, which for us is $E_{\rm steps} = 2
\cdot N_{\rm batches/epoch}$. The constant $P$ in the bound then
uses $E_{\rm steps}$, not $E_{\rm epochs}$.

---

## 6. Summary for defense

**What we add to FLSNN's theory**:
1. Closed-form $\zeta^2$ for 2-source Dirichlet (Prop. 1)
2. Identification of the scheme-ranking collapse regime (Cor. 1)
3. Sup-norm vs mean-square clarification (D-1 §5.1, D-3 §1)
4. α-sensitivity table for practitioners (D-3 §4)

**What we DON'T add**:
- New convergence theorem — we use FLSNN's Thm 2 verbatim
- New topology constants ($\rho, \tilde\tau, C, C_1$) — FLSNN's
- Empirical measurement of $\sigma^2, G^2$ — caveat; v3 work

**Main theoretical claim**: "In our v2-A regime, FLSNN's Theorem 2
predicts scheme-rank spread $\le$ single-seed noise floor, not
RelaySum > Gossip > AllReduce as observed in FLSNN Fig. 5. The
reversal is a corollary of FLSNN's own theorem, not a contradiction."

**Main defensive line**: "Every ingredient is either (a) cited and
verified against a primary source, or (b) derived transparently in
§D1–D3 with justification per step. The novelty is the specific
combination, not any individual step. The main caveat (caveat iii
in §IV.D) is that we do not directly measure $\sigma^2, G$, so the
magnitude argument is qualitative. The rate argument is rigorous."
