"""Exact binomial (Clopper-Pearson) sample-size and precision calculations for the enriched field study
in docs/protocol.md section 11. Prints every number quoted there."""
import numpy as np
from scipy.stats import beta, binom, poisson


def cp(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def power_lower(n, p, floor, alpha=0.05):
    """P(two-sided CP lower bound on a proportion > floor) when the true proportion is p."""
    k = np.arange(n + 1)
    lo = np.array([cp(int(x), n, alpha)[0] for x in k])
    return binom.pmf(k, n, p)[lo > floor].sum()


def min_n(p, floor, power=0.8, start=10, stop=20000, step=1):
    """Smallest n with power >= target that also holds for the next 10 n (binomial power is saw-toothed)."""
    for n in range(start, stop, step):
        if all(power_lower(m, p, floor) >= power for m in range(n, n + 10)):
            return n
    return None


print("== Sensitivity, enriched cancers (n = 150)")
for p in (0.85, 0.90, 0.95):
    k = round(p * 150)
    lo, hi = cp(k, 150)
    print(f"  expected sens {p:.2f}: observed {k}/150 -> 95% CI {lo:.3f}-{hi:.3f}")
for p, floor in ((0.90, 0.80), (0.90, 0.85), (0.95, 0.90), (0.92, 0.85)):
    print(f"  power(n=150, true {p}, lower bound > {floor}) = {power_lower(150, p, floor):.3f}; "
          f"min n for 80% = {min_n(p, floor)}; for 90%: ", end="")
    n90 = next(n for n in range(10, 3000) if all(power_lower(m, p, floor) >= 0.9 for m in range(n, n + 10)))
    print(n90)

print("== Specificity / recall, screening-camp women (3,000 enrolled)")
for prev in (0.002, 0.005):
    exp_ca = 3000 * prev
    print(f"  prevalence {prev*1000:.0f}/1000: expected camp cancers {exp_ca:.0f}; "
          f"P(>=10 cancers)={1-poisson.cdf(9, exp_ca):.2f}")
n_neg = 2985
for recall in (0.03, 0.05, 0.10):
    k = round(recall * n_neg)
    lo, hi = cp(k, n_neg)
    print(f"  true false-recall {recall:.2f}: {k}/{n_neg} -> 95% CI {lo:.4f}-{hi:.4f} "
          f"(false recalls per 1,000: {1000*recall:.0f}, CI {1000*lo:.0f}-{1000*hi:.0f})")
for spec, floor in ((0.95, 0.93), (0.95, 0.94), (0.90, 0.88)):
    print(f"  power(n={n_neg}, true spec {spec}, lower bound > {floor}) = {power_lower(n_neg, spec, floor):.3f}")
for spec, floor in ((0.95, 0.94), (0.90, 0.88)):
    n = next(n for n in range(200, 20000, 25) if power_lower(n, spec, floor) >= 0.8 and power_lower(n + 25, spec, floor) >= 0.8)
    print(f"  min non-cancer women for 80% power, true spec {spec}, lower bound > {floor}: ~{n}")

print("== Certification at a new site (zero / one / two misses, one-sided 95%)")
for target in (0.85, 0.90, 0.95):
    out = []
    for misses in (0, 1, 2):
        n = misses + 1
        while beta.ppf(0.05, n - misses, misses + 1) < target:
            n += 1
        out.append(n)
    print(f"  sensitivity >= {target}: n cancers = {out}")
for rmax in (0.05, 0.10):
    for obs in (0.03, 0.04, 0.07, 0.08):
        if obs >= rmax:
            continue
        n = 50
        while beta.ppf(0.95, round(obs * n) + 1, n - round(obs * n)) > rmax:
            n += 10
        print(f"  recall <= {rmax:.2f} certified (one-sided 95% upper bound) when observed {obs:.2f}: n normals ~{n}")

print("== Paired comparison on cancers: device (+CBE) vs CBE alone, exact McNemar, two-sided alpha 0.05")
from scipy.stats import binomtest
rng = np.random.default_rng(0)


def mcnemar_power(n, p10, p01, sims=4000):
    hits = 0
    for _ in range(sims):
        cells = rng.multinomial(n, [p10, p01, 1 - p10 - p01])
        b, c = cells[0], cells[1]
        if b + c > 0 and b > c and binomtest(b, b + c, 0.5).pvalue < 0.05:
            hits += 1
    return hits / sims


for label, p10, p01 in (("+10 points (disc. 13%/3%)", 0.13, 0.03), ("+15 points (disc. 18%/3%)", 0.18, 0.03),
                        ("+20 points (disc. 22%/2%)", 0.22, 0.02)):
    print(f"  {label}: power with 150 cancers = {mcnemar_power(150, p10, p01):.3f}; "
          f"with 100 = {mcnemar_power(100, p10, p01):.3f}")
