"""Power of the confirmatory intersection-union test (IUT) for CBE + AI vs CBE alone.

Because the CBE + AI rule recalls a superset of the women CBE recalls, added sensitivity
d_sens = P(CBE-, AI+ | cancer) >= 0 and added recall d_rec = P(CBE-, AI+ | cancer-free) >= 0.
The test passes only if BOTH hold:
  H1a: d_sens > delta     (exact one-sided 97.5% Clopper-Pearson lower bound on the enriched cancers > delta)
  H1b: d_rec  < cap       (exact one-sided 97.5% Clopper-Pearson upper bound on camp women < cap)
An IUT controls the familywise type I error at the level of each component test, so no
multiplicity adjustment is needed. The two samples are independent, so power = product.
"""
import numpy as np
from scipy.stats import beta, binom


def lower(k, n, a=0.025):
    return beta.ppf(a, k, n - k + 1) if k > 0 else 0.0


def upper(k, n, a=0.025):
    return beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0


def p_lower_gt(n, p, delta):
    k = np.arange(n + 1)
    return binom.pmf(k, n, p)[np.array([lower(int(x), n) for x in k]) > delta].sum()


def p_upper_lt(n, p, cap):
    k = np.arange(n + 1)
    return binom.pmf(k, n, p)[np.array([upper(int(x), n) for x in k]) < cap].sum()


N_CA, N_NEG, CAP = 150, 2985, 0.05
print(f"Enriched cancers n={N_CA}; camp cancer-free women n={N_NEG}; added-recall cap {CAP:.0%}\n")
print("H1a power (added sensitivity lower bound > delta):")
for delta in (0.0, 0.03, 0.05):
    print(f"  delta={delta:.2f}: " + "  ".join(f"true +{d:.2f} -> {p_lower_gt(N_CA, d, delta):.3f}"
                                         for d in (0.08, 0.10, 0.12, 0.15, 0.20)))
print("H1b power (added recall upper bound < 5%):")
print("  " + "  ".join(f"true {r:.3f} -> {p_upper_lt(N_NEG, r, CAP):.3f}" for r in (0.030, 0.035, 0.040, 0.042, 0.045)))
print("\nJoint IUT power (product), delta = 0.05:")
rows = [(d, r, p_lower_gt(N_CA, d, 0.05) * p_upper_lt(N_NEG, r, CAP)) for d in (0.10, 0.12, 0.15, 0.20)
        for r in (0.030, 0.035, 0.040)]
for d, r, pw in rows:
    print(f"  true added sens +{d:.2f}, true added recall {r:.3f}: power {pw:.3f}")
print("\nCancers needed for 80% / 90% H1a power at delta = 0.05:")
for d in (0.10, 0.12, 0.15):
    out = []
    for target in (0.8, 0.9):
        n = next(n for n in range(30, 2000) if all(p_lower_gt(m, d, 0.05) >= target for m in range(n, n + 10)))
        out.append(n)
    print(f"  true +{d:.2f}: {out[0]} / {out[1]}")
print("\nCamp women needed for 80% / 90% H1b power (cap 5%):")
for r in (0.035, 0.040):
    out = []
    for target in (0.8, 0.9):
        n = next(n for n in range(500, 40000, 50) if p_upper_lt(n, r, CAP) >= target and p_upper_lt(n + 50, r, CAP) >= target)
        out.append(n)
    print(f"  true {r:.3f}: ~{out[0]} / ~{out[1]}")
print("\nExtra recalls per extra cancer detected (screening prevalence), at true added recall 4%:")
for prev in (0.002, 0.005):
    for d in (0.10, 0.15):
        print(f"  prevalence {prev*1000:.0f}/1000, added sens +{d:.2f}: extra cancers/1000 = {1000*prev*d:.2f}, "
              f"extra recalls/1000 = {1000*0.04*(1-prev):.0f}, ratio = {0.04*(1-prev)/(prev*d):.0f}")

print("\nStratum precision (non-palpable or <= 2 cm, imaging-detected; >= 40% of 150 -> n >= 60):")
for n in (60, 75):
    for s in (0.60, 0.70, 0.80):
        k = round(s * n)
        print(f"  n={n}, AI sensitivity {s:.2f}: {k}/{n} -> 95% CI {lower(k, n):.3f}-{upper(k, n):.3f}")

print("\nCertification lower bounds (one-sided 95% Clopper-Pearson) with 0 / 1 / 2 misses:")
for n in (29, 46, 59, 93, 124, 149):
    print(f"  n={n}: " + "  ".join(f"{m} miss -> {beta.ppf(0.05, n - m, m + 1):.3f}" for m in (0, 1, 2)))

print("\nRun-in certification of the added-recall cap (one-sided 95% upper bound < 5%):")
for n in (300, 400, 500, 600):
    kmax = max(k for k in range(n) if beta.ppf(0.95, k + 1, n - k) < CAP)
    print(f"  n={n} run-in camp women: certified if <= {kmax} AI-only recalls (observed added recall <= {kmax/n:.3f})")
