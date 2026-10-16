"""Build every table and figure of the paper from ``results/``.

Usage:  python -m experiments.report [--out paper/generated]

Tables are written as LaTeX row bodies (``tab_*.tex``), figures as PDF, and every number quoted in the
text is collected in ``numbers.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from predictx import cohorts  # noqa: E402
from predictx.evaluation import corrected_resampled_ttest, per_split_metrics  # noqa: E402
from predictx.learners import ENSEMBLE_MEMBERS, LITERATURE_STANDARD  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
COHORTS = ["heart", "diabetes", "breast", "liver", "kidney"]
NAME = {"heart": "Coronary", "diabetes": "Diabetes", "breast": "Breast", "liver": "Liver", "kidney": "Kidney"}
LABEL = {"LR": "Logistic regression", "SVM": "SVM (RBF)", "KNN": "$k$-NN", "NB": "Naive Bayes",
         "RF": "Random forest", "MLP": "MLP", "XGB": "XGBoost (tuned)", "CAT": "CatBoost", "PFN": "TabPFN v2",
         "HARBOR": "HARBOR", "ENS": "PredictX"}
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9893"
TABLE_LAMBDAS = {"information": [0.0, 0.02, 0.05, 0.12], "decision": [0.0, 0.02]}
NUMBERS: dict = {}


def fmt(x: float, d: int = 3) -> str:
    """Signed number for LaTeX tables; values that round to zero are shown as +0."""
    x = 0.0 if abs(x) < 0.5 * 10 ** (-d) else x
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{d}f}"


def load(cohort: str, features: str = "full", protocol: str = "cv") -> pd.DataFrame:
    """Wide prediction table (one column per model) with the PredictX ensembles added."""
    frames = []
    for path in sorted(glob.glob(os.path.join(ROOT, "predictions", f"{cohort}-{features}-{protocol}", "*.csv"))):
        model = os.path.basename(path)[:-4]
        d = pd.read_csv(path).drop_duplicates(["rep", "fold", "row"])
        frames.append(d.set_index(["rep", "fold", "row", "y"])["p"].rename(model))
    wide = pd.concat(frames, axis=1, join="inner").reset_index()
    if all(m in wide for m in ENSEMBLE_MEMBERS):
        wide["ENS"] = wide[list(ENSEMBLE_MEMBERS)].mean(axis=1)
    aux = [m + "+aux" for m in ENSEMBLE_MEMBERS]
    if all(m in wide for m in aux):
        wide["ENS+aux"] = wide[aux].mean(axis=1)
    return wide


def models_of(wide: pd.DataFrame) -> list:
    """Model columns of a wide prediction table."""
    return [c for c in wide.columns if c not in ("rep", "fold", "row", "y")]


def summary(wide: pd.DataFrame) -> pd.DataFrame:
    """Mean AUROC, accuracy and Brier score of every model over all splits."""
    return per_split_metrics(wide, models_of(wide)).groupby("model")[["auroc", "accuracy", "brier"]].mean()


def paired(wide: pd.DataFrame, a: str, b: str, test_fraction: float = 0.2):
    """Corrected resampled t interval for the mean AUROC difference ``a - b`` over splits."""
    m = per_split_metrics(wide, [a, b]).pivot_table(index=["rep", "fold"], columns="model", values="auroc")
    return corrected_resampled_ttest(m[a] - m[b], test_fraction)


def _star(interval) -> str:
    """Marker for a confidence interval that excludes zero."""
    return "$^{*}$" if interval[1] > 0 or interval[2] < 0 else ""


def short(policy: str) -> str:
    """Compact panel label, e.g. ``fixed:EXT+FLU`` -> ``Ex+Fl``."""
    return policy.split(":")[1].replace("EXT", "Ex").replace("FLU", "Fl").replace("THA", "Th")


def best_literature(s: pd.DataFrame) -> str:
    """Literature-standard method with the highest mean AUROC."""
    return s.loc[list(LITERATURE_STANDARD)].auroc.idxmax()


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)

    def write(name: str, lines: list) -> None:
        with open(os.path.join(out, name), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    W = {c: load(c) for c in COHORTS}
    S = {c: summary(W[c]) for c in COHORTS}
    primary = {c: ("ENS+aux" if "ENS+aux" in S[c].index else "ENS") for c in COHORTS}

    # ---- Table: AUROC under P1
    order = list(LITERATURE_STANDARD) + ["CAT", "PFN", "HARBOR", "ENS"]
    lines = []
    for m in order:
        if m == "CAT":
            lines.append("\\midrule")
        cells = []
        for c in COHORTS:
            base = S[c].drop([i for i in S[c].index if "+" in i])
            v, best = S[c].loc[m, "auroc"], base.auroc.max()
            cells.append(f"\\textbf{{{v:.3f}}}" if abs(v - best) < 5e-4 else f"{v:.3f}")
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\midrule", "PredictX + cross-site & " + f"\\textbf{{{S['heart'].loc['ENS+aux', 'auroc']:.3f}}}"
              + " & -- & -- & -- & -- \\\\"]
    write("tab_auroc.tex", lines)

    # ---- Table: accuracy and Brier score of PredictX and the best literature-standard method
    lines = []
    for c in COHORTS:
        b, e = best_literature(S[c]), primary[c]
        lines.append(f"{NAME[c]} & {LABEL[b]} & {100 * S[c].loc[b, 'accuracy']:.1f} & {S[c].loc[b, 'brier']:.3f} & "
                     f"{100 * S[c].loc[e, 'accuracy']:.1f} & {S[c].loc[e, 'brier']:.3f} \\\\")
    write("tab_accuracy.tex", lines)

    # ---- Table: paired differences of PredictX against every comparator
    comparators = list(LITERATURE_STANDARD) + ["CAT", "PFN", "HARBOR"]
    wtl = {m: [0, 0, 0] for m in comparators}
    lines = []
    for m in comparators:
        row = []
        for c in COHORTS:
            mean, lo, hi, _ = paired(W[c], primary[c], m)
            k = 0 if lo > 0 else (2 if hi < 0 else 1)
            wtl[m][k] += 1
            row.append(fmt(mean) + ("$^{\\uparrow}$" if k == 0 else "$^{\\downarrow}$" if k == 2 else ""))
        lines.append(f"{LABEL[m]} & " + " & ".join(row) + f" & {wtl[m][0]}/{wtl[m][1]}/{wtl[m][2]} \\\\")
    write("tab_paired.tex", lines)
    NUMBERS["wins_ties_losses"] = [int(sum(v[i] for v in wtl.values())) for i in range(3)]

    # ---- Figure: PredictX minus best literature-standard method
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    labels = []
    for i, c in enumerate(COHORTS):
        b = best_literature(S[c])
        mean, lo, hi, p = paired(W[c], primary[c], b)
        ax.plot([lo, hi], [-i, -i], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(mean, -i, "o", color=BLUE, ms=5, mec="white", mew=1)
        labels.append(f"{NAME[c]} (vs {b})")
        NUMBERS[f"{c}_vs_best"] = dict(best=b, best_auroc=S[c].loc[b, "auroc"], ours=S[c].loc[primary[c], "auroc"],
                                       delta=mean, lo=lo, hi=hi, p=p)
    ax.axvline(0, color=GREY, lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks([-i for i in range(len(COHORTS))])
    ax.set_yticklabels(labels)
    ax.set_xlabel("$\\Delta$AUROC, PredictX $-$ best literature-standard method (95% CI)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(os.path.join(out, "fig_delta.pdf"), bbox_inches="tight")

    # ---- Table: published protocol (P2)
    published = {  # Hollmann et al., ICLR 2023, Table 2 (ROC AUC, five 50/50 splits)
        "diabetes": {"CatBoost": .8383, "XGBoost": .8378, "AutoGluon": .8391, "TabPFN n.e.": .8412, "TabPFN": .8410,
                     "TabPFN+AutoGluon": .8427},
        "breast": {"CatBoost": .9931, "XGBoost": .9896, "AutoGluon": .9933, "TabPFN n.e.": .9931, "TabPFN": .9934,
                   "TabPFN+AutoGluon": .9940},
        "liver": {"CatBoost": .7279, "XGBoost": .7171, "AutoGluon": .7230, "TabPFN n.e.": .7412, "TabPFN": .7379,
                  "TabPFN+AutoGluon": .7326},
    }
    lines = []
    for c in ("diabetes", "breast", "liver"):
        s = summary(load(c, protocol="half"))
        values = list(published[c].values()) + [s.loc["PFN", "auroc"], s.loc["CAT", "auroc"], s.loc["ENS", "auroc"]]
        best = max(values)
        lines.append(f"{NAME[c]} & " + " & ".join(f"\\textbf{{{v:.4f}}}" if v == best else f"{v:.4f}" for v in values)
                     + " \\\\")
        NUMBERS[f"{c}_half"] = dict(ours=s.loc["ENS", "auroc"], pfn=s.loc["PFN", "auroc"],
                                    best_published=max(published[c].values()))
    write("tab_published.tex", lines)

    # ---- Table: leave-one-member-out
    lines = []
    for c in COHORTS:
        w = W[c].copy()
        for drop in ENSEMBLE_MEMBERS:
            w[f"ENS-{drop}"] = w[[m for m in ENSEMBLE_MEMBERS if m != drop]].mean(axis=1)
        lines.append(f"{NAME[c]} & " + " & ".join(fmt(paired(w, "ENS", f"ENS-{d}")[0], 4) for d in ENSEMBLE_MEMBERS)
                     + " \\\\")
    write("tab_members.tex", lines)

    # ---- Table: cross-site training on the coronary endpoint
    lines = []
    for m in ("LR", "CAT", "PFN", "HARBOR", "ENS"):
        mean, lo, hi, _ = paired(W["heart"], f"{m}+aux", m)
        lines.append(f"{LABEL[m]} & {S['heart'].loc[m, 'auroc']:.3f} & {S['heart'].loc[m + '+aux', 'auroc']:.3f} & "
                     f"{fmt(mean)} \\\\")
        NUMBERS[f"crosssite_{m}"] = mean
    write("tab_crosssite.tex", lines)

    # ---- Table: deployed web-form models
    lines = []
    for c in COHORTS:
        features = "web" if c in ("liver", "kidney") else "full"
        s = summary(load(c, features))
        e = "ENS+aux" if c == "heart" else "ENS"
        b = best_literature(s)
        n_fields = len(cohorts.load_columns(c))
        lines.append(f"{NAME[c]} & {n_fields} & {s.loc[e, 'auroc']:.3f} & {100 * s.loc[e, 'accuracy']:.1f} & "
                     f"{LABEL[b]} & {s.loc[b, 'auroc']:.3f} \\\\")
        NUMBERS[f"{c}_web"] = dict(auroc=s.loc[e, "auroc"], acc=s.loc[e, "accuracy"], best=b,
                                   best_auroc=s.loc[b, "auroc"])
    write("tab_web.tex", lines)

    # ---- Table: controlled missingness
    path = os.path.join(ROOT, "missingness")
    if os.path.isdir(path):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(path, "*.csv")))])
        lines = []
        for data in ("WDBC", "ILPD"):
            for mech in ("MCAR", "MNAR"):
                g = d[(d.data == data) & (d.mechanism == mech)].pivot_table(index=["rep", "fold"], columns="model",
                                                                            values="auroc")
                if g.empty:
                    continue
                mean = g.mean()
                dv = corrected_resampled_ttest(g["two_channel"] - g["value_only"], 0.2)
                dm = corrected_resampled_ttest(g["two_channel"] - g["mask_concat"], 0.2)
                lines.append(f"{data} & {mech} & {mean['mask_only']:.3f} & {mean['LR']:.3f} & "
                             f"{mean['value_only']:.3f} & {mean['mask_concat']:.3f} & {mean['two_channel']:.3f} & "
                             f"{fmt(dv[0])} [{fmt(dv[1])}, {fmt(dv[2])}]{_star(dv)} & "
                             f"{fmt(dm[0])} [{fmt(dm[1])}, {fmt(dm[2])}]{_star(dm)} \\\\")
                NUMBERS[f"miss_{data}_{mech}"] = dict(k=len(g), tc_minus_v0=dv, tc_minus_mc=dm,
                                                      **{m: mean[m] for m in g.columns})
            lines.append("\\midrule")
        write("tab_missingness.tex", lines[:-1])

    # ---- Acquisition: table and frontier
    acq = pd.read_csv(os.path.join(ROOT, "acquisition.csv"))
    g = acq.groupby("policy")[["cost", "n_tests", "auroc", "acc"]].mean()
    fixed = g[g.index.str.startswith("fixed")].sort_values("cost")
    voi = {}
    for k in ("information", "decision"):
        rows = g[g.index.str.startswith(f"voi-{k}")]
        voi[k] = rows.assign(lam=[float(i.split(":")[1]) for i in rows.index]).sort_values("lam")
    lines = [f"Fixed: {short(n)} & {r.cost:.2f} & {r.n_tests:.2f} & {r.auroc:.3f} & {100 * r.acc:.1f} \\\\"
             for n, r in fixed.iterrows()]
    for k, title in (("information", "information"), ("decision", "decision")):
        lines.append("\\midrule")
        shown = voi[k][voi[k].lam.isin(TABLE_LAMBDAS[k])]  # the figure shows the full sweep
        lines += [f"VoI ({title}), $\\lambda={r.lam:g}$ & {r.cost:.2f} & {r.n_tests:.2f} & {r.auroc:.3f} & "
                  f"{100 * r.acc:.1f} \\\\" for _, r in shown.iterrows()]
    write("tab_acquisition.tex", lines)
    full, best = fixed.loc["fixed:EXT+FLU+THA"], voi["information"].iloc[0]
    NUMBERS["acquisition"] = dict(full_cost=full.cost, full_auroc=full.auroc, full_acc=full.acc, voi_cost=best.cost,
                                  voi_auroc=best.auroc, voi_acc=best.acc, voi_tests=best.n_tests,
                                  baseline_auroc=fixed.loc["fixed:none"].auroc)

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(voi["information"].cost, voi["information"].auroc, "-o", color=BLUE, lw=2, ms=4, mec="white", mew=0.8,
            label="VoI policy (information utility)")
    ax.plot(voi["decision"].cost, voi["decision"].auroc, "-s", color=ORANGE, lw=2, ms=4, mec="white", mew=0.8,
            label="VoI policy (decision utility)")
    ax.plot(fixed.cost, fixed.auroc, "D", color=GREY, ms=4, label="Fixed test panels")
    offsets = {"none": (4, -9), "Ex+Fl+Th": (-14, -11), "Ex": (-4, 5), "Ex+Fl": (4, 2), "Ex+Th": (4, -9),
               "Fl": (4, 2), "Th": (4, -9), "Fl+Th": (-6, -11)}
    for n, r in fixed.iterrows():
        ax.annotate(short(n), (r.cost, r.auroc), textcoords="offset points", xytext=offsets[short(n)], fontsize=6,
                    color="#52514e")
    ax.set_xlabel("Mean testing cost per patient")
    ax.set_ylabel("AUROC")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_acquisition.pdf"))

    # ---- quoted numbers
    for c in COHORTS:
        base = S[c].drop([i for i in S[c].index if "+" in i and i != primary[c]])
        base = base.drop("ENS") if primary[c] != "ENS" else base
        NUMBERS[f"{c}_brier_rank"] = int(base.brier.rank().loc[primary[c]])
        NUMBERS[f"{c}_auroc_rank"] = int(base.auroc.rank(ascending=False, method="min").loc[primary[c]])
        NUMBERS[f"{c}_n_models"] = len(base)
    for c in COHORTS:
        NUMBERS[f"{c}_summary"] = S[c].round(4).to_dict("index")
    with open(os.path.join(out, "numbers.json"), "w") as fh:
        json.dump(NUMBERS, fh, indent=1, default=float)
    print(f"wrote tables, figures and numbers.json to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "paper", "generated"))
    main(parser.parse_args().out)
