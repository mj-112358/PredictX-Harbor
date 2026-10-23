"""Build every table, figure and quoted number of the paper from ``results/``.

    python -m experiments.report [--out paper/generated]

Writes LaTeX table bodies ``tab_*.tex``, figures ``fig_*.pdf``, ``numbers.json`` and ``macros.tex``
(one ``\\newcommand`` per number quoted in the text). Sections whose inputs are missing are skipped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sweepscreen.data.loaders import LOADERS  # noqa: E402
from sweepscreen.evaluation.detection import average_precision, froc  # noqa: E402
from sweepscreen.evaluation.metrics import auroc, bootstrap_ci, sens_at_spec  # noqa: E402
from sweepscreen.safety import PlattCalibrator, calibration_slope_intercept, expected_calibration_error  # noqa: E402
from sweepscreen.safety.calibration import reliability_curve  # noqa: E402

from .common import ROOT, parser, read_predictions  # noqa: E402

SITE_ORDER = ["BUSI", "BUS-BRA", "BrEaST", "UDIAT", "BreastVid"]
# validated categorical palette (fixed order, one hue per site); markers give a second channel
COLOR = dict(zip(SITE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKER = dict(zip(SITE_ORDER, ["o", "s", "^", "D", "v"]))
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
COUNTRY = {"BUSI": "Egypt", "BUS-BRA": "Brazil", "BrEaST": "Poland", "UDIAT": "Spain", "BreastVid": "China"}
NUM: dict = {}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def f3(x, d=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci(t, d=3):
    est, lo, hi = t
    return "--" if not np.isfinite(est) else f"{est:.{d}f} ({lo:.{d}f}--{hi:.{d}f})"


def macro_name(key: str) -> str:
    words = re.split(r"[^A-Za-z]+", key.replace("0", "zero").replace("1", "one").replace("2", "two")
                     .replace("3", "three").replace("4", "four").replace("5", "five").replace("6", "six")
                     .replace("7", "seven").replace("8", "eight").replace("9", "nine"))
    return "N" + "".join(w[:1].upper() + w[1:] for w in words if w)


def put(key, value, d=3):
    NUM[key] = value
    return value


def ordered(sites):
    return [s for s in SITE_ORDER if s in sites] + sorted(s for s in sites if s not in SITE_ORDER)


# ---------------------------------------------------------------------------------------------- datasets
def datasets(processed: Path, out: Path):
    t = pd.read_csv(processed / "records.csv")
    lines = []
    for site in ordered(set(t.site)):
        g = t[t.site == site]
        boxes = g.boxes.map(lambda b: len(json.loads(b))).sum()
        meta = [c for c, col in (("age", "age"), ("BI-RADS", "birads"), ("device", "device")) if g[col].notna().any()]
        unit = "frames" if g.video_id.notna().any() else "images"
        lines.append(f"{site} & {COUNTRY.get(site, '')} & {len(g):,} {unit} & {g.patient_id.nunique():,} & "
                     f"{(g.label == 'normal').sum():,} & {(g.label == 'benign').sum():,} & {(g.label == 'malignant').sum():,} & "
                     f"{int(g.lesion_present.sum()):,} & {boxes:,} & {', '.join(meta) or '--'} \\\\")
        put(f"{site}_records", int(len(g)))
        put(f"{site}_patients", int(g.patient_id.nunique()))
        if g.video_id.notna().any():
            put(f"{site}_videos", int(g.video_id.nunique()))
    (out / "tab_datasets.tex").write_text("\n".join(lines) + "\n")
    put("total_records", int(len(t)))
    put("total_sites", int(t.site.nunique()))
    put("discordant_records", int(t.discordant.sum()))
    put("cross_site_records", int(t.cross_site.sum()))
    dd = json.loads((processed / "dedup_summary.json").read_text())
    rows = [f"{r['site']} & {r['records']:,} & {r['dup_clusters_size_gt1']:,} & {r['records_in_dup_clusters']:,} & "
            f"{r['discordant_records']:,} & {r['cross_site_records']:,} \\\\" for r in
            sorted(dd["sites"], key=lambda r: ordered([x["site"] for x in dd["sites"]]).index(r["site"]))]
    (out / "tab_dedup.tex").write_text("\n".join(rows) + "\n")
    put("dedup_max_hamming", dd["max_hamming"])


# ---------------------------------------------------------------------------------------------- LOSO
def site_metrics(pred: pd.DataFrame, n_boot: int):
    m = {}
    g = pred.group_id.values
    if pred.lesion.nunique() == 2:
        m["det_auroc"] = bootstrap_ci(auroc, pred.lesion.values, pred.frame_prob.values, g, n_boot)
        m["det_sens95"] = bootstrap_ci(lambda a, b: sens_at_spec(a, b, 0.95), pred.lesion.values, pred.frame_prob.values, g, n_boot)
    les = pred[pred.lesion == 1]
    preds = [[(tuple(c[0]), c[1]) for c in cands] for cands in pred.candidates]
    gts = [[tuple(b) for b in gb] for gb in pred.gt_boxes]
    fr = froc(preds, gts)
    m["froc"] = fr
    m["ap50"] = average_precision(preds, gts, 0.5)
    lm = les[les.label != "normal"]
    if lm.label.nunique() == 2:
        y = (lm.label == "malignant").astype(int).values
        m["mal_auroc_gt"] = bootstrap_ci(auroc, y, lm.mal_gt.values, lm.group_id.values, n_boot)
        m["mal_auroc_det"] = bootstrap_ci(auroc, y, lm.mal_det.values, lm.group_id.values, n_boot)
        m["mal_sens90"] = bootstrap_ci(lambda a, b: sens_at_spec(a, b, 0.90), y, lm.mal_gt.values, lm.group_id.values, n_boot)
    return m


def loso(results: Path, out: Path, n_boot: int):
    files = {p.stem[5:]: p for p in (results / "predictions").glob("loso_*.jsonl")}
    if not files:
        return {}
    allm, lines, shift = {}, [], []
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for site in ordered(files):
        pred = read_predictions(files[site])
        m = site_metrics(pred, n_boot)
        allm[site] = m
        fr = m["froc"]
        lines.append(f"{site} & {ci(m.get('det_auroc', (np.nan,) * 3))} & {ci(m.get('det_sens95', (np.nan,) * 3))} & "
                     f"{f3(fr['at'][0.5])} & {f3(fr['at'][1.0])} & {f3(m['ap50'])} & "
                     f"{ci(m.get('mal_auroc_gt', (np.nan,) * 3))} & {ci(m.get('mal_auroc_det', (np.nan,) * 3))} \\\\")
        for k in ("det_auroc", "mal_auroc_gt", "mal_auroc_det", "det_sens95", "mal_sens90"):
            if k in m:
                put(f"loso_{site}_{k}", m[k][0])
                put(f"loso_{site}_{k}_lo", m[k][1])
                put(f"loso_{site}_{k}_hi", m[k][2])
        put(f"loso_{site}_ap50", m["ap50"])
        put(f"loso_{site}_froc_half", fr["at"][0.5])
        put(f"loso_{site}_froc_one", fr["at"][1.0])
        x, y = np.asarray(fr["fppi"]), np.asarray(fr["sensitivity"])
        keep = x <= 4
        ax.plot(x[keep], y[keep], color=COLOR.get(site, INK), linewidth=2, label=site)
        if keep.any():
            ax.plot(x[keep][-1], y[keep][-1], MARKER.get(site, "o"), color=COLOR.get(site, INK), markersize=5)
        ip = results / "predictions" / f"internal_{site}.jsonl"
        if ip.exists():
            mi = site_metrics(read_predictions(ip), n_boot)
            shift.append(f"{site} & {f3(mi['ap50'])} & {f3(m['ap50'])} & {ci(mi.get('mal_auroc_gt', (np.nan,) * 3))} & "
                         f"{ci(m.get('mal_auroc_gt', (np.nan,) * 3))} \\\\")
            put(f"internal_{site}_ap50", mi["ap50"])
            if "mal_auroc_gt" in mi:
                put(f"internal_{site}_mal_auroc_gt", mi["mal_auroc_gt"][0])
    (out / "tab_loso.tex").write_text("\n".join(lines) + "\n")
    (out / "tab_shift.tex").write_text("\n".join(shift) + "\n")
    style(ax)
    ax.set_xlabel("False positives per frame", fontsize=8, color=INK)
    ax.set_ylabel("Lesion sensitivity", fontsize=8, color=INK)
    ax.set_xscale("symlog", linthresh=0.125)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_froc.pdf")
    plt.close(fig)
    return allm


# ---------------------------------------------------------------------------------------------- calibration
def calibration(results: Path, out: Path, m_local: int = 50, draws: int = 50):
    files = {p.stem[5:]: p for p in (results / "predictions").glob("loso_*.jsonl")}
    lines = []
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    rng = np.random.default_rng(0)
    for site in ordered(files):
        pred = read_predictions(files[site])
        lm = pred[(pred.lesion == 1) & (pred.label != "normal")]
        if lm.label.nunique() < 2 or len(lm) < m_local + 20:
            continue
        if lm.video_id.notna().any():          # one frame per video so units are independent
            lm = lm.groupby("video_id").sample(1, random_state=0)
        y = (lm.label == "malignant").astype(int).values
        p = lm.mal_gt.values
        slope, inter = calibration_slope_intercept(p, y)
        ece = expected_calibration_error(p, y)
        after = []
        if len(lm) >= m_local + 20:
            for _ in range(draws):
                idx = rng.permutation(len(y))
                a, b = idx[:m_local], idx[m_local:]
                if len(np.unique(y[a])) < 2:
                    continue
                cal = PlattCalibrator().fit(p[a], y[a])
                q = cal(p[b])
                after.append((*calibration_slope_intercept(q, y[b]), expected_calibration_error(q, y[b])))
        after = np.median(np.asarray(after), axis=0) if after else (np.nan,) * 3
        lines.append(f"{site} & {len(y):,} & {f3(slope, 2)} & {f3(inter, 2)} & {f3(ece)} & "
                     f"{f3(after[0], 2)} & {f3(after[1], 2)} & {f3(after[2])} \\\\")
        put(f"cal_{site}_slope", slope)
        put(f"cal_{site}_intercept", inter)
        put(f"cal_{site}_ece", ece)
        put(f"cal_{site}_ece_after", float(after[2]))
        rc = reliability_curve(p, y, 8)
        ax.plot([r[0] for r in rc], [r[1] for r in rc], marker=MARKER.get(site, "o"), color=COLOR.get(site, INK),
                linewidth=2, markersize=5, label=site)
    put("cal_local_cases", m_local)
    (out / "tab_calibration.tex").write_text("\n".join(lines) + "\n")
    style(ax)
    ax.set_xlabel("Predicted malignancy probability", fontsize=8, color=INK)
    ax.set_ylabel("Observed malignant fraction", fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_reliability.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- robustness
def robustness(results: Path, out: Path):
    files = sorted((results / "robustness").glob("*.jsonl"))
    if not files:
        return
    import joblib

    rows = []
    for f in files:
        site = f.stem
        rob = read_predictions(f)
        base = read_predictions(results / "predictions" / f"loso_{site}.jsonl").set_index("uid")
        thr = joblib.load(results / "models" / f"loso_{site}.joblib").frame_threshold
        base = base.loc[rob.uid.unique()]
        b_sens = float((base[base.lesion == 1].frame_prob >= thr).mean())
        les_b = base[(base.lesion == 1) & (base.label != "normal")]
        b_mal = auroc((les_b.label == "malignant").astype(int), les_b.mal_gt) if les_b.label.nunique() == 2 else np.nan
        for (deg, lv), g in rob.groupby(["degradation", "level"]):
            les = g[(g.lesion == 1) & (g.label != "normal")]
            mal = auroc((les.label == "malignant").astype(int), les.mal_gt) if les.label.nunique() == 2 else np.nan
            rows.append({"site": site, "degradation": deg, "level": lv,
                         "d_sens": float((g[g.lesion == 1].frame_prob >= thr).mean()) - b_sens,
                         "d_mal": mal - b_mal, "flagged": float(1 - g.adequate.mean())})
    r = pd.DataFrame(rows)
    agg = r.groupby(["degradation", "level"]).agg(d_sens=("d_sens", "mean"), d_mal=("d_mal", "mean"),
                                                   flagged=("flagged", "mean")).reset_index()
    from sweepscreen.preprocess.augment import LEVELS
    lines = []
    for deg in LEVELS:
        for lv in LEVELS[deg]:
            a = agg[(agg.degradation == deg) & np.isclose(agg.level, lv)]
            if len(a):
                a = a.iloc[0]
                lines.append(f"{deg.replace('_', ' ')} & {lv:g} & {a.d_sens:+.3f} & {f3(a.d_mal) if np.isnan(a.d_mal) else f'{a.d_mal:+.3f}'} & {a.flagged:.3f} \\\\")
                put(f"rob_{deg}_{lv:g}_dsens", float(a.d_sens))
                put(f"rob_{deg}_{lv:g}_flagged", float(a.flagged))
    (out / "tab_robustness.tex").write_text("\n".join(lines) + "\n")
    worst = agg.loc[agg.d_sens.idxmin()]
    put("rob_worst_degradation", f"{worst.degradation.replace('_', ' ')} ({worst.level:g})")
    put("rob_worst_dsens", float(worst.d_sens))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    for k, deg in enumerate(LEVELS):
        a = agg[agg.degradation == deg].sort_values("level", key=lambda s: [LEVELS[deg].index(v) if v in LEVELS[deg] else 0 for v in s])
        ax.plot(range(1, len(a) + 1), a.d_sens, marker="osD^v<>"[k], color=palette[k], linewidth=2, markersize=5,
                label=deg.replace("_", " "))
    style(ax)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_xticks([1, 2, 3], ["mild", "moderate", "severe"])
    ax.set_ylabel("Change in lesion sensitivity", fontsize=8, color=INK)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_robustness.pdf")
    plt.close(fig)


def _robustness_drop(results: Path) -> pd.DataFrame:
    """Mean change in lesion sensitivity per degradation/level (pooled over sites) for one results folder."""
    import joblib

    rows = []
    for f in sorted((results / "robustness").glob("*.jsonl")):
        site = f.stem
        rob = read_predictions(f)
        base = read_predictions(results / "predictions" / f"loso_{site}.jsonl").set_index("uid").loc[rob.uid.unique()]
        thr = joblib.load(results / "models" / f"loso_{site}.joblib").frame_threshold
        b = float((base[base.lesion == 1].frame_prob >= thr).mean())
        for (deg, lv), g in rob.groupby(["degradation", "level"]):
            rows.append({"site": site, "degradation": deg, "level": lv,
                         "d_sens": float((g[g.lesion == 1].frame_prob >= thr).mean()) - b})
    return pd.DataFrame(rows)


def ablation(results: Path, out: Path):
    """Augmentation ablation: clean LOSO AP50 and mean sensitivity change under degradation, with vs without."""
    other = results / "ablation_noaug"
    if not (other / "predictions").exists():
        return
    lines = []
    for site in ordered({p.stem[5:] for p in (other / "predictions").glob("loso_*.jsonl")}):
        aps = []
        for res in (results, other):
            pr = read_predictions(res / "predictions" / f"loso_{site}.jsonl")
            aps.append(average_precision([[(tuple(c[0]), c[1]) for c in cs] for cs in pr.candidates],
                                         [[tuple(b) for b in gb] for gb in pr.gt_boxes], 0.5))
        drops = []
        for res in (results, other):
            d = _robustness_drop(res) if (res / "robustness").exists() else pd.DataFrame()
            d = d[d.site == site] if len(d) else d
            drops.append(float(d.d_sens.mean()) if len(d) else float("nan"))
        lines.append(f"{site} & {f3(aps[1])} & {f3(aps[0])} & {f3(drops[1])} & {f3(drops[0])} \\")
        put(f"abl_{site}_ap_noaug", aps[1])
        put(f"abl_{site}_ap_aug", aps[0])
        put(f"abl_{site}_drop_noaug", drops[1])
        put(f"abl_{site}_drop_aug", drops[0])
    (out / "tab_ablation.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------- quality, video, certify, runtime
def quality(results: Path, out: Path):
    p = results / "quality.json"
    if not p.exists():
        return
    q = json.loads(p.read_text())
    modes = [k[7:] for k in next(iter(q.values())) if k.startswith("reject_")]
    lines = [f"{s} & {f3(v['pass_rate'])} & " + " & ".join(f3(v.get(f'reject_{m}')) for m in modes) + " \\\\"
             for s, v in ((s, q[s]) for s in ordered(q))]
    (out / "tab_quality.tex").write_text("\n".join(lines) + "\n")
    put("q_min_pass", min(v["pass_rate"] for v in q.values()))
    put("q_max_pass", max(v["pass_rate"] for v in q.values()))
    rej = [v[f"reject_{m}"] for v in q.values() for m in modes if f"reject_{m}" in v]
    put("q_min_reject", min(rej) if rej else float("nan"))


def video(results: Path, out: Path):
    p = results / "video.json"
    if not p.exists():
        return
    v = json.loads(p.read_text())
    lines = []
    for w, r in v["windows"].items():
        lines.append(f"{w} & {r['clips_detected']}/{v['n_videos']} ({f3(r['clip_detection_rate'])}) & "
                     f"{f3(r['lesion_free_fp_rate'])} & {f3(r['median_frames_to_detection'], 0)} \\\\")
        put(f"vid_w{w}_det", r["clip_detection_rate"])
        put(f"vid_w{w}_fp", r["lesion_free_fp_rate"])
    (out / "tab_video.tex").write_text("\n".join(lines) + "\n")
    put("vid_n", v["n_videos"])
    put("vid_mal_auroc_det", v["video_malignancy_auroc_detected_box"][0])
    put("vid_mal_auroc_gt", v["video_malignancy_auroc_gt_box"][0])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ws = [int(w) for w in v["windows"]]
    ax.plot(ws, [v["windows"][str(w)]["clip_detection_rate"] for w in ws], marker="o", color=COLOR["BUSI"],
            linewidth=2, label="clips detected")
    ax.plot(ws, [v["windows"][str(w)]["lesion_free_fp_rate"] for w in ws], marker="s", color=COLOR["BUS-BRA"],
            linewidth=2, label="lesion-free frames flagged")
    style(ax)
    ax.set_xlabel("Persistence window (frames)", fontsize=8, color=INK)
    ax.set_ylabel("Rate", fontsize=8, color=INK)
    ax.set_xticks(ws)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_video.pdf")
    plt.close(fig)


def certify(results: Path, out: Path):
    p = results / "certify.json"
    if not p.exists():
        return
    c = json.loads(p.read_text())
    lines = []
    for site in ordered(c):
        for r in c[site]["rows"]:
            lines.append(f"{site} & {r['target']:.2f} & {r['method']} & {r['n'] or '--'} & {f3(r['threshold_found'], 2)} & "
                         f"{f3(r['mean_sensitivity'])} & {f3(r['coverage'], 2)} & {f3(r['mean_specificity'])} \\\\")
    (out / "tab_certify.tex").write_text("\n".join(lines) + "\n")
    rows = pd.DataFrame([dict(r, site=s) for s in c for r in c[s]["rows"]])
    for (tau, meth), g in rows.groupby(["target", "method"]):
        put(f"cert_{meth}_{int(tau * 100)}_coverage", float(g.coverage.mean()))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for k, (meth, mk) in enumerate((("binomial", "o"), ("conformal", "s"))):
        g = rows[(rows.method == meth) & (rows.target == 0.95)].groupby("n").coverage.mean()
        if len(g):
            ax.plot(g.index, g.values, marker=mk, color=["#2a78d6", "#eb6834"][k], linewidth=2, label=meth)
    nv = rows[(rows.method == "naive") & (rows.target == 0.95)].coverage.mean()
    if np.isfinite(nv):
        ax.axhline(nv, color=MUTED, linestyle="--", linewidth=1, label="naive transfer")
    ax.axhline(0.95, color="#c3c2b7", linewidth=1)
    style(ax)
    ax.set_xlabel("Local positives used for certification", fontsize=8, color=INK)
    ax.set_ylabel("Fraction of draws meeting 0.95", fontsize=8, color=INK)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_certify.pdf")
    plt.close(fig)


def runtime(results: Path, out: Path):
    p = results / "runtime.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    keys = [("preprocess_ms", "Preprocessing (crop, inpaint, normalise, resize)"), ("quality_ms", "Quality gate"),
            ("encoder_batch1_ms", "Encoder, batch 1"), ("encoder_batch16_ms_per_frame", "Encoder, batch 16 (per frame)"),
            ("detector_head_ms", "Detector head + boxes"), ("malignancy_ms", "Malignancy head")]
    lines = [f"{label} & {r[k]:.1f} \\\\" for k, label in keys if r.get(k) is not None]
    (out / "tab_runtime.tex").write_text("\n".join(lines) + "\n")
    for k in ("end_to_end_frames_per_second", "bundle_mb", "encoder_weights_mb", "peak_rss_mb", "threads",
              "encoder_batch1_ms", "encoder_batch16_ms_per_frame"):
        if r.get(k) is not None:
            put(f"rt_{k}", r[k])


def write_numbers(out: Path):
    (out / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    lines = []
    seen = set()
    for k, v in sorted(NUM.items()):
        name = macro_name(k)
        if name in seen:
            raise ValueError(f"macro collision for {k}")
        seen.add(name)
        if isinstance(v, (int, np.integer)):
            text = f"{int(v):,}".replace(",", "{,}")
        elif isinstance(v, float):
            text = "--" if not np.isfinite(v) else f"{v:.3f}"
        else:
            text = str(v)
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    (out / "macros.tex").write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "paper" / "generated")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    datasets(args.processed, args.out)
    loso(args.results, args.out, 50 if args.fast else args.n_boot)
    calibration(args.results, args.out, m_local=3 if args.fast else 50)
    robustness(args.results, args.out)
    ablation(args.results, args.out)
    quality(args.results, args.out)
    video(args.results, args.out)
    certify(args.results, args.out)
    runtime(args.results, args.out)
    write_numbers(args.out)
    print(f"wrote {len(NUM)} numbers and tables to {args.out}")


if __name__ == "__main__":
    main()
