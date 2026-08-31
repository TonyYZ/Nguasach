"""Regenerate every manuscript figure from ``results/`` — one command, no hidden state.

    python notebooks/figures.py [--results results] [--out figures]

Pair to a notebook with ``jupytext --to ipynb notebooks/figures.py`` if desired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9})
INK, ACCENT, NULLC = "#1b1b1b", "#3b6ea5", "#b0b0b0"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(fig, out: Path, name: str):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def fig_retrieval(acc, out: Path):
    pairs = sorted(acc["pairs"], key=lambda r: r["acc_mean"])
    labels = [f"{r['source'][:3]}→{r['target'][:3]}" for r in pairs]
    vals = [r["acc_mean"] for r in pairs]
    lo = [r["acc_mean"] - r["boot_ci95"][0] for r in pairs]
    hi = [r["boot_ci95"][1] - r["acc_mean"] for r in pairs]
    nullm = [r["null_mean"] for r in pairs]
    y = range(len(pairs))
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(pairs) + 1))
    ax.barh(y, vals, xerr=[lo, hi], color=ACCENT, height=0.62, error_kw=dict(lw=0.9))
    ax.scatter(nullm, y, color=INK, marker="|", s=90, zorder=5, label="permutation null")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"top-{acc['k']} retrieval accuracy (10-fold CV, 95% bootstrap CI)")
    ax.set_title(f"Phonetic-form retrieval — {acc['config']}")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out, "fig1_retrieval")


def fig_baselines(acc, base, out: Path):
    kinds = ["editdist", "orth", "feat"]
    prs = {(r["source"], r["target"]): r for r in acc["pairs"]}
    brs = {(r["baseline"], r["source"], r["target"]): r for r in base["rows"]}
    pairs = [k for k in prs if k[1] != "Semantics"]
    pairs.sort(key=lambda k: -prs[k]["acc_mean"])
    x = range(len(pairs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(pairs)), 3.4))
    ax.bar([i - 1.5 * w for i in x], [prs[k]["acc_mean"] for k in pairs], w,
           label="phonetic", color=ACCENT)
    for j, kind in enumerate(kinds):
        ax.bar([i + (j - 0.5) * w for i in x],
               [brs.get((kind, *k), {}).get("acc_mean", 0) for k in pairs], w, label=kind)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s[:3]}→{t[:3]}" for s, t in pairs], rotation=45, ha="right")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("Phonetic vs. control representations (CSLS-corrected)")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    _save(fig, out, "fig2_baselines")


def fig_mantel(mant, out: Path):
    fm = [r for r in mant["rows"] if r["analysis"] == "form~meaning"]
    ff = [r for r in mant["rows"] if r["analysis"] == "form~form"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, rows, title in ((axes[0], fm, "form ~ meaning (within language)"),
                            (axes[1], ff, "form ~ form (verified core)")):
        rows = sorted(rows, key=lambda r: r["r"])
        y = list(range(len(rows)))
        ax.barh([i + 0.2 for i in y], [r["r"] for r in rows], 0.38,
                color=ACCENT, label="Mantel r")
        ax.barh([i - 0.2 for i in y], [r["r_partial_orth"] for r in rows], 0.38,
                color=INK, label="partial | orthography")
        xr = max((max(r["r"], r["r_partial_orth"]) for r in rows), default=0.01)
        for i, r in zip(y, rows):
            if r["p_perm"] < 0.05:
                ax.text(r["r"] + xr * 0.02, i + 0.2, "*", va="center", fontsize=8)
            if r["p_partial"] < 0.05:
                ax.text(r["r_partial_orth"] + xr * 0.02, i - 0.2, "*", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([r["unit"] for r in rows])
        ax.axvline(0, color=NULLC, lw=0.8)
        ax.set_title(title, pad=8)
        ax.set_xlabel("correlation")
        ax.margins(x=0.15)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Mantel correlations — n={mant['n_subsample']} concepts, "
                 f"* p<.05  ({mant['config']})", y=1.02)
    _save(fig, out, "fig3_mantel")


def fig_mantel_sweep(mant, out: Path):
    """Exploratory: form~meaning raw vs partial across all languages."""
    fm = [r for r in mant["rows"] if r["analysis"] == "form~meaning"]
    if len(fm) < 8:
        return
    fm = sorted(fm, key=lambda r: r["r"])
    y = list(range(len(fm)))
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(fm) + 1))
    ax.barh([i + 0.2 for i in y], [r["r"] for r in fm], 0.38, color=ACCENT, label="Mantel r")
    ax.barh([i - 0.2 for i in y], [r["r_partial_orth"] for r in fm], 0.38, color=INK,
            label="partial | orthography")
    for i, r in zip(y, fm):
        if r.get("orth_control_degenerate"):
            ax.text(0.0005, i - 0.2, " (logographic)", va="center", fontsize=7, style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels([r["unit"] for r in fm])
    ax.axvline(0, color=NULLC, lw=0.8)
    ax.set_xlabel("form ~ meaning correlation")
    ax.set_title(f"Within-language sound–meaning systematicity, {len(fm)} languages\n"
                 f"(n={mant['n_subsample']}, all raw r significant at p≤.005)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, out, "fig5_mantel_sweep")


_FAMILY = {
    "Hungarian": "Uralic", "Finnish": "Uralic",
    "Greek": "IE-Hellenic", "Russian": "IE-Slavic",
    "German": "IE-Germanic", "English": "IE-Germanic",
    "Spanish": "IE-Romance", "Italian": "IE-Romance", "French": "IE-Romance",
    "Irish": "IE-Celtic", "Welsh": "IE-Celtic", "Hindi": "IE-Indic",
    "Chinese": "Sino-Tibetan", "Vietnamese": "Austroasiatic",
    "Japanese": "Japonic", "Korean": "Koreanic", "Thai": "Kra-Dai",
    "Indonesian": "Austronesian", "Turkish": "Turkic",
    "Arabic": "Afro-Asiatic", "Hebrew": "Afro-Asiatic", "Swahili": "Atlantic-Congo",
}
_FAM_ORDER = ["IE-Germanic", "IE-Romance", "IE-Celtic", "IE-Slavic", "IE-Hellenic",
              "IE-Indic", "Uralic", "Turkic", "Afro-Asiatic", "Atlantic-Congo",
              "Sino-Tibetan", "Kra-Dai", "Austroasiatic", "Austronesian",
              "Japonic", "Koreanic"]


def _ff_pairs(mant):
    return [r for r in mant["rows"] if r["analysis"] == "form~form"]


def fig_form_form_matrix(mant, out: Path):
    """Cross-language form~form partial-r (| orthography), languages blocked by family."""
    import numpy as np

    ff = _ff_pairs(mant)
    if len(ff) < 8:
        return
    langs = sorted({l for r in ff for l in r["unit"].split("~")},
                   key=lambda l: (_FAM_ORDER.index(_FAMILY.get(l, "")) if _FAMILY.get(l) in _FAM_ORDER else 99, l))
    pos = {l: i for i, l in enumerate(langs)}
    n = len(langs)
    M = np.full((n, n), np.nan)
    degen = np.zeros((n, n), bool)
    for r in ff:
        a, b = r["unit"].split("~")
        i, j = pos[a], pos[b]
        M[i, j] = M[j, i] = r["r_partial_orth"]
        if r.get("orth_control_degenerate"):
            degen[i, j] = degen[j, i] = True

    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(langs, rotation=90, fontsize=7)
    ax.set_yticklabels(langs, fontsize=7)
    # family block separators
    bounds = []
    for k in range(1, n):
        if _FAMILY.get(langs[k]) != _FAMILY.get(langs[k - 1]):
            bounds.append(k - 0.5)
    for bnd in bounds:
        ax.axhline(bnd, color="#222", lw=0.8)
        ax.axvline(bnd, color="#222", lw=0.8)
    for i in range(n):
        for j in range(n):
            if degen[i, j]:
                ax.text(j, i, "·", ha="center", va="center", fontsize=6, color="#666")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="partial Mantel r  (form~form | orthography)")
    ax.set_title(f"Cross-language form–form similarity, orthography partialled\n"
                 f"n={mant['n_subsample']} concepts · {len(ff)} pairs · "
                 f"black lines = family boundaries · · = orth control degenerate")
    _save(fig, out, "fig7_form_form_matrix")


def fig_form_form_dist(mant, out: Path):
    """Same-family vs different-family partial-r, to show the different-family cluster."""
    import numpy as np

    ff = _ff_pairs(mant)
    if len(ff) < 8 or not any("same_family" in r for r in ff):
        return
    same = [r["r_partial_orth"] for r in ff if r.get("same_family")]
    diff = [r["r_partial_orth"] for r in ff if not r.get("same_family")]
    sig_diff = [r["r_partial_orth"] for r in ff
                if not r.get("same_family") and r.get("q_partial", 1) < 0.05]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    rng = np.random.default_rng(0)
    for x, vals, c, lab in ((0, same, ACCENT, f"same family (n={len(same)})"),
                            (1, diff, INK, f"different family (n={len(diff)})")):
        xs = x + (rng.random(len(vals)) - 0.5) * 0.28
        ax.scatter(xs, vals, s=10, color=c, alpha=0.55, label=lab)
        ax.plot([x - 0.2, x + 0.2], [np.median(vals)] * 2, color=c, lw=2)
    ax.axhline(0, color=NULLC, lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["same\nfamily", "different\nfamily"])
    ax.set_ylabel("partial Mantel r  (| orthography)")
    ax.set_title(f"Form–form structure by genealogy\n"
                 f"different-family median = {np.median(diff):+.4f}, "
                 f"{len(sig_diff)}/{len(diff)} sig. at q<.05")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "fig8_form_form_dist")


def fig_strata(csv_path: Path, out: Path):
    if not csv_path.exists():
        return
    import csv as _csv

    rows = list(_csv.DictReader(csv_path.open(encoding="utf-8")))
    order = ["leipzig_jakarta", "swadesh", "pos_adj", "no_loanword", "all",
             "pos_noun", "pos_verb"]
    rows = [r for k in order for r in rows if r["stratum"] == k]
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(rows) + 1))
    ax.barh(y, [float(r["median_partial_r"]) for r in rows], color=ACCENT)
    for i, r in zip(y, rows):
        ax.text(0.0005, i, f"  n={r['n_concepts']}, {r['n_sig_partial_q05']}/{r['n_languages_ok']} sig",
                va="center", fontsize=7)
    ax.axvline(0, color=NULLC, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["stratum"].replace("_", " ") for r in rows])
    ax.set_xlabel("median partial-Mantel r (form ~ meaning, 20 languages)")
    ax.set_title("Sound–meaning systematicity by concept stratum")
    _save(fig, out, "fig6_strata")


def fig_association(assoc, out: Path):
    sig = [c for c in assoc["cells"] if c["significant"]]
    if not sig:
        print("association: no significant cells — skipping fig4")
        return
    langs = sorted({c["language"] for c in sig})
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(sig) + 1))
    sig = sorted(sig, key=lambda c: (c["language"], c["z"]))
    y = range(len(sig))
    ax.barh(y, [c["z"] for c in sig],
            color=[ACCENT if c["z"] > 0 else INK for c in sig])
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{c['language'][:3]} /{c['phoneme']}/ · {c['pole']}" for c in sig])
    ax.set_xlabel("z-score (phoneme rate at pole vs. across poles)")
    ax.set_title(f"Phoneme–meaning associations, q<0.10 ({', '.join(langs)})")
    _save(fig, out, "fig4_association")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    O.mkdir(exist_ok=True)

    acc = _load(R / "accuracy_by_pair.json")
    base = _load(R / "baselines.json")
    mant = _load(R / "mantel.json")
    assoc = _load(R / "association_z.json")

    if acc:
        fig_retrieval(acc, O)
        if base:
            fig_baselines(acc, base, O)
    if mant:
        fig_mantel(mant, O)
        fig_mantel_sweep(mant, O)
        fig_form_form_matrix(mant, O)
        fig_form_form_dist(mant, O)
    fig_strata(R / "strata.csv", O)
    if assoc:
        fig_association(assoc, O)


if __name__ == "__main__":
    main()
