"""Assemble the self-contained HTML deliverables from templates + data.

    python notebooks/build_reports.py

* ``results/project_explainer.html`` = ``results/_explainer_template.html`` with
  every ``__FIG_<name>__`` placeholder replaced by a base64 data URI of
  ``figures/exploratory/<file>.png``.
* ``results/etym_table.html`` = ``results/_etym_template.html`` with the
  ``/*__ETYM_DATA__*/`` marker replaced by the inlined ``results/etym_table.json``.

Both outputs are single files with no external dependencies except Google Fonts.
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# placeholder token -> figure png, relative to figures/. The explainer narrates
# the *confirmatory* retrieval + baselines and the *exploratory* Mantel results.
EXPLAINER_FIGS = {
    "retrieval": "fig1_retrieval.png",
    "baselines": "fig2_baselines.png",
    "sweep": "exploratory/fig5_mantel_sweep.png",
    "formform_matrix": "exploratory/fig7_form_form_matrix.png",
    "formform_dist": "exploratory/fig8_form_form_dist.png",
    "strata": "exploratory/fig6_strata.png",
    "pooled": "exploratory/fig9_association_pooled.png",
}


def _data_uri(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii")


def build_explainer(etym_url: str = "etym_table.html") -> Path:
    tpl = (ROOT / "results" / "_explainer_template.html").read_text(encoding="utf-8")
    for token, rel in EXPLAINER_FIGS.items():
        png = ROOT / "figures" / rel
        if not png.exists():
            raise FileNotFoundError(f"{png} — run notebooks/figures.py first")
        marker = f"__FIG_{token}__"
        if marker not in tpl:
            raise KeyError(f"{marker} not in explainer template")
        tpl = tpl.replace(marker, _data_uri(png))
    tpl = tpl.replace("__ETYM_URL__", etym_url)     # relative link; swap for a hosted URL if published
    out = ROOT / "results" / "project_explainer.html"
    out.write_text(tpl, encoding="utf-8")
    return out


def build_etym() -> Path:
    tpl = (ROOT / "results" / "_etym_template.html").read_text(encoding="utf-8")
    data = (ROOT / "results" / "etym_table.json").read_text(encoding="utf-8")
    if "/*__ETYM_DATA__*/" not in tpl:
        raise KeyError("/*__ETYM_DATA__*/ marker not in etym template")
    out = ROOT / "results" / "etym_table.html"
    out.write_text(tpl.replace("/*__ETYM_DATA__*/", data), encoding="utf-8")
    return out


# ---------------------------------------------------------------- results browser
# (token in _results_template.html) -> (figures/ path, one-line caption)
RESULTS_FIGS = [
    ("fig1_retrieval.png", "confirmatory retrieval accuracy vs null"),
    ("fig2_baselines.png", "phonetic map vs editdist / orth / feat baselines"),
    ("exploratory/fig1_retrieval.png", "exploratory (English-anchored) retrieval"),
    ("exploratory/fig5_mantel_sweep.png", "within-language form~meaning, raw vs partial, 22 languages"),
    ("exploratory/fig7_form_form_matrix.png", "cross-language form~form partial-r, blocked by family"),
    ("exploratory/fig8_form_form_dist.png", "same- vs different-family (macro grain) partial-r"),
    ("exploratory/fig6_strata.png", "systematicity by concept stratum"),
    ("exploratory/fig9_association_pooled.png", "pooled phoneme x pole cells clearing FDR + family guard"),
]


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _split_mantel(m):
    if not m:
        return None
    rows = m["rows"]
    return {
        "n_subsample": m.get("n_subsample"),
        "form_meaning": [r for r in rows if r["analysis"] == "form~meaning"],
        "form_form": [r for r in rows if r["analysis"] == "form~form"],
    }


def _strata():
    for p in (ROOT / "results" / "paper_exploratory" / "strata.csv",
              ROOT / "results" / "strata.csv"):
        if p.exists():
            rows = list(csv.reader(p.read_text(encoding="utf-8").splitlines()))
            return {"header": rows[0], "rows": rows[1:]}
    return None


def build_results() -> Path:
    conf, expl = ROOT / "results" / "paper_confirmatory", ROOT / "results" / "paper_exploratory"
    c_acc = _load_json(conf / "accuracy_by_pair.json")
    e_acc = _load_json(expl / "accuracy_by_pair.json")
    e_assoc = _load_json(expl / "association_z.json")
    data = {
        "built": _dt.date.today().isoformat(),
        "blurb": ("Every result table from the frozen runs, unfiltered and sortable. "
                  "Retrieval tests cross-lingual phonetic alignability; Mantel tests "
                  "within- and cross-language form/meaning structure; the association "
                  "layer maps phonemes to the 18 trigram poles."),
        "fingerprints": {
            "confirmatory": (c_acc or {}).get("config_fingerprint"),
            "exploratory": (e_acc or {}).get("config_fingerprint"),
        },
        "confirmatory": {
            "retrieval": c_acc,
            "baselines": _load_json(conf / "baselines.json"),
        },
        "exploratory": {
            "retrieval": e_acc,
            "mantel": _split_mantel(_load_json(expl / "mantel.json")),
            "association": e_assoc,
            "pooled": (e_assoc or {}).get("pooled"),
        },
        "strata": _strata(),
        "figures": [
            {"name": rel.split("/")[-1].replace(".png", ""),
             "caption": cap, "src": _data_uri(ROOT / "figures" / rel)}
            for rel, cap in RESULTS_FIGS if (ROOT / "figures" / rel).exists()
        ],
    }
    tpl = (ROOT / "results" / "_results_template.html").read_text(encoding="utf-8")
    if "/*__RESULTS_DATA__*/" not in tpl:
        raise KeyError("/*__RESULTS_DATA__*/ marker not in results template")
    out = ROOT / "results" / "results_browser.html"
    out.write_text(tpl.replace("/*__RESULTS_DATA__*/",
                               json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    return out


if __name__ == "__main__":
    for fn in (build_explainer, build_etym, build_results):
        p = fn()
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size:,} bytes)")
