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


if __name__ == "__main__":
    for fn in (build_explainer, build_etym):
        p = fn()
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size:,} bytes)")
