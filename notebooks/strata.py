"""Mantel `form~meaning` across concept strata: full / no-loanword / Swadesh /
Leipzig-Jakarta / by part of speech.

    python notebooks/strata.py            # writes results/strata.csv + .md

Runs on the exploratory config's interim data (all 22 languages must have been
built by `nguasach run phonetics --config configs/paper_exploratory.yaml`).
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np

from nguasach.config import Config
from nguasach import mantel

BASE = Config.load("configs/paper_exploratory.yaml")
BASE = dataclasses.replace(BASE, mantel_form_form=False, mantel_cap=1200)

STRATA = {
    "all": {},
    "no_loanword": {"exclude_loanwords": True},
    "swadesh": {"mantel_subset": "swadesh.txt"},
    "leipzig_jakarta": {"mantel_subset": "leipzig_jakarta.txt"},
    "pos_verb": {"mantel_subset": "pos_verb.txt"},
    "pos_noun": {"mantel_subset": "pos_noun.txt"},
    "pos_adj": {"mantel_subset": "pos_adj.txt"},
}

rows = []
for name, over in STRATA.items():
    cfg = dataclasses.replace(BASE, **over)
    out = mantel.run(cfg)
    fm = [r for r in out["rows"] if r["analysis"] == "form~meaning"]
    ok = [r for r in fm if not r["orth_control_degenerate"]]
    rp = np.array([r["r_partial_orth"] for r in ok])
    rr = np.array([r["r"] for r in ok])
    n_sig = sum(1 for r in ok if r["p_partial"] < 0.05)
    n_sig_q = sum(1 for r in ok if r.get("q_partial", 1) < 0.05)
    rows.append({
        "stratum": name, "n_concepts": out["n_subsample"],
        "n_languages_ok": len(ok),
        "median_r": round(float(np.median(rr)), 4),
        "median_partial_r": round(float(np.median(rp)), 4),
        "min_partial_r": round(float(rp.min()), 4),
        "max_partial_r": round(float(rp.max()), 4),
        "n_sig_partial_p05": n_sig,
        "n_sig_partial_q05": n_sig_q,
    })
    print(f"{name:16} n={out['n_subsample']:4}  median partial r={rows[-1]['median_partial_r']:+.4f}"
          f"  sig(q<.05) {n_sig_q}/{len(ok)}")

rdir = Path("results")
with (rdir / "strata.csv").open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

md = ["# form~meaning across concept strata (exploratory config, 22 languages)\n",
      "| stratum | n | languages | median r | median partial r | partial r range | sig p<.05 | sig q<.05 |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['stratum']} | {r['n_concepts']} | {r['n_languages_ok']} | "
              f"{r['median_r']:+.4f} | **{r['median_partial_r']:+.4f}** | "
              f"[{r['min_partial_r']:+.4f}, {r['max_partial_r']:+.4f}] | "
              f"{r['n_sig_partial_p05']} | {r['n_sig_partial_q05']} |")
(rdir / "strata.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\nwrote results/strata.csv + results/strata.md")
