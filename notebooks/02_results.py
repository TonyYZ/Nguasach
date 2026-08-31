# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---
# %% [markdown]
# # Nguasach — every result, in one place
#
# Loads the frozen result files and shows **all** of them: retrieval accuracy for
# every language pair, Mantel form~meaning and form~form for every language and
# every pair, control baselines, the phoneme-pole association cells, and the
# concept-strata breakdown. Nothing is filtered for significance.
#
# Run it top to bottom (`python notebooks/02_results.py`), or cell by cell in an
# editor that understands `# %%` blocks. To pair it to a real notebook:
# `pip install jupytext && jupytext --to ipynb notebooks/02_results.py`.
#
# | config | where | what |
# |---|---|---|
# | `paper_confirmatory` | `results/paper_confirmatory/` | 4 verified languages, full pairwise, + baselines |
# | `paper_exploratory`  | `results/paper_exploratory/`  | 22 languages, English-anchored retrieval, all-pairs Mantel |
# | *(scratch)* | `results/` | whatever the last `nguasach run` wrote |

# %%
import json
from pathlib import Path

import pandas as pd

pd.set_option("display.max_rows", 300)
pd.set_option("display.width", 160)

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
CONF = ROOT / "results" / "paper_confirmatory"
EXPL = ROOT / "results" / "paper_exploratory"


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# %% [markdown]
# ## 1. Retrieval — cross-lingual phonetic alignment
#
# A ridge map is fit from the source language's phonetic-similarity space to the
# target's, then top-`k` nearest-neighbour retrieval scores a hit when the true
# translation is returned. `null_mean` is the label-permutation null (re-fit each
# iteration); `p_perm` is the empirical p; `q_fdr` is Benjamini-Hochberg within
# the confirmatory / exploratory family. `k = 100`.
#
# **`target = Semantics`** is the actual sound -> meaning test. Language -> language
# pairs are dominated by cognates (English->German 0.46 vs English->Chinese 0.08).

# %%
def retrieval_table(d):
    rows = [{
        "source": r["source"], "target": r["target"], "family": r["family"],
        "acc": round(r["acc_mean"], 3), "null": round(r["null_mean"], 3),
        "acc-null": round(r["acc_mean"] - r["null_mean"], 3),
        "boot_lo": round(r["boot_ci95"][0], 3), "boot_hi": round(r["boot_ci95"][1], 3),
        "p_perm": r["p_perm"], "q_fdr": round(r.get("q_fdr", float("nan")), 4),
    } for r in d["pairs"]]
    return pd.DataFrame(rows).sort_values(["target", "acc"], ascending=[True, False])


conf_acc = load(CONF / "accuracy_by_pair.json")
expl_acc = load(EXPL / "accuracy_by_pair.json")
print(f"confirmatory: {len(conf_acc['pairs'])} pairs   exploratory: {len(expl_acc['pairs'])} pairs")

# %% [markdown]
# ### 1a. Confirmatory — {English, Chinese, French, Irish}, full pairwise
# %%
retrieval_table(conf_acc)

# %% [markdown]
# ### 1b. Exploratory — 22 languages, English-anchored (`align_scope: english`)
#
# Not a 22x22 matrix: English->each, each->English, each->Semantics = 64 pairs.
# The full 22-language object is the Mantel form~form matrix in section 2c.
# %%
retrieval_table(expl_acc)

# %% [markdown]
# ### 1c. The sound -> meaning rows only (`target = Semantics`)
# %%
retrieval_table(expl_acc).query("target == 'Semantics'").sort_values("acc", ascending=False)

# %% [markdown]
# ## 2. Mantel — distance-matrix correlation
#
# No learned map. `r` correlates the pairwise form-distance matrix with the
# meaning- (or other form-) distance matrix; `r_partial_orth` residualises both
# on the orthographic edit-distance matrix first. p-values are by concept-label
# permutation of one matrix. `orth_control_degenerate` = the orthographic control
# carried no usable variance (cross-script / logographic) so the partial ~ raw.

# %%
mant = load(EXPL / "mantel.json")
mdf = pd.DataFrame(mant["rows"])
print("config:", mant["config"], "  n_subsample:", mant["n_subsample"],
      "  subset:", mant.get("subset"), "  exclude_loanwords:", mant.get("exclude_loanwords", False))

# %% [markdown]
# ### 2a. form ~ meaning, within language (all 22)
# %%
(mdf[mdf.analysis == "form~meaning"]
 [["unit", "n", "r", "p_perm", "r_partial_orth", "p_partial", "q_partial",
   "orth_control_degenerate"]]
 .sort_values("r", ascending=False).reset_index(drop=True))

# %% [markdown]
# ### 2b. form ~ form, cross-language — summary
# %%
ff = mdf[mdf.analysis == "form~form"].copy()
print(f"{len(ff)} pairs  |  median r_partial_orth = {ff.r_partial_orth.median():+.4f}")
print(f"significant at q<.05: {(ff.q_partial < 0.05).sum()} / {len(ff)}")
print()
print("by genealogy:")
print(ff.assign(kind=ff.same_family.map({True: "same family", False: "different family"}))
      .groupby("kind").r_partial_orth.agg(["count", "median", "min", "max"]))

# %% [markdown]
# ### 2c. form ~ form — every pair, sorted by residual structure
# %%
(ff[["unit", "families", "same_family", "r", "r_partial_orth", "p_partial",
     "q_partial", "orth_control_degenerate"]]
 .sort_values("r_partial_orth", ascending=False).reset_index(drop=True))

# %% [markdown]
# ## 3. Control baselines (confirmatory only)
#
# Same retrieval machinery, no learned map. `editdist` = normalised Levenshtein
# similarity of spellings (the cognate / borrowing control). `orth` = character
# n-gram cosine. `feat` = mean panphon articulatory-feature cosine. The learned
# phonetic map must beat these.
# %%
base = load(CONF / "baselines.json")
if base:
    bdf = pd.DataFrame(base["rows"] if "rows" in base else base.get("pairs", []))
    display_cols = [c for c in ["source", "target", "baseline", "acc_mean",
                                "null_mean", "p_perm", "q_fdr"] if c in bdf.columns]
    print(bdf[display_cols].to_string(index=False))
else:
    print("no baselines.json under", CONF)

# %% [markdown]
# ## 4. Phoneme <-> pole association
#
# Each concept is assigned to its nearest of 18 trigram poles (ridge
# phonetic->semantic projection); per pole, each phoneme's rate is z-scored
# across poles; permutation p, BH-FDR across all phoneme x pole cells.
#
# **Nothing clears FDR** at this granularity — the raw z-scores are shown as a
# descriptive / browsable artefact (see also the etym visualiser).
# %%
assoc = load(EXPL / "association_z.json")
adf = pd.DataFrame(assoc["cells"])
print("languages:", sorted(adf.language.unique()))
print("total cells:", len(adf), "  significant at q<.10:", int(adf.significant.sum()))
print("\nstrongest |z| (count >= 3), any language:")
(adf[adf["count"] >= 3].reindex(adf[adf["count"] >= 3].z.abs().sort_values(ascending=False).index)
 [["language", "pole", "phoneme", "z", "count", "p_perm", "q_fdr"]].head(30).reset_index(drop=True))

# %% [markdown]
# ### 4a. Per-language pole occupancy (how many concepts land on each pole)
# %%
pd.DataFrame(assoc["per_language"]).T[["n_phonemes", "n_significant_cells"]]

# %% [markdown]
# ### 4b. Cross-linguistically **pooled** phoneme x pole test
#
# Per-language z-scores averaged over the languages where the cell is attested
# (>= 3 tokens), null = the same per-language concept->pole permutations pooled
# the same way. `significant` requires BH-FDR < .10 **and** the bias to show in
# >= 4 macro-families (Indo-European counted once) with >= 75% sign concordance,
# a guard against phylogenetic non-independence the permutation null ignores.
# %%
pooled = assoc.get("pooled")
if pooled:
    print(f"{pooled['n_cells_tested']} cells tested  |  "
          f"{pooled['n_fdr_only']} clear FDR<.10  |  "
          f"{pooled['n_significant']} also family-robust")
    pdf = pd.DataFrame(pooled["cells"])
    display_cols = ["pole", "phoneme", "mean_z", "n_langs", "n_families",
                    "family_sign_concord", "total_count", "q_fdr", "significant"]
    print(pdf[display_cols].head(40).to_string(index=False))
else:
    print("no pooled block — re-run `nguasach run associate`")

# %% [markdown]
# ## 5. Concept strata — systematicity by subset
#
# Mantel form~meaning re-run on Swadesh-207, Leipzig-Jakarta-100, and POS subsets.
# %%
sp = EXPL / "strata.csv"
if not sp.exists():
    sp = ROOT / "results" / "strata.csv"
print(Path(sp).read_text(encoding="utf-8") if Path(sp).exists() else "no strata.csv")

# %% [markdown]
# ## 6. Figures
#
# Regenerate with `python notebooks/figures.py --results results --out figures/exploratory`.
# %%
FIGDIR = ROOT / "figures" / "exploratory"
for name in ["fig1_retrieval", "fig2_baselines", "fig3_mantel", "fig5_mantel_sweep",
             "fig7_form_form_matrix", "fig8_form_form_dist", "fig6_strata"]:
    p = FIGDIR / f"{name}.png"
    print(("  ok  " if p.exists() else " MISS ") + str(p))

# %% [markdown]
# In an editor with inline plots:
# %%
try:
    from IPython.display import Image, display
    for name in ["fig1_retrieval", "fig3_mantel", "fig7_form_form_matrix",
                 "fig8_form_form_dist", "fig6_strata"]:
        p = FIGDIR / f"{name}.png"
        if p.exists():
            display(Image(filename=str(p)))
except ImportError:
    print("run inside IPython/Jupyter to see inline figures")
