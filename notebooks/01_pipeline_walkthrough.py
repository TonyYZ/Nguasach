# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---
# %% [markdown]
# # Nguasach pipeline walkthrough
#
# Runs the whole DAG on `configs/smoke.yaml` (5 languages, 200 concepts) and
# inspects each stage's output. For the full analysis use
# `nguasach all --config configs/paper_confirmatory.yaml`.
#
# Pair this file to a notebook with `jupytext --to ipynb 01_pipeline_walkthrough.py`.

# %%
import json
import os

os.environ.setdefault("PYTHONUTF8", "1")
from nguasach.config import Config
from nguasach import data, ipa, phonetics, semantics, crossval, mantel, association, baselines, report

cfg = Config.load("configs/smoke.yaml")
cfg.name, cfg.fingerprint()

# %% [markdown]
# ## 1. `data` — canonical table, folds, leakage report

# %%
rep = data.run(cfg)
print(json.dumps(rep["integrity"], indent=2))
df = data.load_raw(cfg)
df.head()

# %% [markdown]
# The randomized k-fold partition is seeded; `leakage_report` confirms
# train/test disjointness and measures homograph surface collisions.

# %%
folds = data.make_folds(len(df), cfg.folds, cfg.seed)
data.leakage_report(df, folds)["per_fold"][0]

# %% [markdown]
# ## 2. `translate-qc` → 3. `ipa` → 4. `phonetics`

# %%
from nguasach import translate_qc

translate_qc.run(cfg)
ipa.run(cfg)["written"]

# %%
ph = phonetics.run(cfg)
{k: v["n_feature_bigrams"] for k, v in ph["languages"].items()}

# %% [markdown]
# ## 5. `semantics` — per-concept word2vec keys → PCA space
# (needs `model.txt`; skipped automatically if absent)

# %%
try:
    sem = semantics.run(cfg)
    print(sem["n_keys_unresolved"], "unresolved of", sem["n_concepts"])
except FileNotFoundError as e:
    print("skipped:", e)

# %% [markdown]
# ## 6. `align` — ridge map + CSLS retrieval, one pair

# %%
pair = crossval.load_pair_data(cfg, "French", "English")
obs = crossval.score_pair(pair, crossval._folds(cfg), k=cfg.k, map_kind="ridge",
                          alpha=cfg.ridge_alpha, csls_k=cfg.csls_k).summary()
{k: obs[k] for k in ("acc_mean", "acc_clean_mean", "mean_rank")}

# %% [markdown]
# ## 7. Full run + report

# %%
for stage in ("align", "mantel", "associate", "baselines", "report"):
    mod = {"align": crossval, "mantel": mantel, "associate": association,
           "baselines": baselines, "report": report}[stage]
    mod.run(cfg)

print(open(cfg.paths.resolve("results") / "summary.md", encoding="utf-8").read())
