# attic/ — quarantined exploratory offshoots

These scripts were part of the original working directory but are **not on the
critical path** for the confirmatory (phonetic→semantic alignment retrieval) or
interpretability (phoneme–meaning association) pipeline described in the plan.
They are kept for reference, not deleted.

| file | what it was exploring |
|------|-----------------------|
| `conceptnet5.py` | pulling relational data from ConceptNet 5 |
| `visual_genome.py` | Visual Genome scene-graph captions as a semantic source |
| `ActivityNet_Captions.py` | ActivityNet dense-caption corpus ingest |
| `trigramTraining.py`, `trigramTrainingSeq.py` | character/phoneme trigram sequence models |
| `contrastiveLearning.py` | contrastive objective over word pairs |
| `translateByFrameNet.py` | FrameNet-mediated translation |
| `tryVerbNet.py` | VerbNet class lookups |
| `translate2Iti.py` | mapping into a constructed language ("Iti") |
| `interactiveTrigrams.py` | interactive trigram probe |

Also relevant but left in place at repo root for now (data/results, to be moved
to `legacy_results/` during the data reorg): `etymTable*.csv`, `things*.txt`,
`thingsDict*`, `frameCorpus.txt`, `frameDict.txt`, `frameReference.txt`,
`*Hex.txt`, `*ZScore.txt`, the `*.png` heatmaps, `nodes.html`.
