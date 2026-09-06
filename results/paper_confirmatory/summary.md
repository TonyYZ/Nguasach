# Results — config `paper_confirmatory` (`4f80d42508a72fb3`)

map=ridge, k=100, folds=10, null_iters=300, bootstrap_iters=2000


## Confirmatory — retrieval (BH-FDR within this family)

| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |
|---|---|---|---|---|---|---|---|
| French → English | 0.418 [0.397, 0.436] | 0.409 | 0.048 | 0.0033 | 0.0033 * | 487 | 79 |
| English → French | 0.417 [0.394, 0.437] | 0.412 | 0.049 | 0.0033 | 0.0033 * | 493 | 197 |
| English → Irish | 0.264 [0.253, 0.274] | 0.267 | 0.050 | 0.0033 | 0.0033 * | 627 | 176 |
| Irish → English | 0.254 [0.240, 0.269] | 0.248 | 0.048 | 0.0033 | 0.0033 * | 629 | 79 |
| Irish → French | 0.197 [0.184, 0.209] | 0.184 | 0.047 | 0.0033 | 0.0033 * | 693 | 197 |
| French → Irish | 0.193 [0.183, 0.207] | 0.182 | 0.049 | 0.0033 | 0.0033 * | 689 | 176 |
| Irish → Chinese | 0.092 [0.082, 0.102] | 0.090 | 0.047 | 0.0033 | 0.0033 * | 843 | 32 |
| French → Chinese | 0.090 [0.077, 0.104] | 0.086 | 0.048 | 0.0033 | 0.0033 * | 850 | 32 |
| Chinese → Irish | 0.090 [0.082, 0.098] | 0.086 | 0.050 | 0.0033 | 0.0033 * | 850 | 176 |
| Chinese → French | 0.086 [0.073, 0.101] | 0.086 | 0.049 | 0.0033 | 0.0033 * | 872 | 197 |
| Chinese → English | 0.076 [0.064, 0.087] | 0.075 | 0.049 | 0.0033 | 0.0033 * | 864 | 79 |
| English → Chinese | 0.072 [0.058, 0.084] | 0.070 | 0.049 | 0.0033 | 0.0033 * | 854 | 32 |

12/12 pairs significant at q<0.05.


## Exploratory — retrieval (BH-FDR within this family)

| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |
|---|---|---|---|---|---|---|---|
| English → Semantics | 0.206 [0.189, 0.221] | 0.175 | 0.050 | 0.0033 | 0.0033 * | 633 | 79 |
| French → Semantics | 0.198 [0.179, 0.217] | 0.184 | 0.050 | 0.0033 | 0.0033 * | 636 | 79 |
| Irish → Semantics | 0.193 [0.178, 0.209] | 0.183 | 0.050 | 0.0033 | 0.0033 * | 642 | 79 |
| Chinese → Semantics | 0.171 [0.155, 0.186] | 0.170 | 0.051 | 0.0033 | 0.0033 * | 695 | 79 |

4/4 pairs significant at q<0.05.


## Baseline comparison — phonetic vs editdist, feat, orth

| pair | phonetic acc | editdist acc | feat acc | orth acc | null |
|---|---|---|---|---|---|
| French → English | 0.418 | 0.451 | 0.306 | 0.431 | 0.048 |
| English → French | 0.417 | 0.452 | 0.292 | 0.447 | 0.049 |
| English → Irish | 0.264 | 0.298 | 0.218 | 0.284 | 0.050 |
| Irish → English | 0.254 | 0.297 | 0.220 | 0.275 | 0.048 |
| English → Semantics | 0.206 | — | — | — | 0.050 |
| French → Semantics | 0.198 | — | — | — | 0.050 |
| Irish → French | 0.197 | 0.238 | 0.147 | 0.201 | 0.047 |
| French → Irish | 0.193 | 0.241 | 0.154 | 0.205 | 0.049 |
| Irish → Semantics | 0.193 | — | — | — | 0.050 |
| Chinese → Semantics | 0.171 | — | — | — | 0.051 |
| Irish → Chinese | 0.092 | 0.000 | 0.053 | 0.000 | 0.047 |
| French → Chinese | 0.090 | 0.000 | 0.067 | 0.000 | 0.048 |
| Chinese → Irish | 0.090 | 0.000 | 0.060 | 0.000 | 0.050 |
| Chinese → French | 0.086 | 0.000 | 0.068 | 0.000 | 0.049 |
| Chinese → English | 0.076 | 0.000 | 0.068 | 0.000 | 0.049 |
| English → Chinese | 0.072 | 0.000 | 0.068 | 0.000 | 0.049 |

phonetic − editdist is the retrieval accuracy not explained by raw orthographic string overlap (cognates / borrowing).


## Form–meaning correlation (Mantel, n=1968 concepts)

| analysis | unit | r | p | r \| orthography | p (partial) | note |
|---|---|---|---|---|---|---|
| form~meaning | English | +0.0215* | 0.0033 | +0.0202* | 0.0033 |  |
| form~meaning | Chinese | +0.0166* | 0.0033 | +0.0039* | 0.0033 | orth control degenerate |
| form~meaning | French | +0.0209* | 0.0033 | +0.0151* | 0.0033 |  |
| form~meaning | Irish | +0.0202* | 0.0033 | +0.0182* | 0.0033 |  |
| form~form | English~Chinese | +0.0039* | 0.0033 | +0.0033* | 0.0033 | orth control degenerate |
| form~form | English~French | +0.0585* | 0.0033 | +0.0522* | 0.0033 |  |
| form~form | English~Irish | +0.0276* | 0.0033 | +0.0243* | 0.0033 |  |
| form~form | Chinese~French | +0.0046* | 0.0033 | +0.0018* | 0.0133 | orth control degenerate |
| form~form | Chinese~Irish | +0.0055* | 0.0033 | +0.0023* | 0.0033 | orth control degenerate |
| form~form | French~Irish | +0.0189* | 0.0033 | +0.0160* | 0.0033 |  |

A within-language *form~meaning* r that stays significant in the *| orthography* column is sound–meaning systematicity not attributable to spelling / cognate overlap.


## Phoneme–meaning association (18 poles, null_iters=300)

Significant phoneme×pole cells (q<0.10): **0** total.

