# Results — config `paper_confirmatory` (`d4fdffa3635611fa`)

map=ridge, k=100, folds=10, null_iters=300, bootstrap_iters=2000


## Confirmatory — retrieval (BH-FDR within this family)

| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |
|---|---|---|---|---|---|---|---|
| English → French | 0.423 [0.404, 0.443] | 0.415 | 0.051 | 0.0033 | 0.0033 * | 453 | 176 |
| French → English | 0.419 [0.401, 0.439] | 0.408 | 0.052 | 0.0033 | 0.0033 * | 447 | 81 |
| Irish → English | 0.264 [0.247, 0.280] | 0.257 | 0.051 | 0.0033 | 0.0033 * | 606 | 81 |
| English → Irish | 0.262 [0.248, 0.273] | 0.265 | 0.053 | 0.0033 | 0.0033 * | 607 | 160 |
| Irish → French | 0.199 [0.190, 0.208] | 0.185 | 0.051 | 0.0033 | 0.0033 * | 650 | 176 |
| French → Irish | 0.196 [0.185, 0.209] | 0.190 | 0.053 | 0.0033 | 0.0033 * | 647 | 160 |
| Irish → Chinese | 0.098 [0.086, 0.111] | 0.097 | 0.051 | 0.0033 | 0.0033 * | 797 | 26 |
| Chinese → Irish | 0.094 [0.083, 0.107] | 0.088 | 0.053 | 0.0033 | 0.0033 * | 802 | 160 |
| Chinese → French | 0.090 [0.073, 0.105] | 0.088 | 0.052 | 0.0033 | 0.0033 * | 795 | 176 |
| French → Chinese | 0.088 [0.077, 0.100] | 0.084 | 0.052 | 0.0033 | 0.0033 * | 791 | 26 |
| Chinese → English | 0.083 [0.074, 0.090] | 0.082 | 0.052 | 0.0033 | 0.0033 * | 823 | 81 |
| English → Chinese | 0.080 [0.073, 0.086] | 0.079 | 0.053 | 0.0033 | 0.0033 * | 822 | 26 |

12/12 pairs significant at q<0.05.


## Exploratory — retrieval (BH-FDR within this family)

| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |
|---|---|---|---|---|---|---|---|
| French → Semantics | 0.214 [0.202, 0.225] | 0.194 | 0.054 | 0.0033 | 0.0033 * | 584 | 81 |
| English → Semantics | 0.212 [0.189, 0.231] | 0.177 | 0.054 | 0.0033 | 0.0033 * | 604 | 81 |
| Irish → Semantics | 0.208 [0.194, 0.225] | 0.196 | 0.053 | 0.0033 | 0.0033 * | 611 | 81 |
| Chinese → Semantics | 0.170 [0.156, 0.185] | 0.169 | 0.054 | 0.0033 | 0.0033 * | 663 | 81 |

4/4 pairs significant at q<0.05.


## Baseline comparison — phonetic vs editdist, feat, orth

| pair | phonetic acc | editdist acc | feat acc | orth acc | null |
|---|---|---|---|---|---|
| English → French | 0.423 | 0.466 | 0.306 | 0.457 | 0.051 |
| French → English | 0.419 | 0.464 | 0.320 | 0.444 | 0.052 |
| Irish → English | 0.264 | 0.310 | 0.230 | 0.288 | 0.051 |
| English → Irish | 0.262 | 0.311 | 0.229 | 0.296 | 0.053 |
| French → Semantics | 0.214 | — | — | — | 0.054 |
| English → Semantics | 0.212 | — | — | — | 0.054 |
| Irish → Semantics | 0.208 | — | — | — | 0.053 |
| Irish → French | 0.199 | 0.252 | 0.157 | 0.215 | 0.051 |
| French → Irish | 0.196 | 0.250 | 0.167 | 0.221 | 0.053 |
| Chinese → Semantics | 0.170 | — | — | — | 0.054 |
| Irish → Chinese | 0.098 | 0.000 | 0.055 | 0.000 | 0.051 |
| Chinese → Irish | 0.094 | 0.000 | 0.061 | 0.000 | 0.053 |
| Chinese → French | 0.090 | 0.000 | 0.074 | 0.000 | 0.052 |
| French → Chinese | 0.088 | 0.000 | 0.071 | 0.000 | 0.052 |
| Chinese → English | 0.083 | 0.000 | 0.070 | 0.000 | 0.052 |
| English → Chinese | 0.080 | 0.000 | 0.071 | 0.000 | 0.053 |

phonetic − editdist is the retrieval accuracy not explained by raw orthographic string overlap (cognates / borrowing).


## Form–meaning correlation (Mantel, n=700 concepts)

| analysis | unit | r | p | r \| orthography | p (partial) | note |
|---|---|---|---|---|---|---|
| form~meaning | English | +0.0258* | 0.0033 | +0.0243* | 0.0033 |  |
| form~meaning | Chinese | +0.0169* | 0.0033 | +0.0043* | 0.0233 |  |
| form~meaning | French | +0.0212* | 0.0033 | +0.0144* | 0.0033 |  |
| form~meaning | Irish | +0.0172* | 0.0033 | +0.0148* | 0.0033 |  |
| form~form | English~Chinese | +0.0036 | 0.0930 | +0.0034 | 0.1030 |  |
| form~form | English~French | +0.0537* | 0.0033 | +0.0479* | 0.0033 |  |
| form~form | English~Irish | +0.0336* | 0.0033 | +0.0293* | 0.0033 |  |
| form~form | Chinese~French | +0.0044* | 0.0266 | +0.0022 | 0.3023 |  |
| form~form | Chinese~Irish | +0.0064* | 0.0033 | +0.0036 | 0.0963 |  |
| form~form | French~Irish | +0.0185* | 0.0033 | +0.0152* | 0.0033 |  |

A within-language *form~meaning* r that stays significant in the *| orthography* column is sound–meaning systematicity not attributable to spelling / cognate overlap.


## Phoneme–meaning association (18 poles, null_iters=300)

Significant phoneme×pole cells (q<0.10): **0** total.

