# Results — config `paper_confirmatory` (`b771395b7199fec9`)

map=ridge, k=100, folds=10, null_iters=1000, bootstrap_iters=2000


## Confirmatory — retrieval (BH-FDR within this family)

| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |
|---|---|---|---|---|---|---|---|
| French → English | 0.400 [0.379, 0.422] | 0.388 | 0.043 | 0.0010 | 0.0010 * | 503 | 81 |
| English → French | 0.398 [0.380, 0.419] | 0.388 | 0.044 | 0.0010 | 0.0010 * | 506 | 176 |
| Irish → English | 0.240 [0.228, 0.251] | 0.232 | 0.042 | 0.0010 | 0.0010 * | 665 | 81 |
| English → Irish | 0.235 [0.226, 0.243] | 0.236 | 0.042 | 0.0010 | 0.0010 * | 672 | 160 |
| Irish → French | 0.182 [0.172, 0.193] | 0.170 | 0.043 | 0.0010 | 0.0010 * | 706 | 176 |
| French → Irish | 0.180 [0.171, 0.190] | 0.172 | 0.041 | 0.0010 | 0.0010 * | 711 | 160 |
| Irish → Chinese | 0.085 [0.075, 0.096] | 0.085 | 0.043 | 0.0010 | 0.0010 * | 860 | 26 |
| Chinese → Irish | 0.084 [0.072, 0.096] | 0.076 | 0.042 | 0.0010 | 0.0010 * | 872 | 160 |
| French → Chinese | 0.077 [0.064, 0.091] | 0.072 | 0.044 | 0.0010 | 0.0010 * | 854 | 26 |
| Chinese → French | 0.072 [0.058, 0.085] | 0.069 | 0.044 | 0.0010 | 0.0010 * | 859 | 176 |
| English → Chinese | 0.070 [0.065, 0.075] | 0.070 | 0.044 | 0.0010 | 0.0010 * | 895 | 26 |
| Chinese → English | 0.068 [0.060, 0.078] | 0.068 | 0.044 | 0.0010 | 0.0010 * | 893 | 81 |

12/12 pairs significant at q<0.05.


## Exploratory — retrieval (BH-FDR within this family)

| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |
|---|---|---|---|---|---|---|---|
| English → Semantics | 0.204 [0.184, 0.221] | 0.169 | 0.052 | 0.0010 | 0.0010 * | 625 | 81 |
| French → Semantics | 0.201 [0.186, 0.214] | 0.181 | 0.052 | 0.0010 | 0.0010 * | 605 | 81 |
| Irish → Semantics | 0.195 [0.181, 0.209] | 0.183 | 0.052 | 0.0010 | 0.0010 * | 631 | 81 |
| Chinese → Semantics | 0.166 [0.149, 0.184] | 0.161 | 0.052 | 0.0010 | 0.0010 * | 686 | 81 |

4/4 pairs significant at q<0.05.


## Baseline comparison — phonetic vs editdist, feat, orth

| pair | phonetic acc | editdist acc | feat acc | orth acc | null |
|---|---|---|---|---|---|
| French → English | 0.400 | 0.255 | 0.565 | 0.500 | 0.043 |
| English → French | 0.398 | 0.265 | 0.590 | 0.485 | 0.044 |
| Irish → English | 0.240 | 0.120 | 0.455 | 0.475 | 0.042 |
| English → Irish | 0.235 | 0.120 | 0.485 | 0.445 | 0.042 |
| English → Semantics | 0.204 | — | 0.585 | 0.555 | 0.052 |
| French → Semantics | 0.201 | — | 0.475 | 0.505 | 0.052 |
| Irish → Semantics | 0.195 | — | 0.515 | 0.630 | 0.052 |
| Irish → French | 0.182 | 0.080 | 0.465 | 0.470 | 0.043 |
| French → Irish | 0.180 | 0.055 | 0.465 | 0.490 | 0.041 |
| Chinese → Semantics | 0.166 | — | 0.545 | 0.560 | 0.052 |
| Irish → Chinese | 0.085 | 0.000 | 0.495 | 0.500 | 0.043 |
| Chinese → Irish | 0.084 | 0.000 | 0.465 | 0.515 | 0.042 |
| French → Chinese | 0.077 | 0.000 | 0.535 | 0.390 | 0.044 |
| Chinese → French | 0.072 | 0.000 | 0.570 | 0.355 | 0.044 |
| English → Chinese | 0.070 | 0.000 | 0.555 | 0.530 | 0.044 |
| Chinese → English | 0.068 | 0.000 | 0.560 | 0.485 | 0.044 |

phonetic − editdist is the retrieval accuracy not explained by raw orthographic string overlap (cognates / borrowing).


## Phoneme–meaning association (18 poles, null_iters=1000)

Significant phoneme×pole cells (q<0.10): **9** total.

| language | pole | phoneme | z | n | q_FDR |
|---|---|---|---|---|---|
| Chinese | 坤卦-dynamic | /j/ | +3.99 | 4 | 0.0779 |
| Chinese | 巽卦-dynamic | /t͡sʰ/ | +3.81 | 10 | 0.0779 |
| Chinese | 中卦-static | /s/ | +3.52 | 15 | 0.0779 |
| Chinese | 中卦-static | /ɯ/ | +3.21 | 18 | 0.0779 |
| Chinese | 中卦-static | /ʌ/ | +3.21 | 18 | 0.0779 |
| Chinese | 坤卦-static | /i/ | +2.71 | 78 | 0.0779 |
| Chinese | 坤卦-dynamic | /x/ | +2.47 | 41 | 0.0779 |
| Chinese | 兌卦-dynamic | /f/ | +2.40 | 21 | 0.0779 |
| Chinese | 艮卦-dynamic | /ɕ/ | +1.93 | 42 | 0.0779 |
