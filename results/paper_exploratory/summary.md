# Results — config `paper_exploratory` (`7dfa4915905ccd4c`)

map=ridge, k=100, folds=10, null_iters=200, bootstrap_iters=1000


## Confirmatory — retrieval (BH-FDR within this family)

| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |
|---|---|---|---|---|---|---|---|
| English → French | 0.423 [0.406, 0.442] | 0.415 | 0.052 | 0.0050 | 0.0050 * | 453 | 176 |
| French → English | 0.419 [0.401, 0.439] | 0.408 | 0.052 | 0.0050 | 0.0050 * | 447 | 81 |
| Irish → English | 0.264 [0.247, 0.280] | 0.257 | 0.051 | 0.0050 | 0.0050 * | 606 | 81 |
| English → Irish | 0.262 [0.248, 0.273] | 0.265 | 0.053 | 0.0050 | 0.0050 * | 607 | 160 |
| Chinese → English | 0.083 [0.074, 0.090] | 0.082 | 0.052 | 0.0050 | 0.0050 * | 823 | 81 |
| English → Chinese | 0.080 [0.073, 0.086] | 0.079 | 0.053 | 0.0050 | 0.0050 * | 822 | 26 |

6/6 pairs significant at q<0.05.


## Exploratory — retrieval (BH-FDR within this family)

| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |
|---|---|---|---|---|---|---|---|
| English → German | 0.461 [0.440, 0.478] | 0.458 | 0.053 | 0.0050 | 0.0050 * | 431 | 168 |
| German → English | 0.456 [0.437, 0.471] | 0.442 | 0.051 | 0.0050 | 0.0050 * | 430 | 81 |
| Spanish → English | 0.387 [0.368, 0.409] | 0.375 | 0.052 | 0.0050 | 0.0050 * | 476 | 81 |
| English → Spanish | 0.384 [0.362, 0.408] | 0.380 | 0.053 | 0.0050 | 0.0050 * | 466 | 182 |
| English → Italian | 0.382 [0.365, 0.402] | 0.374 | 0.052 | 0.0050 | 0.0050 * | 474 | 182 |
| Italian → English | 0.377 [0.356, 0.396] | 0.361 | 0.052 | 0.0050 | 0.0050 * | 472 | 81 |
| English → Hindi | 0.318 [0.298, 0.341] | 0.293 | 0.055 | 0.0050 | 0.0050 * | 538 | 199 |
| Hindi → English | 0.317 [0.301, 0.337] | 0.285 | 0.052 | 0.0050 | 0.0050 * | 538 | 81 |
| English → Welsh | 0.299 [0.276, 0.320] | 0.271 | 0.057 | 0.0050 | 0.0050 * | 523 | 139 |
| Welsh → English | 0.291 [0.265, 0.319] | 0.259 | 0.053 | 0.0050 | 0.0050 * | 529 | 79 |
| English → Japanese | 0.269 [0.257, 0.281] | 0.263 | 0.159 | 0.0050 | 0.0050 * | 622 | 99 |
| German → Semantics | 0.264 [0.247, 0.279] | 0.245 | 0.054 | 0.0050 | 0.0050 * | 504 | 81 |
| Spanish → Semantics | 0.258 [0.245, 0.274] | 0.240 | 0.053 | 0.0050 | 0.0050 * | 527 | 81 |
| Russian → Semantics | 0.255 [0.236, 0.271] | 0.237 | 0.053 | 0.0050 | 0.0050 * | 522 | 81 |
| Hungarian → Semantics | 0.253 [0.233, 0.272] | 0.219 | 0.058 | 0.0050 | 0.0050 * | 536 | 75 |
| Swahili → Semantics | 0.248 [0.231, 0.264] | 0.232 | 0.062 | 0.0050 | 0.0050 * | 474 | 77 |
| Turkish → Semantics | 0.242 [0.220, 0.264] | 0.227 | 0.055 | 0.0050 | 0.0050 * | 558 | 79 |
| Vietnamese → Semantics | 0.240 [0.223, 0.258] | 0.229 | 0.055 | 0.0050 | 0.0050 * | 555 | 79 |
| Indonesian → Semantics | 0.240 [0.219, 0.261] | 0.223 | 0.057 | 0.0050 | 0.0050 * | 532 | 79 |
| Japanese → English | 0.237 [0.226, 0.249] | 0.238 | 0.046 | 0.0050 | 0.0050 * | 641 | 81 |
| Finnish → Semantics | 0.235 [0.211, 0.260] | 0.217 | 0.054 | 0.0050 | 0.0050 * | 540 | 81 |
| Thai → Semantics | 0.235 [0.209, 0.256] | 0.223 | 0.053 | 0.0050 | 0.0050 * | 586 | 81 |
| Russian → English | 0.233 [0.219, 0.246] | 0.213 | 0.051 | 0.0050 | 0.0050 * | 621 | 81 |
| Italian → Semantics | 0.232 [0.221, 0.244] | 0.214 | 0.054 | 0.0050 | 0.0050 * | 545 | 81 |
| English → Russian | 0.228 [0.212, 0.243] | 0.220 | 0.054 | 0.0050 | 0.0050 * | 614 | 158 |
| Korean → Semantics | 0.223 [0.204, 0.240] | 0.208 | 0.053 | 0.0050 | 0.0050 * | 560 | 81 |
| Greek → Semantics | 0.222 [0.210, 0.236] | 0.204 | 0.053 | 0.0050 | 0.0050 * | 549 | 81 |
| English → Hungarian | 0.218 [0.199, 0.240] | 0.180 | 0.058 | 0.0050 | 0.0050 * | 598 | 190 |
| Hungarian → English | 0.217 [0.202, 0.234] | 0.181 | 0.056 | 0.0050 | 0.0050 * | 601 | 75 |
| French → Semantics | 0.214 [0.202, 0.225] | 0.194 | 0.054 | 0.0050 | 0.0050 * | 584 | 81 |
| English → Semantics | 0.212 [0.188, 0.230] | 0.177 | 0.054 | 0.0050 | 0.0050 * | 604 | 81 |
| Irish → Semantics | 0.208 [0.194, 0.225] | 0.196 | 0.052 | 0.0050 | 0.0050 * | 611 | 81 |
| Arabic → Semantics | 0.207 [0.193, 0.223] | 0.190 | 0.054 | 0.0050 | 0.0050 * | 594 | 81 |
| Hindi → Semantics | 0.204 [0.183, 0.225] | 0.168 | 0.055 | 0.0050 | 0.0050 * | 620 | 81 |
| Welsh → Semantics | 0.198 [0.178, 0.221] | 0.164 | 0.057 | 0.0050 | 0.0050 * | 598 | 79 |
| Japanese → Semantics | 0.193 [0.176, 0.213] | 0.186 | 0.053 | 0.0050 | 0.0050 * | 545 | 81 |
| Swahili → English | 0.189 [0.169, 0.209] | 0.167 | 0.060 | 0.0050 | 0.0050 * | 596 | 77 |
| Hebrew → Semantics | 0.188 [0.171, 0.205] | 0.171 | 0.053 | 0.0050 | 0.0050 * | 623 | 81 |
| Korean → English | 0.183 [0.166, 0.201] | 0.169 | 0.052 | 0.0050 | 0.0050 * | 677 | 81 |
| Indonesian → English | 0.183 [0.167, 0.201] | 0.164 | 0.055 | 0.0050 | 0.0050 * | 644 | 79 |
| English → Turkish | 0.182 [0.165, 0.202] | 0.171 | 0.053 | 0.0050 | 0.0050 * | 670 | 205 |
| English → Indonesian | 0.180 [0.161, 0.201] | 0.164 | 0.056 | 0.0050 | 0.0050 * | 672 | 209 |
| Finnish → English | 0.179 [0.161, 0.198] | 0.158 | 0.053 | 0.0050 | 0.0050 * | 663 | 81 |
| English → Finnish | 0.178 [0.164, 0.192] | 0.163 | 0.053 | 0.0050 | 0.0050 * | 662 | 185 |
| Turkish → English | 0.177 [0.160, 0.197] | 0.158 | 0.052 | 0.0050 | 0.0050 * | 681 | 79 |
| English → Korean | 0.177 [0.160, 0.194] | 0.165 | 0.052 | 0.0050 | 0.0050 * | 677 | 102 |
| English → Swahili | 0.174 [0.153, 0.195] | 0.153 | 0.062 | 0.0050 | 0.0050 * | 615 | 363 |
| Chinese → Semantics | 0.170 [0.156, 0.184] | 0.169 | 0.054 | 0.0050 | 0.0050 * | 663 | 81 |
| English → Greek | 0.163 [0.148, 0.182] | 0.150 | 0.053 | 0.0050 | 0.0050 * | 675 | 143 |
| Greek → English | 0.160 [0.143, 0.178] | 0.140 | 0.050 | 0.0050 | 0.0050 * | 693 | 81 |
| Arabic → English | 0.153 [0.141, 0.165] | 0.136 | 0.053 | 0.0050 | 0.0050 * | 716 | 81 |
| English → Arabic | 0.153 [0.136, 0.170] | 0.141 | 0.052 | 0.0050 | 0.0050 * | 715 | 134 |
| English → Thai | 0.149 [0.129, 0.173] | 0.131 | 0.053 | 0.0050 | 0.0050 * | 734 | 175 |
| Thai → English | 0.148 [0.128, 0.168] | 0.126 | 0.051 | 0.0050 | 0.0050 * | 748 | 81 |
| English → Hebrew | 0.142 [0.129, 0.154] | 0.127 | 0.053 | 0.0050 | 0.0050 * | 727 | 139 |
| Hebrew → English | 0.140 [0.133, 0.148] | 0.123 | 0.052 | 0.0050 | 0.0050 * | 740 | 81 |
| Vietnamese → English | 0.116 [0.104, 0.128] | 0.099 | 0.054 | 0.0050 | 0.0050 * | 748 | 79 |
| English → Vietnamese | 0.116 [0.104, 0.128] | 0.092 | 0.055 | 0.0050 | 0.0050 * | 741 | 157 |

58/58 pairs significant at q<0.05.


## Form–meaning correlation (Mantel, n=600 concepts)

| analysis | unit | r | p | r \| orthography | p (partial) | note |
|---|---|---|---|---|---|---|
| form~meaning | Hungarian | +0.0211* | 0.0050 | +0.0206* | 0.0050 |  |
| form~meaning | Finnish | +0.0244* | 0.0050 | +0.0179* | 0.0050 |  |
| form~meaning | Greek | +0.0225* | 0.0050 | +0.0138* | 0.0050 |  |
| form~meaning | Russian | +0.0317* | 0.0050 | +0.0214* | 0.0050 |  |
| form~meaning | German | +0.0341* | 0.0050 | +0.0253* | 0.0050 |  |
| form~meaning | Spanish | +0.0298* | 0.0050 | +0.0243* | 0.0050 |  |
| form~meaning | Italian | +0.0289* | 0.0050 | +0.0184* | 0.0050 |  |
| form~meaning | French | +0.0185* | 0.0050 | +0.0127* | 0.0050 |  |
| form~meaning | Irish | +0.0228* | 0.0050 | +0.0221* | 0.0050 |  |
| form~meaning | Welsh | +0.0180* | 0.0050 | +0.0179* | 0.0050 |  |
| form~meaning | English | +0.0198* | 0.0050 | +0.0189* | 0.0050 |  |
| form~meaning | Chinese | +0.0155* | 0.0050 | +0.0001 | 0.9353 | orth control degenerate |
| form~meaning | Vietnamese | +0.0272* | 0.0050 | +0.0253* | 0.0050 |  |
| form~meaning | Japanese | +0.0389* | 0.0050 | +0.0362* | 0.0050 | orth control degenerate |
| form~meaning | Korean | +0.0259* | 0.0050 | +0.0082* | 0.0050 |  |
| form~meaning | Thai | +0.0205* | 0.0050 | +0.0175* | 0.0050 |  |
| form~meaning | Indonesian | +0.0248* | 0.0050 | +0.0168* | 0.0050 |  |
| form~meaning | Turkish | +0.0248* | 0.0050 | +0.0150* | 0.0050 |  |
| form~meaning | Arabic | +0.0218* | 0.0050 | +0.0202* | 0.0050 |  |
| form~meaning | Hebrew | +0.0134* | 0.0050 | +0.0087* | 0.0050 |  |
| form~meaning | Swahili | +0.0218* | 0.0050 | +0.0169* | 0.0050 |  |
| form~meaning | Hindi | +0.0172* | 0.0050 | +0.0144* | 0.0050 |  |
| form~form | Hungarian~Finnish | +0.0128* | 0.0050 | +0.0115* | 0.0050 |  |
| form~form | Hungarian~Greek | +0.0163* | 0.0050 | +0.0150* | 0.0050 |  |
| form~form | Hungarian~Russian | +0.0208* | 0.0050 | +0.0183* | 0.0050 |  |
| form~form | Hungarian~German | +0.0302* | 0.0050 | +0.0264* | 0.0050 |  |
| form~form | Hungarian~Spanish | +0.0206* | 0.0050 | +0.0183* | 0.0050 |  |
| form~form | Hungarian~Italian | +0.0183* | 0.0050 | +0.0160* | 0.0050 |  |
| form~form | Hungarian~French | +0.0122* | 0.0050 | +0.0101* | 0.0050 |  |
| form~form | Hungarian~Irish | +0.0085* | 0.0050 | +0.0075* | 0.0050 |  |
| form~form | Hungarian~Welsh | +0.0095* | 0.0050 | +0.0073* | 0.0100 |  |
| form~form | Hungarian~English | +0.0281* | 0.0050 | +0.0246* | 0.0050 |  |
| form~form | Hungarian~Chinese | +0.0062* | 0.0149 | +0.0058* | 0.0199 | orth control degenerate |
| form~form | Hungarian~Vietnamese | +0.0068* | 0.0149 | +0.0065* | 0.0199 |  |
| form~form | Hungarian~Japanese | -0.0006 | 0.7910 | +0.0024 | 0.4229 | orth control degenerate |
| form~form | Hungarian~Korean | +0.0091* | 0.0050 | +0.0079* | 0.0050 |  |
| form~form | Hungarian~Thai | +0.0082* | 0.0050 | +0.0070* | 0.0050 |  |
| form~form | Hungarian~Indonesian | +0.0125* | 0.0050 | +0.0116* | 0.0050 |  |
| form~form | Hungarian~Turkish | +0.0183* | 0.0050 | +0.0167* | 0.0050 |  |
| form~form | Hungarian~Arabic | +0.0126* | 0.0050 | +0.0114* | 0.0050 |  |
| form~form | Hungarian~Hebrew | +0.0076* | 0.0050 | +0.0078* | 0.0050 |  |
| form~form | Hungarian~Swahili | +0.0158* | 0.0050 | +0.0145* | 0.0050 |  |
| form~form | Hungarian~Hindi | +0.0110* | 0.0050 | +0.0099* | 0.0050 |  |
| form~form | Finnish~Greek | +0.0105* | 0.0050 | +0.0081* | 0.0050 |  |
| form~form | Finnish~Russian | +0.0204* | 0.0050 | +0.0170* | 0.0050 |  |
| form~form | Finnish~German | +0.0239* | 0.0050 | +0.0195* | 0.0050 |  |
| form~form | Finnish~Spanish | +0.0178* | 0.0050 | +0.0151* | 0.0050 |  |
| form~form | Finnish~Italian | +0.0135* | 0.0050 | +0.0080* | 0.0050 |  |
| form~form | Finnish~French | +0.0076* | 0.0050 | +0.0059* | 0.0100 |  |
| form~form | Finnish~Irish | +0.0093* | 0.0050 | +0.0073* | 0.0050 |  |
| form~form | Finnish~Welsh | +0.0095* | 0.0050 | +0.0070* | 0.0050 |  |
| form~form | Finnish~English | +0.0168* | 0.0050 | +0.0134* | 0.0050 |  |
| form~form | Finnish~Chinese | +0.0062* | 0.0199 | +0.0050 | 0.0547 | orth control degenerate |
| form~form | Finnish~Vietnamese | +0.0016 | 0.4726 | +0.0007 | 0.7960 |  |
| form~form | Finnish~Japanese | +0.0034 | 0.1841 | +0.0055 | 0.1095 | orth control degenerate |
| form~form | Finnish~Korean | +0.0050* | 0.0448 | +0.0048* | 0.0448 |  |
| form~form | Finnish~Thai | +0.0122* | 0.0050 | +0.0106* | 0.0050 |  |
| form~form | Finnish~Indonesian | +0.0141* | 0.0050 | +0.0110* | 0.0050 |  |
| form~form | Finnish~Turkish | +0.0167* | 0.0050 | +0.0138* | 0.0050 |  |
| form~form | Finnish~Arabic | +0.0062* | 0.0100 | +0.0053* | 0.0249 |  |
| form~form | Finnish~Hebrew | +0.0066* | 0.0100 | +0.0053* | 0.0348 |  |
| form~form | Finnish~Swahili | +0.0141* | 0.0050 | +0.0111* | 0.0050 |  |
| form~form | Finnish~Hindi | +0.0098* | 0.0050 | +0.0081* | 0.0050 |  |
| form~form | Greek~Russian | +0.0187* | 0.0050 | +0.0150* | 0.0050 |  |
| form~form | Greek~German | +0.0251* | 0.0050 | +0.0214* | 0.0050 |  |
| form~form | Greek~Spanish | +0.0208* | 0.0050 | +0.0165* | 0.0050 |  |
| form~form | Greek~Italian | +0.0280* | 0.0050 | +0.0222* | 0.0050 |  |
| form~form | Greek~French | +0.0158* | 0.0050 | +0.0142* | 0.0050 |  |
| form~form | Greek~Irish | +0.0114* | 0.0050 | +0.0094* | 0.0050 |  |
| form~form | Greek~Welsh | +0.0105* | 0.0050 | +0.0091* | 0.0050 |  |
| form~form | Greek~English | +0.0158* | 0.0050 | +0.0132* | 0.0050 |  |
| form~form | Greek~Chinese | +0.0038 | 0.0846 | +0.0037 | 0.0995 | orth control degenerate |
| form~form | Greek~Vietnamese | +0.0098* | 0.0050 | +0.0079* | 0.0050 |  |
| form~form | Greek~Japanese | +0.0089* | 0.0050 | +0.0048 | 0.1443 | orth control degenerate |
| form~form | Greek~Korean | +0.0110* | 0.0050 | +0.0071* | 0.0050 |  |
| form~form | Greek~Thai | +0.0089* | 0.0050 | +0.0062* | 0.0100 |  |
| form~form | Greek~Indonesian | +0.0137* | 0.0050 | +0.0108* | 0.0050 |  |
| form~form | Greek~Turkish | +0.0205* | 0.0050 | +0.0182* | 0.0050 |  |
| form~form | Greek~Arabic | +0.0107* | 0.0050 | +0.0083* | 0.0050 |  |
| form~form | Greek~Hebrew | +0.0079* | 0.0050 | +0.0057* | 0.0199 |  |
| form~form | Greek~Swahili | +0.0106* | 0.0050 | +0.0077* | 0.0050 |  |
| form~form | Greek~Hindi | +0.0085* | 0.0100 | +0.0071* | 0.0149 |  |
| form~form | Russian~German | +0.0365* | 0.0050 | +0.0308* | 0.0050 |  |
| form~form | Russian~Spanish | +0.0297* | 0.0050 | +0.0233* | 0.0050 |  |
| form~form | Russian~Italian | +0.0415* | 0.0050 | +0.0343* | 0.0050 |  |
| form~form | Russian~French | +0.0288* | 0.0050 | +0.0244* | 0.0050 |  |
| form~form | Russian~Irish | +0.0172* | 0.0050 | +0.0145* | 0.0050 |  |
| form~form | Russian~Welsh | +0.0133* | 0.0050 | +0.0112* | 0.0050 |  |
| form~form | Russian~English | +0.0237* | 0.0050 | +0.0205* | 0.0050 |  |
| form~form | Russian~Chinese | +0.0038 | 0.0995 | +0.0038 | 0.1045 | orth control degenerate |
| form~form | Russian~Vietnamese | +0.0053* | 0.0249 | +0.0043 | 0.0896 |  |
| form~form | Russian~Japanese | +0.0065* | 0.0100 | +0.0072* | 0.0249 | orth control degenerate |
| form~form | Russian~Korean | +0.0093* | 0.0050 | +0.0053* | 0.0249 |  |
| form~form | Russian~Thai | +0.0088* | 0.0050 | +0.0070* | 0.0100 |  |
| form~form | Russian~Indonesian | +0.0153* | 0.0050 | +0.0119* | 0.0050 |  |
| form~form | Russian~Turkish | +0.0232* | 0.0050 | +0.0184* | 0.0050 |  |
| form~form | Russian~Arabic | +0.0173* | 0.0050 | +0.0159* | 0.0050 |  |
| form~form | Russian~Hebrew | +0.0090* | 0.0050 | +0.0064* | 0.0050 |  |
| form~form | Russian~Swahili | +0.0146* | 0.0050 | +0.0113* | 0.0050 |  |
| form~form | Russian~Hindi | +0.0092* | 0.0050 | +0.0078* | 0.0050 |  |
| form~form | German~Spanish | +0.0357* | 0.0050 | +0.0304* | 0.0050 |  |
| form~form | German~Italian | +0.0374* | 0.0050 | +0.0311* | 0.0050 |  |
| form~form | German~French | +0.0337* | 0.0050 | +0.0293* | 0.0050 |  |
| form~form | German~Irish | +0.0202* | 0.0050 | +0.0173* | 0.0050 |  |
| form~form | German~Welsh | +0.0233* | 0.0050 | +0.0212* | 0.0050 |  |
| form~form | German~English | +0.0674* | 0.0050 | +0.0594* | 0.0050 |  |
| form~form | German~Chinese | +0.0058* | 0.0149 | +0.0052* | 0.0299 | orth control degenerate |
| form~form | German~Vietnamese | +0.0065* | 0.0100 | +0.0062* | 0.0149 |  |
| form~form | German~Japanese | +0.0130* | 0.0050 | +0.0175* | 0.0050 | orth control degenerate |
| form~form | German~Korean | +0.0147* | 0.0050 | +0.0130* | 0.0050 |  |
| form~form | German~Thai | +0.0125* | 0.0050 | +0.0125* | 0.0050 |  |
| form~form | German~Indonesian | +0.0219* | 0.0050 | +0.0187* | 0.0050 |  |
| form~form | German~Turkish | +0.0321* | 0.0050 | +0.0274* | 0.0050 |  |
| form~form | German~Arabic | +0.0108* | 0.0050 | +0.0103* | 0.0050 |  |
| form~form | German~Hebrew | +0.0146* | 0.0050 | +0.0125* | 0.0050 |  |
| form~form | German~Swahili | +0.0230* | 0.0050 | +0.0194* | 0.0050 |  |
| form~form | German~Hindi | +0.0147* | 0.0050 | +0.0137* | 0.0050 |  |
| form~form | Spanish~Italian | +0.1732* | 0.0050 | +0.1516* | 0.0050 |  |
| form~form | Spanish~French | +0.1087* | 0.0050 | +0.0964* | 0.0050 |  |
| form~form | Spanish~Irish | +0.0196* | 0.0050 | +0.0166* | 0.0050 |  |
| form~form | Spanish~Welsh | +0.0215* | 0.0050 | +0.0189* | 0.0050 |  |
| form~form | Spanish~English | +0.0615* | 0.0050 | +0.0546* | 0.0050 |  |
| form~form | Spanish~Chinese | +0.0062* | 0.0149 | +0.0055* | 0.0348 | orth control degenerate |
| form~form | Spanish~Vietnamese | +0.0100* | 0.0050 | +0.0101* | 0.0050 |  |
| form~form | Spanish~Japanese | +0.0124* | 0.0050 | +0.0098* | 0.0199 | orth control degenerate |
| form~form | Spanish~Korean | +0.0055* | 0.0100 | +0.0040 | 0.0796 |  |
| form~form | Spanish~Thai | +0.0099* | 0.0050 | +0.0085* | 0.0050 |  |
| form~form | Spanish~Indonesian | +0.0177* | 0.0050 | +0.0142* | 0.0050 |  |
| form~form | Spanish~Turkish | +0.0227* | 0.0050 | +0.0186* | 0.0050 |  |
| form~form | Spanish~Arabic | +0.0105* | 0.0050 | +0.0090* | 0.0050 |  |
| form~form | Spanish~Hebrew | +0.0097* | 0.0050 | +0.0074* | 0.0050 |  |
| form~form | Spanish~Swahili | +0.0093* | 0.0050 | +0.0063* | 0.0149 |  |
| form~form | Spanish~Hindi | +0.0148* | 0.0050 | +0.0126* | 0.0050 |  |
| form~form | Italian~French | +0.1159* | 0.0050 | +0.1015* | 0.0050 |  |
| form~form | Italian~Irish | +0.0205* | 0.0050 | +0.0169* | 0.0050 |  |
| form~form | Italian~Welsh | +0.0177* | 0.0050 | +0.0143* | 0.0050 |  |
| form~form | Italian~English | +0.0545* | 0.0050 | +0.0467* | 0.0050 |  |
| form~form | Italian~Chinese | +0.0034 | 0.1393 | +0.0029 | 0.2239 | orth control degenerate |
| form~form | Italian~Vietnamese | +0.0046 | 0.0796 | +0.0050 | 0.0547 |  |
| form~form | Italian~Japanese | +0.0052* | 0.0398 | +0.0037 | 0.2637 | orth control degenerate |
| form~form | Italian~Korean | +0.0066* | 0.0050 | +0.0036 | 0.1144 |  |
| form~form | Italian~Thai | +0.0062* | 0.0149 | +0.0048 | 0.0547 |  |
| form~form | Italian~Indonesian | +0.0181* | 0.0050 | +0.0137* | 0.0050 |  |
| form~form | Italian~Turkish | +0.0247* | 0.0050 | +0.0196* | 0.0050 |  |
| form~form | Italian~Arabic | +0.0134* | 0.0050 | +0.0112* | 0.0050 |  |
| form~form | Italian~Hebrew | +0.0028 | 0.2488 | -0.0004 | 0.8856 |  |
| form~form | Italian~Swahili | +0.0181* | 0.0050 | +0.0141* | 0.0050 |  |
| form~form | Italian~Hindi | +0.0153* | 0.0050 | +0.0119* | 0.0050 |  |
| form~form | French~Irish | +0.0167* | 0.0050 | +0.0139* | 0.0050 |  |
| form~form | French~Welsh | +0.0171* | 0.0050 | +0.0149* | 0.0050 |  |
| form~form | French~English | +0.0612* | 0.0050 | +0.0545* | 0.0050 |  |
| form~form | French~Chinese | +0.0064* | 0.0100 | +0.0063* | 0.0100 | orth control degenerate |
| form~form | French~Vietnamese | +0.0082* | 0.0050 | +0.0081* | 0.0050 |  |
| form~form | French~Japanese | +0.0110* | 0.0050 | +0.0131* | 0.0050 | orth control degenerate |
| form~form | French~Korean | +0.0020 | 0.3582 | +0.0009 | 0.6965 |  |
| form~form | French~Thai | +0.0119* | 0.0050 | +0.0105* | 0.0050 |  |
| form~form | French~Indonesian | +0.0158* | 0.0050 | +0.0129* | 0.0050 |  |
| form~form | French~Turkish | +0.0203* | 0.0050 | +0.0173* | 0.0050 |  |
| form~form | French~Arabic | +0.0075* | 0.0100 | +0.0061* | 0.0149 |  |
| form~form | French~Hebrew | +0.0084* | 0.0050 | +0.0061* | 0.0050 |  |
| form~form | French~Swahili | +0.0150* | 0.0050 | +0.0126* | 0.0050 |  |
| form~form | French~Hindi | +0.0137* | 0.0050 | +0.0117* | 0.0050 |  |
| form~form | Irish~Welsh | +0.0225* | 0.0050 | +0.0193* | 0.0050 |  |
| form~form | Irish~English | +0.0318* | 0.0050 | +0.0285* | 0.0050 |  |
| form~form | Irish~Chinese | +0.0037 | 0.0995 | +0.0032 | 0.1244 | orth control degenerate |
| form~form | Irish~Vietnamese | +0.0030 | 0.1841 | +0.0027 | 0.2239 |  |
| form~form | Irish~Japanese | +0.0047* | 0.0498 | +0.0036 | 0.2836 | orth control degenerate |
| form~form | Irish~Korean | +0.0089* | 0.0050 | +0.0076* | 0.0100 |  |
| form~form | Irish~Thai | +0.0040 | 0.1095 | +0.0035 | 0.1891 |  |
| form~form | Irish~Indonesian | +0.0118* | 0.0050 | +0.0105* | 0.0050 |  |
| form~form | Irish~Turkish | +0.0159* | 0.0050 | +0.0149* | 0.0050 |  |
| form~form | Irish~Arabic | +0.0048* | 0.0398 | +0.0040 | 0.0995 |  |
| form~form | Irish~Hebrew | +0.0098* | 0.0050 | +0.0086* | 0.0050 |  |
| form~form | Irish~Swahili | +0.0129* | 0.0050 | +0.0113* | 0.0050 |  |
| form~form | Irish~Hindi | +0.0090* | 0.0050 | +0.0076* | 0.0050 |  |
| form~form | Welsh~English | +0.0318* | 0.0050 | +0.0273* | 0.0050 |  |
| form~form | Welsh~Chinese | +0.0039 | 0.1194 | +0.0031 | 0.2090 | orth control degenerate |
| form~form | Welsh~Vietnamese | +0.0016 | 0.4428 | +0.0012 | 0.5672 |  |
| form~form | Welsh~Japanese | +0.0069* | 0.0100 | +0.0021 | 0.4876 | orth control degenerate |
| form~form | Welsh~Korean | +0.0050* | 0.0448 | +0.0040 | 0.0846 |  |
| form~form | Welsh~Thai | +0.0038 | 0.0945 | +0.0034 | 0.1194 |  |
| form~form | Welsh~Indonesian | +0.0107* | 0.0050 | +0.0093* | 0.0050 |  |
| form~form | Welsh~Turkish | +0.0183* | 0.0050 | +0.0157* | 0.0050 |  |
| form~form | Welsh~Arabic | +0.0094* | 0.0050 | +0.0078* | 0.0050 |  |
| form~form | Welsh~Hebrew | +0.0024 | 0.2935 | +0.0012 | 0.6020 |  |
| form~form | Welsh~Swahili | +0.0174* | 0.0050 | +0.0160* | 0.0050 |  |
| form~form | Welsh~Hindi | +0.0154* | 0.0050 | +0.0141* | 0.0050 |  |
| form~form | English~Chinese | +0.0008 | 0.7065 | +0.0012 | 0.6070 | orth control degenerate |
| form~form | English~Vietnamese | +0.0096* | 0.0050 | +0.0093* | 0.0050 |  |
| form~form | English~Japanese | +0.0102* | 0.0050 | +0.0056 | 0.0547 | orth control degenerate |
| form~form | English~Korean | +0.0121* | 0.0050 | +0.0106* | 0.0050 |  |
| form~form | English~Thai | +0.0113* | 0.0050 | +0.0096* | 0.0050 |  |
| form~form | English~Indonesian | +0.0216* | 0.0050 | +0.0191* | 0.0050 |  |
| form~form | English~Turkish | +0.0224* | 0.0050 | +0.0199* | 0.0050 |  |
| form~form | English~Arabic | +0.0151* | 0.0050 | +0.0138* | 0.0050 |  |
| form~form | English~Hebrew | +0.0088* | 0.0050 | +0.0076* | 0.0050 |  |
| form~form | English~Swahili | +0.0239* | 0.0050 | +0.0207* | 0.0050 |  |
| form~form | English~Hindi | +0.0519* | 0.0050 | +0.0483* | 0.0050 |  |
| form~form | Chinese~Vietnamese | +0.0168* | 0.0050 | +0.0102* | 0.0050 | orth control degenerate |
| form~form | Chinese~Japanese | +0.0067* | 0.0149 | +0.0037 | 0.2139 | orth control degenerate |
| form~form | Chinese~Korean | +0.0242* | 0.0050 | +0.0168* | 0.0050 | orth control degenerate |
| form~form | Chinese~Thai | +0.0071* | 0.0050 | +0.0014 | 0.6219 | orth control degenerate |
| form~form | Chinese~Indonesian | +0.0126* | 0.0050 | +0.0082* | 0.0050 | orth control degenerate |
| form~form | Chinese~Turkish | +0.0074* | 0.0100 | +0.0032 | 0.2090 | orth control degenerate |
| form~form | Chinese~Arabic | +0.0075* | 0.0100 | +0.0028 | 0.2587 | orth control degenerate |
| form~form | Chinese~Hebrew | +0.0066* | 0.0050 | +0.0043 | 0.0846 | orth control degenerate |
| form~form | Chinese~Swahili | +0.0033 | 0.1592 | +0.0004 | 0.8507 | orth control degenerate |
| form~form | Chinese~Hindi | -0.0005 | 0.8259 | -0.0024 | 0.3234 | orth control degenerate |
| form~form | Vietnamese~Japanese | +0.0019 | 0.4577 | -0.0033 | 0.3483 | orth control degenerate |
| form~form | Vietnamese~Korean | +0.0130* | 0.0050 | +0.0120* | 0.0050 |  |
| form~form | Vietnamese~Thai | +0.0065* | 0.0050 | +0.0050* | 0.0249 |  |
| form~form | Vietnamese~Indonesian | +0.0060* | 0.0100 | +0.0048 | 0.0547 |  |
| form~form | Vietnamese~Turkish | +0.0039 | 0.0896 | +0.0030 | 0.1841 |  |
| form~form | Vietnamese~Arabic | +0.0089* | 0.0050 | +0.0077* | 0.0100 |  |
| form~form | Vietnamese~Hebrew | +0.0047* | 0.0498 | +0.0039 | 0.0846 |  |
| form~form | Vietnamese~Swahili | +0.0061* | 0.0149 | +0.0056* | 0.0199 |  |
| form~form | Vietnamese~Hindi | +0.0046 | 0.0647 | +0.0038 | 0.1095 |  |
| form~form | Japanese~Korean | +0.0189* | 0.0050 | +0.0177* | 0.0050 | orth control degenerate |
| form~form | Japanese~Thai | +0.0114* | 0.0050 | +0.0111* | 0.0050 | orth control degenerate |
| form~form | Japanese~Indonesian | +0.0081* | 0.0050 | +0.0074* | 0.0050 | orth control degenerate |
| form~form | Japanese~Turkish | +0.0079* | 0.0050 | +0.0070* | 0.0100 | orth control degenerate |
| form~form | Japanese~Arabic | +0.0056* | 0.0149 | +0.0053* | 0.0149 | orth control degenerate |
| form~form | Japanese~Hebrew | +0.0063* | 0.0149 | +0.0059* | 0.0149 | orth control degenerate |
| form~form | Japanese~Swahili | +0.0071* | 0.0299 | +0.0064* | 0.0348 | orth control degenerate |
| form~form | Japanese~Hindi | +0.0048* | 0.0498 | +0.0044 | 0.0697 | orth control degenerate |
| form~form | Korean~Thai | +0.0039 | 0.0945 | +0.0015 | 0.4726 |  |
| form~form | Korean~Indonesian | +0.0074* | 0.0149 | +0.0020 | 0.4328 |  |
| form~form | Korean~Turkish | +0.0114* | 0.0050 | +0.0056* | 0.0249 |  |
| form~form | Korean~Arabic | +0.0085* | 0.0050 | +0.0059* | 0.0199 |  |
| form~form | Korean~Hebrew | +0.0048 | 0.0597 | +0.0023 | 0.3930 |  |
| form~form | Korean~Swahili | +0.0118* | 0.0050 | +0.0068* | 0.0050 |  |
| form~form | Korean~Hindi | +0.0043 | 0.0945 | +0.0027 | 0.2587 |  |
| form~form | Thai~Indonesian | +0.0173* | 0.0050 | +0.0134* | 0.0050 |  |
| form~form | Thai~Turkish | +0.0118* | 0.0050 | +0.0093* | 0.0050 |  |
| form~form | Thai~Arabic | +0.0090* | 0.0050 | +0.0072* | 0.0050 |  |
| form~form | Thai~Hebrew | +0.0053* | 0.0398 | +0.0037 | 0.1642 |  |
| form~form | Thai~Swahili | +0.0122* | 0.0050 | +0.0105* | 0.0050 |  |
| form~form | Thai~Hindi | +0.0084* | 0.0050 | +0.0065* | 0.0100 |  |
| form~form | Indonesian~Turkish | +0.0247* | 0.0050 | +0.0207* | 0.0050 |  |
| form~form | Indonesian~Arabic | +0.0102* | 0.0050 | +0.0093* | 0.0050 |  |
| form~form | Indonesian~Hebrew | +0.0063* | 0.0100 | +0.0038 | 0.0746 |  |
| form~form | Indonesian~Swahili | +0.0147* | 0.0050 | +0.0113* | 0.0050 |  |
| form~form | Indonesian~Hindi | +0.0039 | 0.1244 | +0.0025 | 0.2687 |  |
| form~form | Turkish~Arabic | +0.0243* | 0.0050 | +0.0223* | 0.0050 |  |
| form~form | Turkish~Hebrew | +0.0094* | 0.0050 | +0.0072* | 0.0050 |  |
| form~form | Turkish~Swahili | +0.0232* | 0.0050 | +0.0190* | 0.0050 |  |
| form~form | Turkish~Hindi | +0.0161* | 0.0050 | +0.0145* | 0.0050 |  |
| form~form | Arabic~Hebrew | +0.0132* | 0.0050 | +0.0101* | 0.0050 |  |
| form~form | Arabic~Swahili | +0.0138* | 0.0050 | +0.0125* | 0.0050 |  |
| form~form | Arabic~Hindi | +0.0135* | 0.0050 | +0.0119* | 0.0050 |  |
| form~form | Hebrew~Swahili | +0.0072* | 0.0050 | +0.0046* | 0.0448 |  |
| form~form | Hebrew~Hindi | +0.0068* | 0.0100 | +0.0057* | 0.0398 |  |
| form~form | Swahili~Hindi | +0.0092* | 0.0050 | +0.0072* | 0.0050 |  |

A within-language *form~meaning* r that stays significant in the *| orthography* column is sound–meaning systematicity not attributable to spelling / cognate overlap.


## Phoneme–meaning association (18 poles, null_iters=200)

Significant phoneme×pole cells (q<0.10): **0** total.

