# Results — config `paper_exploratory` (`91448921b21baaae`)

map=ridge, k=100, folds=10, null_iters=200, bootstrap_iters=1000


## Confirmatory — retrieval (BH-FDR within this family)

| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |
|---|---|---|---|---|---|---|---|
| French → English | 0.418 [0.396, 0.436] | 0.409 | 0.048 | 0.0050 | 0.0050 * | 487 | 79 |
| English → French | 0.417 [0.395, 0.437] | 0.412 | 0.049 | 0.0050 | 0.0050 * | 493 | 197 |
| English → Irish | 0.264 [0.253, 0.274] | 0.267 | 0.050 | 0.0050 | 0.0050 * | 627 | 176 |
| Irish → English | 0.254 [0.240, 0.269] | 0.248 | 0.048 | 0.0050 | 0.0050 * | 629 | 79 |
| Chinese → English | 0.076 [0.065, 0.087] | 0.075 | 0.049 | 0.0050 | 0.0050 * | 864 | 79 |
| English → Chinese | 0.072 [0.059, 0.084] | 0.070 | 0.049 | 0.0050 | 0.0050 * | 854 | 32 |

6/6 pairs significant at q<0.05.


## Exploratory — retrieval (BH-FDR within this family)

| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |
|---|---|---|---|---|---|---|---|
| English → German | 0.445 [0.429, 0.460] | 0.442 | 0.049 | 0.0050 | 0.0050 * | 469 | 145 |
| German → English | 0.441 [0.426, 0.455] | 0.430 | 0.049 | 0.0050 | 0.0050 * | 467 | 79 |
| Italian → English | 0.369 [0.353, 0.384] | 0.355 | 0.048 | 0.0050 | 0.0050 * | 510 | 79 |
| English → Spanish | 0.369 [0.351, 0.387] | 0.362 | 0.048 | 0.0050 | 0.0050 * | 515 | 176 |
| Spanish → English | 0.364 [0.349, 0.383] | 0.356 | 0.048 | 0.0050 | 0.0050 * | 515 | 79 |
| English → Italian | 0.357 [0.340, 0.373] | 0.349 | 0.049 | 0.0050 | 0.0050 * | 506 | 186 |
| English → Welsh | 0.300 [0.280, 0.316] | 0.299 | 0.050 | 0.0050 | 0.0050 * | 571 | 125 |
| Welsh → English | 0.298 [0.279, 0.314] | 0.284 | 0.049 | 0.0050 | 0.0050 * | 580 | 77 |
| English → Japanese | 0.283 [0.266, 0.298] | 0.279 | 0.176 | 0.0050 | 0.0050 * | 669 | 103 |
| German → Semantics | 0.255 [0.239, 0.272] | 0.240 | 0.050 | 0.0050 | 0.0050 * | 533 | 79 |
| Swahili → Semantics | 0.248 [0.237, 0.259] | 0.233 | 0.056 | 0.0050 | 0.0050 * | 526 | 77 |
| Vietnamese → Semantics | 0.243 [0.226, 0.264] | 0.234 | 0.051 | 0.0050 | 0.0050 * | 602 | 79 |
| Italian → Semantics | 0.234 [0.216, 0.253] | 0.220 | 0.050 | 0.0050 | 0.0050 * | 579 | 79 |
| Finnish → Semantics | 0.232 [0.217, 0.249] | 0.216 | 0.050 | 0.0050 | 0.0050 * | 562 | 79 |
| Hungarian → Semantics | 0.232 [0.218, 0.246] | 0.217 | 0.051 | 0.0050 | 0.0050 * | 596 | 77 |
| Russian → Semantics | 0.230 [0.208, 0.252] | 0.220 | 0.051 | 0.0050 | 0.0050 * | 558 | 79 |
| Spanish → Semantics | 0.226 [0.208, 0.244] | 0.218 | 0.051 | 0.0050 | 0.0050 * | 558 | 79 |
| Thai → Semantics | 0.226 [0.202, 0.250] | 0.216 | 0.050 | 0.0050 | 0.0050 * | 621 | 79 |
| Hindi → English | 0.224 [0.205, 0.245] | 0.205 | 0.048 | 0.0050 | 0.0050 * | 677 | 79 |
| Turkish → Semantics | 0.224 [0.206, 0.242] | 0.217 | 0.050 | 0.0050 | 0.0050 * | 588 | 77 |
| Welsh → Semantics | 0.223 [0.207, 0.240] | 0.212 | 0.051 | 0.0050 | 0.0050 * | 606 | 77 |
| English → Hindi | 0.221 [0.205, 0.238] | 0.205 | 0.049 | 0.0050 | 0.0050 * | 676 | 160 |
| Indonesian → Semantics | 0.215 [0.192, 0.235] | 0.203 | 0.052 | 0.0050 | 0.0050 * | 584 | 77 |
| Russian → English | 0.214 [0.193, 0.235] | 0.202 | 0.047 | 0.0050 | 0.0050 * | 671 | 79 |
| Greek → Semantics | 0.213 [0.202, 0.225] | 0.197 | 0.050 | 0.0050 | 0.0050 * | 614 | 79 |
| Korean → Semantics | 0.213 [0.198, 0.229] | 0.201 | 0.050 | 0.0050 | 0.0050 * | 598 | 79 |
| English → Russian | 0.207 [0.187, 0.228] | 0.204 | 0.050 | 0.0050 | 0.0050 * | 666 | 146 |
| English → Semantics | 0.206 [0.189, 0.221] | 0.175 | 0.050 | 0.0050 | 0.0050 * | 633 | 79 |
| English → Finnish | 0.204 [0.191, 0.216] | 0.192 | 0.051 | 0.0050 | 0.0050 * | 677 | 197 |
| Arabic → Semantics | 0.203 [0.186, 0.216] | 0.189 | 0.050 | 0.0050 | 0.0050 * | 612 | 79 |
| Finnish → English | 0.199 [0.186, 0.212] | 0.180 | 0.049 | 0.0050 | 0.0050 * | 682 | 79 |
| Hindi → Semantics | 0.198 [0.185, 0.210] | 0.178 | 0.050 | 0.0050 | 0.0050 * | 635 | 79 |
| French → Semantics | 0.198 [0.179, 0.216] | 0.184 | 0.050 | 0.0050 | 0.0050 * | 636 | 79 |
| Irish → Semantics | 0.193 [0.178, 0.208] | 0.183 | 0.050 | 0.0050 | 0.0050 * | 642 | 79 |
| Japanese → English | 0.190 [0.178, 0.202] | 0.190 | 0.042 | 0.0050 | 0.0050 * | 720 | 79 |
| Hungarian → English | 0.183 [0.169, 0.197] | 0.164 | 0.050 | 0.0050 | 0.0050 * | 702 | 77 |
| English → Hungarian | 0.182 [0.167, 0.196] | 0.159 | 0.051 | 0.0050 | 0.0050 * | 693 | 177 |
| Hebrew → Semantics | 0.180 [0.166, 0.194] | 0.164 | 0.050 | 0.0050 | 0.0050 * | 656 | 79 |
| Japanese → Semantics | 0.176 [0.167, 0.187] | 0.174 | 0.050 | 0.0050 | 0.0050 * | 568 | 79 |
| Chinese → Semantics | 0.171 [0.155, 0.185] | 0.170 | 0.050 | 0.0050 | 0.0050 * | 695 | 79 |
| Swahili → English | 0.171 [0.159, 0.186] | 0.150 | 0.053 | 0.0050 | 0.0050 * | 674 | 77 |
| Turkish → English | 0.170 [0.158, 0.184] | 0.158 | 0.049 | 0.0050 | 0.0050 * | 725 | 77 |
| English → Turkish | 0.169 [0.158, 0.180] | 0.157 | 0.050 | 0.0050 | 0.0050 * | 710 | 219 |
| English → Swahili | 0.165 [0.145, 0.189] | 0.140 | 0.055 | 0.0050 | 0.0050 * | 691 | 350 |
| English → Indonesian | 0.164 [0.150, 0.177] | 0.153 | 0.052 | 0.0050 | 0.0050 * | 748 | 190 |
| Indonesian → English | 0.159 [0.141, 0.180] | 0.147 | 0.051 | 0.0050 | 0.0050 * | 720 | 77 |
| Greek → English | 0.155 [0.143, 0.168] | 0.135 | 0.048 | 0.0050 | 0.0050 * | 746 | 79 |
| Korean → English | 0.155 [0.143, 0.172] | 0.144 | 0.048 | 0.0050 | 0.0050 * | 740 | 79 |
| English → Greek | 0.154 [0.146, 0.162] | 0.142 | 0.050 | 0.0050 | 0.0050 * | 734 | 167 |
| English → Korean | 0.151 [0.138, 0.168] | 0.143 | 0.049 | 0.0050 | 0.0050 * | 747 | 110 |
| Hebrew → English | 0.143 [0.128, 0.157] | 0.125 | 0.049 | 0.0050 | 0.0050 * | 781 | 79 |
| Arabic → English | 0.135 [0.120, 0.150] | 0.116 | 0.049 | 0.0050 | 0.0050 * | 762 | 79 |
| Thai → English | 0.131 [0.114, 0.150] | 0.116 | 0.047 | 0.0050 | 0.0050 * | 818 | 79 |
| English → Hebrew | 0.130 [0.118, 0.138] | 0.112 | 0.050 | 0.0050 | 0.0050 * | 794 | 211 |
| English → Thai | 0.130 [0.114, 0.148] | 0.111 | 0.049 | 0.0050 | 0.0050 * | 796 | 199 |
| English → Arabic | 0.128 [0.114, 0.142] | 0.113 | 0.049 | 0.0050 | 0.0050 * | 774 | 194 |
| Vietnamese → English | 0.113 [0.101, 0.123] | 0.100 | 0.050 | 0.0050 | 0.0050 * | 789 | 79 |
| English → Vietnamese | 0.111 [0.101, 0.120] | 0.092 | 0.050 | 0.0050 | 0.0050 * | 787 | 161 |

58/58 pairs significant at q<0.05.


## Form–meaning correlation (Mantel, n=1968 concepts)

| analysis | unit | r | p | r \| orthography | p (partial) | note |
|---|---|---|---|---|---|---|
| form~meaning | Hungarian | +0.0217* | 0.0050 | +0.0196* | 0.0050 |  |
| form~meaning | Finnish | +0.0254* | 0.0050 | +0.0202* | 0.0050 |  |
| form~meaning | Greek | +0.0229* | 0.0050 | +0.0158* | 0.0050 |  |
| form~meaning | Russian | +0.0275* | 0.0050 | +0.0168* | 0.0050 |  |
| form~meaning | German | +0.0298* | 0.0050 | +0.0225* | 0.0050 |  |
| form~meaning | Spanish | +0.0279* | 0.0050 | +0.0200* | 0.0050 |  |
| form~meaning | Italian | +0.0277* | 0.0050 | +0.0158* | 0.0050 |  |
| form~meaning | French | +0.0209* | 0.0050 | +0.0151* | 0.0050 |  |
| form~meaning | Irish | +0.0202* | 0.0050 | +0.0182* | 0.0050 |  |
| form~meaning | Welsh | +0.0222* | 0.0050 | +0.0194* | 0.0050 |  |
| form~meaning | English | +0.0215* | 0.0050 | +0.0202* | 0.0050 |  |
| form~meaning | Chinese | +0.0166* | 0.0050 | +0.0039* | 0.0050 | orth control degenerate |
| form~meaning | Vietnamese | +0.0244* | 0.0050 | +0.0216* | 0.0050 |  |
| form~meaning | Japanese | +0.0430* | 0.0050 | +0.0408* | 0.0050 | orth control degenerate |
| form~meaning | Korean | +0.0256* | 0.0050 | +0.0053* | 0.0050 |  |
| form~meaning | Thai | +0.0220* | 0.0050 | +0.0190* | 0.0050 |  |
| form~meaning | Indonesian | +0.0253* | 0.0050 | +0.0161* | 0.0050 |  |
| form~meaning | Turkish | +0.0250* | 0.0050 | +0.0146* | 0.0050 |  |
| form~meaning | Arabic | +0.0266* | 0.0050 | +0.0149* | 0.0050 |  |
| form~meaning | Hebrew | +0.0180* | 0.0050 | +0.0114* | 0.0050 |  |
| form~meaning | Swahili | +0.0253* | 0.0050 | +0.0191* | 0.0050 |  |
| form~meaning | Hindi | +0.0205* | 0.0050 | +0.0171* | 0.0050 |  |
| form~form | Hungarian~Finnish | +0.0192* | 0.0050 | +0.0166* | 0.0050 |  |
| form~form | Hungarian~Greek | +0.0173* | 0.0050 | +0.0155* | 0.0050 |  |
| form~form | Hungarian~Russian | +0.0223* | 0.0050 | +0.0191* | 0.0050 |  |
| form~form | Hungarian~German | +0.0267* | 0.0050 | +0.0240* | 0.0050 |  |
| form~form | Hungarian~Spanish | +0.0224* | 0.0050 | +0.0194* | 0.0050 |  |
| form~form | Hungarian~Italian | +0.0222* | 0.0050 | +0.0192* | 0.0050 |  |
| form~form | Hungarian~French | +0.0185* | 0.0050 | +0.0163* | 0.0050 |  |
| form~form | Hungarian~Irish | +0.0138* | 0.0050 | +0.0122* | 0.0050 |  |
| form~form | Hungarian~Welsh | +0.0161* | 0.0050 | +0.0144* | 0.0050 |  |
| form~form | Hungarian~English | +0.0183* | 0.0050 | +0.0159* | 0.0050 |  |
| form~form | Hungarian~Chinese | +0.0049* | 0.0050 | +0.0042* | 0.0050 | orth control degenerate |
| form~form | Hungarian~Vietnamese | +0.0104* | 0.0050 | +0.0094* | 0.0050 |  |
| form~form | Hungarian~Japanese | +0.0055* | 0.0050 | +0.0075* | 0.0050 | orth control degenerate |
| form~form | Hungarian~Korean | +0.0099* | 0.0050 | +0.0085* | 0.0050 |  |
| form~form | Hungarian~Thai | +0.0102* | 0.0050 | +0.0085* | 0.0050 |  |
| form~form | Hungarian~Indonesian | +0.0151* | 0.0050 | +0.0135* | 0.0050 |  |
| form~form | Hungarian~Turkish | +0.0197* | 0.0050 | +0.0175* | 0.0050 |  |
| form~form | Hungarian~Arabic | +0.0111* | 0.0050 | +0.0097* | 0.0050 |  |
| form~form | Hungarian~Hebrew | +0.0125* | 0.0050 | +0.0111* | 0.0050 |  |
| form~form | Hungarian~Swahili | +0.0146* | 0.0050 | +0.0131* | 0.0050 |  |
| form~form | Hungarian~Hindi | +0.0117* | 0.0050 | +0.0105* | 0.0050 |  |
| form~form | Finnish~Greek | +0.0161* | 0.0050 | +0.0137* | 0.0050 |  |
| form~form | Finnish~Russian | +0.0212* | 0.0050 | +0.0176* | 0.0050 |  |
| form~form | Finnish~German | +0.0266* | 0.0050 | +0.0228* | 0.0050 |  |
| form~form | Finnish~Spanish | +0.0199* | 0.0050 | +0.0165* | 0.0050 |  |
| form~form | Finnish~Italian | +0.0193* | 0.0050 | +0.0152* | 0.0050 |  |
| form~form | Finnish~French | +0.0188* | 0.0050 | +0.0160* | 0.0050 |  |
| form~form | Finnish~Irish | +0.0113* | 0.0050 | +0.0095* | 0.0050 |  |
| form~form | Finnish~Welsh | +0.0159* | 0.0050 | +0.0132* | 0.0050 |  |
| form~form | Finnish~English | +0.0175* | 0.0050 | +0.0149* | 0.0050 |  |
| form~form | Finnish~Chinese | +0.0054* | 0.0050 | +0.0047* | 0.0050 | orth control degenerate |
| form~form | Finnish~Vietnamese | +0.0100* | 0.0050 | +0.0090* | 0.0050 |  |
| form~form | Finnish~Japanese | +0.0069* | 0.0050 | +0.0069* | 0.0050 | orth control degenerate |
| form~form | Finnish~Korean | +0.0115* | 0.0050 | +0.0093* | 0.0050 |  |
| form~form | Finnish~Thai | +0.0125* | 0.0050 | +0.0109* | 0.0050 |  |
| form~form | Finnish~Indonesian | +0.0140* | 0.0050 | +0.0112* | 0.0050 |  |
| form~form | Finnish~Turkish | +0.0178* | 0.0050 | +0.0147* | 0.0050 |  |
| form~form | Finnish~Arabic | +0.0123* | 0.0050 | +0.0098* | 0.0050 |  |
| form~form | Finnish~Hebrew | +0.0122* | 0.0050 | +0.0103* | 0.0050 |  |
| form~form | Finnish~Swahili | +0.0148* | 0.0050 | +0.0130* | 0.0050 |  |
| form~form | Finnish~Hindi | +0.0122* | 0.0050 | +0.0106* | 0.0050 |  |
| form~form | Greek~Russian | +0.0186* | 0.0050 | +0.0152* | 0.0050 |  |
| form~form | Greek~German | +0.0189* | 0.0050 | +0.0159* | 0.0050 |  |
| form~form | Greek~Spanish | +0.0240* | 0.0050 | +0.0198* | 0.0050 |  |
| form~form | Greek~Italian | +0.0258* | 0.0050 | +0.0212* | 0.0050 |  |
| form~form | Greek~French | +0.0198* | 0.0050 | +0.0172* | 0.0050 |  |
| form~form | Greek~Irish | +0.0090* | 0.0050 | +0.0072* | 0.0050 |  |
| form~form | Greek~Welsh | +0.0139* | 0.0050 | +0.0116* | 0.0050 |  |
| form~form | Greek~English | +0.0140* | 0.0050 | +0.0121* | 0.0050 |  |
| form~form | Greek~Chinese | +0.0055* | 0.0050 | +0.0052* | 0.0050 | orth control degenerate |
| form~form | Greek~Vietnamese | +0.0110* | 0.0050 | +0.0099* | 0.0050 |  |
| form~form | Greek~Japanese | +0.0091* | 0.0050 | +0.0050* | 0.0050 | orth control degenerate |
| form~form | Greek~Korean | +0.0099* | 0.0050 | +0.0076* | 0.0050 |  |
| form~form | Greek~Thai | +0.0094* | 0.0050 | +0.0080* | 0.0050 |  |
| form~form | Greek~Indonesian | +0.0128* | 0.0050 | +0.0103* | 0.0050 |  |
| form~form | Greek~Turkish | +0.0182* | 0.0050 | +0.0154* | 0.0050 |  |
| form~form | Greek~Arabic | +0.0119* | 0.0050 | +0.0097* | 0.0050 |  |
| form~form | Greek~Hebrew | +0.0122* | 0.0050 | +0.0106* | 0.0050 |  |
| form~form | Greek~Swahili | +0.0129* | 0.0050 | +0.0108* | 0.0050 |  |
| form~form | Greek~Hindi | +0.0112* | 0.0050 | +0.0094* | 0.0050 |  |
| form~form | Russian~German | +0.0315* | 0.0050 | +0.0259* | 0.0050 |  |
| form~form | Russian~Spanish | +0.0308* | 0.0050 | +0.0243* | 0.0050 |  |
| form~form | Russian~Italian | +0.0345* | 0.0050 | +0.0271* | 0.0050 |  |
| form~form | Russian~French | +0.0282* | 0.0050 | +0.0240* | 0.0050 |  |
| form~form | Russian~Irish | +0.0137* | 0.0050 | +0.0113* | 0.0050 |  |
| form~form | Russian~Welsh | +0.0179* | 0.0050 | +0.0150* | 0.0050 |  |
| form~form | Russian~English | +0.0207* | 0.0050 | +0.0179* | 0.0050 |  |
| form~form | Russian~Chinese | +0.0062* | 0.0050 | +0.0056* | 0.0050 | orth control degenerate |
| form~form | Russian~Vietnamese | +0.0109* | 0.0050 | +0.0093* | 0.0050 |  |
| form~form | Russian~Japanese | +0.0110* | 0.0050 | +0.0124* | 0.0050 | orth control degenerate |
| form~form | Russian~Korean | +0.0128* | 0.0050 | +0.0097* | 0.0050 |  |
| form~form | Russian~Thai | +0.0109* | 0.0050 | +0.0095* | 0.0050 |  |
| form~form | Russian~Indonesian | +0.0171* | 0.0050 | +0.0132* | 0.0050 |  |
| form~form | Russian~Turkish | +0.0223* | 0.0050 | +0.0177* | 0.0050 |  |
| form~form | Russian~Arabic | +0.0140* | 0.0050 | +0.0102* | 0.0050 |  |
| form~form | Russian~Hebrew | +0.0130* | 0.0050 | +0.0100* | 0.0050 |  |
| form~form | Russian~Swahili | +0.0152* | 0.0050 | +0.0121* | 0.0050 |  |
| form~form | Russian~Hindi | +0.0130* | 0.0050 | +0.0104* | 0.0050 |  |
| form~form | German~Spanish | +0.0337* | 0.0050 | +0.0285* | 0.0050 |  |
| form~form | German~Italian | +0.0355* | 0.0050 | +0.0299* | 0.0050 |  |
| form~form | German~French | +0.0363* | 0.0050 | +0.0322* | 0.0050 |  |
| form~form | German~Irish | +0.0167* | 0.0050 | +0.0144* | 0.0050 |  |
| form~form | German~Welsh | +0.0225* | 0.0050 | +0.0196* | 0.0050 |  |
| form~form | German~English | +0.0604* | 0.0050 | +0.0534* | 0.0050 |  |
| form~form | German~Chinese | +0.0062* | 0.0050 | +0.0055* | 0.0050 | orth control degenerate |
| form~form | German~Vietnamese | +0.0118* | 0.0050 | +0.0108* | 0.0050 |  |
| form~form | German~Japanese | +0.0085* | 0.0050 | +0.0150* | 0.0050 | orth control degenerate |
| form~form | German~Korean | +0.0141* | 0.0050 | +0.0120* | 0.0050 |  |
| form~form | German~Thai | +0.0112* | 0.0050 | +0.0101* | 0.0050 |  |
| form~form | German~Indonesian | +0.0206* | 0.0050 | +0.0176* | 0.0050 |  |
| form~form | German~Turkish | +0.0235* | 0.0050 | +0.0200* | 0.0050 |  |
| form~form | German~Arabic | +0.0132* | 0.0050 | +0.0105* | 0.0050 |  |
| form~form | German~Hebrew | +0.0151* | 0.0050 | +0.0125* | 0.0050 |  |
| form~form | German~Swahili | +0.0162* | 0.0050 | +0.0142* | 0.0050 |  |
| form~form | German~Hindi | +0.0141* | 0.0050 | +0.0122* | 0.0050 |  |
| form~form | Spanish~Italian | +0.1708* | 0.0050 | +0.1501* | 0.0050 |  |
| form~form | Spanish~French | +0.1027* | 0.0050 | +0.0905* | 0.0050 |  |
| form~form | Spanish~Irish | +0.0187* | 0.0050 | +0.0156* | 0.0050 |  |
| form~form | Spanish~Welsh | +0.0253* | 0.0050 | +0.0215* | 0.0050 |  |
| form~form | Spanish~English | +0.0482* | 0.0050 | +0.0419* | 0.0050 |  |
| form~form | Spanish~Chinese | +0.0070* | 0.0050 | +0.0060* | 0.0050 | orth control degenerate |
| form~form | Spanish~Vietnamese | +0.0106* | 0.0050 | +0.0094* | 0.0050 |  |
| form~form | Spanish~Japanese | +0.0088* | 0.0050 | +0.0069* | 0.0050 | orth control degenerate |
| form~form | Spanish~Korean | +0.0126* | 0.0050 | +0.0102* | 0.0050 |  |
| form~form | Spanish~Thai | +0.0105* | 0.0050 | +0.0088* | 0.0050 |  |
| form~form | Spanish~Indonesian | +0.0200* | 0.0050 | +0.0162* | 0.0050 |  |
| form~form | Spanish~Turkish | +0.0206* | 0.0050 | +0.0166* | 0.0050 |  |
| form~form | Spanish~Arabic | +0.0155* | 0.0050 | +0.0122* | 0.0050 |  |
| form~form | Spanish~Hebrew | +0.0134* | 0.0050 | +0.0109* | 0.0050 |  |
| form~form | Spanish~Swahili | +0.0168* | 0.0050 | +0.0137* | 0.0050 |  |
| form~form | Spanish~Hindi | +0.0138* | 0.0050 | +0.0113* | 0.0050 |  |
| form~form | Italian~French | +0.1186* | 0.0050 | +0.1039* | 0.0050 |  |
| form~form | Italian~Irish | +0.0175* | 0.0050 | +0.0141* | 0.0050 |  |
| form~form | Italian~Welsh | +0.0268* | 0.0050 | +0.0221* | 0.0050 |  |
| form~form | Italian~English | +0.0471* | 0.0050 | +0.0402* | 0.0050 |  |
| form~form | Italian~Chinese | +0.0045* | 0.0050 | +0.0038* | 0.0050 | orth control degenerate |
| form~form | Italian~Vietnamese | +0.0103* | 0.0050 | +0.0094* | 0.0050 |  |
| form~form | Italian~Japanese | +0.0092* | 0.0050 | +0.0075* | 0.0050 | orth control degenerate |
| form~form | Italian~Korean | +0.0107* | 0.0050 | +0.0076* | 0.0050 |  |
| form~form | Italian~Thai | +0.0098* | 0.0050 | +0.0084* | 0.0050 |  |
| form~form | Italian~Indonesian | +0.0175* | 0.0050 | +0.0127* | 0.0050 |  |
| form~form | Italian~Turkish | +0.0213* | 0.0050 | +0.0158* | 0.0050 |  |
| form~form | Italian~Arabic | +0.0151* | 0.0050 | +0.0103* | 0.0050 |  |
| form~form | Italian~Hebrew | +0.0109* | 0.0050 | +0.0081* | 0.0050 |  |
| form~form | Italian~Swahili | +0.0153* | 0.0050 | +0.0111* | 0.0050 |  |
| form~form | Italian~Hindi | +0.0135* | 0.0050 | +0.0107* | 0.0050 |  |
| form~form | French~Irish | +0.0189* | 0.0050 | +0.0160* | 0.0050 |  |
| form~form | French~Welsh | +0.0250* | 0.0050 | +0.0213* | 0.0050 |  |
| form~form | French~English | +0.0585* | 0.0050 | +0.0524* | 0.0050 |  |
| form~form | French~Chinese | +0.0046* | 0.0050 | +0.0041* | 0.0050 | orth control degenerate |
| form~form | French~Vietnamese | +0.0106* | 0.0050 | +0.0098* | 0.0050 |  |
| form~form | French~Japanese | +0.0083* | 0.0050 | +0.0092* | 0.0050 | orth control degenerate |
| form~form | French~Korean | +0.0081* | 0.0050 | +0.0064* | 0.0050 |  |
| form~form | French~Thai | +0.0098* | 0.0050 | +0.0089* | 0.0050 |  |
| form~form | French~Indonesian | +0.0167* | 0.0050 | +0.0142* | 0.0050 |  |
| form~form | French~Turkish | +0.0209* | 0.0050 | +0.0180* | 0.0050 |  |
| form~form | French~Arabic | +0.0117* | 0.0050 | +0.0093* | 0.0050 |  |
| form~form | French~Hebrew | +0.0107* | 0.0050 | +0.0086* | 0.0050 |  |
| form~form | French~Swahili | +0.0134* | 0.0050 | +0.0112* | 0.0050 |  |
| form~form | French~Hindi | +0.0123* | 0.0050 | +0.0106* | 0.0050 |  |
| form~form | Irish~Welsh | +0.0281* | 0.0050 | +0.0247* | 0.0050 |  |
| form~form | Irish~English | +0.0276* | 0.0050 | +0.0241* | 0.0050 |  |
| form~form | Irish~Chinese | +0.0055* | 0.0050 | +0.0048* | 0.0050 | orth control degenerate |
| form~form | Irish~Vietnamese | +0.0084* | 0.0050 | +0.0080* | 0.0050 |  |
| form~form | Irish~Japanese | +0.0039* | 0.0050 | +0.0020 | 0.1692 | orth control degenerate |
| form~form | Irish~Korean | +0.0076* | 0.0050 | +0.0065* | 0.0050 |  |
| form~form | Irish~Thai | +0.0081* | 0.0050 | +0.0075* | 0.0050 |  |
| form~form | Irish~Indonesian | +0.0098* | 0.0050 | +0.0086* | 0.0050 |  |
| form~form | Irish~Turkish | +0.0142* | 0.0050 | +0.0127* | 0.0050 |  |
| form~form | Irish~Arabic | +0.0064* | 0.0050 | +0.0055* | 0.0050 |  |
| form~form | Irish~Hebrew | +0.0072* | 0.0050 | +0.0062* | 0.0050 |  |
| form~form | Irish~Swahili | +0.0129* | 0.0050 | +0.0114* | 0.0050 |  |
| form~form | Irish~Hindi | +0.0104* | 0.0050 | +0.0093* | 0.0050 |  |
| form~form | Welsh~English | +0.0362* | 0.0050 | +0.0310* | 0.0050 |  |
| form~form | Welsh~Chinese | +0.0060* | 0.0050 | +0.0051* | 0.0050 | orth control degenerate |
| form~form | Welsh~Vietnamese | +0.0109* | 0.0050 | +0.0098* | 0.0050 |  |
| form~form | Welsh~Japanese | +0.0076* | 0.0050 | +0.0049* | 0.0050 | orth control degenerate |
| form~form | Welsh~Korean | +0.0113* | 0.0050 | +0.0095* | 0.0050 |  |
| form~form | Welsh~Thai | +0.0109* | 0.0050 | +0.0097* | 0.0050 |  |
| form~form | Welsh~Indonesian | +0.0127* | 0.0050 | +0.0110* | 0.0050 |  |
| form~form | Welsh~Turkish | +0.0166* | 0.0050 | +0.0144* | 0.0050 |  |
| form~form | Welsh~Arabic | +0.0104* | 0.0050 | +0.0091* | 0.0050 |  |
| form~form | Welsh~Hebrew | +0.0098* | 0.0050 | +0.0087* | 0.0050 |  |
| form~form | Welsh~Swahili | +0.0137* | 0.0050 | +0.0119* | 0.0050 |  |
| form~form | Welsh~Hindi | +0.0140* | 0.0050 | +0.0123* | 0.0050 |  |
| form~form | English~Chinese | +0.0039* | 0.0050 | +0.0033* | 0.0050 | orth control degenerate |
| form~form | English~Vietnamese | +0.0086* | 0.0050 | +0.0079* | 0.0050 |  |
| form~form | English~Japanese | +0.0089* | 0.0050 | +0.0042* | 0.0050 | orth control degenerate |
| form~form | English~Korean | +0.0135* | 0.0050 | +0.0120* | 0.0050 |  |
| form~form | English~Thai | +0.0098* | 0.0050 | +0.0085* | 0.0050 |  |
| form~form | English~Indonesian | +0.0160* | 0.0050 | +0.0142* | 0.0050 |  |
| form~form | English~Turkish | +0.0156* | 0.0050 | +0.0140* | 0.0050 |  |
| form~form | English~Arabic | +0.0109* | 0.0050 | +0.0100* | 0.0050 |  |
| form~form | English~Hebrew | +0.0100* | 0.0050 | +0.0089* | 0.0050 |  |
| form~form | English~Swahili | +0.0180* | 0.0050 | +0.0164* | 0.0050 |  |
| form~form | English~Hindi | +0.0245* | 0.0050 | +0.0224* | 0.0050 |  |
| form~form | Chinese~Vietnamese | +0.0162* | 0.0050 | +0.0090* | 0.0050 | orth control degenerate |
| form~form | Chinese~Japanese | +0.0079* | 0.0050 | +0.0051* | 0.0050 | orth control degenerate |
| form~form | Chinese~Korean | +0.0207* | 0.0050 | +0.0142* | 0.0050 | orth control degenerate |
| form~form | Chinese~Thai | +0.0080* | 0.0050 | +0.0028* | 0.0050 | orth control degenerate |
| form~form | Chinese~Indonesian | +0.0078* | 0.0050 | +0.0040* | 0.0050 | orth control degenerate |
| form~form | Chinese~Turkish | +0.0064* | 0.0050 | +0.0026* | 0.0050 | orth control degenerate |
| form~form | Chinese~Arabic | +0.0043* | 0.0050 | +0.0012 | 0.1343 | orth control degenerate |
| form~form | Chinese~Hebrew | +0.0057* | 0.0050 | +0.0028* | 0.0050 | orth control degenerate |
| form~form | Chinese~Swahili | +0.0061* | 0.0050 | +0.0025* | 0.0100 | orth control degenerate |
| form~form | Chinese~Hindi | +0.0053* | 0.0050 | +0.0024* | 0.0100 | orth control degenerate |
| form~form | Vietnamese~Japanese | +0.0053* | 0.0050 | -0.0032* | 0.0299 | orth control degenerate |
| form~form | Vietnamese~Korean | +0.0153* | 0.0050 | +0.0137* | 0.0050 |  |
| form~form | Vietnamese~Thai | +0.0159* | 0.0050 | +0.0134* | 0.0050 |  |
| form~form | Vietnamese~Indonesian | +0.0119* | 0.0050 | +0.0110* | 0.0050 |  |
| form~form | Vietnamese~Turkish | +0.0120* | 0.0050 | +0.0110* | 0.0050 |  |
| form~form | Vietnamese~Arabic | +0.0099* | 0.0050 | +0.0093* | 0.0050 |  |
| form~form | Vietnamese~Hebrew | +0.0095* | 0.0050 | +0.0087* | 0.0050 |  |
| form~form | Vietnamese~Swahili | +0.0104* | 0.0050 | +0.0095* | 0.0050 |  |
| form~form | Vietnamese~Hindi | +0.0083* | 0.0050 | +0.0074* | 0.0050 |  |
| form~form | Japanese~Korean | +0.0182* | 0.0050 | +0.0172* | 0.0050 | orth control degenerate |
| form~form | Japanese~Thai | +0.0100* | 0.0050 | +0.0096* | 0.0050 | orth control degenerate |
| form~form | Japanese~Indonesian | +0.0098* | 0.0050 | +0.0092* | 0.0050 | orth control degenerate |
| form~form | Japanese~Turkish | +0.0092* | 0.0050 | +0.0085* | 0.0050 | orth control degenerate |
| form~form | Japanese~Arabic | +0.0045* | 0.0050 | +0.0040* | 0.0050 | orth control degenerate |
| form~form | Japanese~Hebrew | +0.0038* | 0.0050 | +0.0034* | 0.0050 | orth control degenerate |
| form~form | Japanese~Swahili | +0.0086* | 0.0050 | +0.0081* | 0.0050 | orth control degenerate |
| form~form | Japanese~Hindi | +0.0069* | 0.0050 | +0.0065* | 0.0050 | orth control degenerate |
| form~form | Korean~Thai | +0.0106* | 0.0050 | +0.0084* | 0.0050 |  |
| form~form | Korean~Indonesian | +0.0120* | 0.0050 | +0.0068* | 0.0050 |  |
| form~form | Korean~Turkish | +0.0128* | 0.0050 | +0.0072* | 0.0050 |  |
| form~form | Korean~Arabic | +0.0102* | 0.0050 | +0.0048* | 0.0050 |  |
| form~form | Korean~Hebrew | +0.0090* | 0.0050 | +0.0056* | 0.0050 |  |
| form~form | Korean~Swahili | +0.0114* | 0.0050 | +0.0074* | 0.0050 |  |
| form~form | Korean~Hindi | +0.0113* | 0.0050 | +0.0081* | 0.0050 |  |
| form~form | Thai~Indonesian | +0.0121* | 0.0050 | +0.0101* | 0.0050 |  |
| form~form | Thai~Turkish | +0.0129* | 0.0050 | +0.0111* | 0.0050 |  |
| form~form | Thai~Arabic | +0.0093* | 0.0050 | +0.0081* | 0.0050 |  |
| form~form | Thai~Hebrew | +0.0081* | 0.0050 | +0.0071* | 0.0050 |  |
| form~form | Thai~Swahili | +0.0129* | 0.0050 | +0.0108* | 0.0050 |  |
| form~form | Thai~Hindi | +0.0114* | 0.0050 | +0.0096* | 0.0050 |  |
| form~form | Indonesian~Turkish | +0.0198* | 0.0050 | +0.0151* | 0.0050 |  |
| form~form | Indonesian~Arabic | +0.0115* | 0.0050 | +0.0081* | 0.0050 |  |
| form~form | Indonesian~Hebrew | +0.0104* | 0.0050 | +0.0080* | 0.0050 |  |
| form~form | Indonesian~Swahili | +0.0183* | 0.0050 | +0.0145* | 0.0050 |  |
| form~form | Indonesian~Hindi | +0.0115* | 0.0050 | +0.0092* | 0.0050 |  |
| form~form | Turkish~Arabic | +0.0181* | 0.0050 | +0.0135* | 0.0050 |  |
| form~form | Turkish~Hebrew | +0.0117* | 0.0050 | +0.0090* | 0.0050 |  |
| form~form | Turkish~Swahili | +0.0167* | 0.0050 | +0.0133* | 0.0050 |  |
| form~form | Turkish~Hindi | +0.0137* | 0.0050 | +0.0115* | 0.0050 |  |
| form~form | Arabic~Hebrew | +0.0173* | 0.0050 | +0.0136* | 0.0050 |  |
| form~form | Arabic~Swahili | +0.0153* | 0.0050 | +0.0119* | 0.0050 |  |
| form~form | Arabic~Hindi | +0.0117* | 0.0050 | +0.0100* | 0.0050 |  |
| form~form | Hebrew~Swahili | +0.0115* | 0.0050 | +0.0076* | 0.0050 |  |
| form~form | Hebrew~Hindi | +0.0081* | 0.0050 | +0.0064* | 0.0050 |  |
| form~form | Swahili~Hindi | +0.0144* | 0.0050 | +0.0120* | 0.0050 |  |

A within-language *form~meaning* r that stays significant in the *| orthography* column is sound–meaning systematicity not attributable to spelling / cognate overlap.


## Phoneme–meaning association (18 poles, null_iters=200)

Significant phoneme×pole cells (q<0.10): **0** total.

