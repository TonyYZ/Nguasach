# LLM sense-disambiguation pass — running notes

## Confirmed data bugs (apply as overrides)
- concept 7 `wait`: Spanish/Italian swapped -> Spanish=esperar, Italian=aspettare
- concept 8 `stay`: Spanish/Italian swapped -> Spanish=quedarse, Italian=stare

## Worker 1 (batches 00-03, concepts 0-599): 360 flags
- Hindi: worst column, many bare transliterations of the English word
- Hungarian: systematic action-verb -> action-noun; homograph sense errors (döntetlen, gróf, nyárs, ...)
- Welsh: pervasively broken (mutated preterite / archaic) -- treat whole column low-trust
- Greek/Korean/Spanish/Turkish/Arabic: verb periphrasis (να.../para.../için.../لـ...) -- systemic formatting
- polyseme sense-collapse: like(resemble), rear(raise), fine(thin), cool, odd, even, fair
