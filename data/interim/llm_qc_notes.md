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

## Broken/typo English headwords (fix in nguasach.xlsx English column)
- concept 1258  "merry-go-around"  -> "merry-go-round"
- concept 687   "thrush bird"      -> "song thrush"  (anchor was degenerate)
- concept 688   "swallow bird"     -> "barn swallow" (Fr anchor = "avaler un oiseau")

## Anchor conflicts (user: trust Chinese, French may be unreliable)
- 979 pupil  : Fr=élève(student) vs Ch/Ir=eye-pupil  -> keep eye-pupil sense
- 998 palm   : Fr=palmier(tree)  vs Ch/Ir=hand-palm   -> keep hand-palm sense
- 1063 earth : Ch=泥土 soil ; several langs gave planet -> keep soil sense

## Worker 07-08 (concepts ~1050-1349): 533 flags
- polyseme collapse: mine/may/march/saw/study/net/mortar/ruler/table/earth/paint
- Hindi transliteration leakage ~30 cells; Hungarian+Swahili English-left-as-is
- Welsh pervasively broken (confirmed 3rd time)

## MAJOR: Finnish column off-by-one shift (pre-existing, in frozen original)
Range ~concepts 1328-1435 (electronics / instruments / clothing / writing tools —
the THINGS-database noun expansion). Finnish cell = the word for concept_id+1.
Not a clean block; self-corrects around 1339-1346, 1367, 1405-1406, 1425-1432.
Needs a Finnish re-translation for that range, not overrides.
German column also has scattered wrong-sense in the same region (ruler->Herrscher).

## Worker 09-10 (concepts ~1350-1649): 481 flags
- Finnish shift (above)
- row-wide polyseme collapse: rock(music), order, play(drama), organ(pipe),
  beam(structural), instrument(musical), fan(electric), view(opinion)
- Hungarian/Hindi/Swahili: English-left-as-is continues
- anchor note: 1649 mechanic Ch=機械 "machinery" not the tradesperson
