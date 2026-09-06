"""Build data/interim/concept_additions_2.xlsx — everyday learner concepts
(IDS gaps), with attested IDS + NorthEuraLex forms pre-filled per cell.

    python notebooks/build_additions2.py

NOTE: after the first build the sheet is hand-maintained (the user fills the
verified English/Chinese/French/Irish columns and prunes rows). Re-running this
script overwrites those edits — regenerate only if starting the sheet over.
POS-only duplicates of concepts already frozen in nguasach.xlsx (hunger/hungry,
thirst/thirsty, anger/angry, shame/ashamed, envy/jealous, silence/silent) were
removed from SPEC on the user's instruction. 'cooked' was kept: Concepticon 269
COOKED explicitly contrasts with RAW and is a distinct root in ZH/FI/etc.
old man / old woman / young man / young woman were also dropped (too compositional
given old/young/man/woman are already frozen concepts).

The delivered sheet has hand-fixed Concepticon ids the fuzzy lookup got wrong:
thumb -> 1781 (not 1303 FINGER), master -> 383 (not 1545 HOST),
tell -> 1711 (not 1458 SAY), pregnant -> 1123 (not 3827).
"""
from __future__ import annotations

import csv
from pathlib import Path

import openpyxl
from pyconcepticon import Concepticon

from nguasach.config import ALL_LANGUAGES, Config
from nguasach import data as _data
from nguasach.lexibank_qc import _attestations

df = _data.load_raw(Config.load("configs/paper_exploratory.yaml"))
E = [df.at[i, "English"].strip().lower() for i in range(len(df))]


def anchor_id(word: str):
    return E.index(word) if word in E else None


def pick_anchor(*candidates):
    for c in candidates:
        if c in E:
            return c
    return candidates[-1]


# (English gloss, thematic anchor concept in the current corpus)
SPEC = [
    ("yes", "not"), ("no", "not"), ("enough", "not"),
    ("soon", pick_anchor("again", "not")), ("beside", pick_anchor("with", "not")),
    ("breakfast", "soup"), ("lunch", "soup"), ("dinner", "soup"), ("meal", "soup"),
    ("snack", "soup"), ("dish", pick_anchor("bowl", "plate")),
    ("flour", pick_anchor("sugar", "salt")), ("raw", "cook"), ("cooked", "cook"),
    ("chew", "bite"),
    ("morning", "night"), ("afternoon", "night"), ("evening", "night"),
    ("midday", "night"), ("age", pick_anchor("year", "time")),
    ("parents", "father"), ("relatives", "sister"), ("uncle", "sister"),
    ("aunt", "sister"),
    ("wedding", pick_anchor("marry", "love")), ("divorce", pick_anchor("marry", "love")),
    ("widow", pick_anchor("wife", "husband")), ("widower", "husband"),
    ("twins", pick_anchor("child", "baby")),
    ("guest", pick_anchor("friend", "person")), ("host", pick_anchor("friend", "person")),
    ("stranger", pick_anchor("friend", "person")), ("neighbour", pick_anchor("friend", "person")),
    ("people", pick_anchor("person", "man")), ("village", pick_anchor("town", "city")),
    ("servant", pick_anchor("job", "work")), ("master", pick_anchor("job", "work")),
    ("throat", "neck"), ("chest", pick_anchor("breast", "neck")), ("elbow", "arm"),
    ("jaw", pick_anchor("chin", "face")), ("forehead", "face"), ("eyebrow", "eye"),
    ("eyelash", "eye"), ("eyelid", "eye"), ("heel", "foot"), ("thumb", "finger"),
    ("rib", "bone"), ("nostril", "nose"), ("vein", "blood"),
    ("fever", pick_anchor("disease", "pain")), ("itch", "pain"), ("sneeze", "cough"),
    ("blink", pick_anchor("yawn", "cough")), ("bathe", "wash"),
    ("pregnant", pick_anchor("baby", "child")), ("wound", "pain"), ("bruise", "pain"),
    ("scar", "pain"),
    ("belt", "hat"), ("button", "hat"), ("collar", "hat"), ("glove", "hat"),
    ("boot", "hat"), ("bracelet", "ring"), ("necklace", "ring"), ("earring", "ring"),
    ("towel", pick_anchor("soap", "hat")), ("veil", "hat"), ("ornament", "ring"),
    ("pity", "love"),
    ("grief", "cry"), ("proud", pick_anchor("happy", "love")), ("greedy", "want"),
    ("forgive", "love"), ("blame", "hate"), ("praise", "love"),
    ("quarrel", pick_anchor("fight", "argue")), ("help", "give"),
    ("obey", pick_anchor("follow", "listen")), ("promise", "say"),
    ("threaten", pick_anchor("fight", "hit")), ("refuse", pick_anchor("deny", "say")),
    ("boast", "say"), ("dare", "try"), ("embrace", "kiss"),
    ("climb", "jump"), ("ride", "walk"), ("lift", "carry"), ("flee", "run"),
    ("kneel", "sit"), ("dive", "swim"), ("path", "road"), ("approach", "come"),
    ("coin", "buy"), ("bill", "buy"), ("debt", "buy"), ("owe", "buy"), ("own", "have"),
    ("keep", "have"), ("trade", "sell"), ("wages", pick_anchor("job", "work")),
    ("tax", "buy"), ("merchant", "sell"), ("beggar", pick_anchor("poor", "buy")),
    ("mind", "think"), ("doubt", "think"), ("secret", "know"),
    ("wise", pick_anchor("clever", "know")), ("seem", pick_anchor("look for", "think")),
    ("listen", "hear"),
    ("loud", pick_anchor("sound", "voice")), ("tell", "say"), ("speak", "say"),
    ("mumble", "say"), ("stutter", "say"),
    ("eleven", "ten"), ("twelve", "ten"), ("fifteen", "ten"), ("twenty", "ten"),
]

api = Concepticon("data/raw/concepticon-data")
terms = [s[0] for s in SPEC]
cc = {}
for t, hits in zip(terms, api.lookup(terms, language="en", full_search=False)):
    best = None
    for _t, cid, gloss, sim in hits:
        s = int(getattr(sim, "value", sim))
        if best is None or s < best[2]:
            best = (cid, gloss, s)
    cc[t] = best[:2] if best else ("", "")

want = {v[0] for v in cc.values() if v[0]}

P = "data/raw/ids-cldf/"
idl = {r["ID"]: r["Name"] for r in csv.DictReader(open(P + "languages.csv", encoding="utf-8"))}
idp = {r["ID"]: r.get("Concepticon_ID", "") for r in csv.DictReader(open(P + "parameters.csv", encoding="utf-8"))}
IDS_TO_OURS = {
    "Hungarian": "Hungarian", "Finnish": "Finnish", "Modern Greek": "Greek",
    "Russian": "Russian", "German": "German", "Spanish": "Spanish",
    "Italian": "Italian", "French": "French", "Irish": "Irish", "Welsh": "Welsh",
    "English": "English", "Vietnamese": "Vietnamese", "Central Thai": "Thai",
}
# IDS "Value" is standard modern orthography only for these. Finnish/Greek/
# Russian/Thai/Vietnamese are phonemic transcription (pölü, 'ɣi, zemlja,
# lo:k.3, d̄ất); IDS Hungarian is pre-1922 spelling (cz/y for cs/j) -- all
# carried in real modern orthography by NorthEuraLex instead.
IDS_ORTHO = {"German", "Spanish", "Italian", "French", "English", "Welsh"}
_IDS_FIX = str.maketrans({"ɲ": "n"})   # stray IPA in a few ES/IT cells
ids_forms = {}
for r in csv.DictReader(open(P + "forms.csv", encoding="utf-8")):
    ln = IDS_TO_OURS.get(idl.get(r["Language_ID"], ""))
    pc = idp.get(r["Parameter_ID"])
    if ln in IDS_ORTHO and pc in want and r.get("Form"):
        f = r["Form"].translate(_IDS_FIX).strip()
        if f and f not in ids_forms.get((pc, ln), []):
            ids_forms.setdefault((pc, ln), []).append(f)

nel = _attestations(Path("data/raw/northeuralex-cldf"))

VERIFIED = {"English", "Chinese", "French", "Irish"}
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "additions"
ws.append(list(ALL_LANGUAGES) + ["_concepticon_id", "_concepticon_gloss",
                                 "_insert_after", "_attested_note"])

per_lang = {L: 0 for L in ALL_LANGUAGES if L not in VERIFIED}
for eng, anchor in SPEC:
    ci, cg = cc[eng]
    row = {L: "" for L in ALL_LANGUAGES}
    row["English"] = eng
    notes = []
    for L in ALL_LANGUAGES:
        if L in VERIFIED:
            continue
        idf = ids_forms.get((ci, L), [])
        nlf = sorted(nel.get((ci, L), [])) if ci else []
        row[L] = idf[0] if idf else (nlf[0] if nlf else "")
        if row[L]:
            per_lang[L] += 1
        if idf and nlf:
            import unicodedata as _u
            nf = lambda s: _u.normalize("NFC", s.lower().strip())
            a = {nf(x) for x in idf}
            b = {nf(x) for x in nlf}
            if not (a & b or any(x in y or y in x for x in a for y in b)):
                notes.append(f"{L}: IDS={idf[0]} NEL={nlf[0]}")
    aid = anchor_id(anchor)
    ws.append([row[L] for L in ALL_LANGUAGES]
              + [ci, cg, f"{anchor} (#{aid})", "; ".join(notes)])

out = Path("data/interim/concept_additions_2.xlsx")
wb.save(out)
print(f"wrote {out}: {len(SPEC)} concepts")
print("attested cells pre-filled per language:")
for L, n in sorted(per_lang.items(), key=lambda x: -x[1]):
    print(f"  {L:12} {n}/{len(SPEC)}")
