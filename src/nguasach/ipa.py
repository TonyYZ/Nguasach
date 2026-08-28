"""Stage ``ipa``: each translation -> space-separated IPA phones.

Output per language: ``data/interim/<Lang>V.txt`` in the PSV input format the
vendored ``generate.py`` expects -- one line ``<label>  <p h o n e s>`` (two
spaces), ``<label>`` taken from ``labels.json`` so it stays concept_id-aligned.

Backends (mirrors the original ``pronounce.py`` / ``processChn.py``):

* eSpeak NG ``--tie --ipa`` for the 19 alphabetic languages (bundled binary
  under ``eSpeak NG/``, invoked with ``--path`` so it finds its data).
* Chinese: the bespoke pinyin->IPA map in :mod:`nguasach._zh`.
* Thai, Japanese: eSpeak 1.50 (bundled) has **no voice** for these. The
  original used a web scrape (Thai) and a dictionary (Japanese). Choose a
  backend in the config (``thai_backend`` / ``ja_backend``); until then those
  languages are reported ``pending`` and skipped, not silently wrong.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import REPO_ROOT, Config
from . import data as _data

# ISO/eSpeak voice per language column.
LANG_VOICE: dict[str, str] = {
    "Hungarian": "hu", "Finnish": "fi", "Greek": "el", "Russian": "ru",
    "German": "de", "Spanish": "es", "Italian": "it", "French": "fr",
    "Irish": "ga", "Welsh": "cy", "English": "en", "Vietnamese": "vi",
    "Korean": "ko", "Indonesian": "id", "Turkish": "tr", "Arabic": "ar",
    "Hebrew": "he", "Swahili": "sw", "Hindi": "hi",
}
CUSTOM = {"Chinese", "Thai", "Japanese"}

# eSpeak-ng voice quirks: raw output token -> IPA replacement, applied per
# language before purify(). eSpeak's Irish voice emits a literal uppercase "A"
# for a low back vowel.
LANG_FIXUPS: dict[str, dict[str, str]] = {"Irish": {"A": "ɑ"}}

# Diacritics that must re-attach to the preceding segment after we space out chars.
_COMBINING = ["̪", "ʰ", "ʲ", "̃", "ᵝ"]  # ̪ ʰ ʲ ̃ ᵝ

# Marks eSpeak emits that the articulatory-feature-bigram model does not
# represent: stress, length, syllable/prosody breaks, tone digits + tone
# letters, the non-syllabic / breve diacritics, hyphen and stray '?'. The
# original <Lang>V.txt files contain none of these.
_SUPRA = dict.fromkeys(
    map(ord, "ˈˌːˑ‿.|-?" + "12345" + "˥˦˧˨˩" + "̯̆"), None
)


def purify(s: str) -> str:
    """Port of processAll.purify (+ suprasegmental stripping to match the
    original <Lang>V.txt format): drop stress/length/tone, space every char,
    then glue tie bars and combining diacritics back."""
    s = s.translate(_SUPRA)
    s = " ".join(ch for ch in s.replace(" ", ""))
    s = s.replace(" ͡ ", "͡")          # tie bar joins its neighbours
    for d in _COMBINING:
        s = s.replace(" " + d, d)
    return s.strip()


# --------------------------------------------------------------------- espeak
def _find_espeak(cfg: Config) -> tuple[Path, Path]:
    if cfg.espeak_bin:
        exe = Path(cfg.espeak_bin)
    else:
        exe = REPO_ROOT / "eSpeak NG" / "espeak-ng.exe"
    if not exe.exists():
        raise FileNotFoundError(
            f"espeak-ng not found at {exe}. Install it or set paths/espeak_bin. "
            "The bundled build is under 'eSpeak NG/'."
        )
    data_dir = Path(cfg.espeak_data) if cfg.espeak_data else exe.parent
    return exe, data_dir


def espeak_version(exe: Path, data_dir: Path) -> str:
    out = subprocess.run(
        [str(exe), f"--path={data_dir}", "--version"],
        capture_output=True, text=True, encoding="utf-8",
    )
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "unknown"


def espeak_ipa(words: list[str], voice: str, exe: Path, data_dir: Path) -> list[str]:
    """One eSpeak call for a whole column: newline-separated in, one IPA line out."""
    proc = subprocess.run(
        [str(exe), f"--path={data_dir}", "-v", voice, "-q", "--tie", "--ipa"],
        input="\n".join(words), capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"espeak-ng ({voice}) failed: {proc.stderr.strip()}")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip() != ""]
    if len(lines) != len(words):
        raise RuntimeError(
            f"espeak-ng ({voice}) returned {len(lines)} lines for {len(words)} words"
        )
    return lines


# ------------------------------------------------------------------ stage main
def _write_v(path: Path, labels: list[str], ipas: list[str]) -> int:
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for lab, ipa in zip(labels, ipas):
            ipa = ipa.strip()
            if not ipa:
                continue
            fh.write(f"{lab}  {ipa}\n")
            n += 1
    return n


def run(cfg: Config) -> dict:
    interim = cfg.paths.resolve("interim")
    interim.mkdir(parents=True, exist_ok=True)
    labels = json.loads((interim / "labels.json").read_text(encoding="utf-8"))
    df = _data.load_raw(cfg)

    exe, data_dir = _find_espeak(cfg)
    version = espeak_version(exe, data_dir)

    written, pending, skipped = {}, {}, []
    for lang in cfg.languages:
        out = interim / f"{lang}V.txt"
        words = df[lang].tolist()
        labs = labels[lang]

        if lang in LANG_VOICE:
            raw = espeak_ipa(words, LANG_VOICE[lang], exe, data_dir)
            fix = LANG_FIXUPS.get(lang)
            if fix:
                raw = ["".join(fix.get(ch, ch) for ch in line) for line in raw]
            ipas = [purify(x) for x in raw]
        elif lang == "Chinese":
            from ._zh import chinese_ipa
            ipas = [chinese_ipa(w) for w in words]
        elif lang in ("Thai", "Japanese"):
            backend = cfg.thai_backend if lang == "Thai" else cfg.ja_backend
            frozen = REPO_ROOT / "data" / "raw" / "frozen_ipa" / f"{lang}V.txt"
            if backend == "frozen" and frozen.exists():
                out.write_text(frozen.read_text(encoding="utf-8"), encoding="utf-8")
                written[lang] = sum(1 for _ in frozen.open(encoding="utf-8"))
                continue
            pending[lang] = backend
            continue
        else:
            skipped.append(lang)
            continue

        written[lang] = _write_v(out, labs, ipas)

    report = {
        "stage": "ipa",
        "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "espeak_version": version,
        "espeak_bin": str(exe),
        "written": written,
        "pending_backend": pending,
        "skipped": skipped,
    }
    (interim / "ipa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not pending and not skipped:
        (interim / "ipa.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return report
