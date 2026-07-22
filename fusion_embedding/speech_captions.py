"""Caption templating for the speech/music corpus delta (research memo
docs/research_speech_gap_solutions.md, section 6.2).

Training rotates templates deterministically by example index so both bare-label
and templated prompts work at eval time (GLAP's finding: bare "{label}" is the
eval prompt for speech; sound/music use domain prompts). Kept import-light so the
rotation logic is unit-testable without torch/Modal.
"""

from __future__ import annotations

import re

# Rotation order matters only in that it is stable: index % len(templates).
WORD_TEMPLATES = (
    "{word}",
    'Someone says "{word}".',
    "A voice says {word}.",
)

SENTENCE_TEMPLATES = (
    "{transcript}",
    "A person reads aloud: {transcript}",
    "Someone says: {transcript}",
)

MUSIC_TEMPLATES = (
    "{genre} music.",
    "A {genre} track.",
    "The music is in the style of {genre}.",
)


def normalize_transcript(t: str) -> str:
    """LibriSpeech transcripts are ALL-CAPS with no punctuation; emit natural
    sentence case so the caption region matches the base's text manifold
    (the v0.4 lesson: caption style is a region, don't create an artificial one)."""
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return t
    if t.upper() == t:                      # all-caps source (LibriSpeech)
        t = t.lower()
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?\"'":
        t += "."
    return t


def normalize_word(w: str) -> str:
    return re.sub(r"\s+", " ", w).strip().lower()


def normalize_genre(g: str) -> str:
    g = re.sub(r"\s+", " ", g).strip()
    return g if g.isupper() else g.lower()   # keep acronym genres (IDM, EDM) as-is


def caption_for_word(word: str, idx: int) -> str:
    w = normalize_word(word)
    return WORD_TEMPLATES[idx % len(WORD_TEMPLATES)].format(word=w)


def caption_for_transcript(transcript: str, idx: int) -> str:
    tr = normalize_transcript(transcript)
    tpl = SENTENCE_TEMPLATES[idx % len(SENTENCE_TEMPLATES)]
    if tpl != "{transcript}":
        # embedded sentence keeps its own final punctuation; that reads fine
        tr2 = tr[0].lower() + tr[1:] if tpl.startswith("Someone says") else tr
        return tpl.format(transcript=tr2 if tpl.startswith("Someone says") else tr)
    return tpl.format(transcript=tr)


def caption_for_genre(genre: str, idx: int) -> str:
    g = normalize_genre(genre)
    tpl = MUSIC_TEMPLATES[idx % len(MUSIC_TEMPLATES)]
    if tpl == "{genre} music.":
        return (g[0].upper() + g[1:] if g and not g.isupper() else g) + " music."
    return tpl.format(genre=g)


# ---- audio license filtering -----------------------------------------------
# Keep only tracks whose AUDIO license permits redistribution and commercial
# derivative use: CC-BY and CC-BY-SA any version (plus public domain marks).
# NC and ND variants are dropped (research memo section 4).
#
# The decisive signal is the Creative Commons license CODE, taken from the
# canonical URL (``creativecommons.org/licenses/<code>/``) when present. Deciding
# on the code avoids substring false positives: an earlier version matched a bare
# ``nd`` alternative, which fires on the word "u​nder" that begins every MTG
# ``audio_licenses.txt`` line ("Available under a Creative Commons ..."), so every
# track was wrongly denied. Only when no CC URL is present do we fall back to
# token-boundary title matching.
_CC_URL = re.compile(r"creativecommons\.org/(licenses/([a-z-]+)|(publicdomain))", re.I)
# Title-only fallback: allowed license names, and full-word deny terms (with
# boundaries, so "under"/"France"/"license" never trip the NC/ND check).
_TITLE_ALLOW = re.compile(
    r"(public\s*domain|\bCC0\b|attribution[- ]?share[- ]?alike|"
    r"attribution(?!\s*[- ]?non)(?!\s*[- ]?no))", re.I)
_TITLE_DENY = re.compile(
    r"(non[- ]?commercial|\bNC\b|no[- ]?deriv\w*|\bND\b|"
    r"licenses/by-(?:nc|nd))", re.I)


def license_allowed(license_text: str) -> bool:
    """True iff the license permits commercial redistribution of derivatives:
    CC-BY, CC-BY-SA (any version), or public domain / CC0. NC and ND denied."""
    if not license_text:
        return False
    s = license_text.strip()
    m = _CC_URL.search(s)
    if m:                                    # canonical CC URL -> decide on the code
        if m.group(3):                       # publicdomain/...
            return True
        code = (m.group(2) or "").lower()    # e.g. "by", "by-sa", "by-nc-sa"
        return code in ("by", "by-sa") or code.startswith(("by/", "by-sa/"))
    if _TITLE_DENY.search(s):                # no URL: fall back to title words
        return False
    return bool(_TITLE_ALLOW.search(s))
