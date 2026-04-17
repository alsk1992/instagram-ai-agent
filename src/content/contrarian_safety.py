"""Hard safety gate for contrarian / hot-take mode.

Even in hot-take mode there are topics the agent must NEVER contradict:
medical misinformation, anti-vaccine claims, conspiracy theories, and a
short list of politically-charged framings. This module ships an
opinionated regex blocklist that runs on every piece of contrarian
content before it lands in the queue.

The user-configurable ``cfg.contrarian.avoid_topics`` is a SOFTER layer
— it's fed to the LLM as 'avoid these topics'. This module is the
HARDER layer — content matching any pattern here is rejected outright
regardless of niche, voice, or user overrides.

Scope:
  * Medical misinformation (vaccines, cancer cures, alternative medicine)
  * Anti-science conspiracy tropes (flat earth, moon landing, chemtrails)
  * Politically toxic framings (self-harm encouragement, ethnic disparagement)
  * Dangerous dietary or health advice

False positives are acceptable — the feature is explicitly an opt-in
contrarian mode and a few rejected takes is better than shipping a post
that nukes the page or causes real-world harm.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# Each pattern is matched case-insensitively against the joined caption
# + visible-text + hook content. The label surfaces in the rejection
# log so an operator can tell WHY a take got killed.
_HARD_BLOCK_PATTERNS: list[tuple[str, re.Pattern]] = [
    # ─── Medical / health misinformation ───
    ("vaccines", re.compile(
        r"\bvaccin\w*\s+(cause|caus\w*|fake|useless|dangerous|harm\w*|scam|lie)",
        re.IGNORECASE,
    )),
    ("vaccines_reverse", re.compile(
        # "avoid vaccines", "refuse the vaccine", "don't take vaccines".
        # Negative lookahead excludes benign follow-ups like
        # "vaccine misinformation" / "vaccine hesitancy" / "vaccine
        # advice" where the person is rejecting DISCOURSE about vaccines
        # rather than the vaccines themselves.
        r"\b(avoid|skip|refuse|reject)\s+(the\s+|your\s+|any\s+|all\s+)?"
        r"vaccines?\b(?![-\s]+("
        r"misinformation|disinformation|hesitanc\w+|denial|denialism|"
        r"skeptic\w*|discourse|debate|drama|talk|takes?|advice|"
        r"related|adjacent|content|posts?|conspirac\w+"
        r"))|"
        r"\bdon'?t\s+(take|get|need|trust)\s+(the\s+|your\s+|any\s+|all\s+)?"
        r"vaccines?\b(?![-\s]+("
        r"misinformation|disinformation|hesitanc\w+|advice|related"
        r"))",
        re.IGNORECASE,
    )),
    ("cancer_cure", re.compile(
        r"\b(cure|cures|curing|cured)\s+cancer\b|"
        r"\bcancer\s+is\s+(curable|fake|caused\s+by|not\s+real)",
        re.IGNORECASE,
    )),
    ("big_pharma", re.compile(
        # Expanded: "is lying", "hide", "cover up", "suppress", "hiding the truth".
        r"\bbig\s+pharma\s+"
        r"(hid\w*|covers?(\s+up)?|covered(\s+up)?|suppress\w*|"
        r"(is|are|has\s+been)\s+lying|lied|deceiv\w+|don'?t\s+want)",
        re.IGNORECASE,
    )),
    ("alternative_medicine_over_proven", re.compile(
        # "essential oils cure cancer", "homeopathy heals depression", etc.
        r"\b(essential\s+oils?|homeopath\w*|crystals?|chakras?)\b[^.!?]{0,60}?"
        r"\b(cure\w*|heal\w*|treat\w*)\b[^.!?]{0,60}?"
        r"\b(cancer|diabetes|covid|autism|depression|anxiety|adhd)",
        re.IGNORECASE,
    )),
    ("anti_psychiatry", re.compile(
        r"\b(depression|anxiety|adhd|autism)\s+(is|isn'?t|is\s+not)\s+real\b",
        re.IGNORECASE,
    )),

    # ─── Conspiracy tropes ───
    ("flat_earth", re.compile(r"\bflat\s+earth", re.IGNORECASE)),
    ("moon_landing_hoax", re.compile(
        r"\bmoon\s+landing\s+(was|is)\s+(fake|faked|a\s+hoax)",
        re.IGNORECASE,
    )),
    ("chemtrails", re.compile(r"\bchemtrails?\b", re.IGNORECASE)),
    ("climate_denial", re.compile(
        r"\bclimate\s+change\s+(is|isn'?t|isn't)\s+(real|fake|a\s+hoax|a\s+scam)",
        re.IGNORECASE,
    )),

    # ─── Dangerous dietary / weight-loss ───
    ("eating_disorder_glorification", re.compile(
        # Include bare "starve for X days" and reset/cleanse framings
        # that would otherwise bypass the "yourself" guard.
        r"\bstarv\w*\s+yourself\b|"
        r"\bstarv\w*\b[^.!?]{0,40}\b(days?|week|reset|cleanse|results?)\b|"
        r"\bgo\s+(a\s+)?day\s+without\s+(eating|food)\b|"
        r"\bskip\s+every\s+meal\b",
        re.IGNORECASE,
    )),
    ("extreme_calorie_restriction", re.compile(
        r"\b(under|below|less\s+than)\s+(500|600|700|800)\s+calor\w*\s+(a\s+day|daily|per\s+day)",
        re.IGNORECASE,
    )),

    # ─── Self-harm / suicide ───
    ("self_harm", re.compile(
        r"\b(kill|killing)\s+yourself\s+(is|would|might)",
        re.IGNORECASE,
    )),

    # ─── Ethnic / protected-class disparagement ───
    # Deliberately conservative — flags any broad statement starting with
    # "all" or "every" about a race/religion/gender group.
    ("blanket_group_claim", re.compile(
        # "all men are predators", "every muslim should be deported".
        # Tightened so positive / advocacy phrasings like "all trans
        # people deserve healthcare" DON'T match — requires a verb or
        # adjective that denotes blame, removal, or harm. Plain "are"
        # too broad, so we require a following disparaging token.
        r"\b(all|every|no)\s+"
        r"(men|women|blacks?|whites?|asians?|jews?|muslims?|christians?|"
        r"gays?|trans(gender)?|immigrants?|arabs?|whites|latinos?|latinas?)\s+"
        r"(are\s+(liars?|criminals?|predators?|parasites?|subhuman|trash|"
        r"animals?|dangerous|evil|stupid|violent)|"
        r"should\s+(be\s+)?(deported|removed|banned|eliminat\w+|die|killed)|"
        r"deserve\s+(nothing|to\s+(die|suffer|lose)))",
        re.IGNORECASE,
    )),
]


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    reason: str | None   # pattern label when unsafe, else None

    def __bool__(self) -> bool:
        return self.safe


_ZERO_WIDTH_CHARS = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
}


def _normalise(text: str) -> str:
    r"""Normalise a caption before regex matching:

    * NFKC fold fullwidth ASCII, ligatures, and compatibility variants
      so 'ｃｈｅｍｔｒａｉｌｓ' / 'cﬁ' / etc. match the ASCII patterns.
    * Strip zero-width + non-breaking characters a human reviewer might
      paste accidentally (or maliciously) to bypass the blocklist.
    * Collapse all whitespace so multi-line + multi-space variants
      behave identically to single-space forms in ``\s+`` patterns.
    """
    if not text:
        return ""
    # NFKC handles fullwidth digits/letters, circled chars, ligatures
    s = unicodedata.normalize("NFKC", text)
    # Strip zero-width joiners/non-joiners that otherwise fragment words
    for zw in _ZERO_WIDTH_CHARS:
        s = s.replace(zw, "")
    # Non-breaking space + related → regular space
    s = s.replace("\u00a0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", s.strip())


def check(*parts: str) -> SafetyResult:
    """Test every provided string against the hard blocklist. Returns
    SafetyResult.safe=False on the FIRST match along with the pattern's
    label so the caller can log which guard fired.

    Usage: ``check(caption, visible_text, hook)``.
    """
    joined = _normalise(" || ".join(p for p in parts if p))
    if not joined:
        return SafetyResult(safe=True, reason=None)
    for label, pattern in _HARD_BLOCK_PATTERNS:
        if pattern.search(joined):
            return SafetyResult(safe=False, reason=label)
    return SafetyResult(safe=True, reason=None)
