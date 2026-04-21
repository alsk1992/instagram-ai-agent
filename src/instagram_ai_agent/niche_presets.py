"""Niche presets — 8 pre-cooked niche.yaml starters so non-technical
users don't have to invent palette/voice/persona/hashtags from scratch.

Users pick one during ``ig-agent init`` → wizard fills the blanks, they
edit to taste. Every preset is commercial-safe and ships with:
  * A viable voice persona
  * Hex palette (dark + light + accent)
  * Core + growth hashtag seeds
  * Sensible format mix for the niche
  * Best-hours-UTC that match the target audience's timezone peak
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    niche: str
    sub_topics: list[str]
    target_audience: str
    persona: str
    voice_tone: list[str]
    voice_forbidden: list[str]
    palette: list[str]
    core_hashtags: list[str]
    growth_hashtags: list[str]
    best_hours_utc: list[int]
    watermark_hint: str = "@yourhandle"
    format_weights: dict[str, float] = field(default_factory=dict)
    # Story Highlights defaults — 6-9 categories ordered with
    # highest-tap-rate first. Keywords decide which fresh story each
    # highlight absorbs; icon/color drive the generated cover image.
    highlights: list[dict[str, Any]] = field(default_factory=list)


PRESETS: list[Preset] = [
    Preset(
        key="fitness",
        label="Fitness & calisthenics",
        niche="home calisthenics and bodyweight training",
        sub_topics=["pullups", "mobility", "progressions", "recovery"],
        target_audience="office workers rebuilding strength at home",
        persona="ex-office worker rebuilding fitness from scratch, no-fluff direct talk",
        voice_tone=["direct", "dry humour", "no-nonsense"],
        voice_forbidden=["hustle", "grind mindset", "#blessed"],
        palette=["#0a0a0a", "#f5f5f0", "#c9a961"],
        core_hashtags=["calisthenics", "homeworkout", "bodyweighttraining"],
        growth_hashtags=["fittips", "fitnessover40", "pullupprogression"],
        best_hours_utc=[12, 17, 21],
        format_weights={"meme": 0.25, "quote_card": 0.15, "carousel": 0.25,
                        "reel_stock": 0.25, "reel_ai": 0.05, "photo": 0.05},
        # 8 highlight categories ordered by expected tap rate. Research-
        # consensus layout for fitness/calisthenics growth pages. First
        # 3 slots carry ~80% of taps; Start Here + Free Program are the
        # top conversion drivers.
        highlights=[
            {"name": "Start Here",  "icon": "★",  "color": "#c9a961",
             "keywords": ["start here", "intro", "welcome", "about"]},
            {"name": "Workouts",    "icon": "💪", "color": "#0a0a0a",
             "keywords": ["workout", "routine", "session", "wod"]},
            {"name": "Form",        "icon": "✓",  "color": "#1a3a2e",
             "keywords": ["form", "technique", "how to", "posture", "alignment"]},
            {"name": "Progressions","icon": "↗",  "color": "#34495e",
             "keywords": ["progression", "beginner", "intermediate", "advanced", "ladder"]},
            {"name": "Mobility",    "icon": "~",  "color": "#5d4e6d",
             "keywords": ["mobility", "stretch", "warmup", "flexibility"]},
            {"name": "Home",        "icon": "⌂",  "color": "#3a3a3a",
             "keywords": ["home workout", "no gym", "no equipment", "bodyweight"]},
            {"name": "Transformations","icon": "→", "color": "#8b3a2c",
             "keywords": ["transformation", "before", "after", "week", "progress"]},
            {"name": "FAQ",         "icon": "?",  "color": "#2c3e50",
             "keywords": ["faq", "question", "ask", "q&a", "answered"]},
        ],
    ),
    Preset(
        key="food",
        label="Food / recipes",
        niche="quick healthy weeknight recipes",
        sub_topics=["15-minute meals", "meal prep", "high-protein", "budget cooking"],
        target_audience="time-pressed parents cooking for the family",
        persona="home cook who stopped pretending, shares what actually works on a tuesday",
        voice_tone=["warm", "practical", "honest"],
        voice_forbidden=["gourmet", "restaurant-quality", "michelin"],
        palette=["#1a1410", "#faf4e8", "#d97757"],
        core_hashtags=["weeknightdinner", "easyrecipes", "mealprep"],
        growth_hashtags=["dinneridea", "familydinner", "quickrecipe"],
        best_hours_utc=[11, 16, 22],
        format_weights={"meme": 0.10, "quote_card": 0.10, "carousel": 0.35,
                        "reel_stock": 0.15, "reel_ai": 0.05, "photo": 0.25},
    ),
    Preset(
        key="travel",
        label="Travel / adventure",
        niche="budget solo travel in southeast asia",
        sub_topics=["hostels", "street food", "visa tips", "transport hacks"],
        target_audience="first-time solo travelers on a tight budget",
        persona="seasoned backpacker, warts-and-all advice, skips the instagram-perfect lies",
        voice_tone=["candid", "wry", "specific"],
        voice_forbidden=["paradise", "hidden gem", "bucket list"],
        palette=["#1a2e3a", "#f7f3e8", "#e8b04c"],
        core_hashtags=["solotravel", "backpacking", "southeastasia"],
        growth_hashtags=["travelhack", "budgettravel", "digitalnomad"],
        best_hours_utc=[8, 13, 20],
        format_weights={"meme": 0.15, "quote_card": 0.10, "carousel": 0.30,
                        "reel_stock": 0.20, "reel_ai": 0.05, "photo": 0.20},
    ),
    Preset(
        key="finance",
        label="Personal finance",
        niche="personal finance for people who hate spreadsheets",
        sub_topics=["budgeting", "index investing", "debt payoff", "emergency fund"],
        target_audience="mid-career earners who never got taught money basics",
        persona="someone who figured it out late and wants to save you the years",
        voice_tone=["direct", "numbers-first", "no hype"],
        voice_forbidden=["get rich quick", "this one trick", "passive income"],
        palette=["#0d1b2a", "#f8f8f2", "#7fb069"],
        core_hashtags=["personalfinance", "moneytips", "financialliteracy"],
        growth_hashtags=["budgeting101", "moneymindset", "debtfree"],
        best_hours_utc=[12, 18, 22],
        format_weights={"meme": 0.15, "quote_card": 0.20, "carousel": 0.35,
                        "reel_stock": 0.20, "reel_ai": 0.05, "photo": 0.05},
    ),
    Preset(
        key="mindfulness",
        label="Mindfulness / mental health",
        niche="practical mindfulness for people who can't sit still",
        sub_topics=["breathing techniques", "micro-meditations", "journaling prompts", "sleep"],
        target_audience="overstimulated professionals looking for 5-min resets",
        persona="former burnout case, now allergic to wellness jargon",
        voice_tone=["calm", "plain-spoken", "gentle"],
        voice_forbidden=["manifest", "high vibrations", "raise your frequency"],
        palette=["#2a3b3e", "#faf6f0", "#a8b4a0"],
        core_hashtags=["mindfulness", "mentalhealth", "stressrelief"],
        growth_hashtags=["selfcare", "anxietyrelief", "burnoutrecovery"],
        best_hours_utc=[7, 13, 21],
        format_weights={"meme": 0.10, "quote_card": 0.30, "carousel": 0.25,
                        "reel_stock": 0.20, "reel_ai": 0.10, "photo": 0.05},
    ),
    Preset(
        key="productivity",
        label="Productivity / learning",
        niche="deep-work productivity for knowledge workers",
        sub_topics=["focus systems", "notetaking", "weekly review", "digital minimalism"],
        target_audience="remote workers + students drowning in shallow tasks",
        persona="ex-consultant who fixed his own focus and won't shut up about it",
        voice_tone=["sharp", "systems-first", "deadpan"],
        voice_forbidden=["hustle", "rise-and-grind", "5am club"],
        palette=["#1a1a1a", "#f0eee6", "#e74c3c"],
        core_hashtags=["productivity", "deepwork", "focus"],
        growth_hashtags=["notetaking", "workflow", "timemanagement"],
        best_hours_utc=[7, 12, 18],
        format_weights={"meme": 0.25, "quote_card": 0.20, "carousel": 0.30,
                        "reel_stock": 0.15, "reel_ai": 0.05, "photo": 0.05},
    ),
    Preset(
        key="fashion",
        label="Fashion / style",
        niche="budget capsule wardrobe for men 25-40",
        sub_topics=["core pieces", "fit", "fabrics", "thrift finds"],
        target_audience="men who want to look put-together without thinking about it",
        persona="ex-tech-bro who escaped the fleece vest and figured out fit",
        voice_tone=["stylish", "confident", "pragmatic"],
        voice_forbidden=["fashion-forward", "must-have", "wardrobe essential"],
        palette=["#1c1c1c", "#ede4d3", "#96704d"],
        core_hashtags=["menswear", "capsulewardrobe", "menstyle"],
        growth_hashtags=["mensfashion", "styletips", "outfitoftheday"],
        best_hours_utc=[12, 17, 20],
        format_weights={"meme": 0.15, "quote_card": 0.10, "carousel": 0.30,
                        "reel_stock": 0.10, "reel_ai": 0.05, "photo": 0.30},
    ),
    Preset(
        key="pets",
        label="Pets (dogs / cats)",
        niche="positive-reinforcement dog training for rescue adopters",
        sub_topics=["leash manners", "recall", "crate training", "reactive dogs"],
        target_audience="first-time dog owners with a rescue they don't know how to handle",
        persona="certified trainer who hates dominance-theory BS",
        voice_tone=["warm", "evidence-based", "empathetic"],
        voice_forbidden=["alpha", "pack leader", "dominance"],
        palette=["#2c2416", "#faf4e0", "#d4a72c"],
        core_hashtags=["dogtraining", "rescuedog", "positivereinforcement"],
        growth_hashtags=["dogsofinstagram", "puppytraining", "reactivedog"],
        best_hours_utc=[9, 14, 20],
        format_weights={"meme": 0.20, "quote_card": 0.10, "carousel": 0.25,
                        "reel_stock": 0.15, "reel_ai": 0.05, "photo": 0.25},
    ),
]


def by_key(key: str) -> Preset | None:
    for p in PRESETS:
        if p.key == key:
            return p
    return None


def to_niche_config_fields(p: Preset) -> dict[str, Any]:
    """Return the subset of NicheConfig init kwargs seeded from the preset.
    Caller merges with wizard-collected overrides (account handle, etc.)."""
    return {
        "niche": p.niche,
        "sub_topics": list(p.sub_topics),
        "target_audience": p.target_audience,
        "persona": p.persona,
        "voice_tone": list(p.voice_tone),
        "voice_forbidden": list(p.voice_forbidden),
        "palette": list(p.palette),
        "core_hashtags": list(p.core_hashtags),
        "growth_hashtags": list(p.growth_hashtags),
        "best_hours_utc": list(p.best_hours_utc),
        "format_weights": dict(p.format_weights),
    }
