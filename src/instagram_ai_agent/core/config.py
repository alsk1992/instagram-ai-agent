"""Niche config, env, and paths.

The whole agent's personality lives in niche.yaml — voice, formats, hashtags,
competitors, aesthetic. Load once at process start; everything else reads from
the validated Pydantic model.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

# Package-internal paths — templates ship inside the wheel.
_PKG_ROOT = Path(__file__).resolve().parent.parent  # .../instagram_ai_agent/
TEMPLATES_DIR = _PKG_ROOT / "content" / "templates"

# User-facing paths live in the current working directory so each user
# of an installed package gets their own `data/` and `niche.yaml`.
# In development (`pip install -e .`) this is the repo root; in
# production (`pipx install ...`) this is wherever the user runs
# `ig-agent init`.
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
MEDIA_STAGED = DATA_DIR / "media" / "staged"
MEDIA_POSTED = DATA_DIR / "media" / "posted"
SESSIONS_DIR = DATA_DIR / "sessions"
LUTS_DIR = DATA_DIR / "luts"
LORAS_DIR = DATA_DIR / "loras"
LORA_DATASETS_DIR = DATA_DIR / "lora_datasets"
CONTROLNET_DIR = DATA_DIR / "controlnet"            # reference images
CONTROLNET_MODELS_DIR = DATA_DIR / "controlnet_models"  # .safetensors
FONTS_DIR = DATA_DIR / "fonts"
MUSIC_DIR = DATA_DIR / "music"
MUSIC_CACHE_DIR = DATA_DIR / "music" / "cache"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
DB_PATH = DATA_DIR / "brain.db"
DEVICE_PATH = DATA_DIR / "device.json"
NICHE_PATH = ROOT / "niche.yaml"
ENV_PATH = ROOT / ".env"


class PostFormat(str, Enum):
    MEME = "meme"
    QUOTE_CARD = "quote_card"
    CAROUSEL = "carousel"
    REEL_STOCK = "reel_stock"
    REEL_AI = "reel_ai"
    PHOTO = "photo"
    STORY_QUOTE = "story_quote"
    STORY_ANNOUNCEMENT = "story_announcement"
    STORY_PHOTO = "story_photo"
    STORY_VIDEO = "story_video"


FEED_FORMATS = frozenset({
    "meme", "quote_card", "carousel", "reel_stock", "reel_ai",
    "photo", "human_photo", "story_carousel",
})
STORY_FORMATS = frozenset(
    {"story_quote", "story_announcement", "story_photo", "story_video", "story_human"}
)


class Voice(BaseModel):
    tone: list[str] = Field(..., min_length=1)
    forbidden: list[str] = Field(default_factory=list)
    persona: str = Field(..., min_length=10)
    cta_styles: list[str] = Field(
        default_factory=lambda: ["save for later", "tag a mate", "follow for more"]
    )


class BrandCharacter(BaseModel):
    """Optional persistent brand character for human-photo posts.

    When set, every human_photo / story_human generation conditions on the
    same persona + seed so the "face" looks consistent across the feed.
    Leave disabled to get unique humans per post instead.
    """

    enabled: bool = False
    name: str | None = None                    # internal label, not shown
    age_range: str = "30s"                     # e.g. "mid-20s", "40s", "50+"
    gender: str = "androgynous"                # man | woman | androgynous | non-binary
    ethnicity: str = "unspecified"             # kept deliberately loose to avoid hard pinning
    hair: str = ""                             # e.g. "short brown hair, stubble"
    build: str = ""                            # e.g. "lean athletic"
    wardrobe_style: str = ""                   # e.g. "dark training kit, worn sneakers"
    vibe: str = ""                             # e.g. "tired but determined"
    seed: int = 0                              # stable seed → stable face under Flux
    negative: str = ""                         # things to avoid


class HumanPhoto(BaseModel):
    """Top-level settings for the human-photo pipeline."""

    enabled: bool = False
    model: str = "flux-realism"                # Pollinations model id; falls back chain below
    model_fallbacks: list[str] = Field(default_factory=lambda: ["flux", "turbo"])
    diversity_pool: list[str] = Field(
        default_factory=lambda: [
            "30s man, short brown hair, lean build",
            "20s woman, dark curly hair, athletic",
            "40s man, grey stubble, stocky build",
            "30s woman, short blonde bob, strong",
            "50+ man, weathered face, long grey hair",
            "20s man, shaved head, wiry",
        ]
    )
    character: BrandCharacter = Field(default_factory=BrandCharacter)


class Aesthetic(BaseModel):
    palette: list[str] = Field(..., min_length=2, max_length=6)
    heading_font: str = "Archivo Black"
    body_font: str = "Inter"
    lut: str | None = None
    watermark: str | None = None
    # Film-emulation strength — applied to every AI-generated image so
    # outputs look photographed, not rendered. Biggest single "doesn't
    # read as AI" lever on the visual side.
    # Options: off | subtle | medium | strong. Default medium.
    film_strength: str = "medium"

    @field_validator("palette")
    @classmethod
    def hex_only(cls, v: list[str]) -> list[str]:
        for c in v:
            if not (c.startswith("#") and len(c) in (4, 7)):
                raise ValueError(f"Invalid hex color: {c}")
        return v

    @field_validator("film_strength")
    @classmethod
    def valid_film_strength(cls, v: str) -> str:
        if v not in ("off", "subtle", "medium", "strong"):
            raise ValueError(f"film_strength must be off/subtle/medium/strong, got {v!r}")
        return v


class HumanMimicConfig(BaseModel):
    """Behavioural anti-detection switches. The 2026 IG ML detectors
    flag bot-script patterns (cold posts, instant replies, scripted
    timing) on top of plain request inspection. These switches close
    the biggest gaps. Defaults are safe ON for everything since none
    cost meaningful throughput."""
    # Before posting, open the feed + mark a few posts seen to look
    # like part of a normal session. ~30-60s overhead per post.
    pre_post_scroll: bool = True
    # After posting, enforce a 30-90min silent window before any other
    # write action (comments, follows, likes). Prevents the "posted +
    # liked 5 things within 10 seconds" bot-script signature.
    post_cooldown: bool = True
    # When replying to incoming comments, stagger with a random 5-60min
    # delay per comment so replies don't arrive within seconds.
    comment_reply_delay: bool = True
    # Move hashtags out of the caption into a self-reply on the post.
    # Cleaner caption, fewer 2026-ML downrank penalties on hashtag-
    # heavy captions, same discoverability. Opt-in.
    first_comment_hashtags: bool = False
    # Before a comment/DM hits send, sleep length-proportional to the
    # text so timing looks human (3-5 chars/sec).
    typing_delays: bool = True
    # Reject a caption if it's >85% similar to any of the last 10 posts.
    caption_entropy_check: bool = True
    # Refuse to upload media with off-spec dimensions that IG would
    # re-compress server-side (and downrank).
    aspect_ratio_check: bool = True
    # Recycle the instagrapi Client every 2-4h to reset TCP pool.
    rotate_client: bool = True


class HashtagPools(BaseModel):
    core: list[str] = Field(..., min_length=3)
    growth: list[str] = Field(default_factory=list)
    long_tail: list[str] = Field(default_factory=list)
    per_post: int = Field(default=15, ge=3, le=30)


class FormatMix(BaseModel):
    """Target distribution of feed content formats as probability weights (summed = 1)."""

    meme: float = 0.30
    quote_card: float = 0.15
    carousel: float = 0.25
    reel_stock: float = 0.20
    reel_ai: float = 0.05
    photo: float = 0.05
    human_photo: float = 0.0  # opt-in; wizard enables it
    story_carousel: float = 0.0  # character-consistent narrative carousel

    @field_validator("*")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weights must be non-negative")
        return v

    def normalized(self) -> dict[str, float]:
        total = (
            self.meme
            + self.quote_card
            + self.carousel
            + self.reel_stock
            + self.reel_ai
            + self.photo
            + self.human_photo
            + self.story_carousel
        )
        if total <= 0:
            raise ValueError("at least one feed format weight must be > 0")
        return {
            "meme": self.meme / total,
            "quote_card": self.quote_card / total,
            "carousel": self.carousel / total,
            "reel_stock": self.reel_stock / total,
            "reel_ai": self.reel_ai / total,
            "photo": self.photo / total,
            "human_photo": self.human_photo / total,
            "story_carousel": self.story_carousel / total,
        }


class StoryMix(BaseModel):
    """Target distribution of story formats (summed = 1)."""

    story_quote: float = 0.35
    story_announcement: float = 0.25
    story_photo: float = 0.20
    story_video: float = 0.20
    story_human: float = 0.0  # opt-in when human_photo is enabled

    @field_validator("*")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weights must be non-negative")
        return v

    def normalized(self) -> dict[str, float]:
        total = (
            self.story_quote
            + self.story_announcement
            + self.story_photo
            + self.story_video
            + self.story_human
        )
        if total <= 0:
            raise ValueError("at least one story format weight must be > 0")
        return {
            "story_quote": self.story_quote / total,
            "story_announcement": self.story_announcement / total,
            "story_photo": self.story_photo / total,
            "story_video": self.story_video / total,
            "story_human": self.story_human / total,
        }


class Schedule(BaseModel):
    posts_per_day: int = Field(default=1, ge=0, le=5)
    stories_per_day: int = Field(default=3, ge=0, le=20)
    best_hours_utc: list[int] = Field(default_factory=lambda: [14, 18, 21])

    @field_validator("best_hours_utc")
    @classmethod
    def valid_hours(cls, v: list[int]) -> list[int]:
        for h in v:
            if not 0 <= h < 24:
                raise ValueError(f"Hour out of range: {h}")
        return sorted(set(v))


class Budget(BaseModel):
    """Daily caps — intentionally below IG's unknown thresholds."""

    likes: int = 150
    follows: int = 25
    unfollows: int = 25
    comments: int = 15
    dms: int = 0
    story_views: int = 400


class MusicConfig(BaseModel):
    """Music bed for reels + story videos.

    ``sources`` is an ordered priority chain: first hit wins. ``local`` draws
    from ``data/music/`` (user-populated, CC0 cache). ``pixabay`` tries the
    Pixabay Music API (requires ``PIXABAY_API_KEY``). ``freesound`` tries
    Freesound's free CC0 filter (requires ``FREESOUND_API_KEY``).
    """

    enabled: bool = True
    sources: list[str] = Field(default_factory=lambda: ["local", "pixabay", "freesound"])
    query_template: str = "{niche} instrumental upbeat"
    duck_gain: float = Field(default=0.22, ge=0.0, le=1.0)  # music volume under VO
    vo_gain: float = Field(default=1.0, ge=0.0, le=2.0)
    fade_in_s: float = Field(default=0.4, ge=0.0, le=5.0)
    fade_out_s: float = Field(default=0.8, ge=0.0, le=5.0)
    max_duration_s: int = Field(default=60, ge=5, le=180)
    genres: list[str] = Field(
        default_factory=lambda: ["lofi", "ambient", "upbeat", "motivational", "cinematic"]
    )
    # Beat-sync knobs. When librosa is installed, scene boundaries in reels
    # snap to the nearest musical beat within ``beat_window_s`` seconds,
    # provided the snap wouldn't make any scene shorter than
    # ``beat_min_scene_s``. Set window_s to 0 to disable beat-sync while
    # keeping the music bed itself.
    beat_window_s: float = Field(default=0.3, ge=0.0, le=1.5)
    beat_min_scene_s: float = Field(default=0.8, ge=0.3, le=5.0)

    # Stable Audio Open Small — generative music source. Runs locally
    # (stable-audio-tools + torch, installed via pip `.[stable-audio]`).
    # Add "stable_audio" to ``sources`` to enable it in the fallback
    # chain. ``sao_license_acknowledged`` is a hard gate: the Stability
    # AI Community License requires an Enterprise Licence for orgs with
    # > USD 1M annual revenue, and the user must explicitly acknowledge
    # this before the generator runs for them.
    sao_enabled: bool = False
    sao_license_acknowledged: bool = False
    # Native SAO Small output is ≤ ~11s; longer beds are tiled by
    # looping the generated clip with a short crossfade. Capped at 90s
    # because Instagram reels themselves cap at 90s — anything longer
    # just tiles more seams with no delivered benefit.
    sao_duration_s: int = Field(default=30, ge=5, le=90)
    sao_steps: int = Field(default=8, ge=4, le=100)
    sao_cfg_scale: float = Field(default=6.0, ge=0.0, le=20.0)
    # Runtime device for the torch model. "cpu" works but is slow;
    # "cuda" needs ~4 GB VRAM; "auto" picks cuda when available.
    sao_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    # Optional prompt override. Leave empty to have the agent build a
    # prompt from the niche + scene context on each call.
    sao_prompt_override: str = ""
    # Optional fixed seed for reproducible music across reruns (handy
    # when A/B-testing captions on the same video). None → random.
    sao_seed: int | None = None


class FinishPass(BaseModel):
    """Upscale + face-restore post-gen pass.

    Runs after the generator picks its best-of-N image and before the
    watermark. Upscales via Real-ESRGAN (BSD-3), then (for human subjects
    only) restores facial detail via GFPGAN (Apache-2.0). CodeFormer is
    deliberately NOT used — S-Lab License 1.0 is non-commercial.
    """

    enabled: bool = True
    # 1 = skip upscale (face restore may still run); 2 or 4 = Real-ESRGAN factor.
    upscale_factor: int = Field(default=2, ge=1, le=4)
    face_restore: bool = True
    use_local: bool = True             # try local torch impl first
    use_hf_fallback: bool = False      # opt-in HF Space calls (needs gradio-client)
    hf_upscale_space: str = "Nick088/Real-ESRGAN_Pytorch"
    hf_face_space: str = "Xintao/GFPGAN"
    # Skip if the source is already huge — avoids OOM
    max_input_megapixels: float = Field(default=6.0, ge=0.5)


class RAGConfig(BaseModel):
    """Niche knowledge RAG — embeds files dropped into ``data/knowledge/``
    and injects top-K relevant chunks into caption + critic prompts.

    Embedding provider chain (first available wins):
      1. ``gemini``  — Google's text-embedding-004, free tier, commercial-OK.
      2. ``local``   — sentence-transformers (BAAI/bge-small-en-v1.5).
      3. ``none``    — feature disables itself silently.
    """

    enabled: bool = True
    knowledge_dir: str = "data/knowledge"
    chunk_max_chars: int = Field(default=520, ge=100, le=2000)
    chunk_overlap_chars: int = Field(default=60, ge=0, le=300)
    retrieve_k: int = Field(default=3, ge=1, le=10)
    embedding_provider: str = "auto"   # auto | gemini | local | none
    # Local model — must be commercial-safe (MIT / Apache / OFL etc.)
    local_model: str = "BAAI/bge-small-en-v1.5"
    # Gemini model name — text-embedding-004 returns 768 dims, free-tier OK
    gemini_model: str = "text-embedding-004"
    # Inject this many chars max into prompts so we don't blow context
    max_inject_chars: int = Field(default=900, ge=200, le=4000)


class HookOverlay(BaseModel):
    """Burned-in hook text over the first N seconds of a reel.

    This is the single biggest scroll-stop mechanism on IG reels — big,
    bold, top-third, appears-and-holds-and-fades. Powered by ffmpeg
    ``drawtext`` with alpha expressions (no external deps).
    """

    enabled: bool = True
    duration_s: float = Field(default=2.0, ge=0.6, le=5.0)
    fade_in_s: float = Field(default=0.20, ge=0.0, le=1.0)
    fade_out_s: float = Field(default=0.25, ge=0.0, le=1.0)
    # Font size expressed as a fraction of video height (0.06 ≈ 115px on 1920h)
    font_scale: float = Field(default=0.062, ge=0.02, le=0.12)
    # Top-third anchor as a fraction of video height
    y_ratio: float = Field(default=0.12, ge=0.02, le=0.45)
    # Text payload limits
    max_words: int = Field(default=8, ge=3, le=16)
    max_chars_per_line: int = Field(default=22, ge=10, le=40)
    max_lines: int = Field(default=3, ge=1, le=5)
    # Box behind the text for legibility on busy backgrounds
    box: bool = True
    box_alpha: float = Field(default=0.55, ge=0.0, le=1.0)
    box_borderw: int = Field(default=28, ge=0, le=80)


class CaptionsConfig(BaseModel):
    """Caption rendering style for reels + story videos.

    ``static``  : current behaviour — 3-4 word SRT chunks burned by ffmpeg.
    ``karaoke`` : ASS subtitles with per-word fade + scale-bounce + accent
                  colour. Dominant IG-reel caption style in 2026.
    """

    style: str = "karaoke"  # static | karaoke
    chunk_size: int = Field(default=4, ge=2, le=8)
    karaoke_word_at_a_time: bool = True
    font_scale_peak: float = Field(default=1.18, ge=1.0, le=1.5)
    margin_v_feed: int = Field(default=160, ge=20, le=600)
    margin_v_story: int = Field(default=420, ge=20, le=1000)
    # Override highlight colour (hex). If None, aesthetic.palette[2] is used.
    highlight_colour: str | None = None
    # Prefer WhisperX when its package is importable (optional dep)
    prefer_whisperx: bool = True


class ContrarianConfig(BaseModel):
    """Contrarian / hot-take mode — shifts LLM framing to challenge
    mainstream beliefs in the niche. Not a new post format: a mode that
    rides any format (meme, quote_card, carousel, reel).

    Rolled per generation cycle with probability ``frequency`` when
    ``enabled``. When it fires, the pipeline biases archetype selection
    toward contrarian hooks and threads a ``contrarian=True`` flag into
    every LLM system prompt downstream (captions, slide outlines, reel
    scripts, critic rubric).

    Safety:
      * ``avoid_topics`` is user-configurable; the LLM is told never to
        take contrarian stances on those topics.
      * A separate HARD blocklist in ``src/content/contrarian_safety.py``
        kills specific claim patterns (medical misinformation, toxic
        political takes, anti-vax) regardless of user settings — even
        hot-takes have limits.
    """
    enabled: bool = False
    # Probability a given generation cycle fires as a contrarian post.
    # 0.15 (~1 in 7) is a healthy dose — polarising enough to drive
    # saves + comments without coming across as an angry feed.
    frequency: float = Field(default=0.15, ge=0.0, le=1.0)
    # "moderate" = "unpopular opinion: ..." framing
    # "high"     = "stop doing X, do Y instead" more aggressive framing
    intensity: Literal["moderate", "high"] = "moderate"
    # User-chosen topics where a contrarian take would invite backlash
    # for THIS page. Generators inject these as a don't-be-contrarian-
    # about list in every contrarian system prompt.
    avoid_topics: list[str] = Field(
        default_factory=lambda: [
            "medical claims", "political candidates", "financial advice",
            "religious doctrine", "grief or death",
        ]
    )


class StoryCarouselConfig(BaseModel):
    """Character-consistent narrative carousel. Every slide shows the
    same persona across scenes that tell a story.

    Implementation: **seed-lock**. Every slide is generated with the
    same seed + same persona prompt prefix + a per-slide scene tail.
    Gives ~90% character coherence with zero extra dependencies —
    commercial-safe. Pairs well with a trained brand LoRA (#13): the
    LoRA carries the face, the seed-lock carries pose/framing coherence
    across slides.

    (A StoryDiffusion Consistent-Self-Attention path was evaluated but
    not shipped — its reference code is CC-BY-NC-4.0 and wiring up the
    required custom ComfyUI node without per-build testing is
    speculative. Seed-lock is the real feature.)
    """
    # Number of slides — IG carousels max at 10; we min at 3 so the
    # "narrative" concept makes sense.
    slides: int = Field(default=6, ge=3, le=10)
    # Seed used for every slide in a single generation. Lock for
    # character consistency; randomised per-generation cycle by default.
    # Set a fixed int here to get REPRODUCIBLE carousels. Leaving this
    # as None means each cycle re-rolls the character — good for
    # freshness, bad for A/B testing the same story with two captions.
    seed: int | None = None
    # Template variant for the text overlay on each slide. Reuses the
    # same pack as reel-repurpose carousels (photo_caption.html is the
    # default — full-bleed background + bottom caption card).
    template_variant: str = "photo_caption"


class ControlNetConfig(BaseModel):
    """Reference-image conditioning via ControlNet in ComfyUI.

    When enabled, every AI-generated image is conditioned on a user-
    supplied reference — pose, depth, or canny edges. Lets a single
    brand reference (athlete pullup stance / composed product shot /
    edge sketch) govern every generation without retraining a LoRA.

    Commercial safety:
      * pose   — default preprocessor is DWPose (Apache-2.0). OpenPose
                 is CMU Academic licence (non-commercial) and is blocked
                 when commercial=True via ``_controlnet_commercial_gate``.
      * depth  — Depth-Anything v2 (Apache-2.0), fallback MiDaS (MIT).
      * canny  — OpenCV Canny (Apache-2.0).

    ControlNet weights themselves (Lvmin Zhang's originals, diffusers'
    SDXL ports, etc.) are Apache-2.0 / MIT — commercial-safe across the
    board. The gate only protects against OpenPose slipping in via
    ``preprocessor_override``.
    """
    enabled: bool = False
    mode: Literal["pose", "depth", "canny"] = "pose"
    # Filename under data/controlnet/ — e.g. "pose.png".
    # Empty → feature is dormant (same as enabled=False).
    reference_image: str = ""
    # Filename under data/controlnet_models/ — e.g. "controlnet_canny_sdxl.safetensors".
    model_name: str = ""
    # 0..2 strength. 0.6–0.9 is the usable sweet spot; >1.0 overcooks
    # and ≤ 0 disables the conditioning. Negative values are refused
    # because they SUBTRACT conditioning — almost certainly a user typo.
    strength: float = Field(default=0.75, ge=0.0, le=2.0)
    # Fraction of the denoising curve where ControlNet is active.
    # [0, 1] = full application. Narrowing to [0.0, 0.6] lets the model
    # follow the condition for composition then improvise details.
    start_percent: float = Field(default=0.0, ge=0.0, le=1.0)
    end_percent: float = Field(default=1.0, ge=0.0, le=1.0)
    # Preprocessor override — leave empty to use the mode's default
    # (safe) picker. Setting this explicitly trips the commercial gate
    # for known-NC preprocessors like OpenposePreprocessor.
    preprocessor_override: str = ""

    @model_validator(mode="after")
    def _end_after_start(self) -> "ControlNetConfig":
        if self.end_percent < self.start_percent:
            raise ValueError(
                f"controlnet.end_percent ({self.end_percent}) must be "
                f">= start_percent ({self.start_percent})"
            )
        return self


class LoRAConfig(BaseModel):
    """Brand-consistent image generation via a user-trained LoRA.

    Workflow (external):
      1. `ig-agent lora prepare <images>/ --name X --trigger WORD`
      2. User trains via FluxGym or kohya-ss on a GPU (not bundled — the
         training stack is 10+ GB of torch/xformers/diffusers and needs
         12 GB+ VRAM, so we don't force it into the base install).
      3. `ig-agent lora import <file>.safetensors --name X --trigger WORD`
      4. Activate → every ComfyUI image route prepends the trigger word
         and chains a LoraLoader between the checkpoint and the sampler.

    Licensing — base_model values:
      * "flux-schnell"  Apache-2.0 — commercial-safe.
      * "sdxl"          CreativeML Open RAIL++-M — commercial-safe.
      * "flux-dev"      FLUX.1-dev NON-COMMERCIAL research licence.
                        REJECTED when niche.commercial=True. Users who
                        need dev quality for a monetised page must obtain
                        a paid commercial licence from Black Forest Labs.
    """
    enabled: bool = False
    # Filename under data/loras/ (e.g. "brand_v1.safetensors")
    name: str = ""
    # Word/phrase the LoRA was trained to attach its concept to. Gets
    # prepended to every positive prompt when the LoRA is active.
    trigger_word: str = ""
    # LoRA mixing weight. 0.5–1.2 is the usable range for most LoRAs;
    # >1.0 starts overcooking the base model. Default 0.85 is a safe
    # "visible but not dominant" value.
    strength_model: float = Field(default=0.85, ge=-2.0, le=2.0)
    strength_clip: float = Field(default=0.85, ge=-2.0, le=2.0)
    # Base model family — drives both the ComfyUI workflow AND the
    # commercial gate below.
    base_model: Literal["flux-schnell", "flux-dev", "sdxl"] = "flux-schnell"


class ReelRepurposeConfig(BaseModel):
    """Open-carrusel repurpose: take a posted reel and publish a carousel
    from its scene keyframes. Lets a strong reel breathe a second life
    without burning the LLM/image budget from scratch.
    """
    enabled: bool = False
    # Only consider reels that posted at least N days ago — give the reel
    # time to gather its own engagement before the carousel rides its tail.
    min_reel_age_days: int = Field(default=7, ge=0, le=365)
    max_reel_age_days: int = Field(default=60, ge=1, le=365)
    # 3..10 matches Instagram's carousel limits; default 5 reads well.
    max_slides: int = Field(default=5, ge=3, le=10)
    # Slide template variant under src/content/templates/carousels/.
    # "photo_caption" uses the reel keyframe as a full-bleed background.
    template_variant: str = "photo_caption"

    @model_validator(mode="after")
    def _age_window_nonempty(self) -> "ReelRepurposeConfig":
        if self.max_reel_age_days < self.min_reel_age_days:
            raise ValueError(
                f"reel_repurpose.max_reel_age_days ({self.max_reel_age_days}) "
                f"must be >= min_reel_age_days ({self.min_reel_age_days})"
            )
        return self


class Safety(BaseModel):
    require_review: bool = True
    dedup_hamming_threshold: int = Field(default=8, ge=0, le=32)
    critic_min_score: float = Field(default=0.65, ge=0.0, le=1.0)
    critic_max_regens: int = Field(default=3, ge=0, le=10)
    caption_candidates: int = Field(default=3, ge=1, le=6)
    image_candidates: int = Field(default=1, ge=1, le=4)  # image candidates cost API calls; default 1
    vision_critic: bool = True
    # Two-pass aesthetic scoring: cheap local ensemble (NIMA / LAION) ranks
    # every candidate, vision LLM only runs on the top-K. Disable to revert
    # to the older vision-only ranker.
    local_aesthetic: bool = True
    vision_top_k: int = Field(default=2, ge=1, le=6)
    sleep_min_s: float = 3.0
    sleep_max_s: float = 8.0
    challenge_cooldown_hours: int = 24


class NicheConfig(BaseModel):
    """Top-level niche config — the spine of the whole agent."""

    niche: str = Field(..., min_length=3)
    sub_topics: list[str] = Field(..., min_length=1)
    target_audience: str = Field(..., min_length=5)
    commercial: bool = True  # excludes non-commercial licensed tools

    voice: Voice
    aesthetic: Aesthetic
    hashtags: HashtagPools
    formats: FormatMix = Field(default_factory=FormatMix)
    stories: StoryMix = Field(default_factory=StoryMix)
    human_photo: HumanPhoto = Field(default_factory=HumanPhoto)
    music: MusicConfig = Field(default_factory=MusicConfig)
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    finish: FinishPass = Field(default_factory=FinishPass)
    hook_overlay: HookOverlay = Field(default_factory=HookOverlay)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    schedule: Schedule = Field(default_factory=Schedule)
    budget: Budget = Field(default_factory=Budget)
    safety: Safety = Field(default_factory=Safety)

    # Event calendar — pushes themed-date context into the brain so the
    # pipeline can schedule seasonal / holiday / niche-launch posts.
    holidays_enabled: bool = True
    holiday_country: str = "US"       # ISO 3166-1 alpha-2 — Nager.Date API
    events_lookahead_days: int = Field(default=14, ge=1, le=60)
    events_calendar: list[dict] = Field(
        default_factory=list,
        description="User-defined niche dates. Each item: {date: 'YYYY-MM-DD', label: str, note: str (optional)}.",
    )

    # Reddit question harvester — pulls top question-like posts from
    # niche subreddits so the pipeline can riff on what the community
    # is actually asking.
    reddit_enabled: bool = True
    reddit_subs: list[str] = Field(default_factory=list)
    reddit_posts_per_sub: int = Field(default=10, ge=1, le=50)
    reddit_min_score: int = Field(default=5, ge=0)
    reddit_lookback_hours: int = Field(default=24, ge=1, le=168)

    # Reel → carousel repurpose (Open Carrusel pattern). Runs as a
    # standalone scheduled job; when it finds an eligible reel it
    # enqueues a `carousel` row — the format picker never touches it.
    reel_repurpose: ReelRepurposeConfig = Field(default_factory=ReelRepurposeConfig)

    # Brand LoRA — activated per-image via ComfyUI's LoraLoader node.
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    # Reference-image conditioning via ControlNet — injected into the
    # ComfyUI workflow alongside the LoRA when both are enabled.
    controlnet: ControlNetConfig = Field(default_factory=ControlNetConfig)

    # Character-consistent narrative carousel — same persona across
    # every slide. Seed-locked by default; opt-in StoryDiffusion path
    # is gated by the commercial licence validator below.
    story_carousel: StoryCarouselConfig = Field(default_factory=StoryCarouselConfig)

    # Contrarian / hot-take mode — rolled per generation cycle.
    contrarian: ContrarianConfig = Field(default_factory=ContrarianConfig)

    # Behavioural anti-detection toggles (pre-post scroll, cooldown,
    # typing delay, first-comment hashtags, caption entropy check,
    # aspect ratio check, client rotation). All safe ON by default.
    human_mimic: HumanMimicConfig = Field(default_factory=HumanMimicConfig)

    competitors: list[str] = Field(default_factory=list)
    reference_accounts: list[str] = Field(default_factory=list)
    watch_target: str | None = None  # legacy; prefer watch_targets
    watch_targets: list[str] = Field(default_factory=list)  # multi-target watcher
    rss_feeds: list[str] = Field(default_factory=list)     # niche news / current-events feeds

    # Keyless trend feeds — each is a list of niche-relevant tags/topics.
    # Empty = that feed is disabled. See brain/{hackernews,devto,wiki_otd}.py.
    hackernews_keywords: list[str] = Field(default_factory=list)
    devto_tags: list[str] = Field(default_factory=list)
    wiki_otd_enabled: bool = False

    has_gpu: bool = False
    language: str = "en"

    @model_validator(mode="after")
    def _sao_license_gate(self) -> "NicheConfig":
        """Stable Audio Open Small runs under the Stability AI Community
        Licence — free up to USD 1M annual revenue, Enterprise above.
        We can't check a user's revenue but we can force them to
        acknowledge the terms before we invoke the model."""
        if self.music.sao_enabled and not self.music.sao_license_acknowledged:
            raise ValueError(
                "music.sao_enabled=True requires music.sao_license_acknowledged=True. "
                "Stable Audio Open Small uses the Stability AI Community Licence. "
                "Commercial use requires (a) annual revenue of the using organisation "
                "< USD 1M, and (b) registration for a (free) Community Licence. "
                "Orgs above the revenue threshold must obtain an Enterprise Licence. "
                "Read the terms at https://stability.ai/community-license-agreement "
                "and set sao_license_acknowledged=True in niche.yaml to proceed."
            )
        return self

    @model_validator(mode="after")
    def _controlnet_commercial_gate(self) -> "NicheConfig":
        """Refuse non-commercial preprocessor overrides under commercial=True.

        Substring-matched blocklist so any variant of OpenPose /
        AnimalPose / DensePose (all non-commercial) is caught —
        ``OpenPosePreprocessor_Preview``, ``openpose_full``,
        ``DensePoseEstimator``, etc. DWPose (Apache-2.0) is unaffected
        because "dwpose" shares no substring with the blocklist.
        """
        if not (self.commercial and self.controlnet.enabled):
            return self
        key = (self.controlnet.preprocessor_override or "").strip().lower()
        if not key:
            return self
        bad_substrings = ("openpose", "animalpose", "densepose")
        if any(bad in key for bad in bad_substrings):
            raise ValueError(
                f"controlnet.preprocessor_override={self.controlnet.preprocessor_override!r} "
                "is non-commercial (OpenPose/AnimalPose/DensePose all ship under research-only "
                "licences — CMU Academic or CC-BY-NC-4.0). For pose on a monetised page use "
                "DWPose (Apache-2.0) — leave preprocessor_override empty to use the safe default."
            )
        return self

    @model_validator(mode="after")
    def _lora_commercial_gate(self) -> "NicheConfig":
        """FLUX.1-dev is non-commercial-only. A monetised page (commercial=True)
        that picks base_model='flux-dev' in lora config would silently ship
        images trained + inferred under a licence that forbids the use case.
        Block at load time so the error surfaces in the setup wizard, not
        after the first generation."""
        if self.commercial and self.lora.enabled and self.lora.base_model == "flux-dev":
            raise ValueError(
                "lora.base_model='flux-dev' is incompatible with commercial=True "
                "(FLUX.1-dev is non-commercial research licence). "
                "Use 'flux-schnell' (Apache-2.0) or 'sdxl' (CreativeML Open RAIL++-M) "
                "for monetised pages, or acquire a commercial licence from Black Forest Labs "
                "and set commercial=False with an explicit override."
            )
        return self

    def all_watch_targets(self) -> list[str]:
        """Unify legacy ``watch_target`` with the newer ``watch_targets`` list, dedup."""
        seen: set[str] = set()
        out: list[str] = []
        for u in list(self.watch_targets) + ([self.watch_target] if self.watch_target else []):
            if not u:
                continue
            key = u.lstrip("@").lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(u.lstrip("@"))
        return out


def load_niche(path: Path | None = None) -> NicheConfig:
    p = path or NICHE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"niche.yaml not found at {p}. Run `ig-agent init` to generate it."
        )
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return NicheConfig.model_validate(raw)


def save_niche(cfg: NicheConfig, path: Path | None = None) -> Path:
    p = path or NICHE_PATH
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg.model_dump(mode="json"),
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    return p


def load_env() -> None:
    """Load .env. Silent if missing — CI / live override from actual env."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def ensure_dirs() -> None:
    for d in (
        DATA_DIR,
        MEDIA_STAGED,
        MEDIA_POSTED,
        SESSIONS_DIR,
        LUTS_DIR,
        FONTS_DIR,
        MUSIC_DIR,
        MUSIC_CACHE_DIR,
        KNOWLEDGE_DIR,
        LORAS_DIR,
        LORA_DATASETS_DIR,
        CONTROLNET_DIR,
        CONTROLNET_MODELS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# Typed accessor for LLM provider availability
class LLMProviders(BaseModel):
    openrouter: bool = False
    groq: bool = False
    gemini: bool = False
    cerebras: bool = False

    @classmethod
    def from_env(cls) -> "LLMProviders":
        return cls(
            openrouter=bool(os.environ.get("OPENROUTER_API_KEY")),
            groq=bool(os.environ.get("GROQ_API_KEY")),
            gemini=bool(os.environ.get("GEMINI_API_KEY")),
            cerebras=bool(os.environ.get("CEREBRAS_API_KEY")),
        )

    def any_configured(self) -> bool:
        return any([self.openrouter, self.groq, self.gemini, self.cerebras])


Profile = Literal["cloud", "local", "hybrid"]
