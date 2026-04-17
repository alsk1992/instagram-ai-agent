# Quality Phase — 17-Step Plan

Objective: lift single-agent output from "competent-AI" to "indistinguishable-from-a-small-creator" across reels, images, ideas, and engagement. Every item below landed from the April 2026 research pass. Every listed tool is verified commercial-safe (MIT / Apache / BSD / CC0) unless explicitly flagged.

**Workflow per item:**
1. Implement with max care.
2. Deploy an audit agent to verify the work.
3. Fix anything the audit flags.
4. Only then tick off.

---

## Tier 1 — One-shot visible-quality wins

### 1. Music on reels (Pixabay Music API)
- **Problem:** Reels ship with voiceover over silence → IG downranks + looks bot-generated.
- **Winner:** Pixabay Music API (100 req/60s, 5000/hr, commercial OK, no attribution).
- **Fallback:** cached `SoundSafari/CC0-1.0-Music` corpus on disk for offline / rate-limit cases.
- **Integrate:** `plugins/music.py` (fetch + cache), ffmpeg sidechain/volume-duck under VO in `reel_stock.py` + `reel_ai.py` + `story_video.py`.
- **Env:** `PIXABAY_API_KEY`.
- **Done when:** every new reel has a looping niche-appropriate music bed attenuated (linear gain ≈ 0.22 → ~-13 dB below VO). A true LUFS-relative mix would need a `loudnorm` post pass — deferred.
- **Audit checklist:** confirm voiceover isn't drowned, track is cached for re-use, no copyright-strike risk.

### 2. Kinetic captions (WhisperX + pycaps)
- **Problem:** Subtitles are static SRT burn-in; IG reels expect karaoke-style word-by-word highlight.
- **Winner A (alignment):** `m-bain/whisperX` (BSD-2) — word-level timestamps via wav2vec2 forced align.
- **Winner B (render):** `francozanardi/pycaps` (MIT) — CSS-styled word animations over MoviePy.
- **Alternative route:** `mateusz-kow/auto-subs` (MIT) if we want to keep ffmpeg `subtitles` filter with ASS karaoke tags.
- **Integrate:** swap `faster-whisper` for `WhisperX` in `reel_stock.transcribe_to_srt`, add a `caption_style` flag in niche.yaml (`static` | `karaoke`). When `karaoke`, hand the rendered video off to pycaps for a final overlay pass.
- **Done when:** reels output has per-word highlight animation; caption style is niche-config-switchable.
- **Audit checklist:** captions timing stays synced to voiceover, legibility on a phone frame, no dropped words.

### 3. Upscale + face-restore finish pass
- **Problem:** Pollinations outputs arrive with visible compression/anatomy issues.
- **Winner (upscale):** `xinntao/Real-ESRGAN` (BSD-3) — 2/4× general-purpose.
- **Winner (face restore):** `TencentARC/GFPGAN` (Apache-2.0). **Do NOT use CodeFormer — S-Lab NC.**
- **Integrate:** `plugins/finish_pass.py` — callable post-gen filter; enabled by `human_photo` + `photo` + `story_photo`; toggleable via `safety.finish_pass: true/false`.
- **Fallback:** HF Space `doevent/Face-Real-ESRGAN` for users with no GPU.
- **Done when:** finish_pass enabled and visible quality bump on a sample image; fallback path works when local weights absent.
- **Audit checklist:** memory usage acceptable, cumulative latency ≤ original gen time, output actually looks better (manual before/after).

### 4. Hook overlay on reels (ffmpeg drawtext 0-2s)
- **Problem:** No burned-in hook text on the first 2 seconds — critical scroll-stop mechanism.
- **Winner:** native ffmpeg `drawtext` + eased fade/scale/shake expressions (reference: `scriptituk/xfade-easing` for easing math — MIT).
- **Integrate:** add `hook_text_overlay` step in `reel_stock.py` / `reel_ai.py` after mux, pull the first scene's `line` as overlay text, animate fade-in 0-0.3s + scale 0.8→1.0, hold 0.3-1.8s, fade-out 1.8-2.0s.
- **Done when:** every reel has top-third hook overlay the first 2s; palette-derived colours, heading font.
- **Audit checklist:** overlay doesn't cover subject, disappears before main caption starts, uses niche heading font.

---

## Tier 2 — Compounding quality systems

### 5. Idea bank (SQLite ideas table + prompt corpus ingest)
- **Problem:** Every gen re-derives ideas from scratch.
- **Winner (corpus):** `f/awesome-chatgpt-prompts` (CC0) + `MaxsPrompts/Marketing-Prompts` (Apache-2.0, 4,368 CSV prompts).
- **Winner (meme templates):** `not-lain/meme-dataset` (CC, 300 templates) — avoid ImgFlip575K (no LICENSE, ToS risk).
- **Integrate:** `brain/idea_bank.py` with schema `{archetype, template, hook_formula, format_hint, niche_tag}`. One-off ingest CLI `ig-agent seed-idea-bank`. Pipeline draws from idea_bank alongside trend context.
- **Done when:** pipeline is provably drawing archetypes from the bank (logged), variety metric up.
- **Audit checklist:** no archetype repeats within last 14 picks, CC0 / Apache source attribution preserved in comments.

### 6. Niche RAG (txtai + Chroma + Gemini embeddings)
- **Problem:** LLM hand-waves niche specifics.
- **Winner (framework):** `neuml/txtai` (Apache-2.0) — smallest surface area.
- **Winner (store):** `chroma-core/chroma` (Apache-2.0).
- **Winner (embeddings):** Gemini Embedding (free tier, commercial-safe) primary; `BAAI/bge-m3` (MIT) or `nomic-ai/nomic-embed-text-v1.5` (Apache) local fallback.
- **Integrate:** `data/knowledge/` drop-folder, `brain/rag.py` indexer + `rag.context_for(query, k=3)` helper, wired into `captions.py` + `critic.py` so every generation gets 3 niche-specific knowledge chunks.
- **Done when:** dropping a PDF/MD/URL list into `data/knowledge/` auto-indexes and the next caption cites facts from it.
- **Audit checklist:** retrieval precision, no PII leakage, graceful empty-index behaviour.

### 7. Multi-template pack (memes, quote cards, carousels)
- **Problem:** Single template per format → visual monotony.
- **Deliverable:** 5 meme templates (+ backgrounds), 3 quote-card HTMLs, 3 carousel HTMLs.
- **Source inspiration:** `not-lain/meme-dataset` for meme formats; re-author HTML layouts in-house.
- **Integrate:** existing `list_templates()` in `meme.py` already iterates — just add files. For quote_card/carousel, add template selector logic.
- **Done when:** each format picks a random template per post and every HTML-driven template (quote cards, carousels) renders correctly across arbitrary niche palettes.
- **Note:** Meme backgrounds are procedurally pre-rendered JPGs with hard-coded colours (drake's red/green, expectation_reality's blue/red). These don't re-tint per niche palette — palette propagation is a feature of the HTML-rendered formats (quote_card, carousel) only. Adding palette-aware meme backgrounds is deferred.
- **Audit checklist:** each template sampled ≥1× in a 20-gen dry-run, no truncation, fonts render (base64 embedding works), typed variant strings emit warnings on miss.

### 8. Beat-synced cuts (librosa)
- **Problem:** Reel cuts aren't synced to music beats.
- **Winner:** `librosa/librosa` (ISC) — `onset_detect` + `beat_track`.
- **Avoid:** BeatNet, beat_this (both CC-BY-NC).
- **Integrate:** after music track selection in reel pipeline, compute beat timestamps, snap scene boundaries to nearest beat within a 0.3s window.
- **Done when:** reel edits visibly land on beats in the chosen track.
- **Audit checklist:** no scenes smaller than 0.8s after snap, beat detection stable on quiet intros.

### 9. Aesthetic scoring ensemble (LAION predictor, NIMA opt-in)
- **Problem:** Vision-LLM ranking is slow + costs quota.
- **Commercial-safe winner:** `shunk031/simple-aesthetics-predictor` (MIT, LAION CLIP predictor).
- **Licence reality:** The NIMA / quality-metric side turned out trickier than originally scoped — `pyiqa` (which bundles NIMA, CLIP-IQA, MUSIQ) ships under PolyForm Noncommercial, not BSD-3. The idealo/image-quality-assessment Keras port is Apache-2.0 but is not pip-installable cleanly. Default commercial ensemble ships LAION only; pyiqa is available via an opt-in `[aesthetic-nc]` extra for personal / research users who accept the licence.
- **Integrate:** `content/image_rank.py` runs a two-pass ranker — local LAION score over every candidate, vision LLM only on top-K by local rank.
- **Done when:** image-candidate ranking latency drops; quality stays ≥ vision-only baseline.
- **Audit checklist:** default extra never pulls NC deps, `default_scorers()` only returns commercial-safe entries, vision provider gated.

---

## Tier 3 — Growth + engagement polish

### 10. Event calendar (Nager.Date + niche RSS)
- **Problem:** No seasonal / event-driven content.
- **Winner:** `Nager.Date` API (MIT, no key, global holidays).
- **Integrate:** `brain/events.py` pulls next 14 days' holidays + user-defined niche dates from niche.yaml (`events_calendar: [ {date, label} ]`). Pipeline checks the calendar and boosts priority of on-theme archetypes.
- **Done when:** a holiday week in niche.yaml triggers a themed post generation.
- **Audit checklist:** timezone correctness, no duplicate event triggers.

### 11. Reddit question harvester (PRAW)
- **Problem:** No pipeline for audience questions.
- **Winner:** `praw-dev/praw` (BSD-2) + free Reddit API read tier.
- **Integrate:** `brain/reddit_harvester.py` fetches top r/{niche_sub}/top/day and `r/AskX` where X is niche-relevant, filters by `?` ending, pushes to context_feed at priority 3.
- **Env:** `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`.
- **Done when:** niche.yaml subreddit list → daily question snippets appear in context_feed.
- **Audit checklist:** rate-limit compliance, dedup (no same question twice), sensitive-sub filtering.

### 12. Open Carrusel integration (reel → carousel repurpose)
- **Problem:** Top reels never reborn as carousels/quote cards.
- **Winner:** `Hainrixz/open-carrusel` (MIT) — Claude-driven HTML/CSS → PNG carousels.
- **Integrate:** `brain/repurpose.py` — given a top-performing reel from `posts`, extract scenes, generate a carousel HTML draft via our carousel generator seeded with that content.
- **Done when:** CLI `ig-agent repurpose <ig_media_pk>` produces a new carousel item in the queue.
- **Audit checklist:** original post attribution in meta, PHash distinct from the reel source.

### 13. FluxGym brand LoRA CLI
- **Problem:** Brand-character face drift despite seed-locking.
- **Winner:** `cocktailpeanut/fluxgym` (MIT). Requires GPU.
- **Integrate:** CLI `ig-agent train-character <photos_dir>` wraps FluxGym, stores resulting `.safetensors` path in niche.yaml; `human_photo` loads it via ComfyUI workflow when set.
- **Done when:** trained LoRA path shows up in niche.yaml.human_photo.character.lora and human_photo renders use it when `COMFYUI_URL` is set.
- **Audit checklist:** graceful no-op when no GPU / no LoRA path; non-commercial weights (InsightFace etc.) not used.

---

## Tier 4 — Nice-to-have extras

### 14. Stable Audio Open Small (commercial-safe generative music)
- **Problem:** Dependent on Pixabay library — no bespoke music.
- **Winner:** Stable Audio Open **Small** (Stability Community License — commercial-safe per the 2025 Arm release, verified).
- **Integrate:** optional `music_mode: library|generative` in niche.yaml. When `generative`, prompt Stable Audio Open Small for a niche-themed short loop.
- **Done when:** with GPU available, one reel per day uses a generated track.
- **Audit checklist:** licence verified on disk (`LICENSE` file alongside weights), no 1.0 weights shipped.

### 15. ControlNet pose/depth in ComfyUI workflow
- **Problem:** Human generations don't enforce compositions.
- **Winner:** `IDEA-Research/DWPose` (Apache) + Depth Anything V2 (Apache). Avoid OpenPose (CMU license).
- **Integrate:** ship an alternative ComfyUI workflow that accepts pose/depth reference images; `human_photo.py` can attach a pose reference pulled from the configured aesthetic reference account.
- **Done when:** sample pose reference produces a generation matching the skeleton.
- **Audit checklist:** workflow runs end-to-end on a fresh ComfyUI install.

### 16. StoryDiffusion for carousel character consistency
- **Problem:** Carousel slides featuring humans drift across slides.
- **Winner:** `HVision-NKU/StoryDiffusion` (Apache-2.0).
- **Integrate:** optional carousel mode that renders all N slides in a single StoryDiffusion pass. GPU required.
- **Done when:** 5-slide carousel uses visually-consistent character across slides.
- **Audit checklist:** falls back cleanly to HTML carousel when GPU absent.

### 17. Contrarian / hot-take mode
- **Problem:** Every post is consensus-aligned; contrarian posts outperform in most niches.
- **Integrate:** JSON archetype in idea_bank marked `contrarian: true`, pipeline randomly selects with ~10% probability (configurable), LLM system prompt adds "steelman the opposite" + "majority is wrong because X" framing.
- **Done when:** hot-take variants appear in critic logs, critic doesn't reject them as off-voice.
- **Audit checklist:** ratio honoured over a 30-post window, forbidden phrases still enforced.

---

## License landmines (never ship these)

| Tool | Why | Safe swap |
|---|---|---|
| CodeFormer | S-Lab NC | GFPGAN (Apache) |
| InsightFace weights | NC research-only — contaminates IP-Adapter FaceID, InstantID transitively | FluxGym trained LoRA or PhotoMaker V2 |
| MusicGen (any size) | CC-BY-NC weights | Pixabay API |
| Stable Audio Open 1.0 | NC | Stable Audio Open **Small** |
| BeatNet / beat_this | CC-BY-NC | librosa |
| FLUX.1 dev (hosted as service) | Redistribution restricted | FLUX.1 schnell |
| Postiz | AGPL redistribution trap | Mixpost (MIT) |
| ImgFlip575K | No LICENSE + ToS | not-lain/meme-dataset |
| Cohere trial | Explicitly NC | Gemini free tier |

---

## Status tracker (kept in sync with TaskList)

- [x] 01 Music on reels (Pixabay) — audited ✅
- [x] 02 Kinetic captions (WhisperX + ASS karaoke) — audited ✅
- [x] 03 Upscale + face-restore finish pass — audited ✅
- [x] 04 Hook overlay on reels (0-2s) — audited ✅
- [x] 05 Idea bank (prompt corpus + SQLite) — audited ✅
- [x] 06 Niche RAG (txtai + Chroma + Gemini) — audited ✅
- [x] 07 Multi-template pack — audited ✅
- [x] 08 Beat-synced cuts (librosa) — audited ✅
- [x] 09 Aesthetic scoring ensemble — audited ✅
- [x] 10 Event calendar (Nager.Date) — audited ✅
- [x] 11 Reddit question harvester (PRAW) — audited ✅
- [x] 12 Open Carrusel repurpose — audited ✅
- [x] 13 FluxGym brand LoRA CLI — audited ✅
- [x] 14 Stable Audio Open Small — audited ✅
- [x] 15 ControlNet pose/depth in ComfyUI — audited ✅
- [x] 16 StoryDiffusion carousel consistency — audited ✅ (seed-lock; SD node path deferred)
- [x] 17 Contrarian / hot-take mode — audited ✅
