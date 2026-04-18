# Third-party notices

`instagram-ai-agent` itself is [MIT-licensed](LICENSE). It integrates with a stack of third-party components, each under its own licence. The agent's `commercial=true` mode (default in `niche.yaml`) refuses to load any component whose licence forbids commercial use — the gates are enforced at config-load time.

## Commercial-safe defaults

The out-of-the-box experience uses only commercial-safe components:

| Component | Purpose | Licence |
|---|---|---|
| instagrapi | IG transport | MIT |
| instaloader | public-content scraping | MIT |
| OpenRouter / Groq / Gemini / Cerebras | LLM routing | each provider's terms (free tiers OK for commercial by default) |
| Pollinations Flux | image generation | [OpenAssistant](https://pollinations.ai/) — free, permanent |
| Pexels | stock footage | commercial-safe, attribution optional |
| Pixabay | stock footage + music | pixabay licence — commercial-safe, no attribution |
| Freesound (CC0 filter) | audio | CC0 only (filter enforced) |
| Playwright | HTML rendering | Apache-2.0 |
| edge-tts | voiceover synthesis | MIT (wraps free Azure endpoint) |
| faster-whisper | transcription | MIT |
| WhisperX (opt-in) | kinetic caption alignment | BSD-4-Clause |
| librosa (opt-in) | beat sync | ISC |
| Real-ESRGAN (opt-in) | image upscaling | BSD-3 |
| GFPGAN (opt-in) | face restoration | Apache-2.0 |
| LAION aesthetic predictor (opt-in) | local aesthetic scoring | MIT |
| Archivo Black, Inter fonts | typography | SIL Open Font Licence |

## Opt-in optional components

These components are commercial-safe only under specific conditions — the relevant config validator enforces them:

| Component | Condition | Gate |
|---|---|---|
| FLUX.1-schnell | Apache-2.0 — always commercial-OK | — |
| FLUX.1-dev | non-commercial research licence only | `_lora_commercial_gate` |
| Stable Audio Open Small | Stability AI Community Licence (free under $1M revenue) | `_sao_license_gate` — requires `sao_license_acknowledged=True` |
| StoryDiffusion custom node | Attribution-NonCommercial 4.0 (code) | — (shipping feature uses seed-lock, not the node) |
| DWPose preprocessor | Apache-2.0 — commercial-OK | default for `mode: pose` |
| OpenPose | CMU Academic — non-commercial | `_controlnet_commercial_gate` |
| DensePose | CC-BY-NC-4.0 | `_controlnet_commercial_gate` |
| CodeFormer | S-Lab Licence (non-commercial) | refused — not a dependency |
| BeatNet, beat_this | CC-BY-NC | refused — not a dependency |
| MusicGen | CC-BY-NC | refused — not a dependency |
| pyiqa (aesthetic-nc extra) | PolyForm Noncommercial | opt-in via `[aesthetic-nc]` extra only |
| Coqui XTTS | Coqui Public Model Licence (non-commercial) | refused — not a dependency |

## Where to find each gate in source

```
src/brain/idea_bank.py::is_commercial_license
src/core/config.py::_lora_commercial_gate
src/core/config.py::_controlnet_commercial_gate
src/core/config.py::_sao_license_gate
src/plugins/controlnet.py::_COMMERCIAL_BLOCK_SUBSTRINGS
```

If `commercial=true` in your `niche.yaml` and you attempt to enable a non-commercial component, config load raises a `ValueError` with a pointer to the relevant licence.

## Reporting missing gates

If you discover a dependency whose licence is non-commercial and we haven't gated it, that's a security-adjacent issue — report via [SECURITY.md](SECURITY.md).
