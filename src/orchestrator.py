"""Main orchestrator — APScheduler wiring for the whole agent.

One process runs:
  - generator cycle (every N minutes → fill queue)
  - brain cycles (trend miner, competitor intel, watcher)
  - poster cycle (publishes scheduled approved items)
  - engager cycle (drains engagement queue)
  - health probe (twice a day)
"""
from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.brain import (
    competitor_intel,
    dm_seeder,
    engagement_seeder,
    events as events_mod,
    hashtag_discovery,
    news_feed,
    rag,
    reddit_harvester,
    retro,
    trend_miner,
    watcher,
)
from src.content import pipeline as content_pipeline
from src.content.generators import carousel_repurpose
from src.core import alerts, db
from src.core.config import (
    ROOT,
    NicheConfig,
    ensure_dirs,
    load_env,
    load_niche,
)
from src.core.logging_setup import setup_logging
from src.core.llm import providers_configured
from src.plugins.ig import BackoffActive, IGClient
from src.workers import (
    comment_replier,
    dm_worker,
    engager,
    follow_back,
    health,
    poster,
    story_poster,
    story_viewer,
)

log = setup_logging(logfile=ROOT / "logs" / "orchestrator.log")


class Orchestrator:
    def __init__(self, cfg: NicheConfig):
        self.cfg = cfg
        self.ig = IGClient()
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self._stop = asyncio.Event()

    # ─── jobs ───
    async def job_generate(self) -> None:
        try:
            cid = await content_pipeline.generate_one(self.cfg)
            if cid is not None and not self.cfg.safety.require_review:
                # Auto-promote to approved and schedule
                db.content_update_status(cid, "approved")
                poster.schedule_approved_items(self.cfg)
        except Exception as e:
            log.exception("job_generate failed")
            await alerts.send(f"Generator cycle failed: {e}", level="err")

    async def job_post(self) -> None:
        try:
            await poster.post_next(self.cfg, ig=self.ig)
        except Exception as e:
            log.exception("job_post failed")
            await alerts.send(f"Post cycle failed: {e}", level="err")

    async def job_post_story(self) -> None:
        try:
            await story_poster.post_next(self.cfg, ig=self.ig)
        except Exception as e:
            log.exception("job_post_story failed")
            await alerts.send(f"Story post cycle failed: {e}", level="err")

    def job_engage(self) -> None:
        try:
            engager.run_pass(self.cfg, ig=self.ig)
        except Exception as e:
            log.exception("job_engage failed")

    async def job_trends(self) -> None:
        try:
            await trend_miner.run_once(self.cfg)
        except Exception as e:
            log.exception("job_trends failed")

    async def job_competitors(self) -> None:
        try:
            await competitor_intel.run_once(self.cfg)
        except Exception as e:
            log.exception("job_competitors failed")

    async def job_news(self) -> None:
        try:
            await news_feed.run_once(self.cfg)
        except Exception as e:
            log.exception("job_news failed")

    def job_hashtag_discovery(self) -> None:
        try:
            hashtag_discovery.run_once(self.cfg)
        except Exception as e:
            log.exception("job_hashtag_discovery failed")

    def job_watch(self) -> None:
        try:
            watcher.run_once(self.cfg)
        except Exception as e:
            log.exception("job_watch failed")

    async def job_seed_engagement(self) -> None:
        try:
            results = await engagement_seeder.run_once(self.cfg)
            seeded = sum(results.values())
            if seeded:
                log.info("seeded %d engagement actions: %s", seeded, results)
        except Exception as e:
            log.exception("job_seed_engagement failed")

    def job_seed_dm(self) -> None:
        try:
            n = dm_seeder.run_once(self.cfg)
            if n:
                log.info("dm_seeder added %d discovered contacts", n)
        except Exception as e:
            log.exception("job_seed_dm failed")

    async def job_curate_dm(self) -> None:
        try:
            promoted = await dm_worker.curate_discovered(self.cfg)
            if promoted:
                log.info("dm_curate: %d promoted discovered→targeted", promoted)
        except Exception as e:
            log.exception("job_curate_dm failed")

    async def job_dm(self) -> None:
        try:
            await dm_worker.run_pass(self.cfg, ig=self.ig)
        except Exception as e:
            log.exception("job_dm failed")

    async def job_comment_replies(self) -> None:
        try:
            await comment_replier.run_pass(self.cfg, ig=self.ig)
        except Exception as e:
            log.exception("job_comment_replies failed")

    async def job_follow_back(self) -> None:
        try:
            await follow_back.run_pass(self.cfg, ig=self.ig)
        except Exception as e:
            log.exception("job_follow_back failed")

    def job_reciprocal(self) -> None:
        try:
            follow_back.queue_reciprocal_from_recent_comments()
        except Exception as e:
            log.exception("job_reciprocal failed")

    async def job_retro(self) -> None:
        try:
            await retro.run_once(self.cfg)
        except Exception as e:
            log.exception("job_retro failed")

    async def job_rag_index(self) -> None:
        try:
            if self.cfg.rag.enabled:
                await rag.index_dir(cfg=self.cfg.rag)
        except Exception as e:
            log.exception("job_rag_index failed")

    async def job_events(self) -> None:
        try:
            await events_mod.run_once(self.cfg)
        except Exception as e:
            log.exception("job_events failed")

    async def job_reddit(self) -> None:
        try:
            await reddit_harvester.run_once(self.cfg)
        except Exception as e:
            log.exception("job_reddit failed")

    async def job_keepalive(self) -> None:
        """Light session probe — get_timeline_feed every ~45 min during
        awake hours. Keeps the cookie jar synced with server-side
        rotations and surfaces LoginRequired before a real write fails.
        Per 2026 instagrapi best practices."""
        try:
            await asyncio.to_thread(self.ig.keep_alive)
        except Exception as e:
            log.debug("job_keepalive non-fatal: %s", e)

    async def job_repurpose(self) -> None:
        try:
            await carousel_repurpose.run_once(self.cfg)
        except Exception as e:
            log.exception("job_repurpose failed")

    def job_story_viewer(self) -> None:
        try:
            story_viewer.run_pass(self.cfg, ig=self.ig)
        except Exception as e:
            log.exception("job_story_viewer failed")

    async def job_health(self) -> None:
        try:
            await health.probe(self.cfg, ig=self.ig)
        except Exception as e:
            log.exception("job_health failed")

    async def job_schedule_approved(self) -> None:
        try:
            n = poster.schedule_approved_items(self.cfg)
            if n:
                log.info("scheduled %d approved items", n)
        except Exception as e:
            log.exception("job_schedule_approved failed")

    # ─── wiring ───
    def start(self) -> None:
        # Generator: every 45–60 min
        self.scheduler.add_job(
            self.job_generate,
            IntervalTrigger(minutes=50, jitter=300),
            id="generate",
            max_instances=1,
            coalesce=True,
        )
        # Poster: check every 2 min
        self.scheduler.add_job(
            self.job_post,
            IntervalTrigger(minutes=2),
            id="post",
            max_instances=1,
            coalesce=True,
        )
        # Story poster: stories are higher-volume, different cadence
        if self.cfg.schedule.stories_per_day > 0:
            self.scheduler.add_job(
                self.job_post_story,
                IntervalTrigger(minutes=4),
                id="post_story",
                max_instances=1,
                coalesce=True,
            )
        # Engager: every 5 min
        self.scheduler.add_job(
            self.job_engage,
            IntervalTrigger(minutes=5, jitter=60),
            id="engage",
            max_instances=1,
            coalesce=True,
        )
        # Trend miner: every ~20 min
        self.scheduler.add_job(
            self.job_trends,
            IntervalTrigger(minutes=22, jitter=300),
            id="trends",
            max_instances=1,
            coalesce=True,
        )
        # Competitor intel: every ~30 min
        self.scheduler.add_job(
            self.job_competitors,
            IntervalTrigger(minutes=35, jitter=300),
            id="competitors",
            max_instances=1,
            coalesce=True,
        )
        # News / RSS miner: every ~40 min, only when feeds configured
        if self.cfg.rss_feeds:
            self.scheduler.add_job(
                self.job_news,
                IntervalTrigger(minutes=40, jitter=300),
                id="news",
                max_instances=1,
                coalesce=True,
            )
        # Hashtag discovery: every ~2h, mines competitor captions
        if self.cfg.competitors:
            self.scheduler.add_job(
                self.job_hashtag_discovery,
                IntervalTrigger(hours=2, jitter=900),
                id="hashtag_discovery",
                max_instances=1,
                coalesce=True,
            )
        # Reply to comments on our own posts
        self.scheduler.add_job(
            self.job_comment_replies,
            IntervalTrigger(minutes=18, jitter=120),
            id="comment_replies",
            max_instances=1,
            coalesce=True,
        )
        # Follow-back triage
        self.scheduler.add_job(
            self.job_follow_back,
            IntervalTrigger(minutes=25, jitter=180),
            id="follow_back",
            max_instances=1,
            coalesce=True,
        )
        # Reciprocal engagement signals
        self.scheduler.add_job(
            self.job_reciprocal,
            IntervalTrigger(minutes=30, jitter=180),
            id="reciprocal",
            max_instances=1,
            coalesce=True,
        )
        # Retro learning — refresh post metrics + feed patterns back
        self.scheduler.add_job(
            self.job_retro,
            IntervalTrigger(hours=3, jitter=600),
            id="retro",
            max_instances=1,
            coalesce=True,
        )
        # RAG re-index on a slow cadence so file drops get picked up automatically
        if self.cfg.rag.enabled:
            self.scheduler.add_job(
                self.job_rag_index,
                IntervalTrigger(minutes=30, jitter=300),
                id="rag_index",
                max_instances=1,
                coalesce=True,
            )
        # Event calendar — every 6h, plus a kick at startup so a themed day
        # reaches context even if the orchestrator just booted.
        if self.cfg.holidays_enabled or self.cfg.events_calendar:
            self.scheduler.add_job(
                self.job_events,
                IntervalTrigger(hours=6, jitter=600),
                id="events",
                max_instances=1,
                coalesce=True,
            )
        # Reddit question harvester — every ~45 min, only when user has
        # configured subs AND PRAW creds are available. Free API tier can
        # handle ~100 qpm so this cadence is well within limits.
        if self.cfg.reddit_enabled and self.cfg.reddit_subs:
            self.scheduler.add_job(
                self.job_reddit,
                IntervalTrigger(minutes=45, jitter=300),
                id="reddit",
                max_instances=1,
                coalesce=True,
            )
        # Session keep-alive — lightweight probe every ~45 min so
        # the cookie jar stays synced + LoginRequired surfaces BEFORE
        # a real write fails. Jittered to avoid scripted cadence.
        self.scheduler.add_job(
            self.job_keepalive,
            IntervalTrigger(minutes=45, jitter=600),
            id="keepalive",
            max_instances=1,
            coalesce=True,
        )
        # Reel → carousel repurpose — runs once a day, only picks up
        # reels older than cfg.reel_repurpose.min_reel_age_days so a reel
        # has time to breathe first. Cheap no-op when no candidate exists.
        if self.cfg.reel_repurpose.enabled:
            self.scheduler.add_job(
                self.job_repurpose,
                CronTrigger(hour=9, minute=30, timezone="UTC"),
                id="repurpose",
                max_instances=1,
                coalesce=True,
            )
        # Story-view pass — view stories of users who just engaged with us
        if self.cfg.budget.story_views > 0:
            self.scheduler.add_job(
                self.job_story_viewer,
                IntervalTrigger(minutes=40, jitter=300),
                id="story_viewer",
                max_instances=1,
                coalesce=True,
            )
        # Watcher: every ~3 min
        if self.cfg.all_watch_targets():
            self.scheduler.add_job(
                self.job_watch,
                IntervalTrigger(minutes=3, jitter=30),
                id="watch",
                max_instances=1,
                coalesce=True,
            )
        # Engagement seeder: every ~15 min, fed by brain tables
        self.scheduler.add_job(
            self.job_seed_engagement,
            IntervalTrigger(minutes=15, jitter=120),
            id="seed_engagement",
            max_instances=1,
            coalesce=True,
        )
        # DM pipeline: only schedule if DM budget > 0 (otherwise it's disabled)
        if self.cfg.budget.dms > 0:
            self.scheduler.add_job(
                self.job_seed_dm,
                IntervalTrigger(minutes=45, jitter=300),
                id="seed_dm",
                max_instances=1,
                coalesce=True,
            )
            self.scheduler.add_job(
                self.job_curate_dm,
                IntervalTrigger(minutes=60, jitter=300),
                id="curate_dm",
                max_instances=1,
                coalesce=True,
            )
            self.scheduler.add_job(
                self.job_dm,
                IntervalTrigger(minutes=12, jitter=90),
                id="dm",
                max_instances=1,
                coalesce=True,
            )
        # Re-schedule approved items when auto-approve is on
        self.scheduler.add_job(
            self.job_schedule_approved,
            IntervalTrigger(minutes=10),
            id="schedule_approved",
            max_instances=1,
            coalesce=True,
        )
        # Health probe: 09:00 and 21:00 UTC
        self.scheduler.add_job(
            self.job_health,
            CronTrigger(hour="9,21", minute=0, timezone="UTC"),
            id="health",
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.start()
        log.info(
            "Orchestrator started. niche=%s commercial=%s formats=%s providers=%s",
            self.cfg.niche,
            self.cfg.commercial,
            {k: v for k, v in self.cfg.formats.normalized().items() if v > 0},
            providers_configured(),
        )

    async def prewarm(self) -> bool:
        """Login once at startup so misconfiguration fails fast.

        Returns True on success, False if we entered cooldown (but we stay up to
        let the cooldown expire). Raises on fatal auth errors.
        """
        try:
            await asyncio.to_thread(self.ig.login)
            log.info("IG login verified at startup (user=%s)", self.ig.username)
            return True
        except BackoffActive as e:
            log.warning("Starting in cooldown — %s", e)
            await alerts.send(f"Starting in cooldown: {e}", level="warn")
            return False
        except Exception as e:
            log.error("IG login failed at startup: %s", e)
            await alerts.send(f"🚨 Login failed at startup: {e}", level="err")
            raise

    async def run_forever(self) -> None:
        self.start()
        # Light "kick" jobs at startup so we don't wait for the first interval
        asyncio.create_task(self.job_trends())
        # Kick events at startup so a themed day reaches context immediately
        if self.cfg.holidays_enabled or self.cfg.events_calendar:
            asyncio.create_task(self.job_events())
        # Kick Reddit harvester so first-boot context isn't dry for 45 min
        if self.cfg.reddit_enabled and self.cfg.reddit_subs:
            asyncio.create_task(self.job_reddit())
        await self._stop.wait()

    def request_stop(self) -> None:
        self._stop.set()


async def amain() -> None:
    load_env()
    ensure_dirs()
    db.init_db()
    cfg = load_niche()

    if not providers_configured():
        log.error("No LLM providers configured. Set OPENROUTER_API_KEY (or GROQ/GEMINI/CEREBRAS).")
        sys.exit(2)

    orch = Orchestrator(cfg)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, orch.request_stop)

    # Fail fast on auth / config issues before scheduling jobs.
    try:
        await orch.prewarm()
    except Exception as e:
        log.error("Startup aborted: %s", e)
        sys.exit(3)

    await alerts.send(f"🟢 ig-agent started for niche={cfg.niche!r}", level="info")
    try:
        await orch.run_forever()
    finally:
        orch.scheduler.shutdown(wait=False)
        await alerts.send("🔴 ig-agent stopped", level="info")


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
