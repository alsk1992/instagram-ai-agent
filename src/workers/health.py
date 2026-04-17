"""Health worker — snapshots own-account metrics so we notice shadow-bans / drops."""
from __future__ import annotations

from statistics import mean

from src.brain.scraper import PublicScraper
from src.core import alerts, db
from src.core.config import NicheConfig
from src.core.logging_setup import get_logger
from src.plugins.ig import BackoffActive, IGClient

log = get_logger(__name__)


async def probe(cfg: NicheConfig, ig: IGClient | None = None) -> dict | None:
    cl = ig or IGClient()
    try:
        cl.login()
    except BackoffActive as e:
        log.warning("health: cooldown — %s", e)
        return None
    except Exception as e:
        log.error("health: login failed — %s", e)
        return None

    info = cl.self_info()
    followers = int(info.get("follower_count", 0) or 0)
    following = int(info.get("following_count", 0) or 0)
    media_count = int(info.get("media_count", 0) or 0)

    # Engagement rate: average likes/comments across last 9 posts vs followers
    engagement_rate = 0.0
    user_id = info.get("pk") or info.get("id")
    if user_id and followers > 0:
        try:
            medias = cl.user_medias(str(user_id), amount=9)
            if medias:
                er = mean((m["likes"] + m["comments"]) / followers for m in medias)
                engagement_rate = round(er, 4)
        except Exception as e:
            log.debug("health: metrics fetch failed: %s", e)

    # Shadow-ban probe: do our last posts appear when an *anonymous* viewer
    # searches one of our core hashtags? We cannot use our own session for
    # this — IG is more permissive to self-requests and gives false negatives.
    shadowbanned = False
    core = [t for t in cfg.hashtags.core if t]
    if core and user_id:
        probe_tag = core[0]
        try:
            # Our recent PKs (auth required)
            our_medias = cl.user_medias(str(user_id), amount=3)
            our_pks = {m["pk"] for m in our_medias}

            # Anonymous hashtag view
            anon_scraper = PublicScraper()
            top = anon_scraper.hashtag_posts(probe_tag, limit=60)
            top_pks = {p.ig_pk for p in top}
            shadowbanned = bool(our_pks) and not (our_pks & top_pks)
        except Exception as e:
            log.debug("health: shadowban probe failed: %s", e)

    db.health_record(followers, following, media_count, engagement_rate, shadowbanned)
    snapshot = {
        "followers": followers,
        "following": following,
        "media_count": media_count,
        "engagement_rate": engagement_rate,
        "shadowbanned": shadowbanned,
    }

    if shadowbanned:
        await alerts.send(
            f"⚠️ Shadow-ban suspected: recent posts not appearing in #{probe_tag} top.",
            level="warn",
        )

    log.info(
        "health: followers=%d following=%d media=%d ER=%.3f shadowban=%s",
        followers, following, media_count, engagement_rate, shadowbanned,
    )
    return snapshot
