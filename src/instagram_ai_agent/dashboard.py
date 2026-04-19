"""Local read-only dashboard — FastAPI + Jinja templates.

Launch with ``ig-agent dashboard``. Binds 127.0.0.1 by default. Optional
HTTP Basic auth via env ``DASH_USER`` / ``DASH_PASS``.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from instagram_ai_agent.core import db
from instagram_ai_agent.core.config import ROOT, ensure_dirs, load_env, load_niche
from instagram_ai_agent.core.llm import providers_configured
from instagram_ai_agent.core.warmup import effective_caps

TEMPLATE_DIR = Path(__file__).parent / "dashboard_templates"


def create_app() -> FastAPI:
    load_env()
    ensure_dirs()
    db.init_db()

    app = FastAPI(title="ig-agent dashboard", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    user = os.environ.get("DASH_USER")
    pw = os.environ.get("DASH_PASS")
    security = HTTPBasic()

    def _auth(creds: HTTPBasicCredentials = Depends(security)) -> str:
        if user and pw:
            ok = secrets.compare_digest(creds.username, user) and secrets.compare_digest(creds.password, pw)
            if not ok:
                raise HTTPException(
                    status.HTTP_401_UNAUTHORIZED,
                    "invalid credentials",
                    headers={"WWW-Authenticate": "Basic"},
                )
        return creds.username

    # Auth is only enforced when both vars are set
    auth_dep = [Depends(_auth)] if (user and pw) else []

    @app.get("/", response_class=HTMLResponse, dependencies=auth_dep)
    async def index(request: Request) -> HTMLResponse:
        ctx = _build_context()
        # Starlette 1.x signature: (request, name, context, ...)
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.get("/api/state", dependencies=auth_dep)
    async def api_state() -> JSONResponse:
        return JSONResponse(_build_context(for_json=True))

    @app.get("/api/queue", dependencies=auth_dep)
    async def api_queue(status_filter: str | None = None) -> JSONResponse:
        items = db.content_list(status=status_filter, limit=100)
        for item in items:
            item.pop("meta", None)
        return JSONResponse(items)

    @app.get("/review", response_class=HTMLResponse, dependencies=auth_dep)
    async def review_page(request: Request) -> HTMLResponse:
        """Visual review of every pending item — approve/reject in one click."""
        pending = db.content_list(status="pending_review", limit=200)
        items: list[dict] = []
        for p in pending:
            media_paths = p.get("media_paths") or []
            media_urls = [
                url for url in (_media_preview_url([mp]) for mp in media_paths) if url
            ]
            critic_overall = None
            try:
                import json as _json
                meta = _json.loads(p.get("meta") or "{}")
                if isinstance(meta, dict):
                    score = meta.get("critic_score")
                    if isinstance(score, (int, float)):
                        critic_overall = round(float(score), 2)
            except Exception:
                pass
            if critic_overall is None and p.get("critic_score") is not None:
                try:
                    critic_overall = round(float(p["critic_score"]), 2)
                except Exception:
                    critic_overall = None
            items.append({
                "id": p["id"],
                "format": p["format"],
                "caption": p.get("caption") or "",
                "media_urls": media_urls,
                "critic_score": critic_overall,
                "critic_notes": (p.get("critic_notes") or "")[:240],
                "created_at": p.get("created_at"),
            })
        return templates.TemplateResponse(request, "review.html", {"items": items})

    @app.post("/api/queue/{cid}/approve", dependencies=auth_dep)
    async def api_approve(cid: int) -> JSONResponse:
        db.content_update_status(cid, "approved")
        return JSONResponse({"id": cid, "status": "approved"})

    @app.post("/api/queue/{cid}/reject", dependencies=auth_dep)
    async def api_reject(cid: int) -> JSONResponse:
        db.content_update_status(cid, "rejected")
        return JSONResponse({"id": cid, "status": "rejected"})

    @app.get("/media/{path:path}", dependencies=auth_dep)
    async def media(path: str) -> FileResponse:
        """Serve staged or posted media for preview in the dashboard."""
        target = (ROOT / "data" / "media" / path).resolve()
        media_root = (ROOT / "data" / "media").resolve()
        try:
            target.relative_to(media_root)
        except ValueError:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "outside media root")
        if not target.exists() or not target.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
        return FileResponse(str(target))

    # Static dir is optional; only mount if it exists
    static_dir = Path(__file__).parent / "dashboard_static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def _build_context(*, for_json: bool = False) -> dict[str, Any]:
    cfg = load_niche()
    warmup = effective_caps(cfg)

    # Queue grouping
    q_rows = db.content_list(status=None, limit=500)
    queue_by_status: dict[str, int] = {}
    for r in q_rows:
        queue_by_status[r["status"]] = queue_by_status.get(r["status"], 0) + 1

    # Scheduled items preview (next 10)
    upcoming = [
        {
            "id": r["id"],
            "format": r["format"],
            "scheduled_for": r.get("scheduled_for"),
            "caption": (r["caption"] or "")[:140],
            "media_preview": _media_preview_url(r.get("media_paths") or []),
        }
        for r in sorted(
            [r for r in q_rows if r["status"] == "approved" and r.get("scheduled_for")],
            key=lambda r: r["scheduled_for"] or "",
        )[:10]
    ]

    # Recent posts
    conn = db.get_conn()
    posts = [dict(r) for r in conn.execute(
        "SELECT ig_media_pk, format, caption, posted_at, likes, comments, reach "
        "FROM posts ORDER BY posted_at DESC LIMIT 12"
    ).fetchall()]

    # Action log tail
    action_tail = [dict(r) for r in conn.execute(
        "SELECT action, target, result, at FROM action_log ORDER BY id DESC LIMIT 25"
    ).fetchall()]

    # Engagement queue depth
    eng_pending = conn.execute(
        "SELECT COUNT(*) c FROM engagement_queue WHERE status='pending'"
    ).fetchone()["c"]

    # Latest health + backoff
    health = db.health_latest()
    backoff_until = db.state_get("backoff_until")
    backoff_reason = db.state_get("backoff_reason")

    return {
        "niche": cfg.niche,
        "audience": cfg.target_audience,
        "commercial": cfg.commercial,
        "queue_by_status": queue_by_status,
        "queue_total": sum(queue_by_status.values()),
        "upcoming": upcoming,
        "posts": posts,
        "action_tail": action_tail,
        "eng_pending": eng_pending,
        "health": health,
        "backoff_until": backoff_until,
        "backoff_reason": backoff_reason,
        "warmup": {
            "day": warmup.day,
            "phase": warmup.phase_label,
            "multiplier": warmup.multiplier,
            "caps": warmup.caps,
            "allow_posts": warmup.allow_posts,
            "allow_dms": warmup.allow_dms,
        },
        "formats": cfg.formats.normalized(),
        "stories": cfg.stories.normalized(),
        "providers": providers_configured(),
    }


def _media_preview_url(paths: list[str]) -> str | None:
    """Convert an absolute media path into a /media/... URL for the dashboard."""
    if not paths:
        return None
    p = Path(paths[0])
    media_root = (ROOT / "data" / "media").resolve()
    try:
        rel = p.resolve().relative_to(media_root)
    except (ValueError, OSError):
        return None
    return f"/media/{rel.as_posix()}"
