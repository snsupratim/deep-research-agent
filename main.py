"""
Research Agent — FastAPI entry point
Rate-limited, CORS-enabled, MongoDB-backed
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from db import close_db, init_db
from routers import reports, websocket

load_dotenv()

# ── Rate limiter (shared across routers) ────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="Deep Research Agent API",
    version="1.0.0",
    description="Autonomous multi-source intelligence report generator",
    lifespan=lifespan,
)

# ── Middleware ───────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(reports.router, prefix="/api/v1")
app.include_router(websocket.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "research-agent"}
