"""
Async MongoDB client via Motor.
Collections:
  - reports      : full report documents
  - source_cache : cached web / arxiv fetch results
"""
import os

from motor.motor_asyncio import AsyncIOMotorClient

_client: AsyncIOMotorClient | None = None
_db = None


async def init_db():
    global _client, _db
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "research_agent")
    _client = AsyncIOMotorClient(uri)
    _db = _client[db_name]

    # Indexes
    await _db.reports.create_index("query")
    await _db.reports.create_index("created_at")
    await _db.source_cache.create_index("url", unique=True)
    await _db.source_cache.create_index(
        "fetched_at", expireAfterSeconds=86400  # 24-hour TTL
    )
    print("✅  MongoDB connected and indexes ensured")


async def close_db():
    if _client:
        _client.close()


def get_db():
    if _db is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _db
