"""GhostSig FastAPI entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="GhostSig API",
    description="Behavioral fingerprinting for coordinated inauthentic behavior detection.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ghostsig"}


@app.get("/campaigns")
async def list_campaigns():
    """List all detected CIB campaigns."""
    # TODO: query campaigns table
    return {"campaigns": [], "total": 0}


@app.get("/accounts/{account_id}/fingerprint")
async def get_fingerprint(account_id: str):
    """Get behavioral fingerprint for a specific account."""
    # TODO: query account_fingerprints table
    return {"account_id": account_id, "fingerprint": None}


@app.get("/fingerprints")
async def list_fingerprints():
    """List all computed behavioral fingerprints."""
    # TODO: paginated query
    return {"fingerprints": [], "total": 0}
