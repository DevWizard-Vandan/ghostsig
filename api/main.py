"""GhostSig FastAPI – full REST API with auth, Pydantic v2 models, 10 endpoints."""

import os
import tempfile
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

import psycopg
from fastapi import FastAPI, Depends, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ghostsig.api")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")
API_KEY = os.getenv("GHOSTSIG_API_KEY", "ghostsig-dev-key")

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------
def get_db():
    conn = psycopg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pydantic v2 response models
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    accounts_count: int
    campaigns_count: int


class CampaignSummary(BaseModel):
    campaign_id: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    tier: Optional[str] = None
    account_count: Optional[int] = None
    platform_count: Optional[int] = None
    platform_list: Optional[list[str]] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    operator_id: Optional[str] = None


class MemberAccount(BaseModel):
    account_id: str
    similarity: Optional[float] = None


class CampaignDetail(CampaignSummary):
    evidence_json: Optional[dict] = None
    detected_at: Optional[str] = None
    member_accounts: list[MemberAccount] = Field(default_factory=list)


class AccountFingerprint(BaseModel):
    account_id: str
    platform: Optional[str] = None
    mean_interval_sec: Optional[float] = None
    std_interval_sec: Optional[float] = None
    coefficient_of_variation: Optional[float] = None
    event_count: Optional[int] = None
    char_entropy_mean: Optional[float] = None
    char_entropy_std: Optional[float] = None
    word_entropy_mean: Optional[float] = None
    word_entropy_std: Optional[float] = None
    entropy_sample_count: Optional[int] = None
    operator_id: Optional[str] = None
    campaign_id: Optional[str] = None
    confidence: Optional[float] = None
    tier: Optional[str] = None


class ScreenRequest(BaseModel):
    account_ids: list[str]


class ScreenResult(BaseModel):
    account_id: str
    campaign_id: Optional[str] = None
    confidence: Optional[float] = None
    tier: Optional[str] = None


class OperatorSummary(BaseModel):
    operator_id: str
    campaign_count: int


class PaginatedResponse(BaseModel):
    data: list = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tier(confidence: Optional[float]) -> str:
    if confidence is None:
        return "UNKNOWN"
    if confidence > 0.7:
        return "HIGH"
    if confidence >= 0.4:
        return "REVIEW"
    return "LIKELY_FP"


def _iso(dt) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GhostSig API",
    description="Behavioral fingerprinting for coordinated inauthentic behavior detection.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health(conn=Depends(get_db)):
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM account_fingerprints;")
        accounts_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM campaigns;")
        campaigns_count = cur.fetchone()[0]
    return HealthResponse(
        status="ok",
        version="0.2.0",
        accounts_count=accounts_count,
        campaigns_count=campaigns_count,
    )


@app.get("/campaigns", response_model=PaginatedResponse)
async def list_campaigns(
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    platform: Optional[str] = Query(None),
    tier: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _key: str = Depends(verify_api_key),
    conn=Depends(get_db),
):
    conditions = ["confidence >= %s"]
    params: list = [min_confidence]

    if platform:
        conditions.append("%s = ANY(platform_list)")
        params.append(platform)

    if tier:
        if tier == "HIGH":
            conditions.append("confidence > 0.7")
        elif tier == "REVIEW":
            conditions.append("confidence >= 0.4 AND confidence <= 0.7")
        elif tier == "LIKELY_FP":
            conditions.append("confidence < 0.4")

    where = " AND ".join(conditions)

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM campaigns WHERE {where};", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""
            SELECT campaign_id, label, confidence, account_count, platform_count,
                   platform_list, first_seen, last_seen, operator_id
            FROM campaigns
            WHERE {where}
            ORDER BY confidence DESC NULLS LAST
            LIMIT %s OFFSET %s;
            """,
            params + [limit, offset],
        )
        rows = cur.fetchall()

    data = []
    for r in rows:
        c_id, label, conf, acnt, pcnt, plist, fs, ls, oid = r
        data.append(
            CampaignSummary(
                campaign_id=str(c_id),
                label=label,
                confidence=conf,
                tier=_tier(conf),
                account_count=acnt,
                platform_count=pcnt,
                platform_list=plist or [],
                first_seen=_iso(fs),
                last_seen=_iso(ls),
                operator_id=oid,
            ).model_dump()
        )

    return PaginatedResponse(data=data, total=total, limit=limit, offset=offset)


@app.get("/campaigns/{campaign_id}", response_model=CampaignDetail)
async def get_campaign(
    campaign_id: str,
    _key: str = Depends(verify_api_key),
    conn=Depends(get_db),
):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT campaign_id, label, confidence, account_count, platform_count,
                   platform_list, first_seen, last_seen, operator_id, evidence_json, detected_at
            FROM campaigns WHERE campaign_id = %s;
            """,
            (campaign_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Campaign not found")

    c_id, label, conf, acnt, pcnt, plist, fs, ls, oid, ej, det = row

    with conn.cursor() as cur:
        cur.execute(
            "SELECT account_id, similarity FROM campaign_accounts WHERE campaign_id = %s;",
            (campaign_id,),
        )
        members = [MemberAccount(account_id=r[0], similarity=r[1]) for r in cur.fetchall()]

    return CampaignDetail(
        campaign_id=str(c_id),
        label=label,
        confidence=conf,
        tier=_tier(conf),
        account_count=acnt,
        platform_count=pcnt,
        platform_list=plist or [],
        first_seen=_iso(fs),
        last_seen=_iso(ls),
        operator_id=oid,
        evidence_json=ej,
        detected_at=_iso(det),
        member_accounts=members,
    )


@app.get("/campaigns/{campaign_id}/evidence")
async def get_evidence(
    campaign_id: str,
    _key: str = Depends(verify_api_key),
):
    from attribution.evidence_cards import generate_json

    try:
        payload = generate_json(campaign_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return payload


@app.get("/campaigns/{campaign_id}/pdf")
async def get_pdf(
    campaign_id: str,
    _key: str = Depends(verify_api_key),
):
    from attribution.evidence_cards import generate_pdf

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        generate_pdf(campaign_id, tmp_path)
    except ValueError as exc:
        os.unlink(tmp_path)
        raise HTTPException(status_code=404, detail=str(exc))

    return FileResponse(
        tmp_path,
        media_type="application/pdf",
        filename=f"campaign_{campaign_id}.pdf",
    )


@app.get("/accounts/{account_id}/fingerprint", response_model=AccountFingerprint)
async def get_fingerprint(
    account_id: str,
    _key: str = Depends(verify_api_key),
    conn=Depends(get_db),
):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.account_id, f.platform, f.mean_interval_sec, f.std_interval_sec,
                   f.coefficient_of_variation, f.event_count,
                   f.char_entropy_mean, f.char_entropy_std,
                   f.word_entropy_mean, f.word_entropy_std,
                   f.entropy_sample_count, f.operator_id,
                   ca.campaign_id, c.confidence
            FROM account_fingerprints f
            LEFT JOIN campaign_accounts ca ON f.account_id = ca.account_id
            LEFT JOIN campaigns c ON ca.campaign_id = c.campaign_id
            WHERE f.account_id = %s
            LIMIT 1;
            """,
            (account_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Account not found")

    (
        aid, plat, mi, si, cv, ec, cem, ces, wem, wes, esc, oid, cid, conf
    ) = row

    return AccountFingerprint(
        account_id=aid,
        platform=plat,
        mean_interval_sec=mi,
        std_interval_sec=si,
        coefficient_of_variation=cv,
        event_count=ec,
        char_entropy_mean=cem,
        char_entropy_std=ces,
        word_entropy_mean=wem,
        word_entropy_std=wes,
        entropy_sample_count=esc,
        operator_id=oid,
        campaign_id=str(cid) if cid else None,
        confidence=conf,
        tier=_tier(conf),
    )


@app.get("/fingerprints", response_model=PaginatedResponse)
async def list_fingerprints(
    platform: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _key: str = Depends(verify_api_key),
    conn=Depends(get_db),
):
    conditions = []
    params: list = []

    if platform:
        conditions.append("f.platform = %s")
        params.append(platform)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM account_fingerprints f {where};", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""
            SELECT f.account_id, f.platform, f.mean_interval_sec, f.event_count,
                   f.char_entropy_mean, f.operator_id
            FROM account_fingerprints f
            {where}
            ORDER BY f.account_id
            LIMIT %s OFFSET %s;
            """,
            params + [limit, offset],
        )
        rows = cur.fetchall()

    data = [
        {
            "account_id": r[0],
            "platform": r[1],
            "mean_interval_sec": r[2],
            "event_count": r[3],
            "char_entropy_mean": r[4],
            "operator_id": r[5],
        }
        for r in rows
    ]

    return PaginatedResponse(data=data, total=total, limit=limit, offset=offset)


@app.post("/screen", response_model=PaginatedResponse)
async def screen_accounts(
    body: ScreenRequest,
    _key: str = Depends(verify_api_key),
    conn=Depends(get_db),
):
    if not body.account_ids:
        return PaginatedResponse(data=[], total=0, limit=len(body.account_ids), offset=0)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.account_id, ca.campaign_id, c.confidence
            FROM account_fingerprints f
            LEFT JOIN campaign_accounts ca ON f.account_id = ca.account_id
            LEFT JOIN campaigns c ON ca.campaign_id = c.campaign_id
            WHERE f.account_id = ANY(%s);
            """,
            (body.account_ids,),
        )
        rows = cur.fetchall()

    data = []
    for aid, cid, conf in rows:
        data.append(
            ScreenResult(
                account_id=aid,
                campaign_id=str(cid) if cid else None,
                confidence=conf,
                tier=_tier(conf),
            ).model_dump()
        )

    return PaginatedResponse(
        data=data, total=len(data), limit=len(body.account_ids), offset=0
    )


@app.get("/operators", response_model=PaginatedResponse)
async def list_operators(
    _key: str = Depends(verify_api_key),
    conn=Depends(get_db),
):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT operator_id, count(*) as campaign_count
            FROM campaigns
            WHERE operator_id IS NOT NULL
            GROUP BY operator_id
            ORDER BY campaign_count DESC;
            """
        )
        rows = cur.fetchall()

    data = [
        OperatorSummary(operator_id=r[0], campaign_count=r[1]).model_dump()
        for r in rows
    ]

    return PaginatedResponse(data=data, total=len(data), limit=len(data), offset=0)


@app.get("/operators/{operator_id}/campaigns", response_model=PaginatedResponse)
async def get_operator_campaigns(
    operator_id: str,
    _key: str = Depends(verify_api_key),
    conn=Depends(get_db),
):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT campaign_id, label, confidence, account_count, platform_list,
                   first_seen, last_seen, operator_id
            FROM campaigns
            WHERE operator_id = %s
            ORDER BY confidence DESC NULLS LAST;
            """,
            (operator_id,),
        )
        rows = cur.fetchall()

    data = []
    for r in rows:
        c_id, label, conf, acnt, plist, fs, ls, oid = r
        data.append(
            CampaignSummary(
                campaign_id=str(c_id),
                label=label,
                confidence=conf,
                tier=_tier(conf),
                account_count=acnt,
                platform_list=plist or [],
                first_seen=_iso(fs),
                last_seen=_iso(ls),
                operator_id=oid,
            ).model_dump()
        )

    return PaginatedResponse(data=data, total=len(data), limit=len(data), offset=0)
