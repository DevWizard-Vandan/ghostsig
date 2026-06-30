"""GhostSig — Analyst Dashboard for Coordinated Inauthentic Behavior Detection.

Four-page Streamlit application consuming the GhostSig FastAPI REST API.
Pages:
  1. Campaign Overview  – metrics, campaign table, evidence drill-down
  2. Account Search     – fingerprint lookup for individual accounts
  3. Operator Network   – pyvis graph of operator → campaign links
  4. Live Ingest        – trigger the full pipeline and view stdout
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

# ── Page config (MUST be the very first Streamlit call) ──────────────────────
st.set_page_config(page_title="GhostSig", layout="wide", page_icon="👻")

# ── Constants ────────────────────────────────────────────────────────────────
PLATFORMS = ["twitter", "reddit", "telegram", "youtube", "github", "gdelt"]
TIERS = ["All", "HIGH", "REVIEW", "LIKELY_FP"]
TIER_COLORS = {"HIGH": "#e74c3c", "REVIEW": "#f39c12", "LIKELY_FP": "#95a5a6"}
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_API_KEY = "ghostsig-dev-key"
REQUEST_TIMEOUT = 30.0


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[str, str, float, list[str], str]:
    """Render the shared sidebar and return user-selected config values."""
    with st.sidebar:
        st.markdown("# 👻 GhostSig")
        st.caption("Behavioral CIB Detection")
        st.divider()

        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")
        confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.0, 0.01)
        platforms = st.multiselect("Platforms", options=PLATFORMS, default=[])
        tier = st.selectbox("Tier", options=TIERS, index=0)

        st.divider()
        if st.button("🔄 Refresh", width='stretch'):
            st.cache_data.clear()
        st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

        st.divider()
        page = st.radio(
            "Navigation",
            ["📊 Campaign Overview", "🔍 Account Search",
             "🕸️ Operator Network", "⚡ Live Ingest"],
            label_visibility="collapsed",
        )

    return api_url, api_key, confidence, platforms, tier, page  # type: ignore[return-value]


# ── API helpers ──────────────────────────────────────────────────────────────
def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key, "Accept": "application/json"}


def api_get(
    base_url: str,
    api_key: str,
    path: str,
    params: dict | None = None,
    *,
    accept: str | None = None,
) -> httpx.Response:
    """Issue a GET request and return the raw httpx.Response."""
    headers = _headers(api_key)
    if accept:
        headers["Accept"] = accept
    resp = httpx.get(
        f"{base_url.rstrip('/')}{path}",
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp


def api_post(
    base_url: str,
    api_key: str,
    path: str,
    json_data: dict,
) -> httpx.Response:
    """Issue a POST request and return the raw httpx.Response."""
    resp = httpx.post(
        f"{base_url.rstrip('/')}{path}",
        headers=_headers(api_key),
        json=json_data,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp


# ── Cached data-fetching wrappers ───────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def fetch_health(base_url: str, api_key: str) -> dict:
    return api_get(base_url, api_key, "/health").json()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_campaigns(
    base_url: str,
    api_key: str,
    min_confidence: float = 0.0,
    platform: str | None = None,
    tier: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    params: dict = {"min_confidence": min_confidence, "limit": limit, "offset": offset}
    if platform:
        params["platform"] = platform
    if tier and tier != "All":
        params["tier"] = tier
    return api_get(base_url, api_key, "/campaigns", params).json()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_campaign_detail(base_url: str, api_key: str, campaign_id: str) -> dict:
    return api_get(base_url, api_key, f"/campaigns/{campaign_id}").json()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_evidence(base_url: str, api_key: str, campaign_id: str) -> dict:
    return api_get(base_url, api_key, f"/campaigns/{campaign_id}/evidence").json()


def fetch_pdf(base_url: str, api_key: str, campaign_id: str) -> bytes:
    """Fetch campaign PDF report (not cached — binary payload)."""
    resp = api_get(
        base_url, api_key,
        f"/campaigns/{campaign_id}/pdf",
        accept="application/pdf",
    )
    return resp.content


@st.cache_data(ttl=30, show_spinner=False)
def fetch_fingerprint(base_url: str, api_key: str, account_id: str) -> dict:
    return api_get(base_url, api_key, f"/accounts/{account_id}/fingerprint").json()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_operators(base_url: str, api_key: str) -> dict:
    return api_get(base_url, api_key, "/operators").json()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_operator_campaigns(base_url: str, api_key: str, operator_id: str) -> dict:
    return api_get(base_url, api_key, f"/operators/{operator_id}/campaigns").json()


@st.cache_data(ttl=60, show_spinner=False)
def screen_accounts(base_url: str, api_key: str, account_ids: list[str]) -> dict:
    return api_post(base_url, api_key, "/screen", {"account_ids": account_ids}).json()


# ── Page 1: Campaign Overview ───────────────────────────────────────────────
def page_campaign_overview(
    base_url: str, api_key: str, confidence: float,
    platforms: list[str], tier: str,
) -> None:
    st.header("📊 Campaign Overview")

    # ── Top metric cards ─────────────────────────────────────────────────
    try:
        health = fetch_health(base_url, api_key)
    except Exception as exc:
        st.error(f"Failed to reach API /health: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Campaigns", health.get("campaigns_count", "—"))
    c2.metric("🔴 HIGH", _count_by_tier(base_url, api_key, "HIGH"))
    c3.metric("Active Operators", _count_operators(base_url, api_key))
    c4.metric("Accounts Monitored", health.get("accounts_count", "—"))

    st.divider()

    # ── Campaign table ───────────────────────────────────────────────────
    platform_param = ",".join(platforms) if platforms else None
    try:
        campaigns_resp = fetch_campaigns(
            base_url, api_key,
            min_confidence=confidence,
            platform=platform_param,
            tier=tier,
        )
    except Exception as exc:
        st.error(f"Failed to fetch campaigns: {exc}")
        return

    rows = campaigns_resp.get("data", [])
    if not rows:
        st.info("No campaigns match the current filters.")
        return

    display_cols = [
        "label", "confidence", "tier", "account_count",
        "platforms", "first_seen", "last_seen",
    ]
    df = pd.DataFrame(rows)
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].sort_values("confidence", ascending=False)
        if "confidence" in df.columns else df[available],
        width='stretch',
        hide_index=True,
    )

    # ── Campaign drill-down ──────────────────────────────────────────────
    campaign_labels = {
        r.get("campaign_id", r.get("id", str(i))): r.get("label", f"Campaign {i}")
        for i, r in enumerate(rows)
    }
    selected_id = st.selectbox(
        "Select a campaign for details",
        options=list(campaign_labels.keys()),
        format_func=lambda cid: campaign_labels.get(cid, cid),
    )

    if selected_id:
        with st.expander(f"Evidence & details — {campaign_labels[selected_id]}", expanded=True):
            _render_campaign_detail(base_url, api_key, selected_id)


def _count_by_tier(base_url: str, api_key: str, tier: str) -> int | str:
    try:
        resp = fetch_campaigns(base_url, api_key, tier=tier, limit=1)
        return resp.get("total", "—")
    except Exception:
        return "—"


def _count_operators(base_url: str, api_key: str) -> int | str:
    try:
        resp = fetch_operators(base_url, api_key)
        return resp.get("total", "—")
    except Exception:
        return "—"


def _render_campaign_detail(base_url: str, api_key: str, campaign_id: str) -> None:
    """Show evidence JSON, PDF download, and top accounts for a campaign."""
    col_evidence, col_pdf = st.columns([3, 1])

    # Evidence JSON
    with col_evidence:
        st.subheader("Evidence Card")
        try:
            evidence = fetch_evidence(base_url, api_key, campaign_id)
            st.json(evidence)

            # Top accounts table (extracted from evidence)
            accounts = evidence.get("top_accounts") or evidence.get("member_accounts", [])
            if accounts:
                st.subheader("Top Accounts")
                st.dataframe(pd.DataFrame(accounts), width='stretch', hide_index=True)
        except Exception as exc:
            st.warning(f"Could not load evidence: {exc}")

    # PDF download
    with col_pdf:
        st.subheader("Report")
        try:
            pdf_bytes = fetch_pdf(base_url, api_key, campaign_id)
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"ghostsig_campaign_{campaign_id}.pdf",
                mime="application/pdf",
                width='stretch',
            )
        except Exception as exc:
            st.warning(f"PDF not available: {exc}")


# ── Page 2: Account Search ──────────────────────────────────────────────────
def page_account_search(base_url: str, api_key: str) -> None:
    st.header("🔍 Account Search")
    st.caption("Look up the behavioral fingerprint for any monitored account.")

    account_id = st.text_input("Account ID", placeholder="e.g. twitter:1234567890")

    if not account_id:
        st.info("Enter an account ID above to view its fingerprint.")
        return

    try:
        fp = fetch_fingerprint(base_url, api_key, account_id.strip())
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            st.warning("Account not found in the fingerprint database.")
        else:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        return
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return

    st.divider()

    # ── Temporal stats ───────────────────────────────────────────────────
    temporal = fp.get("temporal", fp.get("temporal_stats", {}))
    entropy = fp.get("entropy", fp.get("entropy_stats", {}))
    cluster = fp.get("cluster", fp.get("cluster_membership", {}))

    st.subheader("Temporal Stats")
    if temporal:
        cols = st.columns(min(len(temporal), 4))
        for i, (key, val) in enumerate(temporal.items()):
            cols[i % len(cols)].metric(key.replace("_", " ").title(), _fmt(val))
    else:
        st.caption("No temporal data available.")

    st.subheader("Entropy Stats")
    if entropy:
        cols = st.columns(min(len(entropy), 4))
        for i, (key, val) in enumerate(entropy.items()):
            cols[i % len(cols)].metric(key.replace("_", " ").title(), _fmt(val))
    else:
        st.caption("No entropy data available.")

    st.subheader("Cluster Membership")
    if cluster:
        if isinstance(cluster, dict):
            cols = st.columns(min(len(cluster), 4))
            for i, (key, val) in enumerate(cluster.items()):
                cols[i % len(cols)].metric(key.replace("_", " ").title(), _fmt(val))
        else:
            st.write(cluster)
    else:
        st.caption("No cluster data available.")

    # Raw JSON fallback
    with st.expander("Raw fingerprint JSON"):
        st.json(fp)


def _fmt(val: object) -> str:
    """Format a metric value for display."""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


# ── Page 3: Operator Network ────────────────────────────────────────────────
def page_operator_network(base_url: str, api_key: str) -> None:
    st.header("🕸️ Operator Network")
    st.caption("Visualize links between suspected operators and their campaigns.")

    try:
        operators_resp = fetch_operators(base_url, api_key)
    except Exception as exc:
        st.error(f"Failed to fetch operators: {exc}")
        return

    operators = operators_resp.get("data", [])
    if not operators:
        st.info("No operators detected yet. Run the pipeline to discover operator clusters.")
        return

    # ── Build pyvis network ──────────────────────────────────────────────
    try:
        from pyvis.network import Network  # noqa: WPS433
    except ImportError:
        st.error("pyvis is not installed. Run `pip install pyvis` to enable network graphs.")
        return

    net = Network(
        height="500px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        directed=True,
        notebook=False,
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    for op in operators:
        op_id = op.get("operator_id", "unknown")
        campaign_count = op.get("campaign_count", 0)
        net.add_node(
            f"op:{op_id}",
            label=f"Operator {op_id}\n({campaign_count} campaigns)",
            color="#2ecc71",
            size=30,
            shape="dot",
            title=f"Operator {op_id}",
        )

        # Fetch campaigns for this operator
        try:
            op_campaigns = fetch_operator_campaigns(base_url, api_key, op_id)
            for camp in op_campaigns.get("data", []):
                camp_id = camp.get("campaign_id", camp.get("id", "?"))
                camp_tier = camp.get("tier", "REVIEW")
                color = TIER_COLORS.get(camp_tier, "#3498db")
                net.add_node(
                    f"camp:{camp_id}",
                    label=camp.get("label", camp_id),
                    color=color,
                    size=15,
                    shape="dot",
                    title=f"Tier: {camp_tier}  Confidence: {camp.get('confidence', '—')}",
                )
                net.add_edge(f"op:{op_id}", f"camp:{camp_id}")
        except Exception:
            pass  # silently skip on error — node still visible

    # ── Render to HTML and embed ─────────────────────────────────────────
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".html", mode="w", encoding="utf-8",
    ) as tmp:
        net.save_graph(tmp.name)
        html_content = Path(tmp.name).read_text(encoding="utf-8")

    import streamlit.components.v1 as components  # noqa: WPS433

    components.html(html_content, height=520, scrolling=True)

    # Legend
    st.markdown(
        "**Legend:** "
        "🟢 Operator node  &nbsp;·&nbsp; "
        "🔴 HIGH tier  &nbsp;·&nbsp; "
        "🟡 REVIEW tier  &nbsp;·&nbsp; "
        "⚪ LIKELY_FP tier"
    )


# ── Page 4: Live Ingest ─────────────────────────────────────────────────────
def page_live_ingest(base_url: str, api_key: str) -> None:
    st.header("⚡ Live Ingest")
    st.caption("Trigger the GhostSig pipeline to ingest new data and refresh detections.")

    st.warning(
        "This will run the full pipeline in a subprocess. "
        "Ensure collectors and infrastructure are running."
    )

    if st.button("▶️ Run Full Pipeline", type="primary", width='stretch'):
        st.divider()
        with st.spinner("Running pipeline… this may take a few minutes."):
            try:
                result = subprocess.run(
                    ["python", "-m", "pipeline.full_pipeline", "--once", "--skip-training"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=str(Path(__file__).resolve().parent.parent),
                )
                st.subheader("Pipeline Output")
                if result.stdout:
                    st.code(result.stdout, language="text")
                if result.stderr:
                    with st.expander("stderr", expanded=False):
                        st.code(result.stderr, language="text")
                if result.returncode != 0:
                    st.error(f"Pipeline exited with code {result.returncode}")
                else:
                    st.success("Pipeline completed successfully.")
            except subprocess.TimeoutExpired:
                st.error("Pipeline timed out after 10 minutes.")
            except FileNotFoundError:
                st.error(
                    "Could not find `pipeline.full_pipeline`. "
                    "Make sure the pipeline module exists."
                )
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

        # Refresh campaign count after pipeline run
        st.divider()
        try:
            health = fetch_health.__wrapped__(base_url, api_key)  # bypass cache
            st.metric("Updated Campaign Count", health.get("campaigns_count", "—"))
        except Exception:
            st.info("Could not refresh campaign count.")


# ── Main router ──────────────────────────────────────────────────────────────
def main() -> None:
    api_url, api_key, confidence, platforms, tier, page = render_sidebar()

    if page == "📊 Campaign Overview":
        page_campaign_overview(api_url, api_key, confidence, platforms, tier)
    elif page == "🔍 Account Search":
        page_account_search(api_url, api_key)
    elif page == "🕸️ Operator Network":
        page_operator_network(api_url, api_key)
    elif page == "⚡ Live Ingest":
        page_live_ingest(api_url, api_key)


if __name__ == "__main__":
    main()
