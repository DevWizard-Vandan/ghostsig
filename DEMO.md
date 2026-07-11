# 👻 GhostSig — Product Demo Walkthrough

This guide walks you through GhostSig in 5 steps. No ML or engineering background required.

---

## Step 1 — Start the Stack

Open PowerShell in the project root and run:

```powershell
.\start.ps1
```

This script:
1. Checks that Docker Desktop is running (retries for up to 30 seconds)
2. Starts the infrastructure containers (database, message broker, cache)
3. Opens two new windows — one for the API server, one for the dashboard

When you see:
```
GhostSig running — API: http://localhost:8000/docs | Dashboard: http://localhost:8501
```
you are ready.

> **Prerequisite:** Docker Desktop must be installed and running. Python 3.11+ and the virtual environment must be set up (see [README.md](README.md#quick-start)).

---

## Step 2 — Open the Analyst Dashboard

Open your browser and go to: **http://localhost:8501**

You will see the **GhostSig** dashboard with a sidebar on the left containing:

- **API URL** — leave as `http://localhost:8000`
- **API Key** — leave as `ghostsig-dev-key`
- **Confidence threshold** slider — start at `0.0` to see all campaigns
- **Platform** multiselect — leave blank to see all platforms
- **Tier** selector — try `HIGH` to filter for confirmed campaigns

Click the **🔄 Refresh** button if the page shows no data.

---

## Step 3 — Review the Campaign Overview

The default page is **Campaign Overview**. At the top you will see 4 metric cards:

| Card | What it means |
|------|--------------|
| **Total Campaigns** | All detected coordinated networks |
| **HIGH tier** | Campaigns with confidence > 0.9 |
| **Active Operators** | Distinct operator IDs linking campaigns |
| **Accounts Monitored** | Total accounts in the fingerprint database |

Below the cards is a **campaign table** showing each network with its confidence score, tier (`HIGH` / `REVIEW` / `LIKELY_FP`), account count, and platforms.

**Select a campaign row** to expand the detail panel below the table. You will see:
- The full evidence card (JSON viewer)
- A table of the top flagged accounts in the network
- A button to download the PDF evidence report

---

## Step 4 — Open an Evidence Card and Download the PDF

With a campaign selected and the detail panel expanded:

1. The **JSON evidence card** loads automatically and shows:
   - `confidence` score and tier
   - `timing_overlap_pct` — how uniformly timed the network's posts are
   - `entropy_overlap_pct` — how similar the linguistic patterns are across accounts
   - `device_echo_markers` — shared device fingerprints across the network
   - `top_accounts` — the accounts most characteristic of the campaign

2. Click **📄 Download PDF Report** to save the evidence card as a PDF.

The PDF is suitable for sharing with a Trust & Safety team, regulatory body, or investigation report. It includes the campaign summary, behavioral signal breakdown, and a table of flagged accounts.

> Sample evidence cards are also available statically in [docs/evidence/](docs/evidence/).

---

## Step 5 — Check the API and Test /screen

Open: **http://localhost:8000/docs**

This is the interactive Swagger UI for the GhostSig REST API. All requests require the header `X-API-Key: ghostsig-dev-key`.

**Try the `/screen` endpoint:**

1. Click on `POST /screen` → **Try it out**
2. In the request body, enter a list of account IDs you want to check:
   ```json
   {
     "account_ids": ["twitter:embed_test_user_0", "twitter:embed_test_user_1"]
   }
   ```
3. Click **Execute**
4. The response will tell you which accounts (if any) are members of a known campaign, the campaign ID, confidence score, and tier.

**Other endpoints worth exploring:**

- `GET /campaigns` — list all detected campaigns with filters
- `GET /operators` — see how campaigns are grouped into operator networks
- `GET /campaigns/{id}/evidence` — structured JSON for any campaign
- `GET /campaigns/{id}/pdf` — download the PDF programmatically

---

## What to Look for in a Demo

| Signal | What it indicates |
|--------|------------------|
| Confidence ≥ 0.99 | Near-certain coordinated network |
| `entropy_overlap_pct = 1.0` | All accounts share near-identical linguistic patterns |
| `timing_overlap_pct > 0.7` | 70%+ of accounts post on the same cadence |
| `device_echo_markers ≥ 2` | At least 2 device fingerprints shared across ≥20% of accounts |

---

## Interactive UMAP Visualization

For a visual look at the behavioral clusters, open:

**[docs/umap_clusters.html](docs/umap_clusters.html)**

This is an interactive Plotly chart showing every account fingerprinted in 2D space. Bot clusters appear as tight, high-density groupings; organic accounts are spread loosely. Hover over any point to see the account ID, platform, and cluster label.

---

*For the technical implementation details, see [walkthrough.md](walkthrough.md).*
