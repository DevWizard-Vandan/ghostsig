# GhostSig — Project Context for Claude Code

## One-Liner
Passive OSINT platform detecting Coordinated Inauthentic Behavior (CIB) by fingerprinting public behavioral metadata (timing, entropy, device echoes) — no content, no PII, no auth.

## Architecture (Memorize This Flow)
```
GDELT/Reddit/Twitter → Kafka (raw.events) 
  → Normalize + Features (temporal + entropy) → Parquet + Postgres (account_fingerprints)
  → TemporalEncoder (128-d) + EntropyEncoder (64-d) 
  → FusionEncoder (256-d) → HDBSCAN clusters 
  → Adversarial validation (confidence scores)
  → Operator attribution + Evidence cards
  → FastAPI + Streamlit Dashboard
```

## Key Files You'll Touch
| Layer | Files |
|---|---|
| Infra | `docker-compose.yml`, `infra/postgres/init.sql`, `Makefile` |
| Collectors | `collectors/gdelt_collector.py`, `reddit_collector.py`, `twitter_collector.py` |
| Features | `features/temporal.py`, `features/entropy.py` |
| ML Models | `ml/temporal_encoder.py`, `ml/entropy_encoder.py`, `ml/fusion_encoder.py`, `ml/clustering.py` |
| Pipeline | `pipeline/normalize.py`, `pipeline/full_pipeline.py`, `consumers/raw_to_postgres.py` |
| Training | `ml/train_temporal.py`, `ml/train_entropy.py`, `ml/embed_temporal.py` |
| Attribution | `attribution/operator_linker.py`, `attribution/evidence_cards.py` |
| API | `api/main.py` |
| Dashboard | `dashboard/app.py` |
| Tests | `tests/test_features.py` |

## Tech Stack (Frozen)
- **Python 3.11**, PyTorch 2.3, HuggingFace Transformers
- **Redpanda** (KRaft mode) — not Kafka+Zookeeper
- **Postgres 16 + pgvector** — embeddings stored as `vector(N)`
- **FastAPI** + Pydantic v2, **Streamlit** for dashboard
- **HDBSCAN + UMAP** for clustering
- **XGBoost** for adversarial classifier
- **Render** (API), **Streamlit Cloud** (dashboard) — free tiers

## Constraints
- **Zero cash spend** — only free tiers
- **No private data** — only public, unauthenticated endpoints
- **No new deps without approval** — keep requirements.txt lean
- **Write tests** for every new module
- **Update CLAUDE.md** if architecture changes

## Current Sprint: Week 1 (Infra + Collectors)
Goal: `make up` → all 3 collectors writing to Redpanda `raw.events` → consumer inserting into `raw_events` table.

## Patent Claims (Already Defined — Do Not Change)
1. CIB detection via public behavioral metadata (timing, rhythm, entropy, device echoes) without content access
2. Cross-platform behavioral fingerprinting via unsupervised contrastive learning on public metadata streams
3. Campaign attribution engine linking accounts to operator via shared generative behavioral parameters

## Team Context
- Solo founder (Vandan), CS/AI student, Rust/Python/TS full-stack
- Background: Ritam (multi-agent alpha), WhisperNet (covert signals), WorldQuant, OpenAI Parameter Golf (global rank 20)
- Naming convention: Sanskrit — GhostSig = Gupt (hidden) + Signature

## Success Criteria (MVP)
- Detect ≥3 known CIB campaigns from public test sets
- False positive rate < 15% on organic accounts
- End-to-end latency < 30 min ingestion → cluster
- Analyst explains flagged cluster in < 2 min via UI

---
