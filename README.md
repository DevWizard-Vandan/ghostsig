# 👻 GhostSig

> **Behavioral Fingerprinting for Covert Coordination Detection**

GhostSig is a passive, open-source intelligence platform that detects **Coordinated Inauthentic Behavior (CIB)** networks by fingerprinting public behavioral metadata — timing cadence, cross-platform rhythm, linguistic entropy, and device-echo signals — **without ever accessing private content, user PII, or authenticated APIs.**

---

## 🧠 Core Innovation

| Layer | Traditional Approach | GhostSig Approach |
|---|---|---|
| **Data** | Scrapes posts, images, comments | Public metadata only: timestamps, handles, entropy |
| **Signal** | Content similarity, hashtag co-occurrence | Behavioral fingerprint: timing, burst, entropy drift |
| **Model** | Supervised (needs labeled CIB) | Unsupervised + self-supervised contrastive learning |
| **Attribution** | Account-level | Campaign-level: 500+ accounts → one operator |
| **Explainability** | Black-box scores | Source-attributed evidence cards |

---

## 🏗️ Architecture

```
Ingestion → Normalization → Feature Engineering
                                    ↓
          Behavioral Fingerprint Engine
    [Temporal Encoder | Entropy Encoder | Device Echo]
                                    ↓
                        Fusion Encoder (Transformer)
                                    ↓
              Clustering & Attribution Layer (HDBSCAN)
                                    ↓
                        API + Analyst Dashboard
```

---

## 📦 Data Sources (All Free, Public, No Auth)

- [GDELT 2.0](https://www.gdeltproject.org/) — global event streams, 15-min updates
- Twitter/X public timelines — user_timeline (public accounts only)
- Reddit public JSON — `/.json` endpoints
- Telegram public channels — `@channel` public views
- YouTube public comments — commentThreads API
- GitHub public events API
- Certificate Transparency logs — crt.sh
- Passive DNS — SecurityTrails community, CIRCL

---

## 🚀 MVP Roadmap (8 Weeks)

| Week | Milestone |
|------|----------|
| 1 | Repo, infra, GDELT + Twitter + Reddit collectors |
| 2 | Normalization + feature extraction (temporal, entropy) |
| 3 | Temporal encoder (Point Process Transformer) |
| 4 | Linguistic entropy encoder |
| 5 | Fusion encoder + HDBSCAN clustering |
| 6 | Adversarial validation (organic vs synthetic) |
| 7 | Campaign attribution + evidence packaging |
| 8 | FastAPI + Streamlit analyst dashboard |

---

## 🛠️ Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+, TypeScript |
| ML | PyTorch, HuggingFace, sentence-transformers |
| Clustering | HDBSCAN, UMAP, FAISS |
| Streaming | Apache Kafka (KRaft) / Redpanda |
| API | FastAPI + Pydantic v2 |
| DB | PostgreSQL (Neon free) + Redis (Upstash free) |
| Vector Search | pgvector / Qdrant |
| Frontend | React 18 + Vite + Tailwind + React Flow |
| Deployment | GitHub Actions → Render / Fly.io |

---

## 💰 Cost to Build MVP

**Total cash outlay: ~₹2,400 (< $30)**

All compute, data, ML frameworks, storage, and hosting are free tier.

---

## ⚖️ Legal & Compliance

GhostSig is **compliant by architecture**:
- No private content access
- No PII collection
- No authenticated API bypass
- Compatible with DPDP Act (India), GDPR (EU), CCPA (US)

---

## 📋 Patent Status

3 provisional claims (PCT path defined):
1. CIB detection via public behavioral metadata (timing, rhythm, entropy, device echoes)
2. Cross-platform behavioral fingerprinting via unsupervised contrastive learning
3. Campaign attribution engine linking accounts to operator via shared generative parameters

---

## 👥 Team

3–4 students: ML, Backend, Frontend, OSINT/Data

---

*GhostSig sees the coordination in the silence between posts.*
