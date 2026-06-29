-- GhostSig Database Schema
-- Normalized event table: one row per public post/event

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS raw_events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id      TEXT NOT NULL,          -- platform-namespaced: "twitter:user123"
    platform        TEXT NOT NULL,          -- twitter | reddit | gdelt | telegram | youtube | github
    event_type      TEXT NOT NULL,          -- post | comment | retweet | commit | upload
    event_ts        TIMESTAMPTZ NOT NULL,   -- UTC timestamp of the event
    text_entropy    FLOAT,                  -- character-level entropy of public text
    word_entropy    FLOAT,                  -- word-level entropy
    device_hint     TEXT,                   -- public user-agent / platform header (if available)
    metadata        JSONB,                  -- raw public metadata (no PII)
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_raw_events_platform_account_ts UNIQUE (platform, account_id, event_ts)
);

CREATE INDEX idx_raw_events_account_id ON raw_events(account_id);
CREATE INDEX idx_raw_events_event_ts   ON raw_events(event_ts);
CREATE INDEX idx_raw_events_platform   ON raw_events(platform);

-- Account-level behavioral fingerprints
CREATE TABLE IF NOT EXISTS account_fingerprints (
    account_id                  TEXT PRIMARY KEY,
    platform                    TEXT NOT NULL,
    temporal_embedding          vector(128),        -- output of temporal encoder
    entropy_embedding           vector(64),         -- output of entropy encoder
    fused_embedding             vector(256),        -- output of fusion encoder
    operator_id                 TEXT,               -- ID/hash of operator link
    device_echo_hash            TEXT,               -- JA3 / header fingerprint hash
    burst_periodicity           FLOAT,              -- dominant burst frequency (Hz)
    inter_event_mean            FLOAT,              -- mean inter-post interval (seconds)
    inter_event_std             FLOAT,
    mean_interval_sec           FLOAT,
    std_interval_sec            FLOAT,
    coefficient_of_variation    FLOAT,
    burst_freqs_hz              FLOAT[],
    event_count                 INT,
    char_entropy_mean           FLOAT,
    char_entropy_std            FLOAT,
    word_entropy_mean           FLOAT,
    word_entropy_std            FLOAT,
    entropy_sample_count        INT,
    last_updated                TIMESTAMPTZ DEFAULT NOW()
);

-- Detected CIB campaigns
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    label               TEXT,               -- cluster label or analyst-assigned name
    confidence          FLOAT,              -- 0.0 – 1.0
    account_count       INT,
    platform_count      INT,
    platform_list       TEXT[],             -- array of platforms present in campaign
    first_seen          TIMESTAMPTZ,        -- earliest activity
    last_seen           TIMESTAMPTZ,        -- latest activity
    detected_at         TIMESTAMPTZ DEFAULT NOW(),
    operator_id         TEXT,               -- operator attribution link
    evidence_json       JSONB               -- fingerprint overlap stats, shared markers
);

-- Campaign membership: account → campaign
CREATE TABLE IF NOT EXISTS campaign_accounts (
    campaign_id     UUID REFERENCES campaigns(campaign_id),
    account_id      TEXT,
    similarity      FLOAT,
    PRIMARY KEY (campaign_id, account_id)
);
