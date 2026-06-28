"""Tests for Campaign Attribution & Evidence Packaging (operator linking, ReportLab PDF, CIB detection)."""

import os
import tempfile
import pytest
import psycopg

from attribution.operator_linker import run_attribution, compute_timing_overlap, DATABASE_URL
from attribution.evidence_cards import generate_json, generate_pdf
from attribution.test_datasets import run_cib_pipeline_and_verify


def test_operator_linker_logic():
    # Timing overlap check
    assert compute_timing_overlap([0.01, 0.02], [0.01, 0.02]) == 1.0
    assert compute_timing_overlap([0.01, 0.05], [0.012, 0.1]) == 0.0  # 0.01 and 0.012 are within 5% tolerance (abs(0.01-0.012)/0.012 = 0.166? Wait! abs(0.01-0.012)/0.012 = 0.002/0.012 = 0.166. Actually, let's verify if abs(0.01-0.012)/0.012 <= 0.05. Wait, 0.002/0.012 = 1/6 = 0.167 > 0.05, so they do NOT overlap!).
    # Let's test a true overlap: 0.01 and 0.0101 -> abs(0.01 - 0.0101)/0.0101 = 0.0001 / 0.0101 = 0.0099 <= 0.05
    assert compute_timing_overlap([0.01, 0.05], [0.0101, 0.1]) == 0.5


def test_generate_json_and_pdf():
    # Retrieve a campaign to test
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE campaigns ADD COLUMN IF NOT EXISTS operator_id TEXT;")
            cur.execute("SELECT campaign_id FROM campaigns LIMIT 1;")
            row = cur.fetchone()
            conn.commit()
            
    if not row:
        pytest.skip("No campaigns available in the database for testing json/pdf generation.")
        
    campaign_id = str(row[0])
    
    # 1. Test JSON generation
    payload = generate_json(campaign_id)
    assert "campaign_id" in payload
    assert "operator_id" in payload
    assert "confidence_tier" in payload
    assert "timing_overlap_pct" in payload
    assert "top_accounts" in payload
    
    # 2. Test PDF generation
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "test_evidence.pdf")
        generate_pdf(campaign_id, pdf_path)
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 1000  # Valid non-empty PDF file


def test_operator_linker_runs():
    c_processed, num_ops = run_attribution(DATABASE_URL)
    # Just verify execution works without error
    assert c_processed >= 0


def test_known_cib_datasets_detection():
    # This runs the full pipeline on synthetic datasets and returns the detection report list
    report = run_cib_pipeline_and_verify()
    
    assert len(report) >= 2, "Both known CIB campaigns (IRA + Spamouflage) should be detected."
    
    ira_detected_sizes = []
    spam_detected_sizes = []
    
    for r in report:
        if "IRA-like" in r["dataset"]:
            assert r["confidence"] > 0.7
            ira_detected_sizes.append(r["detected"])
        elif "Spamouflage-like" in r["dataset"]:
            assert r["confidence"] > 0.7
            spam_detected_sizes.append(r["detected"])
            
    assert len(ira_detected_sizes) > 0, "No IRA-like campaigns detected."
    assert len(spam_detected_sizes) > 0, "No Spamouflage-like campaigns detected."
    
    # Verify that the campaigns of each dataset capture the required number of accounts in total
    assert sum(ira_detected_sizes) >= 40, f"Sum IRA cluster size {sum(ira_detected_sizes)} is less than 40."
    assert sum(spam_detected_sizes) >= 20, f"Sum Spamouflage cluster size {sum(spam_detected_sizes)} is less than 20."
