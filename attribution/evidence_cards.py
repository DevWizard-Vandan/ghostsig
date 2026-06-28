"""Script to package campaign evidence into JSON and ReportLab PDF evidence cards."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
import numpy as np
import xgboost as xgb
import psycopg

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evidence_cards")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")
CHECKPOINT_PATH = "checkpoints/adversarial_clf_v1.json"


def generate_json(campaign_id: str) -> dict:
    logger.info(f"Generating JSON evidence payload for campaign {campaign_id}...")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise e
        
    with conn.cursor() as cur:
        # Load campaign details
        cur.execute(
            """
            SELECT campaign_id, label, confidence, account_count, platform_list, 
                   first_seen, last_seen, operator_id, evidence_json
            FROM campaigns
            WHERE campaign_id = %s;
            """,
            (campaign_id,)
        )
        row = cur.fetchone()
        
    if not row:
        conn.close()
        raise ValueError(f"Campaign with ID {campaign_id} not found.")
        
    # Unpack campaign row
    c_id, label, confidence, account_count, platform_list, first_seen, last_seen, op_id, ev_json = row
    
    # Tier mapping
    confidence = confidence if confidence is not None else 0.0
    if confidence > 0.7:
        tier = "HIGH"
    elif confidence >= 0.4:
        tier = "REVIEW"
    else:
        tier = "LIKELY_FP"
        
    # Unpack evidence statistics
    timing_overlap = 0.0
    entropy_overlap = 0.0
    device_markers = 0
    cross_sync = 0
    
    if ev_json:
        timing_overlap = ev_json.get("timing_overlap_pct", 0.0)
        entropy_overlap = ev_json.get("entropy_overlap_pct", 0.0)
        device_markers = ev_json.get("device_echo_markers", 0)
        cross_sync = ev_json.get("cross_platform_sync", 0)

    # Load and score member accounts
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.account_id, f.platform, f.fused_embedding, f.char_entropy_mean, f.mean_interval_sec, f.event_count
            FROM campaign_accounts ca
            JOIN account_fingerprints f ON ca.account_id = f.account_id
            WHERE ca.campaign_id = %s AND f.fused_embedding IS NOT NULL;
            """,
            (campaign_id,)
        )
        member_rows = cur.fetchall()
        
    top_accounts = []
    
    # Load model if available for threat probability
    clf = None
    if os.path.exists(CHECKPOINT_PATH):
        try:
            clf = xgb.XGBClassifier()
            clf.load_model(CHECKPOINT_PATH)
        except Exception as e:
            logger.warning(f"Could not load classifier for threat probability: {e}")
            
    scored_members = []
    for acc_id, platform, fused_emb, char_ent_mean, mean_int, event_cnt in member_rows:
        prob = 0.5  # default fallback
        if clf is not None and fused_emb is not None:
            try:
                emb = np.array(fused_emb, dtype=np.float32).reshape(1, -1)
                prob = float(clf.predict_proba(emb)[0, 1])
            except Exception:
                pass
        scored_members.append({
            "account_id": acc_id,
            "platform": platform,
            "threat_prob": prob,
            "char_entropy_mean": float(char_ent_mean) if char_ent_mean is not None else 0.0,
            "mean_interval_sec": float(mean_int) if mean_int is not None else 0.0,
            "event_count": event_cnt
        })
        
    # Sort descending by threat probability, then select top 5
    scored_members.sort(key=lambda x: x["threat_prob"], reverse=True)
    top_accounts = scored_members[:5]
    
    conn.close()
    
    payload = {
        "campaign_id": str(c_id),
        "operator_id": op_id if op_id else "UNKNOWN",
        "account_count": account_count if account_count else 0,
        "platform_list": platform_list if platform_list else [],
        "confidence": confidence,
        "confidence_tier": tier,
        "timing_overlap_pct": timing_overlap,
        "entropy_overlap_pct": entropy_overlap,
        "device_echo_markers": device_markers,
        "cross_platform_sync": cross_sync,
        "first_seen": first_seen.isoformat() if first_seen else None,
        "last_seen": last_seen.isoformat() if last_seen else None,
        "top_accounts": top_accounts,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
    return payload


def make_progressbar(pct: float, fill_color: colors.Color) -> Table:
    # Beautiful progressbar built with reportlab nested tables
    # Outer width: 150pt
    fill_width = max(1.0, min(150.0, pct * 150.0))
    empty_width = 150.0 - fill_width
    
    bar_data = [['', '']]
    bar_style = TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), fill_color),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ])
    
    bar_tbl = Table(bar_data, colWidths=[fill_width, empty_width], rowHeights=[10], style=bar_style)
    return bar_tbl


def generate_pdf(campaign_id: str, output_path: str) -> str:
    logger.info(f"Generating PDF evidence card for campaign {campaign_id} to {output_path}...")
    data = generate_json(campaign_id)
    
    # Query linked campaigns count
    linked_campaigns_count = 1
    if data["operator_id"] != "UNKNOWN":
        try:
            conn = psycopg.connect(DATABASE_URL)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM campaigns WHERE operator_id = %s;",
                    (data["operator_id"],)
                )
                linked_campaigns_count = cur.fetchone()[0]
            conn.close()
        except Exception:
            pass

    # Document setup: exactly 1 page
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    style_logo = ParagraphStyle(
        'GhostSigLogo',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#0F172A'),
        leading=26
    )
    
    style_title = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=5,
        leading=14
    )
    
    style_body = ParagraphStyle(
        'NormalBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#334155'),
        leading=11
    )
    
    style_body_bold = ParagraphStyle(
        'NormalBodyBold',
        parent=style_body,
        fontName='Helvetica-Bold'
    )
    
    style_badge_text = ParagraphStyle(
        'BadgeText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.white,
        alignment=1,  # Centered
        leading=12
    )

    story = []
    
    # 1. Header Row
    logo_p = Paragraph("GHOSTSIG INTEL", style_logo)
    
    # Color badge based on tier
    if data["confidence_tier"] == "HIGH":
        badge_bg = colors.HexColor('#EF4444')
    elif data["confidence_tier"] == "REVIEW":
        badge_bg = colors.HexColor('#F59E0B')
    else:
        badge_bg = colors.HexColor('#10B981')
        
    badge_tbl = Table(
        [[Paragraph(f"{data['confidence_tier']} RISK", style_badge_text)]],
        colWidths=[110],
        rowHeights=[20],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), badge_bg),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ])
    )
    
    header_tbl = Table([[logo_p, badge_tbl]], colWidths=[380, 120])
    header_tbl.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(header_tbl)
    
    # Divider line
    divider = Table([['']], colWidths=[500], rowHeights=[2], style=TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#4F46E5')),
    ]))
    story.append(divider)
    story.append(Spacer(1, 10))
    
    # 2. Overview Section
    story.append(Paragraph("Campaign Overview", style_title))
    overview_data = [
        [
            Paragraph("Campaign ID:", style_body_bold),
            Paragraph(data["campaign_id"], style_body),
            Paragraph("Operator ID:", style_body_bold),
            Paragraph(data["operator_id"], style_body)
        ],
        [
            Paragraph("Account Count:", style_body_bold),
            Paragraph(str(data["account_count"]), style_body),
            Paragraph("Platforms:", style_body_bold),
            Paragraph(", ".join(data["platform_list"]), style_body)
        ],
        [
            Paragraph("First Seen:", style_body_bold),
            Paragraph(data["first_seen"][:19] if data["first_seen"] else "N/A", style_body),
            Paragraph("Last Seen:", style_body_bold),
            Paragraph(data["last_seen"][:19] if data["last_seen"] else "N/A", style_body)
        ]
    ]
    overview_tbl = Table(overview_data, colWidths=[90, 160, 90, 160])
    overview_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(overview_tbl)
    story.append(Spacer(1, 10))
    
    # 3. Behavioral Fingerprint Section
    story.append(Paragraph("Coordinated Behavioral Fingerprint", style_title))
    
    timing_bar = make_progressbar(data["timing_overlap_pct"], colors.HexColor('#6366F1'))
    entropy_bar = make_progressbar(data["entropy_overlap_pct"], colors.HexColor('#10B981'))
    
    fingerprint_data = [
        [
            Paragraph("Timing Overlap:", style_body_bold),
            timing_bar,
            Paragraph(f"{data['timing_overlap_pct']:.1%}", style_body_bold)
        ],
        [
            Paragraph("Entropy Similarity:", style_body_bold),
            entropy_bar,
            Paragraph(f"{data['entropy_overlap_pct']:.1%}", style_body_bold)
        ],
        [
            Paragraph("Shared Devices:", style_body_bold),
            Paragraph(f"{data['device_echo_markers']} user-agents shared across >= 20% of accounts", style_body),
            Paragraph("", style_body)
        ],
        [
            Paragraph("Cross-Platform Sync:", style_body_bold),
            Paragraph(f"{data['cross_platform_sync']} account pairs co-active within 30-min window", style_body),
            Paragraph("", style_body)
        ]
    ]
    fingerprint_tbl = Table(fingerprint_data, colWidths=[120, 300, 80])
    fingerprint_tbl.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(fingerprint_tbl)
    story.append(Spacer(1, 10))
    
    # 4. Top Accounts Section
    story.append(Paragraph("Top Campaigns Accounts & Verification Metrics", style_title))
    
    accounts_header = [
        Paragraph("Account ID", style_body_bold),
        Paragraph("Platform", style_body_bold),
        Paragraph("Event Count", style_body_bold),
        Paragraph("Linguistic Entropy", style_body_bold),
        Paragraph("Threat Prob", style_body_bold)
    ]
    accounts_table_data = [accounts_header]
    
    for acc in data["top_accounts"][:3]:  # Top 3 to easily fit on exactly 1 page
        accounts_table_data.append([
            Paragraph(acc["account_id"], style_body),
            Paragraph(acc["platform"], style_body),
            Paragraph(str(acc["event_count"]), style_body),
            Paragraph(f"{acc['char_entropy_mean']:.3f}", style_body),
            Paragraph(f"{acc['threat_prob']:.2%}", style_body_bold)
        ])
        
    accounts_tbl = Table(accounts_table_data, colWidths=[160, 80, 80, 100, 80])
    accounts_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(accounts_tbl)
    story.append(Spacer(1, 10))
    
    # 5. Operator Attribution Section
    story.append(Paragraph("Operator Attribution & Intel Sync", style_title))
    attribution_data = [
        [
            Paragraph("Attributed Operator:", style_body_bold),
            Paragraph(data["operator_id"], style_body_bold),
            Paragraph("Linked Campaigns:", style_body_bold),
            Paragraph(f"{linked_campaigns_count} campaigns match this behavior profile", style_body)
        ]
    ]
    attribution_tbl = Table(attribution_data, colWidths=[120, 130, 110, 140])
    attribution_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EEF2F6')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(attribution_tbl)
    story.append(Spacer(1, 20))
    
    # 6. Footer Divider & Footer text
    story.append(Table([['']], colWidths=[500], rowHeights=[0.5], style=TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#94A3B8')),
    ])))
    story.append(Spacer(1, 5))
    
    footer_text = f"Generated at {data['generated_at'][:19]} UTC  |  GhostSig — Behavioral Coordinated Inauthentic Behavior (CIB) Detection"
    style_footer = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor('#64748B'),
        alignment=1
    )
    story.append(Paragraph(footer_text, style_footer))
    
    # Build document
    doc.build(story)
    return output_path


def generate_all(output_dir='docs/evidence/'):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating evidence cards for all HIGH risk campaigns to {output_dir}...")
    
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return
        
    with conn.cursor() as cur:
        # Fetch HIGH confidence campaigns
        cur.execute("SELECT campaign_id, label FROM campaigns WHERE confidence > 0.7;")
        campaigns = cur.fetchall()
        
    conn.close()
    
    logger.info(f"Found {len(campaigns)} HIGH risk campaigns to package.")
    
    generated_count = 0
    for campaign_id, label in campaigns:
        c_str = str(campaign_id)
        # JSON
        json_path = os.path.join(output_dir, f"campaign_{c_str}.json")
        try:
            data = generate_json(c_str)
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            # PDF
            pdf_path = os.path.join(output_dir, f"campaign_{c_str}.pdf")
            generate_pdf(c_str, pdf_path)
            generated_count += 1
            logger.info(f"Generated JSON and PDF evidence cards for campaign {c_str}.")
        except Exception as e:
            logger.error(f"Failed to generate evidence card for campaign {c_str}: {e}")
            
    print(f"\nGenerated {generated_count} JSON+PDF evidence packages in {output_dir}\n")
    return generated_count


def main():
    generate_all()


if __name__ == "__main__":
    main()
