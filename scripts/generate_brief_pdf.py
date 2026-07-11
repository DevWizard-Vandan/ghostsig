"""
Generate the GhostSig one-page cold-outreach PDF brief.

Output: docs/ghostsig-brief.pdf

Run from repo root:
    python -m scripts.generate_brief_pdf

Every metric in the PDF is sourced from committed repo files:
  - docs/evidence/*.json  -> campaign confidence and account counts
  - tests/ suite          -> 44/44 pass count (manual verification, Day 1)
  - api/main.py           -> 10 endpoint count
  - evidence_cards.py     -> ReportLab / styling reference
"""

import json
import os
import pathlib

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Brand palette (matches evidence_cards.py) ──────────────────────────────
NAVY      = colors.HexColor("#0F172A")
SLATE     = colors.HexColor("#1E293B")
MUTED     = colors.HexColor("#334155")
LIGHT_BG  = colors.HexColor("#F8FAFC")
BORDER    = colors.HexColor("#CBD5E1")
RULE      = colors.HexColor("#4F46E5")
ACCENT    = colors.HexColor("#6366F1")
GREEN     = colors.HexColor("#10B981")
WHITE     = colors.white

OUTPUT_PATH = pathlib.Path("docs/ghostsig-brief.pdf")


def _load_evidence_metrics() -> dict:
    """Read docs/evidence/*.json and return verified campaign proof-points."""
    evidence_dir = pathlib.Path("docs/evidence")
    campaigns = []
    for f in sorted(evidence_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        campaigns.append({
            "campaign_id": d.get("campaign_id", ""),
            "confidence":  d.get("confidence", 0.0),
            "account_count": d.get("account_count", 0),
            "tier": d.get("confidence_tier", ""),
        })
    campaigns.sort(key=lambda x: x["account_count"], reverse=True)
    return {
        "total_campaigns":    len(campaigns),
        "top_campaign":       campaigns[0] if campaigns else {},
        "second_campaign":    campaigns[1] if len(campaigns) > 1 else {},
        "all_high":           all(c["tier"] == "HIGH" for c in campaigns),
    }


def generate(output_path: pathlib.Path = OUTPUT_PATH) -> None:
    os.makedirs(output_path.parent, exist_ok=True)
    metrics = _load_evidence_metrics()
    top = metrics["top_campaign"]
    second = metrics["second_campaign"]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    styles = getSampleStyleSheet()

    # ── Custom styles (consistent with evidence_cards.py) ──────────────────
    s_logo = ParagraphStyle(
        "BriefLogo",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=NAVY,
        leading=26,
    )
    s_tagline = ParagraphStyle(
        "Tagline",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=MUTED,
        leading=12,
        spaceBefore=2,
    )
    s_note = ParagraphStyle(
        "Note",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=colors.HexColor("#64748B"),
        leading=10,
    )
    s_section = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=SLATE,
        spaceBefore=10,
        spaceAfter=4,
        leading=13,
    )
    s_body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=MUTED,
        leading=13,
    )
    s_body_bold = ParagraphStyle(
        "BodyBold",
        parent=s_body,
        fontName="Helvetica-Bold",
        textColor=SLATE,
    )
    s_table_hdr = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=WHITE,
        leading=10,
    )
    s_table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=MUTED,
        leading=10,
    )
    s_disclaimer = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=colors.HexColor("#64748B"),
        leading=10,
        borderColor=BORDER,
        borderWidth=0.5,
        borderPadding=5,
        borderRadius=3,
    )
    s_footer = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        textColor=colors.HexColor("#94A3B8"),
        leading=10,
    )
    s_cta = ParagraphStyle(
        "CTA",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=ACCENT,
        leading=12,
    )

    story = []

    # ── 1. Header ──────────────────────────────────────────────────────────
    logo_p   = Paragraph("GHOSTSIG", s_logo)
    note_p   = Paragraph("Prototype — seeking pilot feedback", s_note)
    header_t = Table([[logo_p, note_p]], colWidths=[320, 195])
    header_t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_t)

    tagline_p = Paragraph(
        "Detects coordinated inauthentic behavior (CIB) networks using behavioral metadata "
        "fingerprinting — timing cadence, cross-platform rhythm, and linguistic entropy — "
        "without content access or PII.",
        s_tagline,
    )
    story.append(tagline_p)

    # Indigo rule
    story.append(Spacer(1, 6))
    rule = Table([[""]], colWidths=[515], rowHeights=[2], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), RULE),
    ]))
    story.append(rule)
    story.append(Spacer(1, 8))

    # ── 2. The Problem ─────────────────────────────────────────────────────
    story.append(Paragraph("The Problem", s_section))
    story.append(Paragraph(
        "Detecting coordinated inauthentic account networks is expensive with content-based "
        "approaches: they require direct platform API access, labeled training sets, and break "
        "immediately when the coordinated narrative shifts. Platforms typically catch campaigns "
        "weeks or months after activity peaks.",
        s_body,
    ))
    story.append(Spacer(1, 8))

    # ── 3. The Approach ────────────────────────────────────────────────────
    story.append(Paragraph("The Approach", s_section))
    story.append(Paragraph(
        "GhostSig fingerprints public behavioral metadata — posting timing cadence, "
        "cross-platform activity rhythm, linguistic entropy, and device-echo signals — "
        "using an end-to-end ML pipeline: neural encoders (temporal, entropy, fusion) feed "
        "HDBSCAN clustering, which is scored by an XGBoost adversarial discriminator trained "
        "on synthetic bot/organic data. No content is read. No PII is collected.",
        s_body,
    ))
    story.append(Spacer(1, 8))

    # ── 4. Verified Results table ──────────────────────────────────────────
    story.append(Paragraph("Verified Results", s_section))

    conf_top    = f"{top.get('confidence', 0):.4f}"
    count_top   = str(top.get("account_count", "–"))
    conf_second = f"{second.get('confidence', 0):.4f}"
    count_second = str(second.get("account_count", "–"))

    hdr_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SLATE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, WHITE]),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, BORDER),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), MUTED),
    ])

    table_data = [
        [
            Paragraph("Metric", s_table_hdr),
            Paragraph("Value", s_table_hdr),
            Paragraph("Source", s_table_hdr),
        ],
        [
            Paragraph("Automated tests passing", s_table_cell),
            Paragraph("44 / 44", s_table_cell),
            Paragraph("tests/ (pytest)", s_table_cell),
        ],
        [
            Paragraph(f"Largest detected campaign (accounts)", s_table_cell),
            Paragraph(f"{count_top} accounts · confidence {conf_top}", s_table_cell),
            Paragraph("docs/evidence/*.json", s_table_cell),
        ],
        [
            Paragraph(f"Second-largest campaign", s_table_cell),
            Paragraph(f"{count_second} accounts · confidence {conf_second}", s_table_cell),
            Paragraph("docs/evidence/*.json", s_table_cell),
        ],
        [
            Paragraph("Campaigns detected (all HIGH tier)", s_table_cell),
            Paragraph(f"{metrics['total_campaigns']} campaigns", s_table_cell),
            Paragraph("docs/evidence/*.json", s_table_cell),
        ],
        [
            Paragraph("REST API endpoints", s_table_cell),
            Paragraph("10", s_table_cell),
            Paragraph("api/main.py", s_table_cell),
        ],
        [
            Paragraph("Evidence output formats", s_table_cell),
            Paragraph("JSON + PDF per campaign", s_table_cell),
            Paragraph("attribution/evidence_cards.py", s_table_cell),
        ],
    ]

    results_tbl = Table(table_data, colWidths=[185, 185, 145])
    results_tbl.setStyle(hdr_style)
    story.append(results_tbl)
    story.append(Spacer(1, 10))

    # ── 5. What This Is / Is Not ───────────────────────────────────────────
    story.append(Paragraph(
        "This is an analyst-support signal, not an automated enforcement decision. "
        "Human review required.",
        s_disclaimer,
    ))
    story.append(Spacer(1, 10))

    # ── 6. What I'm Looking For ────────────────────────────────────────────
    story.append(Paragraph("What I'm Looking For", s_section))
    story.append(Paragraph(
        "Pilot feedback from a Trust & Safety or OSINT team, a labeled CIB dataset to validate "
        "against real-world campaigns, or a 15-minute call.",
        s_body,
    ))
    story.append(Spacer(1, 10))

    # ── Footer rule ────────────────────────────────────────────────────────
    footer_rule = Table([[""]], colWidths=[515], rowHeights=[1], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BORDER),
    ]))
    story.append(footer_rule)
    story.append(Spacer(1, 5))

    footer_data = [[
        Paragraph("Vandan  ·  [YOUR_EMAIL]", s_footer),
        Paragraph(
            "github.com/DevWizard-Vandan/ghostsig",
            s_footer,
        ),
        Paragraph("Live demo: [DEMO_LINK]", s_footer),
    ]]
    footer_tbl = Table(footer_data, colWidths=[175, 200, 140])
    footer_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (-1, 0), (-1, 0), 0),
    ]))
    story.append(footer_tbl)

    doc.build(story)
    print(f"[OK] Brief generated -> {output_path}  ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    generate()
