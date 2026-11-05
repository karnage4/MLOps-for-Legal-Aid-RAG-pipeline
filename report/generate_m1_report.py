"""Generate the final Milestone 1 PDF report with verified visual evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

_EARLY_REPO = Path(__file__).resolve().parents[1]
_MPL_CONFIG = _EARLY_REPO / "tmp" / "pdfs" / "matplotlib"
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "tmp" / "pdfs" / "m1_report_assets"
OUTPUT = REPO / "output" / "pdf" / "legal_aid_rag_milestone1_report.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2867A8")
TEAL = colors.HexColor("#198F8A")
GOLD = colors.HexColor("#D79516")
GREEN = colors.HexColor("#2C8C5A")
RED = colors.HexColor("#B34747")
INK = colors.HexColor("#22303C")
MUTED = colors.HexColor("#5F6B76")
LIGHT = colors.HexColor("#F3F6F8")
PALE_BLUE = colors.HexColor("#EAF2FA")
PALE_GOLD = colors.HexColor("#FFF6E3")
WHITE = colors.white


def load_json(relative: str):
    with (REPO / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


legacy = load_json("artifacts/rag_evaluation_summary.json")
genai = load_json("artifacts/genai_evaluation_comparison.json")
ablation = load_json("artifacts/inference_ablation/comparison.json")


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
)
styles.add(
    ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        textColor=MUTED,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        name="Section",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=8,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="Subsection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=BLUE,
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="BodyReport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.4,
        leading=13.4,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallReport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        name="TableHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.6,
        leading=10,
        textColor=WHITE,
    )
)
styles.add(
    ParagraphStyle(
        name="Caption",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=7.7,
        leading=10,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=9,
    )
)
styles.add(
    ParagraphStyle(
        name="Callout",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.2,
        leading=13,
        textColor=NAVY,
        leftIndent=4,
        rightIndent=4,
    )
)
styles.add(
    ParagraphStyle(
        name="ReportCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=INK,
    )
)


def p(text: str, style: str = "BodyReport") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"- {text}", styles["BodyReport"])


def section(title: str):
    return [Spacer(1, 2 * mm), p(title, "Section"), HRFlowable(width="100%", color=BLUE, thickness=0.7), Spacer(1, 2 * mm)]


def subsection(title: str):
    return [p(title, "Subsection")]


def table(data, widths, header=True, font_size=7.6, repeat_rows=1):
    converted = []
    for row_index, row in enumerate(data):
        converted_row = []
        for cell in row:
            if hasattr(cell, "wrap"):
                converted_row.append(cell)
            else:
                style = "TableHeader" if header and row_index == 0 else "SmallReport"
                converted_row.append(p(str(cell), style))
        converted.append(converted_row)
    result = LongTable(converted, colWidths=widths, repeatRows=repeat_rows if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D3DC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    for row_index in range(1 if header else 0, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT))
    result.setStyle(TableStyle(commands))
    return result


def callout(text: str, color=PALE_BLUE):
    box = Table([[p(text, "Callout")]], colWidths=[170 * mm])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return box


def fit_image(path: Path, max_width=170 * mm, max_height=118 * mm):
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def figure(path: Path, caption: str, max_height=118 * mm):
    return KeepTogether([fit_image(path, max_height=max_height), p(caption, "Caption")])


def code_box(lines: str):
    rows = [[Paragraph(line.replace(" ", "&nbsp;"), styles["ReportCode"])] for line in lines.splitlines()]
    box = Table(rows, colWidths=[170 * mm])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7F9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#B9C5CF")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return box


def architecture_drawing():
    d = Drawing(500, 270)
    boxes = [
        (10, 190, 94, 48, "Legal PDF corpus", GREEN),
        (125, 190, 94, 48, "Validate + chunk", BLUE),
        (240, 210, 105, 42, "Dense embeddings", TEAL),
        (240, 154, 105, 42, "BM25 sparse index", GOLD),
        (370, 182, 118, 52, "Semantic / hybrid\nretrieval + rerank", NAVY),
        (370, 92, 118, 52, "HF generation\nGroq fallback", RED),
        (240, 92, 105, 52, "MLflow traces,\nruns, judges", BLUE),
        (125, 92, 94, 52, "FastAPI +\nGradio UI", TEAL),
        (10, 92, 94, 52, "Grounded legal\nanswer", GREEN),
        (188, 18, 124, 42, "Held-out evaluation\nand ablation", GOLD),
    ]
    for x, y, w, h, label, color in boxes:
        d.add(Rect(x, y, w, h, rx=7, ry=7, fillColor=colors.Color(color.red, color.green, color.blue, alpha=0.12), strokeColor=color, strokeWidth=1.2))
        lines = label.split("\n")
        start = y + h / 2 + (len(lines) - 1) * 6
        for index, line in enumerate(lines):
            d.add(String(x + w / 2, start - index * 13, line, textAnchor="middle", fontName="Helvetica-Bold", fontSize=8, fillColor=INK))

    def arrow(x1, y1, x2, y2, color=NAVY):
        d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.2))
        if abs(x2 - x1) >= abs(y2 - y1):
            direction = 1 if x2 > x1 else -1
            d.add(Polygon([x2, y2, x2 - 6 * direction, y2 + 3, x2 - 6 * direction, y2 - 3], fillColor=color, strokeColor=color))
        else:
            direction = 1 if y2 > y1 else -1
            d.add(Polygon([x2, y2, x2 + 3, y2 - 6 * direction, x2 - 3, y2 - 6 * direction], fillColor=color, strokeColor=color))

    arrow(104, 214, 125, 214)
    arrow(219, 214, 240, 231)
    arrow(219, 214, 240, 175)
    arrow(345, 231, 370, 214)
    arrow(345, 175, 370, 202)
    arrow(429, 182, 429, 144)
    arrow(370, 118, 345, 118)
    arrow(240, 118, 219, 118)
    arrow(125, 118, 104, 118)
    arrow(288, 92, 250, 60)
    arrow(429, 92, 312, 39)
    return d


def make_charts():
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})

    ranking = genai["ranking"]
    labels = [item["label"].replace("recursive", "rec").replace("semantic", "sem").replace("hybrid", "hyb").replace("fixed", "fix") for item in ranking]
    y = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.barh([v + 0.22 for v in y], [r["legal_answer_quality"] for r in ranking], height=0.21, label="Legal quality", color="#2867A8")
    ax.barh(y, [r["retrieval_relevance"] for r in ranking], height=0.21, label="Retrieval relevance", color="#198F8A")
    ax.barh([v - 0.22 for v in y], [r["correctness"] for r in ranking], height=0.21, label="Correctness", color="#D79516")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean judge score")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", ncol=3, frameon=False)
    ax.set_title("Managed MLflow GenAI comparison (3 held-out queries per profile)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "genai_profile_chart.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ranking = ablation["ranking"]
    labels = [r["label"] for r in ranking]
    metrics = ["correctness", "legal_answer_quality", "retrieval_relevance", "retrieval_groundedness", "retrieval_sufficiency"]
    metric_labels = ["Correctness", "Legal quality", "Retrieval relevance", "Groundedness", "Sufficiency"]
    x = list(range(len(labels)))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    palette = ["#17324D", "#2867A8", "#198F8A", "#D79516", "#B34747"]
    for index, (metric, label, color) in enumerate(zip(metrics, metric_labels, palette)):
        offset = (index - 2) * width
        ax.bar([v + offset for v in x], [r[metric] for r in ranking], width=width, label=label, color=color)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean judge score")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper center", ncol=3, frameon=False)
    ax.set_title("Controlled prompt, model, and temperature ablation", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "inference_ablation_chart.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def prepare_screenshot_crops():
    crops = {
        "corpus_drive.png": ("corpus_drive_cropped.png", 470),
        "mlflow_final_home.png": ("mlflow_final_home_cropped.png", 720),
    }
    for source_name, (target_name, crop_height) in crops.items():
        source_path = ASSETS / source_name
        target_path = ASSETS / target_name
        with PILImage.open(source_path) as source:
            image = source.convert("RGB")
            image.crop((0, 0, image.width, min(crop_height, image.height))).save(target_path)


def page_decor(canvas, doc):
    page = canvas.getPageNumber()
    canvas.saveState()
    if page > 1:
        canvas.setStrokeColor(colors.HexColor("#D5DEE5"))
        canvas.setLineWidth(0.4)
        canvas.line(20 * mm, A4[1] - 14 * mm, A4[0] - 20 * mm, A4[1] - 14 * mm)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(20 * mm, A4[1] - 11 * mm, "LEGAL-AID RAG - MILESTONE 1")
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 11 * mm, "Model, API, MLflow, and GenAI Evaluation")
        canvas.line(20 * mm, 14 * mm, A4[0] - 20 * mm, 14 * mm)
        canvas.drawString(20 * mm, 10 * mm, "Branch: 8-m1-submission | Evidence baseline: a12cd29")
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {page}")
    canvas.restoreState()


def build_story():
    story = []

    story.extend(
        [
            Spacer(1, 24 * mm),
            HRFlowable(width="35%", color=GOLD, thickness=4, hAlign="CENTER"),
            Spacer(1, 8 * mm),
            p("Legal-Aid RAG MLOps Pipeline", "ReportTitle"),
            p("Milestone 1: Model, API, MLflow Tracking, and GenAI Evaluation", "ReportSubtitle"),
            Spacer(1, 16 * mm),
            callout(
                "A reproducible legal-inheritance retrieval-augmented generation system with validated corpus preparation, eight retrieval profiles, MLflow GenAI datasets and judges, controlled inference ablations, FastAPI serving, and a local Gradio interface.",
                PALE_GOLD,
            ),
            Spacer(1, 18 * mm),
            table(
                [
                    ["Team member", "ERP ID", "Primary responsibility"],
                    ["Karam Hussain", "27857", "Modeling, evaluation, MLflow experiment design, and champion selection"],
                    ["Aadesh Panjwani", "29303", "Data pipeline, retrieval implementation, API, and generation metadata logging"],
                    ["Muhammad Rayyan Khan", "28410", "Testing, manual verification, reproducibility, and quality assurance"],
                ],
                [45 * mm, 25 * mm, 100 * mm],
            ),
            Spacer(1, 14 * mm),
            p("Submission branch: <b>8-m1-submission</b><br/>Report date: 5 November 2026<br/>Repository: github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline", "ReportSubtitle"),
            PageBreak(),
        ]
    )

    story += section("Executive summary")
    story += [
        p(
            "This milestone converts the previous notebook and application prototype into a reproducible RAG delivery pipeline. The original architecture is preserved: local BM25 retrieval is combined with Pinecone semantic retrieval, candidates are fused and reranked, Hugging Face remains the primary generation provider, and Groq is used as a clean fallback. MLflow records conventional experiments, nested query-retrieval-generation traces, a managed evaluation dataset, versioned GenAI judges, and controlled comparisons."
        ),
        p(
            "The traditional supervised-training requirement is explicitly treated as exempt because the project uses pretrained embedding, reranking, and generation models. In its place, the system performs inference evaluation across retrieval parameters, prompts, models, and temperatures, matching the instructor's RAG-specific guidance."
        ),
        callout(
            "Final status: all eight retrieval configurations and all four controlled inference-ablation configurations completed successfully. The managed six-judge comparison selected fixed-hybrid-top5, while the generation ablation selected structured_legal_v2 with Qwen at temperature 0.1.",
            PALE_BLUE,
        ),
    ]
    story += subsection("Milestone compliance matrix")
    compliance = [
        ["Requirement", "Status", "Evidence"],
        ["README pitch, architecture, quick start, dataset", "Complete", "README includes reproducible setup, architecture, corpus details, and public Drive access"],
        ["Separate data preparation and validation", "Complete", "prepare_bm25.py validates fields, empty values, ranges, duplicates, strategy, and persisted alignment"],
        ["Training and model artifact", "RAG exemption", "Inference evaluation and clearly named prepared/evaluation artifacts replace unrelated supervised training"],
        ["At least three MLflow experiment runs", "Complete", "8 retrieval runs, 8 managed GenAI runs, and 4 controlled generation ablations"],
        ["FastAPI prediction service", "Complete", "/health, /predict, Pydantic validation, component cache, and /docs"],
        ["Unit tests", "Complete", "Preparation, loading, fallback, tracing, evaluation, API, and Gradio coverage"],
        ["Contribution document", "Complete", "Names, ERP IDs, task division, and detailed responsibilities"],
    ]
    story.append(table(compliance, [48 * mm, 26 * mm, 96 * mm]))
    story.append(PageBreak())

    story += section("1. System architecture and design decisions")
    story += [
        p(
            "The pipeline separates preparation, indexing, retrieval, generation, evaluation, and serving. Fixed and recursive chunk stores are isolated through matching BM25 artifacts and Pinecone namespaces. Retrieval can run dense semantic search only or a hybrid path that combines BM25 and dense candidates using reciprocal-rank fusion before cross-encoder reranking."
        ),
        architecture_drawing(),
        p("Figure 1. End-to-end architecture of the final Milestone 1 RAG pipeline.", "Caption"),
    ]
    story += subsection("Key architectural guarantees")
    for text in [
        "Retrieval behavior remains compatible with the original project; configurable profiles are layered around the same semantic and hybrid components.",
        "Every application query produces one MLflow CHAIN trace with RETRIEVER and LLM child spans, preserving parent-child nesting.",
        "Hugging Face is attempted first in production auto mode. Groq uses GROQ_API_KEY from .env only when the primary provider is unavailable.",
        "Evaluation runs can request direct Groq generation for controlled model comparisons without changing the production fallback default.",
        "Secrets, full prompts, and full retrieved documents are not returned by the public API response contract.",
    ]:
        story.append(bullet(text))
    story += subsection("Technology stack")
    story.append(
        table(
            [
                ["Layer", "Technology", "Purpose"],
                ["Preparation", "Python, PDF extraction, validation", "Create fixed and recursive chunks and aligned BM25 corpora"],
                ["Dense retrieval", "Sentence Transformers + Pinecone", "Semantic candidate retrieval with isolated namespaces"],
                ["Sparse retrieval", "BM25", "Exact-term and legal phrase matching"],
                ["Reranking", "Cross-encoder MiniLM", "Improve final candidate ordering"],
                ["Generation", "Hugging Face + Groq", "Primary/fallback answer generation"],
                ["MLOps", "MLflow 3.17.x", "Runs, nested traces, datasets, judges, assessments, and artifacts"],
                ["Serving", "FastAPI + Gradio", "Validated REST API and interactive local interface"],
            ],
            [33 * mm, 54 * mm, 83 * mm],
        )
    )
    story += subsection("Operational modes")
    story.append(
        table(
            [
                ["Mode", "Behavior"],
                ["Production auto", "Hugging Face first; bounded Groq fallback if credits, provider availability, or transient requests fail"],
                ["Controlled evaluation", "Direct requested Groq model to isolate prompt, model, and temperature effects"],
                ["Semantic retrieval", "Pinecone dense candidates only, followed by the shared response pipeline"],
                ["Hybrid retrieval", "BM25 plus Pinecone candidates, reciprocal-rank fusion, and cross-encoder reranking"],
            ],
            [42 * mm, 128 * mm],
        )
    )
    story += subsection("Milestone design rationale")
    story.append(
        p(
            "The implementation avoids introducing an unrelated classifier simply to satisfy a traditional training rubric. Instead, it treats retrieval configuration and generation behavior as the model system under evaluation, preserving architectural validity while producing reproducible MLflow evidence."
        )
    )
    story.append(PageBreak())

    story += section("2. Dataset access, preparation, and validation")
    story += [
        p(
            "The source corpus contains legal inheritance material grouped into five categories: core statutory laws, Islamic fiqh-based material, research and legal analysis, government and legal reports, and case law. The folder is publicly viewable for the instructor and teaching assistant. Raw PDFs remain outside Git to keep source distribution separate from code and to avoid repository bloat."
        ),
        figure(ASSETS / "corpus_drive_cropped.png", "Figure 2. Anonymous Google Drive view showing the five public corpus folders.", 76 * mm),
        p(
            'Public corpus: <link href="https://drive.google.com/drive/folders/103phLHYLT7Pv464nQ47e-lUl_J3DKquT?usp=drive_link" color="#2867A8">Google Drive folder</link>. Anonymous HTTP and visual access were verified on 5 November 2026.'
        ),
    ]
    story += subsection("Preparation outputs")
    story.append(
        table(
            [
                ["Item", "Verified result"],
                ["Input files", "50 supplied PDFs across five folders"],
                ["Accepted files", "46 PDFs with sufficient machine-readable content"],
                ["Skipped files", "4 PDFs below the minimum extractable-text threshold"],
                ["Recursive chunks", "3,593 validated chunks"],
                ["Fixed chunks", "883 validated chunks"],
                ["Sparse artifacts", "Strategy-specific BM25 JSON corpora"],
                ["Dense storage", "Recursive default namespace and isolated fixed namespace in Pinecone"],
            ],
            [55 * mm, 115 * mm],
        )
    )
    story += subsection("Validation controls")
    for text in [
        "Reject malformed JSON and non-list chunk collections.",
        "Require non-empty id, text, and source fields.",
        "Reject negative or non-integer chunk indexes.",
        "Reject duplicate IDs and chunking-strategy mismatches.",
        "Persist, reload, and revalidate BM25 alignment before downstream use.",
    ]:
        story.append(bullet(text))
    story.append(PageBreak())

    story += section("3. MLflow integration and experiment lineage")
    story += [
        p(
            "MLflow is configured through a shared module and defaults to a repository-local SQLite tracking store. The same experiment captures smoke tests, offline comparisons, nested RAG traces, managed datasets, judge definitions, and controlled evaluation artifacts. Optional MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID variables support migration to a shared server."
        ),
        figure(ASSETS / "mlflow_final_home_cropped.png", "Figure 3. Repository-local MLflow 3.17.2 server showing the legal-aid-rag experiment.", 85 * mm),
    ]
    story += subsection("Nested trace contract")
    story.append(
        table(
            [
                ["Span", "Type", "Recorded evidence"],
                ["legal_aid_rag_query", "CHAIN", "Query, retrieval profile, temperature, requested provider/model, final response metadata"],
                ["hybrid_legal_retrieval", "RETRIEVER", "Strategy, namespace, top-K, sparse/dense candidate counts, reranking, latency"],
                ["legal_generation", "LLM", "Provider attempts, actual provider/model, fallback flag, latency, sanitized outcomes"],
            ],
            [42 * mm, 25 * mm, 103 * mm],
        )
    )
    story += subsection("Managed GenAI assets")
    for text in [
        "Dataset legal-aid-rag-held-out-v1 contains three idempotently registered held-out questions and human-reviewed expected facts.",
        "Six pinned judges assess query relevance, retrieval relevance, retrieval groundedness, correctness, retrieval sufficiency, and legal-answer quality.",
        "Automatic trace evaluation remains off; the registered judges are invoked explicitly on the controlled dataset to prevent unbounded provider usage.",
        "Partial or timed-out runs are marked invalid and excluded from ranking until every expected prediction and judge value is present.",
    ]:
        story.append(bullet(text))
    story.append(PageBreak())

    story += section("4. Retrieval-parameter evaluation")
    story += [
        p(
            "The first experiment suite evaluates the exact Cartesian product of two chunking strategies, two retrieval modes, and two top-K values. Every profile uses the same three questions, the same generation temperature (0.2), and the same evaluation formulas. All 24 query executions completed successfully."
        )
    ]
    legacy_rows = [["Configuration", "Faithfulness", "Relevancy", "Avg total ms", "Run ID"]]
    for label, item in legacy["experiments"].items():
        summary = item["summary"]
        legacy_rows.append(
            [
                label,
                f'{summary["faithfulness_mean"]:.4f}',
                f'{summary["relevancy_mean"]:.4f}',
                f'{summary["avg_total_ms"]:,.1f}',
                item["run_id"],
            ]
        )
    story.append(table(legacy_rows, [43 * mm, 25 * mm, 23 * mm, 25 * mm, 54 * mm], font_size=6.9))
    story.append(Spacer(1, 4 * mm))
    story.append(
        callout(
            "Legacy evaluator winner: fixed-hybrid-top3 (run ba6464e78f014a6db1d21af5c39d508f). It tied for highest faithfulness at 1.0000 and had higher relevancy than the other 1.0000-faithfulness profile.",
            PALE_GOLD,
        )
    )
    story += subsection("Interpretation")
    story.append(
        p(
            "The results show that retrieval quality is not determined by one universal setting. Recursive hybrid top-3 produced the highest mean relevancy (0.7860), while fixed configurations achieved the highest faithfulness. This justified keeping chunking, mode, and top-K configurable rather than hard-coding a single profile throughout the application."
        )
    )
    story.append(PageBreak())

    story += section("5. Managed six-judge GenAI comparison")
    story += [
        p(
            "The second evaluation suite reuses the same eight profiles but applies the MLflow-managed dataset, reviewed expectations, and all six pinned judges. Ranking prioritizes correctness, retrieval sufficiency, legal-answer quality, grounding, relevance, and latency under strict completeness rules."
        ),
        figure(ASSETS / "genai_profile_chart.png", "Figure 4. Quality metrics across all eight complete managed GenAI profiles.", 92 * mm),
    ]
    genai_rows = [["Profile", "Correct", "Sufficient", "Legal quality", "Grounded", "Relevant", "Run ID"]]
    for item in genai["ranking"]:
        genai_rows.append(
            [
                item["label"],
                f'{item["correctness"]:.3f}',
                f'{item["retrieval_sufficiency"]:.3f}',
                f'{item["legal_answer_quality"]:.3f}',
                f'{item["retrieval_groundedness"]:.3f}',
                f'{item["retrieval_relevance"]:.3f}',
                item["run_id"],
            ]
        )
    story.append(table(genai_rows, [36 * mm, 17 * mm, 19 * mm, 19 * mm, 18 * mm, 17 * mm, 44 * mm], font_size=6.3))
    story.append(Spacer(1, 4 * mm))
    story.append(
        callout(
            "Managed GenAI winner: fixed-hybrid-top5 (run e6034e3cda2f4582ab5fac02132b4687). It was the only profile to reach correctness 0.333 while also achieving legal-answer quality 1.000 and query relevance 1.000.",
            PALE_BLUE,
        )
    )
    story += subsection("Why the two retrieval winners differ")
    story.append(
        p(
            "The legacy suite selects fixed-hybrid-top3 using faithfulness, relevancy, and latency. The managed GenAI suite selects fixed-hybrid-top5 using a broader rubric that includes reviewed expected facts, legal framing, sufficiency, and correctness. The results are therefore complementary rather than contradictory. The latter profile is frozen for the controlled generation ablation because it won the richer six-judge comparison."
        )
    )
    story += subsection("Pinned judge rubric")
    story.append(
        table(
            [
                ["Judge", "Purpose", "Expected facts"],
                ["relevance_to_query", "Does the answer address the question?", "No"],
                ["retrieval_relevance", "Are retrieved chunks relevant to the query?", "No"],
                ["retrieval_groundedness", "Is the answer supported by retrieved evidence?", "No"],
                ["legal_answer_quality", "Does the answer communicate conditions, uncertainty, and responsible framing?", "No"],
                ["correctness", "Does the answer cover the approved expected facts?", "Yes"],
                ["retrieval_sufficiency", "Can retrieval support the approved expected facts?", "Yes"],
            ],
            [48 * mm, 95 * mm, 27 * mm],
        )
    )
    story.append(PageBreak())

    story += section("6. Prompt, model, parameter, and temperature ablation")
    story += [
        p(
            "The instructor's RAG-specific update requires inference comparisons across prompts, models, parameters, and temperature. Parameter variation is covered by the eight retrieval profiles. A separate four-run ablation freezes retrieval to fixed-hybrid-top5 and changes exactly one generation variable at a time. Direct Groq mode is used for controlled model identity while production auto mode remains Hugging Face first."
        ),
        figure(ASSETS / "inference_ablation_chart.png", "Figure 5. Controlled prompt, model, and temperature results on the same dataset and judge lineage.", 92 * mm),
    ]
    ablation_rows = [["Run", "Prompt", "Model", "Temp", "Correct", "Legal", "Relevance", "Latency ms"]]
    configs = {
        "baseline-low-temp": ("strict_grounded_v1", "Qwen 3.8-27B", "0.1"),
        "prompt-structured": ("structured_legal_v2", "Qwen 3.8-27B", "0.1"),
        "model-b": ("strict_grounded_v1", "GPT-OSS-120B", "0.1"),
        "temperature-high": ("strict_grounded_v1", "Qwen 3.8-27B", "0.7"),
    }
    by_label = {item["label"]: item for item in ablation["ranking"]}
    for label in ["baseline-low-temp", "prompt-structured", "model-b", "temperature-high"]:
        item = by_label[label]
        prompt_name, model_name, temp = configs[label]
        ablation_rows.append(
            [
                label,
                prompt_name,
                model_name,
                temp,
                f'{item["correctness"]:.3f}',
                f'{item["legal_answer_quality"]:.3f}',
                f'{item["retrieval_relevance"]:.3f}',
                f'{item["avg_total_ms"]:,.0f}',
            ]
        )
    story.append(table(ablation_rows, [28 * mm, 36 * mm, 31 * mm, 12 * mm, 17 * mm, 14 * mm, 17 * mm, 19 * mm], font_size=6.4))
    story += subsection("Controlled findings")
    for text in [
        "Prompt: structured_legal_v2 improved correctness from 0.000 to 0.333, legal quality from 0.667 to 1.000, and retrieval relevance from 0.667 to 0.733.",
        "Model: GPT-OSS-120B was faster than Qwen in this sample but reduced legal quality to 0.333 and retrieval relevance to 0.600.",
        "Temperature: raising 0.1 to 0.7 reduced retrieval sufficiency from 0.667 to 0.333 and relevance from 0.667 to 0.600, while groundedness increased to 1.000.",
        "Selected generation configuration: structured_legal_v2 with qwen/qwen3.8-27b at temperature 0.1.",
    ]:
        story.append(bullet(text))
    story.append(
        p(
            "These scores are milestone evidence from three held-out examples, not statistically generalizable claims. Expanding the reviewed dataset is the most important next evaluation improvement."
        )
    )
    story.append(PageBreak())

    story += section("7. FastAPI prediction service")
    story += [
        p(
            "The service exposes /health and /predict through FastAPI. Pydantic rejects blank queries, unknown fields, top-K values outside 3-10, and temperatures outside 0-1. Retriever components are loaded once and cached at application startup. Errors are mapped to compact HTTP responses without returning secrets, full prompts, or complete retrieved documents."
        ),
        figure(ASSETS / "fastapi_docs.png", "Figure 6. Auto-generated FastAPI documentation showing health, prediction, and validated schemas.", 118 * mm),
    ]
    story += subsection("Request example")
    story.append(
        code_box(
            'POST /predict\n'
            '{\n'
            '  "query": "Who inherits without a will?",\n'
            '  "chunking_strategy": "fixed",\n'
            '  "retrieval_mode": "hybrid",\n'
            '  "top_k": 5,\n'
            '  "temperature": 0.2\n'
            '}'
        )
    )
    story += subsection("Response evidence")
    story.append(
        p(
            "The compact response includes the answer, actual provider and model, fallback flag, retrieval profile, source identifiers, trace ID, and retrieval/generation/total timing. Manual acceptance confirmed both /health and /predict through /docs."
        )
    )
    story.append(PageBreak())

    story += section("8. Local Gradio interface")
    story += [
        p(
            "Miras provides a local interface similar to the earlier Hugging Face application while using the same traced RAG boundary. Users can select recursive or fixed chunks, semantic or hybrid retrieval, top-K from 3 to 10, generation temperature from 0 to 1, and whether to display retrieved context."
        ),
        figure(ASSETS / "gradio_ui.png", "Figure 7. Miras interface showing chunking, retrieval, top-K, temperature, and context controls.", 126 * mm),
        p(
            "The interface displays an explicit general-information disclaimer and surfaces provider, retrieval, and timing metadata after prediction. Unit, HTTP, live RAG, fallback, and nested-trace verification have passed."
        ),
    ]
    story.append(PageBreak())

    story += section("9. Testing, validation, and operational quality")
    story += [
        callout(
            "Verification status: the full repository regression completed with 91 passing tests and one pre-existing MLflow SQLAlchemy deprecation warning. The final focused evaluation regression completed with 14 passing tests after the high-temperature run.",
            PALE_GOLD,
        ),
        Spacer(1, 3 * mm),
    ]
    story.append(
        table(
            [
                ["Test area", "Representative coverage"],
                ["Preparation", "Missing values, invalid ranges, duplicate IDs, strategy mismatch, persisted alignment"],
                ["Embedding/index", "Batch preparation, namespace routing, loader inputs, failure handling"],
                ["Retrieval", "Semantic/hybrid execution, top-K profiles, namespace guards, live smoke checks"],
                ["Generation", "HF-first order, Groq fallback, retries, provider metadata, secret exclusion"],
                ["Tracing", "Single root chain and nested retriever/generation parent IDs"],
                ["Evaluation", "Aggregation, incomplete judge output, strict reuse, best-run selection, eight-profile matrix"],
                ["FastAPI", "Cached loading, request validation, success response, provider failures, /health"],
                ["Gradio", "Input mapping, chunk selector, response formatting, HTTP startup"],
                ["Inference ablation", "Config isolation, lineage matching, partial-run exclusion, ranking and deltas"],
            ],
            [42 * mm, 128 * mm],
        )
    )
    story += subsection("Quality safeguards")
    for text in [
        ".env, local MLflow databases, generated outputs, and raw corpus files are excluded from version control.",
        "Tracked diffs were scanned for credential-shaped values before submission commits.",
        "Judge and provider failures use bounded retries and explicit partial-run status rather than silent score omission.",
        "The replacement Groq key remained local; no API key appears in commits or report evidence.",
        "A transient Pinecone DNS failure was diagnosed at the trace span and excluded before the successful retry.",
    ]:
        story.append(bullet(text))
    story.append(PageBreak())

    story += section("10. Reproducibility and execution guide")
    story += [
        p("The following sequence reproduces setup, credential-free tests, local services, preparation, and controlled evaluations. Cloud-backed steps require the three keys documented in .env.example."),
        code_box(
            "git clone https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline.git\n"
            "cd MLOps-for-Legal-Aid-RAG-pipeline\n"
            "python -m venv .venv\n"
            ".venv\\Scripts\\Activate.ps1\n"
            "python -m pip install -r requirements.txt\n"
            "Copy-Item .env.example .env\n"
            "pytest -q\n"
            "uvicorn src.app:app --host 127.0.0.1 --port 8000\n"
            "python gradio_app.py"
        ),
    ]
    story += subsection("Evaluation commands")
    story.append(
        code_box(
            "python -m src.prepare_bm25 --strategy fixed\n"
            "python -m src.evaluation\n"
            "python -m src.genai_dataset --creator \"Karam Hussain\"\n"
            "python -m src.genai_expectations --reviewer \"Karam Hussain\"\n"
            "python -m src.genai_judges\n"
            "python -m src.genai_comparison\n"
            "python -m src.inference_ablation"
        )
    )
    story += subsection("Required environment variables")
    story.append(
        table(
            [
                ["Variable", "Purpose"],
                ["HF_API_TOKEN", "Primary Hugging Face generation provider"],
                ["PINECONE_API_KEY", "Dense retrieval from the configured Pinecone index"],
                ["GROQ_API_KEY", "Generation fallback and controlled direct-Groq evaluations"],
                ["MLFLOW_TRACKING_URI", "Optional override for a shared MLflow server"],
                ["GROQ_JUDGE_MODEL", "Optional judge model override; verified default is openai/gpt-oss-20b"],
            ],
            [55 * mm, 115 * mm],
        )
    )
    story.append(PageBreak())

    story += section("11. Team contributions")
    contribution_rows = [
        ["Member", "ERP", "Responsibility", "Verified deliverables"],
        [
            "Karam Hussain",
            "27857",
            "Modeling and evaluation",
            "Retrieval experiments, held-out dataset and judges, MLflow tracking and nested tracing, eight-profile comparison, prompt/model/temperature ablation, champion selection, findings documentation, and technical integration.",
        ],
        [
            "Aadesh Panjwani",
            "29303",
            "Pipeline and API",
            "Data preparation and validation, fixed/recursive artifacts, BM25 and Pinecone support, semantic/hybrid retrieval, FastAPI endpoints and docs, generation metadata logging, and architecture documentation.",
        ],
        [
            "Muhammad Rayyan Khan",
            "28410",
            "Quality and operations",
            "Unit and integration tests, focused/full pytest execution, result and warning review, MLflow trace/judge/artifact verification, manual FastAPI and Gradio validation, reproducibility, and final QA.",
        ],
    ]
    story.append(table(contribution_rows, [31 * mm, 16 * mm, 38 * mm, 85 * mm]))
    story += subsection("Collaboration model")
    story.append(
        p(
            "Responsibilities are separated by ownership but verified collaboratively. Modeling decisions are supported by pipeline instrumentation; pipeline changes are protected by automated and manual QA; and final selections are backed by tracked run IDs and reproducible artifacts rather than undocumented judgment."
        )
    )
    story.append(PageBreak())

    story += section("12. Limitations, risks, and next steps")
    story.append(
        table(
            [
                ["Current limitation or risk", "Mitigation / next action"],
                ["Only three human-reviewed evaluation questions", "Expand the managed dataset across more inheritance scenarios and document strata"],
                ["LLM judges can vary and are provider-quota sensitive", "Retain pinned versions, strict completeness, retries, and add human review on larger samples"],
                ["Hugging Face credits/provider availability caused fallback", "Keep provider abstraction and monitor primary/fallback success rates"],
                ["Pinecone and external APIs require network connectivity", "Add availability monitoring, clearer retry budgets, and optional offline retrieval fixtures"],
                ["Legal answers are informational, not legal advice", "Preserve disclaimers, source visibility, uncertainty language, and qualified-lawyer guidance"],
                ["Local MLflow SQLite is single-user oriented", "Move to a shared tracking server and artifact store for team deployment"],
                ["No hosted Space in this milestone", "Defer deployment until platform choice and tracking topology are approved"],
            ],
            [65 * mm, 105 * mm],
        )
    )
    story += subsection("Recommended next milestone")
    for text in [
        "Increase dataset coverage and add reviewer provenance for every expectation.",
        "Package the API and UI with a production-ready deployment configuration.",
        "Run automated CI tests and artifact validation on pull requests.",
        "Deploy shared MLflow tracking with durable artifacts and access controls.",
        "Add monitoring for provider fallback rate, latency, retrieval drift, and answer-quality regressions.",
    ]:
        story.append(bullet(text))
    story.append(PageBreak())

    story += section("13. Evidence index and conclusion")
    story += subsection("Key run IDs")
    story.append(
        table(
            [
                ["Evidence", "Selected configuration", "MLflow run ID"],
                ["Legacy retrieval evaluator", "fixed-hybrid-top3", "ba6464e78f014a6db1d21af5c39d508f"],
                ["Managed six-judge comparison", "fixed-hybrid-top5", "e6034e3cda2f4582ab5fac02132b4687"],
                ["Ablation baseline", "strict prompt, Qwen, temp 0.1", "f8eb6177b0a44c2e91cbd27926ca3efd"],
                ["Selected structured prompt", "structured prompt, Qwen, temp 0.1", "1528a96621d54a93819f01eab6b600a0"],
                ["Model comparison", "strict prompt, GPT-OSS-120B, temp 0.1", "6db207d1847d4d96af0f42905fcf1bf1"],
                ["Temperature comparison", "strict prompt, Qwen, temp 0.7", "caf08bcd4194455f9efc030ffc2a9634"],
            ],
            [51 * mm, 65 * mm, 54 * mm],
        )
    )
    story += subsection("Tracked evidence")
    for text in [
        "artifacts/rag_evaluation_summary.json - complete legacy eight-profile comparison.",
        "artifacts/genai_evaluation_comparison.json - complete managed six-judge ranking.",
        "artifacts/inference_ablation/comparison.json - complete prompt/model/temperature comparison.",
        "README.md - setup, architecture, dataset access, commands, evaluation results, and training exemption.",
        "MLOPS_LOG.md - chronological implementation and manual-verification record.",
        "CONTRIBUTION.md - final member-to-task allocation.",
    ]:
        story.append(bullet(text))
    story += subsection("Conclusion")
    story.append(
        p(
            "Milestone 1 is complete under the RAG-specific evaluation interpretation. The project now has a validated and reproducible corpus pipeline, explicit retrieval alternatives, tracked experiments, nested GenAI traces, a reusable dataset with versioned judges, complete inference comparisons, a served prediction API, a local interactive interface, documented tests, and auditable team contributions. The report evidence also makes limitations visible: the current held-out sample is intentionally small, external providers remain operational dependencies, and deployment is deferred. Within those boundaries, the repository provides a clean foundation for the next MLOps milestone."
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", color=GOLD, thickness=1.2))
    story.append(Spacer(1, 4 * mm))
    story.append(
        p(
            'Repository: <link href="https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline" color="#2867A8">MLOps-for-Legal-Aid-RAG-pipeline</link><br/>'
            'Submission branch: <b>8-m1-submission</b><br/>'
            'Public corpus: <link href="https://drive.google.com/drive/folders/103phLHYLT7Pv464nQ47e-lUl_J3DKquT?usp=drive_link" color="#2867A8">Google Drive corpus folder</link>',
            "SmallReport",
        )
    )
    return story


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    prepare_screenshot_crops()
    make_charts()
    required_images = [
        "corpus_drive_cropped.png",
        "mlflow_final_home_cropped.png",
        "fastapi_docs.png",
        "gradio_ui.png",
        "genai_profile_chart.png",
        "inference_ablation_chart.png",
    ]
    missing = [name for name in required_images if not (ASSETS / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing report assets: {missing}")

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="Legal-Aid RAG MLOps Pipeline - Milestone 1 Report",
        author="Karam Hussain, Aadesh Panjwani, Muhammad Rayyan Khan",
        subject="Milestone 1 model, API, MLflow tracking, and GenAI evaluation report",
    )
    document.build(build_story(), onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUTPUT)


if __name__ == "__main__":
    main()
