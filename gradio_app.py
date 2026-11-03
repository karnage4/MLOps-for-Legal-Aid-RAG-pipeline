"""Local Gradio interface for the MLflow-traced Legal-Aid RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Some model dependencies import Matplotlib on first use. Keep its generated
# font cache inside the ignored project cache so Windows user-profile
# permissions cannot stall the first prediction.
PROJECT_ROOT = Path(__file__).resolve().parent
MATPLOTLIB_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import gradio as gr

from src.app import (
    build_profile,
    load_retriever_components,
    run_rag_query,
)


APP_TITLE = "Miras — Inheritance Law Assistant"
DEFAULT_QUESTION = "What share does a daughter receive in Islamic inheritance?"
EXAMPLE_QUESTIONS = (
    "Who inherits when there is no will?",
    DEFAULT_QUESTION,
    "Can a non-Muslim inherit from a Muslim estate?",
    "How are debts settled before distributing the estate?",
    "Can a minor be an heir?",
)

CUSTOM_CSS = """
:root {
    --miras-ink: #20170c;
    --miras-cream: #fbf7ef;
    --miras-paper: #f3eadb;
    --miras-gold: #b88922;
    --miras-rust: #8b3f24;
    --miras-sage: #4f6b48;
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    background:
        radial-gradient(circle at 85% 5%, rgba(184, 137, 34, 0.11), transparent 26rem),
        var(--miras-cream) !important;
}

#miras-hero {
    position: relative;
    overflow: hidden;
    padding: 2.25rem 2.5rem;
    margin-bottom: 1.15rem;
    border: 1px solid rgba(184, 137, 34, 0.5);
    border-radius: 18px;
    color: #fffaf0;
    background: linear-gradient(135deg, #171008 0%, #2a1d0f 60%, #3b2712 100%);
    box-shadow: 0 18px 45px rgba(32, 23, 12, 0.16);
}

#miras-hero::after {
    content: "§";
    position: absolute;
    right: 2.2rem;
    top: -1.4rem;
    font-family: Georgia, serif;
    font-size: 10rem;
    color: rgba(212, 168, 67, 0.10);
}

#miras-hero h1 {
    margin: 0;
    color: #fffaf0;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    letter-spacing: -0.04em;
}

#miras-hero p {
    max-width: 680px;
    margin: 0.55rem 0 0;
    color: #d7b76d;
    font-size: 0.96rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

#settings-panel, #question-panel, #answer-panel, #details-panel {
    border: 1px solid rgba(110, 81, 36, 0.18) !important;
    border-radius: 16px !important;
    background: rgba(255, 255, 255, 0.82) !important;
    box-shadow: 0 10px 28px rgba(50, 34, 16, 0.07);
}

#settings-panel { padding: 1rem; }
#question-panel { padding: 1.15rem; }
#answer-panel { border-left: 5px solid var(--miras-gold) !important; }

#ask-button {
    color: #fffaf0 !important;
    border: none !important;
    background: linear-gradient(135deg, #20170c, #5a351a) !important;
}

#ask-button:hover { background: linear-gradient(135deg, #5a351a, #8b3f24) !important; }

.miras-kicker {
    color: var(--miras-gold);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.miras-note {
    margin-top: 1rem;
    padding: 0.9rem 1rem;
    border-radius: 10px;
    border: 1px solid rgba(184, 137, 34, 0.35);
    color: #5d4725;
    background: #fff8e8;
    font-size: 0.84rem;
    line-height: 1.5;
}

@media (prefers-color-scheme: dark) {
    .gradio-container {
        background: radial-gradient(circle at 85% 5%, #322713, transparent 26rem), #120e09 !important;
    }
    #settings-panel, #question-panel, #answer-panel, #details-panel {
        background: rgba(34, 27, 18, 0.96) !important;
        border-color: rgba(212, 168, 67, 0.25) !important;
    }
    .miras-note { color: #ead7ad; background: #2b2114; }
}
"""


def _clean_error(value: Any) -> str:
    """Return a compact provider-safe error without credentials or tracebacks."""
    return " ".join(str(value or "Unknown provider error").split())[:240]


def _format_context(chunks: list[dict[str, Any]], *, visible: bool) -> str:
    if not visible:
        return "Retrieved context is hidden. Enable ‘Show retrieved context’ and ask again to inspect it."

    sections = []
    for position, chunk in enumerate(chunks, 1):
        source = str(chunk.get("source") or "Unknown source")
        score = chunk.get("rerank_score")
        if score is None:
            score = chunk.get("semantic_score", chunk.get("rrf_score"))
        score_text = f" · score {float(score):.4f}" if isinstance(score, (int, float)) else ""
        text = " ".join(str(chunk.get("text") or "").split())
        sections.append(f"[{position}] {source}{score_text}\n{text[:3000]}")
    return "\n\n".join(sections)


def run_prediction(
    query: str,
    retrieval_mode: str,
    top_k: int,
    temperature: float,
    show_context: bool,
) -> tuple[str, str, str, dict[str, Any]]:
    """Run the unchanged RAG boundary and adapt its result for the UI."""
    question = " ".join(str(query or "").split())
    if len(question) < 3:
        raise ValueError("Enter a legal question containing at least 3 characters.")
    if len(question) > 1000:
        raise ValueError("Questions are limited to 1,000 characters.")
    if retrieval_mode not in {"hybrid", "semantic"}:
        raise ValueError("Retrieval mode must be hybrid or semantic.")

    profile = build_profile(retrieval_mode=retrieval_mode, top_k=int(top_k))
    result = run_rag_query(
        question,
        load_retriever_components(),
        temperature=float(temperature),
        profile=profile,
    )
    chunks = list(result.get("retrieval", {}).get("chunks") or [])
    if not chunks:
        raise RuntimeError("No relevant legal context was retrieved. Try rephrasing the question.")

    generation = result.get("generation") or {}
    if generation.get("error") or not generation.get("answer"):
        raise RuntimeError(
            "Answer generation failed: " + _clean_error(generation.get("error"))
        )

    timings = result.get("timings") or {}
    sources = sorted(
        {str(chunk["source"]) for chunk in chunks if chunk.get("source") is not None}
    )
    provider = str(generation.get("provider") or "unknown")
    model = str(generation.get("model_used") or "unknown")
    fallback_used = bool(generation.get("fallback_used"))

    status_text = (
        f"Provider: **{provider}** · Model: **{model}** · "
        f"Fallback: **{'yes' if fallback_used else 'no'}** · "
        f"Retrieved: **{len(chunks)} chunks / {len(sources)} sources**"
    )
    details = {
        "profile": {
            "chunking_strategy": "recursive",
            "retrieval_mode": retrieval_mode,
            "top_k": int(top_k),
            "temperature": float(temperature),
        },
        "generation": {
            "provider": provider,
            "model": model,
            "fallback_used": fallback_used,
        },
        "timings_ms": {
            "retrieval": int(timings.get("retrieval_ms") or 0),
            "generation": int(timings.get("generation_ms") or 0),
            "total": int(timings.get("total_ms") or 0),
        },
        "sources": sources,
    }
    return (
        str(generation["answer"]),
        status_text,
        _format_context(chunks, visible=bool(show_context)),
        details,
    )


def submit_prediction(*args: Any) -> tuple[str, str, str, dict[str, Any]]:
    """Translate safe application errors into a Gradio user notification."""
    try:
        return run_prediction(*args)
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error("The RAG service could not complete this request.") from exc


def build_demo() -> gr.Blocks:
    """Build the local/Hugging Face-compatible Gradio Blocks application."""
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(
            """
            <section id="miras-hero">
              <div class="miras-kicker">Legal inheritance · retrieval augmented generation</div>
              <h1>⚖ Miras</h1>
              <p>Grounded inheritance-law research with visible sources and MLflow tracing</p>
            </section>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280, elem_id="settings-panel"):
                gr.Markdown("### Research settings")
                retrieval_mode = gr.Radio(
                    choices=[("Hybrid + reranking", "hybrid"), ("Semantic only", "semantic")],
                    value="hybrid",
                    label="Retrieval mode",
                )
                top_k = gr.Slider(3, 10, value=5, step=1, label="Chunks retrieved")
                temperature = gr.Slider(
                    0.0,
                    1.0,
                    value=0.2,
                    step=0.05,
                    label="Generation temperature",
                    info="Lower values are preferred for legal-information answers.",
                )
                show_context = gr.Checkbox(value=True, label="Show retrieved context")
                gr.HTML(
                    """
                    <div class="miras-note">
                      <strong>General information only.</strong><br>
                      This assistant is a research aid, not personalized legal advice.
                    </div>
                    """
                )

            with gr.Column(scale=3, min_width=520):
                with gr.Group(elem_id="question-panel"):
                    query = gr.Textbox(
                        label="Your legal question",
                        placeholder="What share does a daughter receive in Islamic inheritance?",
                        lines=4,
                        max_lines=8,
                        max_length=1000,
                    )
                    with gr.Row():
                        ask = gr.Button("Ask Miras →", variant="primary", elem_id="ask-button")
                        clear = gr.ClearButton(value="Clear", components=[query])
                    gr.Examples(
                        examples=[[question] for question in EXAMPLE_QUESTIONS],
                        inputs=[query],
                        label="Example questions",
                    )

                with gr.Group(elem_id="answer-panel"):
                    answer = gr.Textbox(label="Answer", lines=12, interactive=False)
                    status = gr.Markdown("Provider and retrieval details will appear here.")

                with gr.Accordion("Retrieved context", open=False, elem_id="details-panel"):
                    context = gr.Textbox(show_label=False, lines=14, interactive=False)
                with gr.Accordion("Run metadata", open=False):
                    metadata = gr.JSON(label="Pipeline details")

        inputs = [query, retrieval_mode, top_k, temperature, show_context]
        outputs = [answer, status, context, metadata]
        ask.click(submit_prediction, inputs=inputs, outputs=outputs, api_name="predict")
        query.submit(submit_prediction, inputs=inputs, outputs=outputs)

    return demo.queue(default_concurrency_limit=1)


demo = build_demo()


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=False,
        theme=gr.themes.Base(primary_hue="amber", neutral_hue="stone"),
        css=CUSTOM_CSS,
    )
