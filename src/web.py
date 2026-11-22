"""Production entry point: the JSON API plus the Miras Gradio interface.

``src.app:app`` remains the API on its own. This module mounts the Gradio
interface from ``gradio_app.py`` at ``/ui`` and redirects ``/`` to it, so the
public URL opens the interface while ``/docs``, ``/health``, ``/predict``, and
``/metrics`` keep working. The interface reuses the retrievers the API loads
at startup, so the models are loaded once per process, and each question
asked in the interface is counted in ``/metrics`` as ``POST /ui/predict``.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

import gradio_app
from src.app import create_app

UI_PATH = "/ui"
UI_PREDICT_ROUTE = f"{UI_PATH}/predict"


def create_web_app(**api_options: Any) -> FastAPI:
    """Create the API with the Gradio interface mounted at ``/ui``."""
    application = create_app(**api_options)

    def get_components(strategy: str) -> dict[str, Any]:
        retrievers = getattr(application.state, "retrievers", None) or {}
        if strategy not in retrievers:
            raise RuntimeError(
                "The legal search index is still loading. Try again in a minute."
            )
        return retrievers[strategy]

    def record(status_code: int, elapsed_ms: float) -> None:
        application.state.monitor.observe(
            "POST", UI_PREDICT_ROUTE, status_code, elapsed_ms
        )

    @application.get("/", include_in_schema=False)
    def open_interface() -> RedirectResponse:
        return RedirectResponse(url=f"{UI_PATH}/")

    demo = gradio_app.build_demo(get_components=get_components, record=record)
    return gr.mount_gradio_app(
        application,
        demo,
        path=UI_PATH,
        theme=gradio_app.THEME,
        css=gradio_app.CUSTOM_CSS,
        show_error=False,
    )


app = create_web_app()
