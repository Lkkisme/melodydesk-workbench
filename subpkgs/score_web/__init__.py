"""Subpackage 4: Dual-Track alphaTex Builder, Yoopu-Style Combined Score SVG/HTML Renderer, and FastAPI Web Workbench."""
from .alphatex_builder import to_alphatex
from .yoopu_svg_renderer import render_publication_score_html
from .server import create_app


def __getattr__(name: str):
    if name == "app":
        from . import server
        return server.app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "to_alphatex",
    "render_publication_score_html",
    "app",
    "create_app",
]
