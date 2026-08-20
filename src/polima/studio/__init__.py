"""PoLiMa Studio, the SOM-resident robot operations cockpit."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Import Flask only when the board-side Studio is actually started."""
    from .app import create_app as factory

    return factory(*args, **kwargs)
