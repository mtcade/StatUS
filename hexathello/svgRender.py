"""
    Uses the library `svg.py` to render board states to an SVG of hexagons
"""

import hexathello.engine as engine

import svg

def from_boardState(
    boardState: engine.BoardState,
    width: int,
    height: int
    ) -> svg.SVG:
    return svg.SVG(
        width = width,
        height = height
    )
#/def from_boardState

