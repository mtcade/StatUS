"""
    Uses the library `svg.py` to render board states to an SVG of hexagons
"""

import hexathello.engine as engine
from typing import Literal, Optional, Protocol, Self, TypedDict
import svg

import numpy as np

def centerPoint_from_cellStatus(
    qr: engine.QRTuple,
    hexagon_radius: float,
    canvas_width: float,
    canvas_height: float
    ) -> tuple[ float, float ]:
    from math import sqrt
    
    canvas_center: tuple[ float, float ] = (
        canvas_width*0.5, canvas_height*0.5
    )
    
    canvas_offset: tuple[ float, float ] = (
        hexagon_radius*(sqrt(3)*qr[0] + sqrt(3)*qr[1]*0.5),
        hexagon_radius*3*qr[1]*0.5
    )
    
    return (
        canvas_center[0] + canvas_offset[0],
        canvas_center[1] + canvas_offset[1]
    )
#/def center_point_from_cellStatus

def hexagon_svg_list(
    centerPoint: tuple[ float, float ],
    hexagon_radius: float
    ) -> list[ tuple[ float, float ] ]:
    """
        Get the svg canvas points as a series of x,y tuples
    """
    from math import sqrt
    # Get the x offset for the up/down offsets, and
    # four side points,
    half_radius: float = 0.5*hexagon_radius
    x_radius: float = 0.5*sqrt(3)*hexagon_radius
    
    points_list: list[ tuple[ float, float ] ] = [
        ( centerPoint[0], centerPoint[1]-hexagon_radius ), # 12:00
        ( centerPoint[0]+x_radius, centerPoint[1]-half_radius ), # 2:00
        ( centerPoint[0]+x_radius, centerPoint[1]+half_radius ), # 4:00
        ( centerPoint[0], centerPoint[1]+hexagon_radius ), # 6:00
        ( centerPoint[0]-x_radius, centerPoint[1]+half_radius ), # 8:00
        ( centerPoint[0]-x_radius, centerPoint[1]-half_radius )  # 10:00
    ]
    return points_list
#/def hexagonPoints

def hexagon_from_cellStatus(
    cellStatus: engine.CellStatus,
    size: int,
    hexagon_radius: float,
    stroke: str,
    fill: str,
    stroke_width: float,
    canvas_width: float,
    canvas_height: float
    ) -> svg.Polygon:
    """
        Literal hexagonal cell
    """
    
    center_svg: tuple[ float, float ] = centerPoint_from_cellStatus(
        qr = ( cellStatus["q"], cellStatus["r"] ),
        hexagon_radius = hexagon_radius,
        canvas_width = canvas_width,
        canvas_height = canvas_height
    )
    
    points_list: list[ tuple[ float, float ] ] = hexagon_svg_list(
        centerPoint = center_svg,
        hexagon_radius = hexagon_radius
    )
    # Unrap point list into a one level list
    
    points: list[ float ] = sum(
        [
            [ point[0], point[1] ] for point in points_list
        ],
        []
    )
    
    return svg.Polygon(
        points = points,
        stroke = stroke,
        fill = fill,
        stroke_width = stroke_width
    )
#/def hexagon_from_cellStatus

def polygon_fill_for_owner(
    owner: int | None,
    ) -> str:
    if owner is None:
        return "#00AA77"
    elif owner == 0:
        return "#FFFFFF"
    elif owner == 1:
        return "#000000"
    else:
        raise Exception("Unrecognized owner={}".format(owner))
    #
#/def polygon_fill_for_owner

def from_boardState(
    boardState: engine.BoardState,
    size: int,
    hexagon_radius: float,
    hexagon_stroke: str,
    hexagon_stroke_width: float,
    canvas_width: float,
    canvas_height: float
    ) -> svg.SVG:
    """
        Initializes svg with the given dimensions, filling in the hexagons
    """
    return svg.SVG(
        width = canvas_width,
        height = canvas_height,
        elements = [
            hexagon_from_cellStatus(
                cellStatus,
                size = size,
                hexagon_radius = hexagon_radius,
                stroke = hexagon_stroke,
                fill = polygon_fill_for_owner( cellStatus["owner"] ),
                stroke_width = hexagon_stroke_width,
                canvas_width = canvas_width,
                canvas_height = canvas_height
            ) for cellStatus in boardState.values()
        ]
    )
#/def from_boardState

def canvasSize_for_gameSize(
    size: int,
    hexagon_radius: float
    ) -> tuple[ float, float ]:
    """
    :param int size: Number of hexagons per side of board
    :param float hexagon_radius: Distance from center of hexagon to a point
    :returns: Tuple of ( canvas_width, canvas_height )
    :rtype: tuple[ float, float ]
    
    The center row of the pointy topped board is `2*size-1` hexagons
    """
    from math import sqrt
    return (
        (2*size-1)*sqrt(3)*hexagon_radius,
        ((2*size-1)*3*0.5 + 0.5)*hexagon_radius
    )
#/def canvasSize_for_gameSize

def from_boardState_with_hexagonRadius(
    boardState: engine.BoardState | np.ndarray,
    size: int,
    hexagon_radius: float,
    hexagon_stroke: str,
    hexagon_stroke_width: float,
    player_count: int | None = None,
    hexagonGridHelper: engine.HexagonGridHelper | None = None
    ) -> svg.SVG:
    """
        Calculate canvas size to fit the game, the get the game svg of that size. Alternative to using ``.from_boardState()``
    """
    canvasSize: tuple[ float, float ] = canvasSize_for_gameSize(
        size = size,
        hexagon_radius = hexagon_radius
    )
    
    # Convert boardState to
    if isinstance( boardState, np.ndarray ):
        boardState = engine.get_boardState_from_vector(
            boardState,
            hexagonGridHelper = hexagonGridHelper,
            player_count = player_count,
            size = size
        )
    #/if isinstance( boardState, np.ndarray )
    
    return from_boardState(
        boardState = boardState,
        size = size,
        hexagon_radius = hexagon_radius,
        hexagon_stroke = hexagon_stroke,
        hexagon_stroke_width = hexagon_stroke_width,
        canvas_width = canvasSize[0],
        canvas_height = canvasSize[1]
    )
#/def from_boardState_with_hexagonRadius
