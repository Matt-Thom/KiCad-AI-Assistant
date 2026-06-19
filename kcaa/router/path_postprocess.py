"""
Path post-processing: convert an ordered list of :class:`RouteNode` from A\\*
into a sequence of ``(segment ...)`` S-expression nodes with mitered corners.

What this does
--------------

A\\* returns a polyline with arbitrary angles (often 90°).  PCB tracks prefer
**45° miters** at corners — that's both more manufacturable and visually
clean. This module walks the polyline and inserts intermediate points so
each interior corner becomes a 45° cut:

    A                A
     \\                \
      \\      →         *--C   (C is the mitered corner vertex)
       \\              /
        B            B

The miter is bounded by ``min(distance to prev, distance to next,
max_miter)`` so it doesn't overshoot a tight corner.

The post-processor is layer-aware; on output each segment carries its
layer and the trace width.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sexpdata

from kcaa.router.visibility_graph import RouteNode


@dataclass
class OutputSegment:
    """A single track segment ready to be emitted as a S-expression node."""

    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    layer: str
    net: str


@dataclass
class OutputVia:
    """A single via ready to be emitted as a S-expression node."""

    x: float
    y: float
    diameter: float
    drill: float
    layers: tuple[str, str]  # exactly two layers (a through-via)
    net: str


def postprocess(
    path: list[RouteNode],
    width: float,
    layer: str,
    net: str,
    max_miter_mm: float = 1.0,
) -> list[OutputSegment]:
    """Convert an A\\* polyline into mitered OutputSegments.

    The first and last segments keep their original orientation (so the
    track enters/exits the pad in the right direction). Interior corners
    get a 45° cut, length-bounded by ``max_miter_mm``.

    Args:
        path: Ordered list of :class:`RouteNode` from start to goal.
        width: Trace width in mm.
        layer: Copper layer name (e.g. ``"F.Cu"``).
        net: Net name.
        max_miter_mm: Maximum length of any single miter cut in mm.

    Returns:
        A list of :class:`OutputSegment` ready for serialization.
    """
    if len(path) < 2:
        return []

    # Simplify collinear runs first — A* often includes a vertex
    # exactly along a straight edge.
    pts = _simplify_collinear([(p.x, p.y) for p in path])
    if len(pts) < 2:
        return []

    out: list[OutputSegment] = []
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        out.append(OutputSegment(x1=x1, y1=y1, x2=x2, y2=y2, width=width, layer=layer, net=net))

    # Apply mitering in-place on the segments by splitting interior corners.
    out = _apply_miters(out, max_miter_mm)
    return out


# ---------------------------------------------------------------------------
# S-expression emission helpers
# ---------------------------------------------------------------------------


def emit_segment_nodes(segments: list[OutputSegment]) -> list[list[Any]]:
    """Convert a list of :class:`OutputSegment` into raw ``segment`` sexp nodes."""
    nodes: list[list[Any]] = []
    for s in segments:
        nodes.append(
            [
                sexpdata.Symbol("segment"),
                [sexpdata.Symbol("start"), s.x1, s.y1],
                [sexpdata.Symbol("end"), s.x2, s.y2],
                [sexpdata.Symbol("width"), s.width],
                [sexpdata.Symbol("layer"), s.layer],
                [sexpdata.Symbol("net"), s.net],
            ]
        )
    return nodes


def emit_via_nodes(vias: list[OutputVia]) -> list[list[Any]]:
    """Convert a list of :class:`OutputVia` into raw ``via`` sexp nodes."""
    nodes: list[list[Any]] = []
    for v in vias:
        layers = [sexpdata.Symbol("layers"), v.layers[0], v.layers[1]]
        nodes.append(
            [
                sexpdata.Symbol("via"),
                [sexpdata.Symbol("at"), v.x, v.y],
                [sexpdata.Symbol("size"), v.diameter],
                [sexpdata.Symbol("drill"), v.drill],
                layers,
                [sexpdata.Symbol("net"), v.net],
            ]
        )
    return nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simplify_collinear(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop intermediate vertices that lie on the line between their neighbors."""
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        px, py = out[-1]
        cx, cy = pts[i]
        nx, ny = pts[i + 1]
        # Cross product of (c - p) and (n - c) — collinear if ~0.
        cross = (cx - px) * (ny - cy) - (cy - py) * (nx - cx)
        if abs(cross) > 1e-9:
            out.append((cx, cy))
    out.append(pts[-1])
    return out


def _apply_miters(segments: list[OutputSegment], max_miter_mm: float) -> list[OutputSegment]:
    """At each interior join, replace two segments with three by inserting a 45° miter.

    Only interior corners whose two edges are axis-aligned (which is the
    A\\* output we generate) can be mitered cleanly. Diagonal corners are
    passed through unmodified.
    """
    if len(segments) < 2:
        return segments

    out: list[OutputSegment] = []
    for i in range(len(segments) - 1):
        a = segments[i]
        b = segments[i + 1]
        # Axis-aligned edges only.
        if not (_is_horizontal(a) or _is_vertical(a)):
            out.append(a)
            continue
        if not (_is_horizontal(b) or _is_vertical(b)):
            out.append(a)
            continue
        # Determine if the corner is convex (right turn for CW-positive y-down).
        if not _is_convex_corner(a, b):
            out.append(a)
            continue

        # Compute miter length: distance from a.end along a's axis to the
        # perpendicular through b.start. Capped by half the shorter adjacent
        # segment and by max_miter_mm.
        ax, ay = a.x2, a.y2  # shared corner
        miter_len = _compute_miter_length(a, b, max_miter_mm)
        if miter_len <= 0:
            out.append(a)
            continue
        if _is_horizontal(a):
            mx = ax + (miter_len if b.x2 > ax else -miter_len)
            my = ay
        else:
            mx = ax
            my = ay + (miter_len if b.y2 > ay else -miter_len)
        # Replace a's end with the miter point and insert a new segment
        # from miter point to b.start. Append the shortened a now, then
        # append the new segment.
        out.append(
            OutputSegment(x1=a.x1, y1=a.y1, x2=mx, y2=my, width=a.width, layer=a.layer, net=a.net)
        )
        # We will append a "fake" b whose start is (mx, my); the loop's
        # next iteration will then patch b's start to the miter point too.
        segments[i + 1] = OutputSegment(
            x1=mx, y1=my, x2=b.x2, y2=b.y2, width=b.width, layer=b.layer, net=b.net
        )
    out.append(segments[-1])
    return out


def _is_horizontal(s: OutputSegment) -> bool:
    return abs(s.y2 - s.y1) < 1e-9


def _is_vertical(s: OutputSegment) -> bool:
    return abs(s.x2 - s.x1) < 1e-9


def _is_convex_corner(a: OutputSegment, b: OutputSegment) -> bool:
    """True if the shared endpoint forms a convex (non-overlapping) join."""
    # Direction of a as it enters the corner.
    adx, ady = a.x2 - a.x1, a.y2 - a.y1
    # Direction of b as it leaves the corner.
    bdx, bdy = b.x2 - b.x1, b.y2 - b.y1
    # Cross product z-component (positive = left turn in screen-y-down coords,
    # which we treat as convex here).
    cross = adx * bdy - ady * bdx
    return cross > 0


def _compute_miter_length(a: OutputSegment, b: OutputSegment, cap: float) -> float:
    """Length of the 45° miter cut at the join of two axis-aligned segments."""
    # Available length is how far a travels before its x or y matches b.start's.
    if _is_horizontal(a):
        avail_a = abs(a.x2 - a.x1)
    else:
        avail_a = abs(a.y2 - a.y1)
    if _is_horizontal(b):
        avail_b = abs(b.x2 - b.x1)
    else:
        avail_b = abs(b.y2 - b.y1)
    # Miter can't exceed the shorter leg.
    return min(avail_a, avail_b, cap)
