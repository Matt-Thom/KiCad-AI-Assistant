"""
PCB routing tools for the KiCad MCP server.

Exposes the no-shove PNS router as an MCP tool that connects two pads with
a track on a single layer.  The tool writes the resulting segments and (if
present) vias back to the .kicad_pcb file, with the usual ``.bak`` backup.

This is the no-shove variant: if a route is blocked, the tool fails rather
than displacing existing tracks.  Use the placement / edit tools to clear
the path first, or call with a different layer.
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import Context, FastMCP
import sexpdata

from kcaa.router.path_postprocess import OutputSegment, OutputVia
from kcaa.router.router import (
    RouteFailure,
    RouteRequest,
    auto_route_pair,
    connect_with_via,
)
from kcaa.utils.pcb_sexp_utils import load_pcb, save_pcb

log = logging.getLogger(__name__)


def register_pcb_routing_tools(mcp: FastMCP) -> None:
    """Register PCB routing tools with the MCP server."""

    @mcp.tool()
    async def pcb_route_pad_to_pad(
        pcb_path: str,
        ref_a: str,
        pad_a: str,
        ref_b: str,
        pad_b: str,
        net: str,
        ctx: Context | None,
        layer: str = "F.Cu",
        width: float | None = None,
        target_layer: str | None = None,
        via_pairs: tuple[tuple[str, str], ...] | None = None,
    ) -> dict[str, Any]:
        """Connect two pads with an obstacle-avoiding track, optionally across layers.

        Uses the no-shove PNS router: if the path is blocked by an existing
        track or footprint courtyard, the call fails rather than moving
        anything.  Run the placement tools first to clear the way, or call
        again with a different ``layer``.

        PCB coordinates: mm, +X right, **+Y down**, rotation
        **clockwise-positive** (KiCad PCB convention).

        The track's width defaults to the net's netclass ``track_width`` from
        the matching ``.kicad_pro`` (or 0.25 mm if no project file is
        found).  Clearance is taken from the board's effective design rules
        (see :mod:`kcaa.utils.pcb_design_rules`).

        Args:
            pcb_path: Absolute path to the .kicad_pcb file.
            ref_a: Reference designator of the first footprint (e.g. ``"R1"``).
            pad_a: Pad number on ``ref_a`` (e.g. ``"1"``).
            ref_b: Reference designator of the second footprint.
            pad_b: Pad number on ``ref_b``.
            net: Net name to assign to the new segments.
            ctx: MCP context (unused).
            layer: Starting copper layer (``"F.Cu"`` by default).
            width: Override the netclass track width (mm).  ``None`` uses the
                DRC default for the net.
            target_layer: Destination copper layer.  When ``None`` (default)
                the route stays on ``layer``.  When set, the router may
                insert through-hole vias to reach the destination.
            via_pairs: Optional tuple of ``(from_layer, to_layer)`` pairs
                the router is allowed to use as via transitions.  Defaults
                to ``(("F.Cu", "B.Cu"),)`` when ``target_layer`` differs
                from ``layer``; ignored otherwise.

        Returns:
            dict with:
                segment_count: number of segments written.
                segments: list of dicts ``{x1, y1, x2, y2, width, layer, net}``.
                via_count: number of vias written (0 for single-layer).
                vias: list of dicts ``{x, y, diameter, drill, layers, net}``.
                layers_used: ordered list of layers touched by the path.
                start: ``(x, y)`` exit point of pad_a.
                end: ``(x, y)`` entry point of pad_b.
                backup_path: path to the ``.bak`` created before writing.
                pcb_path: echo of the input path.

            Or ``{"error": "<message>"}`` on failure.
        """
        if target_layer is None:
            target_layer = layer
        if via_pairs is None and target_layer != layer:
            via_pairs = (("F.Cu", "B.Cu"),)
        req = RouteRequest(
            pcb_path=pcb_path,
            ref_a=ref_a,
            pad_a=pad_a,
            ref_b=ref_b,
            pad_b=pad_b,
            net=net,
            start_layer=layer,
            end_layer=target_layer,
            width=width,
            via_pairs=via_pairs or (),
        )
        try:
            result = auto_route_pair(req)
        except RouteFailure as exc:
            return {"error": str(exc)}
        except (FileNotFoundError, ValueError) as exc:
            return {"error": f"Routing input error: {exc}"}

        # Load the PCB and append the new segments and vias.
        data = load_pcb(pcb_path)
        for seg in result.segments:
            data.append(_segment_to_sexp(seg))
        for via in result.vias:
            data.append(_via_to_sexp(via))
        try:
            backup_path = save_pcb(pcb_path, data)
        except OSError as exc:
            return {"error": f"Failed to write PCB file: {exc}"}

        return {
            "segment_count": len(result.segments),
            "segments": [
                {
                    "x1": s.x1,
                    "y1": s.y1,
                    "x2": s.x2,
                    "y2": s.y2,
                    "width": s.width,
                    "layer": s.layer,
                    "net": s.net,
                }
                for s in result.segments
            ],
            "via_count": len(result.vias),
            "vias": [
                {
                    "x": v.x,
                    "y": v.y,
                    "diameter": v.diameter,
                    "drill": v.drill,
                    "layers": [v.layers[0], v.layers[1]],
                    "net": v.net,
                }
                for v in result.vias
            ],
            "layers_used": list(result.layers_used),
            "start": list(result.start),
            "end": list(result.end),
            "backup_path": backup_path,
            "pcb_path": pcb_path,
        }

    @mcp.tool()
    async def pcb_connect_with_via(
        pcb_path: str,
        x: float,
        y: float,
        net: str,
        ctx: Context,
        diameter: float = 0.8,
        drill: float = 0.4,
        layers: tuple[str, str] = ("F.Cu", "B.Cu"),
    ) -> dict[str, Any]:
        """Insert a single through-hole via at a given point.

        PCB coordinates: mm, +X right, **+Y down**.  Use this after
        ``pcb_route_pad_to_pad`` calls on two different layers to stitch
        them together.

        Args:
            pcb_path: Absolute path to the .kicad_pcb file.
            x: Via x coordinate (mm).
            y: Via y coordinate (mm).
            net: Net name.
            ctx: MCP context.
            diameter: Pad diameter of the via (mm).  Default 0.8.
            drill: Drill diameter (mm).  Default 0.4.
            layers: Two-element tuple of copper layers the via connects.

        Returns:
            dict with ``via`` info and ``backup_path``.
        """
        via = OutputVia(
            x=x,
            y=y,
            diameter=diameter,
            drill=drill,
            layers=tuple(layers),
            net=net,
        )
        data = load_pcb(pcb_path)
        data.append(_via_to_sexp(via))
        try:
            backup_path = save_pcb(pcb_path, data)
        except OSError as exc:
            return {"error": f"Failed to write PCB file: {exc}"}
        return {
            "via": {
                "x": via.x,
                "y": via.y,
                "diameter": via.diameter,
                "drill": via.drill,
                "layers": list(via.layers),
                "net": via.net,
            },
            "backup_path": backup_path,
            "pcb_path": pcb_path,
        }


# ---------------------------------------------------------------------------
# S-expression emission (board-format strings)
# ---------------------------------------------------------------------------


def _segment_to_sexp(seg: OutputSegment) -> list:
    """Build a (segment ...) node in the standard board format."""
    return [
        sexpdata.Symbol("segment"),
        [sexpdata.Symbol("start"), seg.x1, seg.y1],
        [sexpdata.Symbol("end"), seg.x2, seg.y2],
        [sexpdata.Symbol("width"), seg.width],
        [sexpdata.Symbol("layer"), seg.layer],
        [sexpdata.Symbol("net"), seg.net],
    ]


def _via_to_sexp(via: OutputVia) -> list:
    """Build a (via ...) node in the standard board format."""
    layers_node = [sexpdata.Symbol("layers"), via.layers[0], via.layers[1]]
    return [
        sexpdata.Symbol("via"),
        [sexpdata.Symbol("at"), via.x, via.y],
        [sexpdata.Symbol("size"), via.diameter],
        [sexpdata.Symbol("drill"), via.drill],
        layers_node,
        [sexpdata.Symbol("net"), via.net],
    ]


# Re-exported for callers that want to assemble multi-layer routes by hand.
__all__ = [
    "register_pcb_routing_tools",
    "connect_with_via",
]
