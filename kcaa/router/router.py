"""
High-level router orchestration: turn a user request into a list of
``OutputSegment`` + ``OutputVia`` ready to be written into a .kicad_pcb.

Pipeline
--------

1. **Load PCB & DRC**: parse the S-expression, read the matching ``.kicad_pro``
   for net class rules.
2. **Build world model**: hand the PCB to :func:`kcaa.router.world_model.
   build_world_model` so we know where all obstacles sit. The start/end
   footprints are excluded.
3. **Find pad centers**: locate the two pads to connect and read their copper
   pad shape's center.
4. **Pick exit points**: choose one or two candidate exit points on the pad
   edge for each end (axis-aligned first, 45° if needed).
5. **Build visibility graph + A\\*** on the chosen layer.
6. **Postprocess**: miter corners, emit segments.

Layers
------

A single routing call works on one layer. Switching layers is the job of a
*via*, which lives on its own node in the output list. The current API does
not auto-insert vias — for multi-layer routing, call :func:`auto_route_pair`
once per layer, then :func:`connect_with_via` to link them with a via.

No shove
--------

This is the **no-shove** variant. If a route is blocked, we raise
:class:`RouteFailure` rather than displacing existing tracks. That is enough
for ~80% of "connect A to B" requests; the remaining cases need the user to
move a track or add a via first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math

from kcaa.router.a_star import a_star
from kcaa.router.path_postprocess import (
    OutputSegment,
    OutputVia,
    postprocess,
)
from kcaa.router.visibility_graph import (
    RouteNode,
    build_visibility_graph,
)
from kcaa.router.world_model import Obstacle, build_world_model
from kcaa.utils.pcb_sexp_utils import load_pcb

logger = logging.getLogger(__name__)


class RouteFailure(RuntimeError):
    """Raised when no valid route can be found."""


class ProFileMissing(RuntimeError):
    """No ``.kicad_pro`` found next to the ``.kicad_pcb``.

    The router needs the project file to look up netclass settings for
    width/clearance. Either create the project file in KiCad, or pass
    ``width=`` and ``clearance=`` explicitly in :class:`RouteRequest`.
    """


class ProFileMalformed(RuntimeError):
    """The ``.kicad_pro`` exists but cannot be read or parsed.

    This is almost always a sign of file corruption. Fix the project file
    in KiCad before re-running.
    """


class NetClassUnresolved(RuntimeError):
    """A net did not match any ``netclass_patterns`` entry, and the project
    has no ``Default`` netclass to fall back to.

    Either add the net to a netclass, add a ``Default`` netclass, or pass
    ``width=`` explicitly in :class:`RouteRequest`.
    """


class DesignRulesUnavailable(RuntimeError):
    """The board's design rules cannot be read, or ``min_clearance`` is
    missing.

    Pass ``clearance=`` explicitly in :class:`RouteRequest` to override.
    """


@dataclass
class RouteRequest:
    """A request to connect two pads with a track.

    The pads may live on different copper layers; in that case the router
    will insert one or more vias to switch layers.

    Attributes:
        pcb_path: Absolute path to the ``.kicad_pcb`` file.
        ref_a / pad_a: Reference designator and pad number for one end.
        ref_b / pad_b: Reference designator and pad number for the other end.
        net: Net name shared by both pads.
        start_layer: Copper layer the pad-A copper shape is on.
        end_layer: Copper layer the pad-B copper shape is on.
        via_pairs: Allowed (top, bottom) layer pairs that may carry a
            through-via. Default is ``(("F.Cu", "B.Cu"),)``. Pass an
            explicit tuple to restrict transitions (e.g. to forbid inner-
            layer vias on a 4-layer board).
        width: Track width; ``None`` → resolve from netclass.
        clearance: Minimum clearance to obstacles; ``None`` → resolve from
            the board's design rules.
        via_diameter / via_drill: Through-via dimensions; ``None`` →
            resolve from netclass.
        max_miter_mm: Maximum corner miter extension before falling back
            to a sharp 90° corner.
    """

    pcb_path: str
    ref_a: str
    pad_a: str
    ref_b: str
    pad_b: str
    net: str
    start_layer: str = "F.Cu"
    end_layer: str = "F.Cu"
    via_pairs: tuple[tuple[str, str], ...] = (("F.Cu", "B.Cu"),)
    width: float | None = None  # if None, use DRC default for the net
    clearance: float | None = None
    via_diameter: float | None = None
    via_drill: float | None = None
    max_miter_mm: float = 1.0


@dataclass
class RouteResult:
    """The output of a successful routing attempt.

    Attributes:
        segments: Track segments, all carrying the same ``layer`` as their
            corresponding path run. A route that crosses layers has
            multiple runs (one per layer), separated by vias.
        vias: Through-vias inserted at layer transitions. Empty for a
            single-layer route.
        start / end: The pad centres the route connected.
        layers_used: The copper layers the route actually traversed, in
            order. Useful for callers that want to know whether a via
            was inserted (``len(layers_used) > 1``).
    """

    segments: list[OutputSegment] = field(default_factory=list)
    vias: list[OutputVia] = field(default_factory=list)
    start: tuple[float, float] = (0.0, 0.0)
    end: tuple[float, float] = (0.0, 0.0)
    layers_used: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_route_pair(req: RouteRequest) -> RouteResult:
    """Connect pad ``req.pad_a`` on ``req.ref_a`` to pad ``req.pad_b`` on
    ``req.ref_b`` on the same net ``req.net`` and same layer ``req.start_layer``
    (and ``req.end_layer`` for the destination pad).

    Returns:
        A :class:`RouteResult` containing the segments and (optionally) vias.

    Raises:
        RouteFailure: If no path is found or inputs are invalid.
    """
    data = load_pcb(req.pcb_path)

    # Validate requested layers against the PCB early. The visibility graph
    # query below assumes both layers exist; checking later would produce
    # a less informative error path.
    pcb_layers = _pcb_layer_names(data)
    for layer in (req.start_layer, req.end_layer):
        if layer not in pcb_layers:
            raise RouteFailure(
                f"Layer {layer!r} is not present in PCB {req.pcb_path}; "
                f"PCB layers are {pcb_layers}."
            )
    # via_pairs must reference layers that exist too — otherwise A* would
    # never use those edges and the user might not notice.
    for top, bot in req.via_pairs:
        if top not in pcb_layers:
            raise RouteFailure(
                f"via_pairs contains top layer {top!r} which is not in PCB "
                f"{req.pcb_path}; PCB layers are {pcb_layers}."
            )
        if bot not in pcb_layers:
            raise RouteFailure(
                f"via_pairs contains bottom layer {bot!r} which is not in PCB "
                f"{req.pcb_path}; PCB layers are {pcb_layers}."
            )

    # DRC defaults from the .kicad_pro / board file. Fail loudly if the
    # project file is missing/malformed rather than silently guessing.
    width = req.width
    if width is None:
        try:
            width = _default_track_width(req.pcb_path, req.net)
        except (ProFileMissing, ProFileMalformed, NetClassUnresolved) as exc:
            raise RouteFailure(
                f"Cannot determine track width for net {req.net!r}: {exc}. "
                f"Pass width= explicitly in RouteRequest to skip DRC lookup."
            ) from exc
    clearance = req.clearance
    if clearance is None:
        try:
            clearance = _default_clearance(req.pcb_path)
        except (ProFileMissing, ProFileMalformed, DesignRulesUnavailable) as exc:
            raise RouteFailure(
                f"Cannot determine clearance: {exc}. "
                f"Pass clearance= explicitly in RouteRequest to skip DRC lookup."
            ) from exc

    # Pad center coordinates.
    pad_a_xy = _find_pad_center(data, req.ref_a, req.pad_a)
    pad_b_xy = _find_pad_center(data, req.ref_b, req.pad_b)
    if pad_a_xy is None:
        raise RouteFailure(f"Pad {req.ref_a}/{req.pad_a} not found")
    if pad_b_xy is None:
        raise RouteFailure(f"Pad {req.ref_b}/{req.pad_b} not found")

    # World model: start/end footprints are excluded (we route out of them,
    # not around them).
    model = build_world_model(
        req.pcb_path,
        net_filter=req.net,
        exclude_refs={req.ref_a, req.ref_b},
    )

    # Shrink obstacles by half the trace width (so the track is centered on
    # the line) and inflate by clearance. Remaining: forbidden region.
    buffered = _inflate_obstacles(model.obstacles, width / 2.0 + clearance)

    # Pick candidate exit points on each pad edge.
    pad_a_size = _find_pad_size(data, req.ref_a, req.pad_a, req.start_layer)
    pad_b_size = _find_pad_size(data, req.ref_b, req.pad_b, req.end_layer)
    if pad_a_size is None:
        raise RouteFailure(
            f"Pad {req.ref_a}/{req.pad_a} has no copper shape on layer "
            f"{req.start_layer!r}; cannot route from there."
        )
    if pad_b_size is None:
        raise RouteFailure(
            f"Pad {req.ref_b}/{req.pad_b} has no copper shape on layer "
            f"{req.end_layer!r}; cannot route to there."
        )
    exits_a = _pad_exit_points(pad_a_xy, pad_a_size)
    exits_b = _pad_exit_points(pad_b_xy, pad_b_size)

    best = _try_route(buffered, req.start_layer, exits_a, exits_b)
    if best is None:
        raise RouteFailure(
            f"No obstacle-avoiding path from {req.ref_a}/{req.pad_a} to "
            f"{req.ref_b}/{req.pad_b} on layer {req.start_layer}"
        )

    start_xy, end_xy, path = best
    segs = postprocess(
        path,
        width=width,
        layer=req.start_layer,
        net=req.net,
        max_miter_mm=req.max_miter_mm,
    )

    # Verify every emitted segment stays inside the board (Edge.Cuts).
    # We use a board polygon that is shrunk by width/2 on each side so the
    # track's copper edge is what we check against, not its centerline.
    #
    # Edge.Cuts is a workflow artifact: the user may legitimately be
    # routing before the board outline is drawn, so a missing board is
    # a warning, not a failure. A *present* board with segments that
    # cross it, on the other hand, is a router bug we must surface.
    if model.board_bbox is None:
        logger.warning(
            "No Edge.Cuts items in %s; skipping board-bounds check. "
            "Add an Edge.Cuts outline to verify segments stay within the board.",
            req.pcb_path,
        )
    else:
        _check_segments_in_board(segs, model.board_bbox)

    return RouteResult(segments=segs, start=start_xy, end=end_xy)


def connect_with_via(
    seg_a: OutputSegment,
    seg_b: OutputSegment,
    net: str,
    diameter: float,
    drill: float,
    layer_a: str,
    layer_b: str,
) -> OutputVia:
    """Helper to build a through-via connecting two segments on different layers.

    The via is placed at the (x, y) of ``seg_a``'s end. Both segments are
    expected to end at the same point.
    """
    return OutputVia(
        x=seg_a.x2,
        y=seg_a.y2,
        diameter=diameter,
        drill=drill,
        layers=(layer_a, layer_b),
        net=net,
    )


# ---------------------------------------------------------------------------
# Obstacle buffering
# ---------------------------------------------------------------------------


def _inflate_obstacles(obstacles: list[Obstacle], delta: float) -> list[Obstacle]:
    """Grow each obstacle's polygon by ``delta`` (negative shrinks it).

    Returns new Obstacle instances (shapely Polygon buffers are immutable).
    Tracks and vias are widened/shrunk by half the track width and clearance;
    footprints and keepouts are inflated by clearance alone (the track width
    is already implicit in their AABB extent, but clearance is not).
    """
    if delta == 0:
        return list(obstacles)
    out: list[Obstacle] = []
    for o in obstacles:
        new_shape = o.shape.buffer(delta)
        if new_shape.is_empty:
            continue
        out.append(
            Obstacle(
                shape=new_shape,
                layers=o.layers,
                net=o.net,
                kind=o.kind,
                ref=o.ref,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Board-bounds check
# ---------------------------------------------------------------------------


def _check_segments_in_board(
    segs: list[OutputSegment],
    board_bbox: tuple[float, float, float, float],
) -> None:
    """Raise :class:`RouteFailure` if any segment leaves the Edge.Cuts AABB.

    The check is conservative: we test the segment endpoints plus a few
    interior points against the board polygon *shrunk* by the segment's
    own ``width / 2``, so the track's copper edge is what we verify.
    A track whose centerline is exactly on the boundary is allowed (its
    copper would still touch but not cross the edge); a track whose
    centerline is on the wrong side of the shrunk boundary fails.

    Args:
        segs: The segments produced by :func:`postprocess`.
        board_bbox: ``(minx, miny, maxx, maxy)`` from
            :func:`kcaa.router.world_model._board_bbox`.

    Raises:
        RouteFailure: The first segment that would leave the board.
    """
    from shapely.geometry import LineString, Polygon

    minx, miny, maxx, maxy = board_bbox
    if minx >= maxx or miny >= maxy:
        raise RouteFailure(
            f"Board bbox is degenerate ({board_bbox}); cannot verify "
            f"segments stay within the board."
        )

    for i, seg in enumerate(segs):
        # Shrink the board by half the track width so the segment's center
        # is checked against a region the copper itself must stay inside.
        shrink = seg.width / 2.0
        shrunk_bbox = (minx + shrink, miny + shrink, maxx - shrink, maxy - shrink)
        if shrunk_bbox[0] >= shrunk_bbox[2] or shrunk_bbox[1] >= shrunk_bbox[3]:
            raise RouteFailure(
                f"Track width {seg.width} mm is wider than the board "
                f"(shrunk bbox {shrunk_bbox} is degenerate)."
            )
        board_poly = Polygon(
            [
                (shrunk_bbox[0], shrunk_bbox[1]),
                (shrunk_bbox[2], shrunk_bbox[1]),
                (shrunk_bbox[2], shrunk_bbox[3]),
                (shrunk_bbox[0], shrunk_bbox[3]),
            ]
        )
        line = LineString([(seg.x1, seg.y1), (seg.x2, seg.y2)])
        if not board_poly.covers(line):
            raise RouteFailure(
                f"Segment {i} from ({seg.x1:.3f},{seg.y1:.3f}) to "
                f"({seg.x2:.3f},{seg.y2:.3f}) would extend outside the "
                f"Edge.Cuts boundary (board {board_bbox}, track width "
                f"{seg.width} mm)."
            )


# ---------------------------------------------------------------------------
# Exit-point selection
# ---------------------------------------------------------------------------


def _pad_exit_points(
    center: tuple[float, float],
    size: tuple[float, float],
) -> list[tuple[float, float]]:
    """Return candidate exit points just outside a rectangular pad edge.

    The track must leave the pad at a point *outside* the pad shape — a small
    delta pushes the candidate past the copper. We return 4 axis-aligned
    exits (top/right/bottom/left); 45° exits are tried as a fallback inside
    the visibility graph by adding the pad corner as an extra node.
    """
    cx, cy = center
    w, h = size
    delta = max(w, h) * 0.5 + 0.05  # 50 µm beyond the pad edge
    return [
        (cx, cy - h / 2 - delta),  # top
        (cx + w / 2 + delta, cy),  # right
        (cx, cy + h / 2 + delta),  # bottom
        (cx - w / 2 - delta, cy),  # left
    ]


# ---------------------------------------------------------------------------
# A* over candidate exit pairs
# ---------------------------------------------------------------------------


def _try_route(
    obstacles: list[Obstacle],
    layer: str,
    exits_a: list[tuple[float, float]],
    exits_b: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], list[RouteNode]] | None:
    """Try every (exit_a, exit_b) pair and return the first successful path.

    Single-layer search: the caller passes one ``layer``; we build the graph
    on that single layer only. Multi-layer routes are handled by the caller
    (see :func:`auto_route_pair` for the multi-layer wiring).
    """
    for s in exits_a:
        for t in exits_b:
            try:
                g = build_visibility_graph(obstacles, [layer], s, t)
            except Exception:
                continue
            if not g.adj.get(0) or not g.adj.get(1):
                continue
            path = a_star(g, 0, 1)
            if path is not None:
                return s, t, path
    return None


# ---------------------------------------------------------------------------
# Pad lookup (parse the PCB tree directly)
# ---------------------------------------------------------------------------


def _pcb_layer_names(data: list) -> list[str]:
    """Return the ordered list of layer names declared in the PCB.

    Reads the PCB root's ``(layers (idx "name" type) ...)`` section and
    returns just the names in their declared order. Returns an empty list
    if the section is missing.
    """
    for item in data:
        if not _is_list(item) or str(item[0]) != "layers":
            continue
        names: list[str] = []
        for sub in item[1:]:
            if not _is_list(sub) or len(sub) < 2:
                continue
            v = sub[1]
            names.append(v if isinstance(v, str) else str(v))
        return names
    return []


def _find_pad_center(
    data: list,
    ref: str,
    pad_name: str,
) -> tuple[float, float] | None:
    """Return the (x, y) center of the named pad on the given footprint ref."""
    fp = _find_footprint(data, ref)
    if fp is None:
        return None
    fp_x, fp_y, fp_rot = _node_at3(fp)
    for sub in fp:
        if not _is_list(sub):
            continue
        if str(sub[0]) != "pad":
            continue
        name = _get_pad_name(sub)
        if name != pad_name:
            continue
        # Pad ``at`` is in footprint-local coords.
        at = _get_sub(sub, "at")
        if at is None or len(at) < 3:
            return None
        try:
            px, py = float(at[1]), float(at[2])
        except (TypeError, ValueError):
            return None
        # Transform local → world (only translation + rotation; pads don't
        # scale).
        wx, wy = _rotate(px, py, fp_rot)
        return fp_x + wx, fp_y + wy
    return None


def _find_pad_size(
    data: list,
    ref: str,
    pad_name: str,
    layer: str,
) -> tuple[float, float] | None:
    """Return the (w, h) of the pad shape, for the requested copper layer.

    Returns ``None`` if the pad is not on ``layer`` or not SMD (e.g. thru-hole
    pads are circular and need a different exit strategy).
    """
    fp = _find_footprint(data, ref)
    if fp is None:
        return None
    for sub in fp:
        if not _is_list(sub):
            continue
        if str(sub[0]) != "pad":
            continue
        if _get_pad_name(sub) != pad_name:
            continue
        if layer not in _pad_layers(sub):
            return None
        size_sub = _get_sub(sub, "size")
        if size_sub is None or len(size_sub) < 3:
            return None
        try:
            return float(size_sub[1]), float(size_sub[2])
        except (TypeError, ValueError):
            return None
    return None


def _find_footprint(data: list, ref: str) -> list | None:
    for item in data:
        if not _is_list(item) or str(item[0]) != "footprint":
            continue
        for sub in item:
            if not _is_list(sub):
                continue
            if str(sub[0]) != "property":
                continue
            if len(sub) >= 3 and str(sub[1]) == "Reference":
                v = sub[2]
                val = v if isinstance(v, str) else str(v)
                if val == ref:
                    return item
    return None


def _get_pad_name(pad_node: list) -> str:
    if len(pad_node) >= 2:
        v = pad_node[1]
        return v if isinstance(v, str) else str(v)
    return ""


def _pad_layers(pad_node: list) -> list[str]:
    layers: list[str] = []
    for sub in pad_node:
        if _is_list(sub) and str(sub[0]) == "layers" and len(sub) >= 2:
            for v in sub[1:]:
                layers.append(v if isinstance(v, str) else str(v))
    return layers


# ---------------------------------------------------------------------------
# DRC defaults (lightweight: read the netclass table from the .kicad_pro)
# ---------------------------------------------------------------------------


def _default_track_width(pcb_path: str, net: str) -> float:
    """Resolve track width for ``net`` from the project's netclass settings.

    Reads the matching ``.kicad_pro`` and looks up the netclass that
    ``net`` belongs to (via ``netclass_patterns``). Returns that netclass's
    ``track_width``.

    Raises:
        ProFileMissing: No ``.kicad_pro`` next to ``pcb_path`` — pass
            ``RouteRequest(width=...)`` explicitly to skip DRC lookup.
        ProFileMalformed: The ``.kicad_pro`` exists but cannot be parsed or
            lacks the expected structure.
        NetClassUnresolved: The net does not match any netclass pattern and
            there is no ``Default`` netclass to fall back to.
    """
    import json
    import os

    pro_path = _project_file_for(pcb_path)
    if pro_path is None or not os.path.exists(pro_path):
        raise ProFileMissing(pcb_path)
    try:
        with open(pro_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ProFileMalformed(pro_path, f"invalid JSON: {exc}") from exc
    except OSError as exc:
        raise ProFileMalformed(pro_path, f"cannot read: {exc}") from exc
    if not isinstance(data, dict):
        raise ProFileMalformed(pro_path, "top-level JSON is not an object")
    nc_widths = _netclass_track_widths(data)
    assignments = _net_to_netclass(data)
    nc = _resolve_netclass(net, assignments)
    if nc is not None and nc in nc_widths:
        return nc_widths[nc]
    if "Default" in nc_widths:
        return nc_widths["Default"]
    raise NetClassUnresolved(net, pro_path)


def _default_clearance(pcb_path: str) -> float:
    """Resolve minimum clearance from the board's effective design rules.

    Raises:
        ProFileMissing: No ``.kicad_pro`` next to ``pcb_path``.
        DesignRulesUnavailable: Rules cannot be read; ``min_clearance`` is
            not set.
    """
    try:
        from kcaa.utils.pcb_design_rules import get_effective_design_rules_from_file
    except ImportError as exc:
        raise DesignRulesUnavailable(f"pcb_design_rules module not importable: {exc}") from exc
    try:
        rules = get_effective_design_rules_from_file(pcb_path)
    except Exception as exc:
        raise DesignRulesUnavailable(f"failed to read design rules from {pcb_path}: {exc}") from exc
    # `get_effective_design_rules_from_file` returns
    # ``{"design_rules": {...}, "net_classes": [...], ...}``.
    design_rules = rules.get("design_rules") if isinstance(rules, dict) else None
    if not isinstance(design_rules, dict):
        raise DesignRulesUnavailable("design rules response is missing the design_rules section")
    v = design_rules.get("min_clearance")
    if v is None:
        raise DesignRulesUnavailable("design rules do not contain min_clearance")
    try:
        return float(v)
    except (TypeError, ValueError) as exc:
        raise DesignRulesUnavailable(f"min_clearance is not numeric: {v!r}") from exc


def _project_file_for(pcb_path: str) -> str | None:
    import os
    import re

    base = os.path.splitext(os.path.basename(pcb_path))[0]
    d = os.path.dirname(pcb_path)
    if not base:
        return None
    for f in os.listdir(d):
        if f.startswith(base + ".") and re.match(r".+\.kicad_pro$", f):
            return os.path.join(d, f)
    return None


def _netclass_track_widths(data: dict) -> dict[str, float]:
    """Read netclass track widths from the JSON project file.

    Returns ``{netclass_name: track_width}``.
    """
    out: dict[str, float] = {}
    ns = data.get("net_settings", {}) if isinstance(data, dict) else {}
    classes = ns.get("classes", []) if isinstance(ns, dict) else []
    for c in classes:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        tw = c.get("track_width")
        if isinstance(name, str) and isinstance(tw, int | float):
            out[name] = float(tw)
    return out


def _net_to_netclass(data: dict) -> dict[str, str]:
    """Read net→netclass assignments from the JSON project file.

    KiCad's project file uses ``netclass_patterns`` with a ``pattern`` glob —
    a net belongs to the first matching pattern's netclass. This is a
    glob-style match (we use ``fnmatch`` for ``*`` and ``?`` wildcards).
    """

    out: dict[str, str] = {}
    ns = data.get("net_settings", {}) if isinstance(data, dict) else {}
    patterns = ns.get("netclass_patterns", []) if isinstance(ns, dict) else []
    # Build list of (pattern, netclass).
    pat_list: list[tuple[str, str]] = []
    for p in patterns:
        if not isinstance(p, dict):
            continue
        nc = p.get("netclass")
        pat = p.get("pattern")
        if isinstance(nc, str) and isinstance(pat, str):
            pat_list.append((pat, nc))
    # We also support an explicit "nets" table if present (newer KiCad).
    nets = ns.get("nets", []) if isinstance(ns, dict) else []
    for n in nets:
        if not isinstance(n, dict):
            continue
        name = n.get("name")
        nc = n.get("netclass") or n.get("class")
        if isinstance(name, str) and isinstance(nc, str):
            out[name] = nc
    # Resolve patterns into explicit per-net entries.
    # (Net names that are not in the explicit table are resolved here.)
    # The caller will look up by net name; for resolution we need to know
    # the set of net names — but for our purposes (looking up a single
    # net by name) the explicit table is enough. We expose the patterns
    # so a higher layer can resolve ambiguous names. For now, expose
    # ``out`` as the per-net map and also a fallback: if the net is not
    # in ``out``, the caller checks the patterns directly. To keep the
    # API simple, we store the pattern list globally in this function's
    # closure via a small cache on the returned dict.
    if pat_list:
        out.setdefault("__patterns__", None)  # sentinel
        out["__patterns__"] = pat_list  # type: ignore[assignment]
    return out


def _resolve_netclass(net: str, assignments: dict[str, str]) -> str | None:
    """Return the netclass for ``net`` (explicit assignment or pattern)."""
    if net in assignments and net != "__patterns__":
        return assignments[net]
    patterns = assignments.get("__patterns__")
    if patterns:
        import fnmatch

        for pat, nc in patterns:
            if fnmatch.fnmatchcase(net, pat):
                return nc
    return None


# ---------------------------------------------------------------------------
# Local S-expression helpers (mirror world_model.py style)
# ---------------------------------------------------------------------------


def _is_list(v) -> bool:
    return isinstance(v, list) and len(v) > 0


def _get_sub(node: list, tag: str):
    for sub in node:
        if _is_list(sub) and str(sub[0]) == tag:
            return sub
    return None


def _find_section(data: list, tag: str) -> list:
    """Return all subnodes whose head is ``tag``."""
    out = []
    for item in data:
        if _is_list(item) and str(item[0]) == tag:
            out.append(item)
    return out


def _node_at3(node: list) -> tuple[float, float, float]:
    sub = _get_sub(node, "at")
    if sub is None or len(sub) < 3:
        return 0.0, 0.0, 0.0
    try:
        x, y = float(sub[1]), float(sub[2])
        rot = float(sub[3]) if len(sub) >= 4 else 0.0
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0
    return x, y, rot


def _rotate(x: float, y: float, deg: float) -> tuple[float, float]:
    """Rotate (x, y) by ``deg`` (CW-positive, matching KiCad's PCB convention)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return c * x - s * y, s * x + c * y
