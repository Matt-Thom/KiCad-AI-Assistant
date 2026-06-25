"""
Pre-flight checks for adding vias to a PCB.

Used by :func:`kcaa.tools.pcb_routing_tools.pcb_add_vias` before any
write to the .kicad_pcb file.  A single failure rejects the whole
batch; the file is left untouched.

Three independent checks, all producing ``Violation`` records:

1. **Netclass rules** — the via's net has a netclass in the matching
   ``.kicad_pro``; if its ``via_diameter``/``via_drill`` disagree with
   what the user requested, report a mismatch.
2. **Position** — the via's pad ring (radius = diameter/2) must not
   overlap any forbidden obstacle on a layer it occupies.
3. **Board edge** — the via must stay inside the board outline with
   at least ``min_copper_edge_clearance`` (or 0 if not set).

The check is intentionally **strict**:

* ``.kicad_pro`` not found → error (no silent skip).
* net has no resolvable netclass and no ``Default`` → error.
* any footprint / track / via / keepout overlap → error.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import fnmatch
import json
import os
import re
from typing import Any

from shapely.geometry import Point

from kcaa.router.world_model import Obstacle, WorldModel, build_world_model

# Match the project's .kicad_pro next to a .kicad_pcb.
_PROJECT_FILE_RE = re.compile(r".+\.kicad_pro$")


def find_project_file(pcb_path: str) -> str | None:
    """Return the absolute path of the project's ``.kicad_pro``, or ``None``."""
    base = os.path.splitext(os.path.basename(pcb_path))[0]
    d = os.path.dirname(pcb_path)
    if not base or not d or not os.path.isdir(d):
        return None
    for entry in os.listdir(d):
        if entry.startswith(base + ".") and _PROJECT_FILE_RE.match(entry):
            return os.path.join(d, entry)
    return None


@dataclass(frozen=True)
class ProposedVia:
    """The via the user wants to drop.  Same shape as ``OutputVia`` minus net."""

    x: float
    y: float
    diameter: float
    drill: float
    layers: tuple[str, str]
    net: str


@dataclass(frozen=True)
class Violation:
    """One rule violation.  ``index`` is the position in the batch."""

    index: int
    kind: str  # "netclass" | "footprint" | "track" | "via" | "board_edge" | "keepout"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


def check_vias(pcb_path: str, vias: list[ProposedVia]) -> list[Violation]:
    """Run all pre-flight checks.  Returns an empty list when everything's OK.

    Raises nothing on bad data — failures are returned as :class:`Violation`
    records so the caller can format a single error message and reject the
    whole batch.
    """
    violations: list[Violation] = []

    # Netclass checks need the .kicad_pro up front; failure here short-circuits.
    nc_rules = _resolve_netclass_rules(pcb_path, [v.net for v in vias])
    if isinstance(nc_rules, str):
        # Single string means a hard error (no pro file / malformed).
        violations.append(Violation(-1, "project", nc_rules))
        return violations
    # nc_rules is now dict[net_name -> {"via_diameter": float|None, "via_drill": float|None}]
    for i, via in enumerate(vias):
        rule = nc_rules.get(via.net)
        if rule is None:
            violations.append(
                Violation(
                    i,
                    "netclass",
                    f"net {via.net!r} has no resolvable netclass "
                    f"(no matching pattern and no Default netclass)",
                    {"net": via.net},
                )
            )
            continue
        want_d = rule.get("via_diameter")
        want_r = rule.get("via_drill")
        if want_d is not None and abs(via.diameter - want_d) > 1e-3:
            violations.append(
                Violation(
                    i,
                    "netclass",
                    f"net {via.net!r} netclass expects via diameter {want_d} mm, "
                    f"got {via.diameter} mm",
                    {
                        "net": via.net,
                        "expected_diameter": want_d,
                        "actual_diameter": via.diameter,
                    },
                )
            )
        if want_r is not None and abs(via.drill - want_r) > 1e-3:
            violations.append(
                Violation(
                    i,
                    "netclass",
                    f"net {via.net!r} netclass expects via drill {want_r} mm, got {via.drill} mm",
                    {
                        "net": via.net,
                        "expected_drill": want_r,
                        "actual_drill": via.drill,
                    },
                )
            )

    # Position checks: build the world model once (without any of the new vias).
    # build_world_model already treats all existing tracks/vias as obstacles
    # except those on the via's own net — perfect for our needs.
    world = build_world_model(pcb_path)

    # Board-edge clearance, in mm.  0 if KiCad DRC didn't expose it.
    edge_clear = _min_copper_edge_clearance(pcb_path)

    for i, via in enumerate(vias):
        violations.extend(_check_position(i, via, world, edge_clear))

    return violations


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_netclass_rules(
    pcb_path: str, nets: Iterable[str]
) -> dict[str, dict[str, float]] | str:
    """Return ``{net: {"via_diameter": ..., "via_drill": ...}}``.

    Returns a string error message if the project file is missing or
    malformed (caller treats as a hard failure).
    """
    pro_path = find_project_file(pcb_path)
    if pro_path is None:
        return f"no .kicad_pro found next to {pcb_path!r} (cannot resolve netclass rules)"
    try:
        with open(pro_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return f"cannot read {pro_path!r}: {exc}"
    if not isinstance(data, dict):
        return f"{pro_path!r}: top-level JSON is not an object"

    ns = data.get("net_settings", {})
    if not isinstance(ns, dict):
        return f"{pro_path!r}: net_settings is not an object"

    classes_raw = ns.get("classes", [])
    patterns_raw = ns.get("netclass_patterns", [])
    if not isinstance(classes_raw, list) or not isinstance(patterns_raw, list):
        return f"{pro_path!r}: malformed net_settings.classes or netclass_patterns"

    # Class name -> {via_diameter, via_drill, ...}
    class_rules: dict[str, dict[str, float]] = {}
    default_name: str | None = None
    for c in classes_raw:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not isinstance(name, str):
            continue
        rule: dict[str, float] = {}
        for key in ("via_diameter", "via_drill"):
            val = c.get(key)
            if isinstance(val, int | float):
                rule[key] = float(val)
        class_rules[name] = rule
        if name == "Default":
            default_name = name

    # Net -> class name (explicit nets table first, then patterns).
    net_to_class: dict[str, str] = {}
    pat_list: list[tuple[str, str]] = []
    for p in patterns_raw:
        if not isinstance(p, dict):
            continue
        nc = p.get("netclass")
        pat = p.get("pattern")
        if isinstance(nc, str) and isinstance(pat, str):
            pat_list.append((pat, nc))

    nets_table = ns.get("nets", [])
    if isinstance(nets_table, list):
        for n in nets_table:
            if not isinstance(n, dict):
                continue
            name = n.get("name")
            nc = n.get("netclass") or n.get("class")
            if isinstance(name, str) and isinstance(nc, str):
                net_to_class[name] = nc

    # We also support an explicit assignment via class rules themselves,
    # since some project files attach nets to a class directly.
    for cls in classes_raw:
        if not isinstance(cls, dict):
            continue
        cls_name = cls.get("name")
        nets_field = cls.get("nets")
        if isinstance(cls_name, str) and isinstance(nets_field, list):
            for n in nets_field:
                if isinstance(n, str) and n not in net_to_class:
                    net_to_class[n] = cls_name

    out: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    for net in set(nets):
        cls = net_to_class.get(net)
        if cls is None:
            for pat, nc in pat_list:
                if fnmatch.fnmatchcase(net, pat):
                    cls = nc
                    break
        if cls is None and default_name is not None:
            cls = default_name
        if cls is None or cls not in class_rules:
            out[net] = None  # type: ignore[assignment]
        else:
            out[net] = class_rules[cls]
        seen.add(net)
    return out


def _min_copper_edge_clearance(pcb_path: str) -> float:
    """Read min_copper_edge_clearance from .kicad_pro if set, else 0."""
    pro_path = find_project_file(pcb_path)
    if pro_path is None:
        return 0.0
    try:
        with open(pro_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0.0
    if not isinstance(data, dict):
        return 0.0
    rules = data.get("design_rules", {})
    if not isinstance(rules, dict):
        return 0.0
    val = rules.get("min_copper_edge_clearance")
    if isinstance(val, int | float):
        return float(val)
    return 0.0


def _check_position(
    index: int,
    via: ProposedVia,
    world: WorldModel,
    edge_clear: float,
) -> list[Violation]:
    out: list[Violation] = []

    # Board edge: via center must be inside [minx+ec, maxx-ec] x [miny+ec, maxy-ec].
    if world.board_bbox is not None:
        minx, miny, maxx, maxy = world.board_bbox
        if (
            via.x < minx + edge_clear
            or via.x > maxx - edge_clear
            or via.y < miny + edge_clear
            or via.y > maxy - edge_clear
        ):
            out.append(
                Violation(
                    index,
                    "board_edge",
                    f"via at ({via.x}, {via.y}) is outside the board outline "
                    f"({minx}, {miny})-({maxx}, {maxy}) "
                    f"with required edge clearance {edge_clear} mm",
                    {
                        "x": via.x,
                        "y": via.y,
                        "board_bbox": [minx, miny, maxx, maxy],
                        "edge_clearance": edge_clear,
                    },
                )
            )

    # Pad ring on each copper layer the via occupies.
    radius = via.diameter / 2.0
    ring = Point(via.x, via.y).buffer(radius)
    layers = set(via.layers)

    for obs in world.obstacles:
        if not (layers & obs.layers):
            continue
        # Same-net existing track/via isn't a collision.
        if obs.net == via.net and obs.kind in ("track", "via"):
            continue
        if obs.shape.intersects(ring):
            desc = _describe_obstacle(obs, via)
            out.append(
                Violation(
                    index,
                    obs.kind,
                    f"via at ({via.x}, {via.y}) overlaps {obs.kind} {desc}",
                    {
                        "x": via.x,
                        "y": via.y,
                        "obstacle_kind": obs.kind,
                        "obstacle_net": obs.net,
                        "obstacle_ref": obs.ref,
                        "layers": sorted(layers & obs.layers),
                    },
                )
            )
    return out


def _describe_obstacle(obs: Obstacle, via: ProposedVia) -> str:
    """Short description for the error message."""
    if obs.ref:
        return f"on footprint {obs.ref!r}"
    if obs.net:
        return f"on net {obs.net!r}"
    return "(net unknown)"
