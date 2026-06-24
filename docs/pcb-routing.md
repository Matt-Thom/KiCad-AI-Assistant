# PCB Routing Guide

The KiCad MCP server provides a **no-shove PNS router** that connects
two pads with an obstacle-avoiding track.  It is exposed through the
`pcb_route_pad_to_pad` MCP tool.

## Single-layer routing

```python
await pcb_route_pad_to_pad(
    pcb_path="/path/to/board.kicad_pcb",
    ref_a="R1", pad_a="2",
    ref_b="C1", pad_b="2",
    net="VCC",
    layer="F.Cu",        # optional, default "F.Cu"
    width=0.5,           # optional; uses netclass track width if omitted
)
```

The router walks the pad-exit points of `R1.2` and `C1.2` and A*'s a
visibility graph over the obstacle-free regions.  The track is added
to the board on the same layer as the source pad.  Obstacles include:

* Same-net tracks are *not* obstacles (you are extending them).
* Footprint courtyards for components other than the two endpoints.
* Existing tracks of *other* nets.
* Keepout zones on the active layer.

## Multi-layer routing (via insertion)

If the source pad is on a different layer from the destination pad,
the router will insert through-hole vias at every layer transition.
You must supply the destination layer and the set of via pairs the
router is allowed to use:

```python
await pcb_route_pad_to_pad(
    pcb_path="/path/to/board.kicad_pcb",
    ref_a="R1", pad_a="2",
    ref_b="U1", pad_b="5",
    net="VCC",
    layer="F.Cu",                # start layer
    target_layer="In1.Cu",        # end layer
    via_pairs=(("F.Cu", "B.Cu"), ("B.Cu", "In1.Cu")),
)
```

If `target_layer` is supplied and differs from `layer` and
`via_pairs` is omitted, the router defaults to a single F<->B pair.

### Response shape

The tool returns a dict with these keys (in addition to the keys for
the single-layer case):

| Key | Type | Description |
| --- | --- | --- |
| `via_count` | `int` | Number of vias written to the board. |
| `vias` | `list[dict]` | Each dict has `x`, `y`, `diameter`, `drill`, `layers` (`[from, to]`), and `net`. |
| `layers_used` | `list[str]` | Ordered, deduplicated list of layers the path traversed. |

### Via cost

The default via cost is `2.0 + 0.5 * n` millimetres, where `n` is the
number of vias the route has already taken.  A two-via path costs
`2.0 + 2.5 = 4.5 mm` of via overhead.  This biases the router toward
fewer-layer solutions when both are viable.

### Adding a single via manually

For stitching two pre-existing tracks that already live on different
layers, use the standalone via tool:

```python
await pcb_connect_with_via(
    pcb_path="/path/to/board.kicad_pcb",
    x=40.0, y=25.0,
    net="GND",
    diameter=0.8,   # optional, default 0.8
    drill=0.4,      # optional, default 0.4
    layers=("F.Cu", "B.Cu"),  # optional, default F<->B
)
```

## Failure modes

| Failure | Meaning |
| --- | --- |
| `RouteFailure: Pad <ref>/<pad> not found` | Pad name is wrong or pad is on a different layer than requested. |
| `RouteFailure: Pad <ref>/<pad> has no copper shape on layer '<layer>'` | The destination pad does not have a copper shape on the requested `end_layer`. |
| `RouteFailure: No obstacle-avoiding path from <a> to <b> across layers [...]` | No combination of exit points / via transitions yields a clear path.  Check obstacles and via pairs. |
| `RouteFailure: Via at (x, y) would extend outside the board` | At least one via pad does not fit inside the Edge.Cuts board outline.  Move the route or shrink the board. |

The tool returns `{"error": "..."}` on any of the above; the board
file is **not** modified on failure.
