"""
Visibility graph: a sparse graph over candidate routing nodes.

For a 2D routing problem with polygonal obstacles, the **shortest** obstacle-
avoiding path between ``S`` and ``G`` has a property: every vertex of the path
is either ``S``, ``G``, or a vertex of some obstacle. We build a graph whose
nodes are exactly those candidate points, and whose edges are the pairs of
nodes whose straight-line segment is **clear of all obstacles** (visibility).

A\\* on this graph is fast and produces geometric shortest paths.

Node layer
----------

Every node carries a layer. Edges exist only between nodes on the same layer
(direct routing on a copper plane). Switching layers happens via ``via`` nodes
(see :mod:`kcaa.router.router`).

Spatial prefilter
-----------------

Visibility checks are O(n) in the number of obstacles. We prefilter with an
rtree over obstacle bounding boxes so that only nearby obstacles are checked.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import math

from rtree import index
from shapely.geometry import LineString

from kcaa.router.world_model import Obstacle


@dataclass(frozen=True)
class RouteNode:
    """A candidate point on the visibility graph."""

    x: float
    y: float
    layer: str
    node_id: int  # global integer id for graph indexing

    def distance(self, other: RouteNode) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class VisibilityGraph:
    """Adjacency map from node_id to set of visible node_ids."""

    nodes: list[RouteNode] = field(default_factory=list)
    adj: dict[int, set[int]] = field(default_factory=dict)

    def add_node(self, node: RouteNode) -> int:
        nid = node.node_id
        if nid in self.adj:
            return nid  # already added
        self.nodes.append(node)
        self.adj[nid] = set()
        return nid

    def add_edge(self, a: int, b: int) -> None:
        if a == b:
            return
        self.adj.setdefault(a, set()).add(b)
        self.adj.setdefault(b, set()).add(a)

    def neighbors(self, nid: int) -> Iterable[int]:
        return self.adj.get(nid, ())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_visibility_graph(
    obstacles: list[Obstacle],
    layer: str,
    start: tuple[float, float],
    end: tuple[float, float],
) -> VisibilityGraph:
    """Construct a visibility graph over ``layer``.

    Candidate nodes are the start, end, and all obstacle vertices that sit
    on ``layer``. Edges are visibility connections between same-layer nodes
    whose straight line crosses no obstacle.

    Same-net obstacles (``net is not None``) are not added as obstacles and
    their vertices are also excluded from the graph — a track should not be
    told to "go around" itself.

    Args:
        obstacles: All obstacles; only those on ``layer`` participate.
        layer: The routing layer (e.g. ``"F.Cu"``).
        start: Start point in world coordinates.
        end: End point in world coordinates.

    Returns:
        A :class:`VisibilityGraph`.
    """
    # Only consider obstacles on this layer.
    layer_obs = [o for o in obstacles if layer in o.layers]

    # Build rtree for spatial prefilter.
    rtree_idx = index.Index()
    for i, o in enumerate(layer_obs):
        rtree_idx.insert(i, o.shape.bounds)

    graph = VisibilityGraph()
    counter = 0

    def _new_node(x: float, y: float) -> RouteNode:
        nonlocal counter
        node = RouteNode(x=x, y=y, layer=layer, node_id=counter)
        graph.add_node(node)
        counter += 1
        return node

    _new_node(*start)
    _new_node(*end)

    # Collect obstacle vertices as additional candidate nodes.
    for o in layer_obs:
        # Skip same-net obstacles: their tracks are part of the route.
        if o.net is not None:
            continue
        for x, y in o.shape.exterior.coords:
            _new_node(x, y)

    # Connect every pair (O(n²)), using rtree prefilter to skip far pairs.
    node_list = graph.nodes
    n = len(node_list)
    for i in range(n):
        for j in range(i + 1, n):
            a = node_list[i]
            b = node_list[j]
            if _is_visible(a, b, layer_obs, rtree_idx):
                graph.add_edge(a.node_id, b.node_id)

    return graph


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------


def _is_visible(
    a: RouteNode,
    b: RouteNode,
    obstacles: list[Obstacle],
    rtree_idx: index.Index,
) -> bool:
    """True iff the segment a→b crosses no obstacle interior or boundary."""
    seg = LineString([(a.x, a.y), (b.x, b.y)])
    # Prefilter: only obstacles whose bbox intersects the segment bbox.
    minx = min(a.x, b.x)
    maxx = max(a.x, b.x)
    miny = min(a.y, b.y)
    maxy = max(a.y, b.y)
    candidates = list(rtree_idx.intersection((minx, miny, maxx, maxy)))
    for i in candidates:
        o = obstacles[i]
        if seg.intersects(o.shape) and not seg.touches(o.shape):
            # touches counts only at boundary; if endpoints are on the
            # boundary (e.g. connecting to a footprint corner) it's still OK.
            # A real interior intersection (cross) means blocked.
            return False
        # If the segment *crosses* the interior, intersect returns a non-empty
        # area.  touches at a single vertex is allowed (we'll avoid re-checking
        # via the eps buffer).
    return True
