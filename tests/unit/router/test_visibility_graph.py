"""Unit tests for kcaa.router.visibility_graph."""

from __future__ import annotations

from shapely.geometry import Polygon

from kcaa.router.visibility_graph import (
    RouteNode,
    VisibilityGraph,
    build_visibility_graph,
)
from kcaa.router.world_model import Obstacle


def _rect_obstacle(x1: float, y1: float, x2: float, y2: float, layer: str = "F.Cu") -> Obstacle:
    return Obstacle(
        shape=Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]),
        layers=frozenset({layer}),
        net=None,
        kind="footprint",
    )


class TestRouteNode:
    def test_distance(self):
        a = RouteNode(0.0, 0.0, "F.Cu", 0)
        b = RouteNode(3.0, 4.0, "F.Cu", 1)
        assert a.distance(b) == 5.0

    def test_distance_ignores_layer(self):
        a = RouteNode(0.0, 0.0, "F.Cu", 0)
        b = RouteNode(3.0, 4.0, "B.Cu", 1)
        assert a.distance(b) == 5.0


class TestGraphBuilding:
    def test_empty_obstacles_yields_only_start_and_end(self):
        g = build_visibility_graph([], "F.Cu", (0.0, 0.0), (10.0, 10.0))
        assert len(g.nodes) == 2
        assert g.adj[0] == {1}
        assert g.adj[1] == {0}

    def test_obstacle_vertices_become_nodes(self):
        # 5x5 obstacle between S and G
        obs = _rect_obstacle(4.0, 4.0, 6.0, 6.0)
        g = build_visibility_graph([obs], "F.Cu", (0.0, 0.0), (10.0, 10.0))
        # 2 endpoints + 4 unique corner vertices (closed ring yields 5 coords)
        # We accept 6 OR 7 because shapely may dedupe.
        assert 6 <= len(g.nodes) <= 7

    def test_obstacle_blocks_some_edges(self):
        obs = _rect_obstacle(4.0, 4.0, 6.0, 6.0)
        g = build_visibility_graph([obs], "F.Cu", (0.0, 0.0), (10.0, 10.0))
        end_node = next(n for n in g.nodes if (n.x, n.y) == (10.0, 10.0))
        start_node = next(n for n in g.nodes if (n.x, n.y) == (0.0, 0.0))
        assert end_node.node_id not in g.adj[start_node.node_id]

    def test_same_layer_only(self):
        obs_f = _rect_obstacle(4.0, 4.0, 6.0, 6.0, layer="F.Cu")
        obs_b = _rect_obstacle(4.0, 4.0, 6.0, 6.0, layer="B.Cu")
        g = build_visibility_graph([obs_f, obs_b], "F.Cu", (0.0, 0.0), (10.0, 10.0))
        # Only F.Cu obstacle contributes vertices (4 unique corners, possibly 5 with closing point)
        f_cu_vertex_count = sum(1 for n in g.nodes if 4.0 <= n.x <= 6.0 and 4.0 <= n.y <= 6.0)
        assert f_cu_vertex_count >= 4
        # No nodes from the B.Cu obstacle
        assert all(n.layer == "F.Cu" for n in g.nodes)

    def test_same_net_obstacle_excluded(self):
        # A track on the routing net — should not be added as obstacle
        obs = Obstacle(
            shape=Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 0.2), (0.0, 0.2)]),
            layers=frozenset({"F.Cu"}),
            net="VCC",  # same net as the route
            kind="track",
        )
        g = build_visibility_graph([obs], "F.Cu", (0.0, -1.0), (10.0, -1.0))
        # No obstacle vertices added
        assert len(g.nodes) == 2


class TestGraphAddEdge:
    def test_add_node_and_edge(self):
        g = VisibilityGraph()
        n1 = RouteNode(0.0, 0.0, "F.Cu", 0)
        n2 = RouteNode(1.0, 0.0, "F.Cu", 1)
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(0, 1)
        assert 1 in g.adj[0]
        assert 0 in g.adj[1]

    def test_add_edge_to_self_is_noop(self):
        g = VisibilityGraph()
        g.add_node(RouteNode(0.0, 0.0, "F.Cu", 0))
        g.add_edge(0, 0)
        assert g.adj[0] == set()

    def test_neighbors(self):
        g = VisibilityGraph()
        for i in range(3):
            g.add_node(RouteNode(float(i), 0.0, "F.Cu", i))
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        assert set(g.neighbors(0)) == {1, 2}
        assert set(g.neighbors(1)) == {0}
