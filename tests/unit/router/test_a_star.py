"""Unit tests for kcaa.router.a_star."""

from __future__ import annotations

import pytest

from kcaa.router.a_star import a_star
from kcaa.router.visibility_graph import RouteNode, VisibilityGraph


def _linear_graph() -> VisibilityGraph:
    """A → B → C, no obstacles. 0=start, 2=goal."""
    g = VisibilityGraph()
    for i, (x, y) in enumerate([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]):
        g.add_node(RouteNode(x, y, "F.Cu", i))
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    return g


def _disconnected_graph() -> VisibilityGraph:
    """Two disjoint chains: 0-1 and 2-3."""
    g = VisibilityGraph()
    for i, (x, y) in enumerate([(0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (11.0, 0.0)]):
        g.add_node(RouteNode(x, y, "F.Cu", i))
    g.add_edge(0, 1)
    g.add_edge(2, 3)
    return g


class TestAStar:
    def test_linear_path(self):
        path = a_star(_linear_graph(), 0, 2)
        assert path is not None
        assert [n.node_id for n in path] == [0, 1, 2]
        assert path[0].x == 0.0
        assert path[-1].x == 2.0

    def test_unreachable_returns_none(self):
        path = a_star(_disconnected_graph(), 0, 3)
        assert path is None

    def test_start_equals_goal(self):
        g = _linear_graph()
        path = a_star(g, 0, 0)
        assert path is not None
        assert len(path) == 1
        assert path[0].node_id == 0

    def test_picks_shortest_branch(self):
        # Triangle 0-1-2, 0-2 direct. A* should prefer direct edge.
        g = VisibilityGraph()
        for i, (x, y) in enumerate([(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)]):
            g.add_node(RouteNode(x, y, "F.Cu", i))
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)  # direct
        path = a_star(g, 0, 2)
        assert path is not None
        assert [n.node_id for n in path] == [0, 2]

    def test_invalid_start_returns_none(self):
        path = a_star(_linear_graph(), 999, 0)
        assert path is None

    def test_invalid_goal_returns_none(self):
        path = a_star(_linear_graph(), 0, 999)
        assert path is None

    def test_custom_heuristic_is_used(self):
        g = _linear_graph()
        # Zero heuristic — A* becomes Dijkstra; should still find a path
        path = a_star(g, 0, 2, heuristic=lambda a, b: 0.0)
        assert path is not None
        assert path[-1].node_id == 2

    def test_path_length_equals_sum_of_segments(self):
        path = a_star(_linear_graph(), 0, 2)
        assert path is not None
        total = 0.0
        for a, b in zip(path, path[1:]):
            total += a.distance(b)
        assert total == pytest.approx(2.0)
