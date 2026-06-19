"""
A\\* shortest-path search on the visibility graph.

Standard textbook implementation with a Euclidean heuristic. Returns the
ordered list of :class:`kcaa.router.visibility_graph.RouteNode` from start
to goal, or ``None`` if no path exists.
"""

from __future__ import annotations

from collections.abc import Callable
import heapq

from kcaa.router.visibility_graph import RouteNode, VisibilityGraph


def a_star(
    graph: VisibilityGraph,
    start_id: int,
    goal_id: int,
    heuristic: Callable[[RouteNode, RouteNode], float] | None = None,
) -> list[RouteNode] | None:
    """A\\* shortest-path search.

    Args:
        graph: The visibility graph to search.
        start_id: ``node_id`` of the start node.
        goal_id: ``node_id`` of the goal node.
        heuristic: Optional heuristic function. Defaults to Euclidean.

    Returns:
        Ordered list of :class:`RouteNode` from start to goal, or ``None``
        if no path exists.
    """
    if start_id == goal_id:
        node = _node_by_id(graph, start_id)
        return [node] if node else None

    if heuristic is None:
        heuristic = lambda a, b: a.distance(b)  # noqa: E731

    start_node = _node_by_id(graph, start_id)
    goal_node = _node_by_id(graph, goal_id)
    if start_node is None or goal_node is None:
        return None

    open_heap: list[tuple[float, int, RouteNode]] = []
    heapq.heappush(open_heap, (0.0, 0, start_node))
    counter = 0  # tiebreaker to avoid comparing RouteNode

    g_score: dict[int, float] = {start_id: 0.0}
    came_from: dict[int, int] = {}

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current.node_id == goal_id:
            return _reconstruct(graph, came_from, current.node_id)
        for nid in graph.neighbors(current.node_id):
            tentative_g = g_score[current.node_id] + current.distance(_node_by_id(graph, nid))
            if tentative_g < g_score.get(nid, float("inf")):
                came_from[nid] = current.node_id
                g_score[nid] = tentative_g
                neighbor = _node_by_id(graph, nid)
                f = tentative_g + heuristic(neighbor, goal_node)
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _node_by_id(graph: VisibilityGraph, nid: int) -> RouteNode | None:
    if 0 <= nid < len(graph.nodes):
        return graph.nodes[nid]
    return None


def _reconstruct(
    graph: VisibilityGraph,
    came_from: dict[int, int],
    end_id: int,
) -> list[RouteNode]:
    path_ids = [end_id]
    cur = end_id
    while cur in came_from:
        cur = came_from[cur]
        path_ids.append(cur)
    path_ids.reverse()
    return [graph.nodes[nid] for nid in path_ids]
