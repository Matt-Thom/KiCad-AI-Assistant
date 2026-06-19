"""Unit tests for kcaa.router.path_postprocess."""

from __future__ import annotations

from kcaa.router.path_postprocess import (
    OutputSegment,
    OutputVia,
    emit_segment_nodes,
    emit_via_nodes,
    postprocess,
)
from kcaa.router.visibility_graph import RouteNode


def _path(*coords: tuple[float, float], layer: str = "F.Cu") -> list[RouteNode]:
    return [RouteNode(x, y, layer, i) for i, (x, y) in enumerate(coords)]


class TestPostprocess:
    def test_empty_path(self):
        assert postprocess([], 0.25, "F.Cu", "VCC") == []

    def test_single_node(self):
        assert postprocess(_path((0.0, 0.0)), 0.25, "F.Cu", "VCC") == []

    def test_straight_line(self):
        segs = postprocess(_path((0.0, 0.0), (10.0, 0.0)), 0.25, "F.Cu", "VCC")
        assert len(segs) == 1
        assert segs[0].x1 == 0.0
        assert segs[0].x2 == 10.0

    def test_collinear_simplified(self):
        # 3 collinear points should collapse to 1 segment
        segs = postprocess(_path((0.0, 0.0), (5.0, 0.0), (10.0, 0.0)), 0.25, "F.Cu", "VCC")
        assert len(segs) == 1

    def test_l_shape_gets_miter(self):
        # Path: (0,0) → (5,0) → (5,5)
        # Miter cuts a 45° diagonal from (4,0) to (5,5), shortening the first
        # axis-aligned segment. We expect 2 segments after mitering.
        segs = postprocess(
            _path((0.0, 0.0), (5.0, 0.0), (5.0, 5.0)), 0.25, "F.Cu", "VCC", max_miter_mm=1.0
        )
        assert len(segs) == 2
        # First segment shortened by 1 mm → ends at (4, 0)
        assert segs[0].x2 == 4.0
        assert segs[0].y2 == 0.0
        # Second segment is the 45° miter cut from (4,0) to (5,5)
        assert segs[1].x1 == 4.0
        assert segs[1].y1 == 0.0
        assert segs[1].x2 == 5.0
        assert segs[1].y2 == 5.0

    def test_miter_capped_by_max(self):
        # Long L — miter capped at max_miter_mm
        segs = postprocess(
            _path((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)), 0.25, "F.Cu", "VCC", max_miter_mm=2.0
        )
        # First segment is shortened by 2 mm: (0,0)→(8,0)
        assert segs[0].x2 == 8.0

    def test_segments_carry_layer_net_width(self):
        segs = postprocess(_path((0.0, 0.0), (5.0, 0.0)), 0.5, "B.Cu", "GND")
        assert segs[0].layer == "B.Cu"
        assert segs[0].net == "GND"
        assert segs[0].width == 0.5


class TestEmission:
    def test_emit_segment_node(self):
        seg = OutputSegment(0.0, 0.0, 5.0, 0.0, 0.25, "F.Cu", "VCC")
        nodes = emit_segment_nodes([seg])
        assert len(nodes) == 1
        node = nodes[0]
        # First element is the symbol "segment"
        assert str(node[0]) == "segment"
        # Width and layer are present
        flat = str(node)
        assert "0.25" in flat
        assert "F.Cu" in flat
        assert "VCC" in flat

    def test_emit_via_node(self):
        via = OutputVia(1.0, 2.0, 0.8, 0.4, ("F.Cu", "B.Cu"), "GND")
        nodes = emit_via_nodes([via])
        assert len(nodes) == 1
        node = nodes[0]
        assert str(node[0]) == "via"
        flat = str(node)
        assert "F.Cu" in flat
        assert "B.Cu" in flat
        assert "GND" in flat
