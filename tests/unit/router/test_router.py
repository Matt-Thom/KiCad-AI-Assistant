"""
Unit tests for the high-level router orchestration.

Focus: DRC lookup (``_default_track_width``, ``_default_clearance``) and the
exception classes that propagate into :class:`RouteFailure`. The routing
algorithm itself is covered by
:mod:`tests.integration.test_pcb_routing`.

These tests enforce the *fail-loud* contract: when the project file is
missing, malformed, or lacks the netclass info we need, the router must
raise a specific subclass of :class:`RuntimeError` rather than silently
fall back to a guessed value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kcaa.router.path_postprocess import OutputSegment
from kcaa.router.router import (
    DesignRulesUnavailable,
    NetClassUnresolved,
    ProFileMalformed,
    ProFileMissing,
    RouteFailure,
    RouteRequest,
    _check_segments_in_board,
    _default_clearance,
    _default_track_width,
    _project_file_for,
    auto_route_pair,
)

# ---------------------------------------------------------------------------
# _project_file_for
# ---------------------------------------------------------------------------


def test_project_file_for_returns_pro_sibling(tmp_path: Path) -> None:
    pcb = tmp_path / "board.kicad_pcb"
    pro = tmp_path / "board.kicad_pro"
    pcb.write_text("(kicad_pcb)\n")
    pro.write_text("{}\n")
    assert _project_file_for(str(pcb)) == str(pro)


def test_project_file_for_missing_returns_none(tmp_path: Path) -> None:
    pcb = tmp_path / "board.kicad_pcb"
    pcb.write_text("(kicad_pcb)\n")
    assert _project_file_for(str(pcb)) is None


# ---------------------------------------------------------------------------
# _default_track_width
# ---------------------------------------------------------------------------


def _write_pro(tmp_path: Path, payload: dict) -> Path:
    pcb = tmp_path / "board.kicad_pcb"
    pro = tmp_path / "board.kicad_pro"
    pcb.write_text("(kicad_pcb)\n")
    pro.write_text(json.dumps(payload))
    return pcb


def _base_pro() -> dict:
    return {
        "net_settings": {
            "classes": [
                {
                    "name": "Default",
                    "track_width": 0.25,
                    "clearance": 0.2,
                }
            ],
            "netclass_patterns": [],
        }
    }


def test_default_track_width_uses_netclass_pattern(tmp_path: Path) -> None:
    payload = _base_pro()
    payload["net_settings"]["classes"].append(
        {"name": "Power", "track_width": 0.5, "clearance": 0.3}
    )
    payload["net_settings"]["netclass_patterns"].append({"netclass": "Power", "pattern": "VCC"})
    pcb = _write_pro(tmp_path, payload)
    assert _default_track_width(str(pcb), "VCC") == 0.5


def test_default_track_width_falls_back_to_default_class(tmp_path: Path) -> None:
    payload = _base_pro()
    pcb = _write_pro(tmp_path, payload)
    assert _default_track_width(str(pcb), "MysteryNet") == 0.25


def test_default_track_width_raises_when_pro_missing(tmp_path: Path) -> None:
    pcb = tmp_path / "board.kicad_pcb"
    pcb.write_text("(kicad_pcb)\n")
    with pytest.raises(ProFileMissing):
        _default_track_width(str(pcb), "VCC")


def test_default_track_width_raises_on_invalid_json(tmp_path: Path) -> None:
    pcb = tmp_path / "board.kicad_pcb"
    pro = tmp_path / "board.kicad_pro"
    pcb.write_text("(kicad_pcb)\n")
    pro.write_text("{not valid json")
    with pytest.raises(ProFileMalformed):
        _default_track_width(str(pcb), "VCC")


def test_default_track_width_raises_on_non_object_root(tmp_path: Path) -> None:
    pcb = tmp_path / "board.kicad_pcb"
    pro = tmp_path / "board.kicad_pro"
    pcb.write_text("(kicad_pcb)\n")
    pro.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ProFileMalformed):
        _default_track_width(str(pcb), "VCC")


def test_default_track_width_raises_when_no_default_and_no_match(tmp_path: Path) -> None:
    payload = {
        "net_settings": {
            "classes": [{"name": "Power", "track_width": 0.5, "clearance": 0.3}],
            "netclass_patterns": [{"netclass": "Power", "pattern": "VCC"}],
        }
    }
    pcb = _write_pro(tmp_path, payload)
    with pytest.raises(NetClassUnresolved):
        _default_track_width(str(pcb), "MysteryNet")


# ---------------------------------------------------------------------------
# _default_clearance
# ---------------------------------------------------------------------------


def test_default_clearance_reads_from_design_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``get_effective_design_rules_from_file`` returns a sane dict,
    the clearance flows through."""
    fake_rules = {"design_rules": {"min_clearance": 0.3}}
    monkeypatch.setattr(
        "kcaa.utils.pcb_design_rules.get_effective_design_rules_from_file",
        lambda _path: fake_rules,
    )
    assert _default_clearance("/some/path.kicad_pcb") == 0.3


def test_default_clearance_raises_when_section_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_rules = {"net_classes": []}
    monkeypatch.setattr(
        "kcaa.utils.pcb_design_rules.get_effective_design_rules_from_file",
        lambda _path: fake_rules,
    )
    with pytest.raises(DesignRulesUnavailable):
        _default_clearance("/some/path.kicad_pcb")


def test_default_clearance_raises_when_min_clearance_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_rules = {"design_rules": {"min_track_width": 0.2}}
    monkeypatch.setattr(
        "kcaa.utils.pcb_design_rules.get_effective_design_rules_from_file",
        lambda _path: fake_rules,
    )
    with pytest.raises(DesignRulesUnavailable):
        _default_clearance("/some/path.kicad_pcb")


def test_default_clearance_raises_when_reader_explodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_path: str) -> dict:
        raise OSError("kaboom")

    monkeypatch.setattr(
        "kcaa.utils.pcb_design_rules.get_effective_design_rules_from_file",
        boom,
    )
    with pytest.raises(DesignRulesUnavailable):
        _default_clearance("/some/path.kicad_pcb")


# ---------------------------------------------------------------------------
# auto_route_pair — DRC exception → RouteFailure translation
# ---------------------------------------------------------------------------


def _fixture_pcb() -> str:
    return str(
        Path(__file__).resolve().parents[2]
        / "integration"
        / "fixtures"
        / "test_routing_board.kicad_pcb"
    )


def test_auto_route_pair_translates_pro_missing_into_route_failure(tmp_path: Path) -> None:
    """If the .kicad_pro is missing and no width/clearance is given,
    auto_route_pair should raise RouteFailure, not silently pick a number."""
    # Make a temporary copy of the fixture so we don't depend on a sibling
    # .kicad_pro being present (it will be — but we delete it).
    src = _fixture_pcb()
    dst = tmp_path / "board.kicad_pcb"
    dst.write_text(Path(src).read_text())
    # No .kicad_pro next to dst.

    req = RouteRequest(
        pcb_path=str(dst),
        ref_a="R1",
        pad_a="1",
        ref_b="C1",
        pad_b="1",
        net="VCC",
    )
    with pytest.raises(RouteFailure) as excinfo:
        auto_route_pair(req)
    msg = str(excinfo.value)
    assert "track width" in msg.lower()
    assert "width=" in msg  # the hint to pass width= explicitly


def test_auto_route_pair_explicit_width_skips_drc(tmp_path: Path) -> None:
    """When the user passes width= explicitly, missing .kicad_pro must not
    block routing — that's the whole point of the explicit override."""
    src = _fixture_pcb()
    dst = tmp_path / "board.kicad_pcb"
    dst.write_text(Path(src).read_text())
    # No .kicad_pro next to dst.

    req = RouteRequest(
        pcb_path=str(dst),
        ref_a="R1",
        pad_a="1",
        ref_b="C1",
        pad_b="1",
        net="VCC",
        width=0.3,
        clearance=0.2,
    )
    # Should NOT raise — both DRC lookups are skipped.
    result = auto_route_pair(req)
    assert len(result.segments) > 0


# ---------------------------------------------------------------------------
# _check_segments_in_board
# ---------------------------------------------------------------------------


def _seg(x1: float, y1: float, x2: float, y2: float, width: float = 0.25) -> OutputSegment:
    return OutputSegment(x1=x1, y1=y1, x2=x2, y2=y2, width=width, layer="F.Cu", net="VCC")


def test_check_segments_in_board_accepts_segment_inside() -> None:
    segs = [_seg(2.0, 2.0, 4.0, 2.0)]
    # No exception expected.
    _check_segments_in_board(segs, (0.0, 0.0, 10.0, 10.0))


def test_check_segments_in_board_accepts_segment_on_boundary() -> None:
    """A segment whose centerline sits exactly on the Edge.Cuts boundary
    is fine — its copper would just touch the edge, not cross it."""
    segs = [_seg(0.5, 5.0, 5.0, 5.0)]
    _check_segments_in_board(segs, (0.0, 0.0, 10.0, 10.0))


def test_check_segments_in_board_rejects_segment_outside() -> None:
    segs = [_seg(2.0, 2.0, 12.0, 5.0)]  # x2 is past maxx=10
    with pytest.raises(RouteFailure) as excinfo:
        _check_segments_in_board(segs, (0.0, 0.0, 10.0, 10.0))
    assert "outside the Edge.Cuts boundary" in str(excinfo.value)


def test_check_segments_in_board_rejects_segment_through_wall() -> None:
    """A segment whose endpoints are inside but whose centerline would
    not cross a wall — this case should still pass. To trigger failure we
    need an endpoint past the boundary."""
    segs = [_seg(-1.0, 5.0, 5.0, 5.0)]
    with pytest.raises(RouteFailure):
        _check_segments_in_board(segs, (0.0, 0.0, 10.0, 10.0))


def test_check_segments_in_board_rejects_when_width_wider_than_board() -> None:
    """A 20 mm track cannot possibly fit inside a 5 mm board."""
    segs = [_seg(2.0, 2.0, 3.0, 2.0, width=20.0)]
    with pytest.raises(RouteFailure) as excinfo:
        _check_segments_in_board(segs, (0.0, 0.0, 5.0, 5.0))
    msg = str(excinfo.value)
    assert "wider than the board" in msg or "outside" in msg


def test_check_segments_in_board_rejects_degenerate_bbox() -> None:
    segs = [_seg(0.0, 0.0, 1.0, 1.0)]
    with pytest.raises(RouteFailure) as excinfo:
        _check_segments_in_board(segs, (5.0, 5.0, 5.0, 10.0))
    assert "degenerate" in str(excinfo.value)


# ---------------------------------------------------------------------------
# auto_route_pair — board-bounds check integration
# ---------------------------------------------------------------------------


def test_auto_route_pair_warns_when_no_edge_cuts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A PCB with no Edge.Cuts items is a workflow state, not an error.

    The user may be routing before drawing the board outline. We log a
    warning and proceed; final validation is the responsibility of KiCad
    DRC.
    """
    # Build a minimal PCB with the fixture's footprints but strip the
    # Edge.Cuts gr_rect we just added.
    pcb_text = Path(_fixture_pcb()).read_text()
    no_edge_cuts = "\n".join(
        line
        for line in pcb_text.splitlines()
        if "Edge.Cuts" not in line and "edge-cuts-rect" not in line
    )
    dst = tmp_path / "board.kicad_pcb"
    dst.write_text(no_edge_cuts)

    req = RouteRequest(
        pcb_path=str(dst),
        ref_a="R1",
        pad_a="1",
        ref_b="C1",
        pad_b="1",
        net="VCC",
        width=0.3,
        clearance=0.2,
    )
    with caplog.at_level("WARNING", logger="kcaa.router.router"):
        result = auto_route_pair(req)
    assert len(result.segments) > 0
    assert any("Edge.Cuts" in record.message for record in caplog.records), (
        f"Expected a warning mentioning Edge.Cuts; got {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Multi-layer request: layer / pad-layer validation
# ---------------------------------------------------------------------------


def test_unknown_layer_in_pcb_raises_route_failure(tmp_path: Path) -> None:
    """A start_layer / end_layer / via_pair layer that the PCB doesn't declare
    must fail loudly with a RouteFailure that names the offending layer."""
    src = _fixture_pcb()
    dst = tmp_path / "board.kicad_pcb"
    dst.write_text(Path(src).read_text())

    req = RouteRequest(
        pcb_path=str(dst),
        ref_a="R1",
        pad_a="1",
        ref_b="C1",
        pad_b="1",
        net="VCC",
        width=0.3,
        clearance=0.2,
        start_layer="In1.Cu",  # 2-layer fixture has no In1.Cu
    )
    with pytest.raises(RouteFailure) as excinfo:
        auto_route_pair(req)
    assert "In1.Cu" in str(excinfo.value)


def test_pad_missing_on_layer_raises_route_failure(tmp_path: Path) -> None:
    """If a pad has no copper shape on the requested layer, routing from it
    must fail loudly — never silently fall back to a guessed size."""
    src = _fixture_pcb()
    dst = tmp_path / "board.kicad_pcb"
    dst.write_text(Path(src).read_text())

    req = RouteRequest(
        pcb_path=str(dst),
        ref_a="R1",
        pad_a="1",
        ref_b="C1",
        pad_b="1",
        net="VCC",
        width=0.3,
        clearance=0.2,
        end_layer="B.Cu",  # R1.1 / C1.1 are SMD on F.Cu in the fixture
    )
    with pytest.raises(RouteFailure) as excinfo:
        auto_route_pair(req)
    assert "no copper shape" in str(excinfo.value)
