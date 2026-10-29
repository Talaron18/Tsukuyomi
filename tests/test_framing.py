"""flow.framing: which body file a render frames by (the bust view itself
needs a GL context and a Live2D model: checked by hand)."""

import os

from flow import framing


def test_body_path_for(tmp_path, monkeypatch):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(root)
    assert framing.body_path_for("rigs/Hiyori.npz") == "rigs/Hiyori.body.json"
    assert framing.body_path_for(
        None, "x/Hiyori/Hiyori.model3.json") == "rigs/Hiyori.body.json"
    assert framing.body_path_for(str(tmp_path / "none.npz")) is None


class _FaceOnly:
    """A body file with no arm controls and no chest landmark
    (huonululu_5's): hand_reach must not need the chest."""
    variants = [{"present": [], "rest": []}]
    fidx = {}

    def __init__(self, landmarks):
        self.landmarks = landmarks


def test_hand_reach_without_chest(capsys, monkeypatch):
    monkeypatch.setattr(framing, "_WARNED", set())
    r = framing.hand_reach(_FaceOnly({"head": [0.0, 0.5],
                                      "shoulder.y": -0.3}))
    assert r["all"] == 0.0 and r["low"] is None
    assert "no chest landmark" in capsys.readouterr().out
    assert framing.chest_y({"shoulder.y": -0.3}) == -0.7
    # neither chest nor shoulders: the default shoulder line
    assert abs(framing.chest_y({}) - (-0.65)) < 1e-9
    assert framing.chest_y({"chest": [0.0, -0.5]}) == -0.5
    framing.hand_reach(_FaceOnly({}))        # not even a head landmark
