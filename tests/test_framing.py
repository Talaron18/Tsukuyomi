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
