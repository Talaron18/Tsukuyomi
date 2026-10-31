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


# ------------------------------------------------- the camera follows moves

def _move(name, start, seconds):
    return {"name": name, "start": start, "seconds": seconds}


def _quiet(_s):
    pass


def test_follow_whole_body_move_in_bust_and_back():
    toks = framing.follow_tokens([_move("dance", 3.0, 4.0)], [], "bust",
                                 log=_quiet)
    assert [(k["view"], k["t"], k["type"]) for k in toks] == [
        ("full", 2.7, "frame"), ("bust", 7.5, "frame")]
    assert all(k["duration"] == framing.VIEW_EASE and k["follow"]
               for k in toks)
    # Played by the Camera: full by the time the move is 0.3 s old, bust
    # again 0.6 s after the move's tail.
    cam = framing.Camera({"bust": (2.0, 0.1, 0.2), "full": (1.0, 0.0, 0.0)},
                         "bust")
    assert framing.view_tokens(toks, cam) == 2
    assert cam.at(2.6) == (2.0, 0.1, 0.2)
    assert cam.at(2.7 + 0.6) == (1.0, 0.0, 0.0)
    assert cam.at(5.0) == (1.0, 0.0, 0.0)
    assert cam.at(7.5 + 0.6) == (2.0, 0.1, 0.2)


def test_follow_covers_every_whole_body_move_and_not_hand_ones():
    assert set(framing.WHOLE_BODY) == {"dance", "twirl", "shimmy", "bounce",
                                       "cheer"}
    for name in framing.WHOLE_BODY:
        assert len(framing.follow_tokens([_move(name, 1.0, 2.0)], [], "bust",
                                         log=_quiet)) == 2
    for name in ("wave", "clap", "sway", "head_bob", "look_around"):
        assert framing.follow_tokens([_move(name, 1.0, 2.0)], [], "bust",
                                     log=_quiet) == []


def test_follow_only_from_bust():
    dance = [_move("dance", 3.0, 4.0)]
    assert framing.follow_tokens(dance, [], "full", log=_quiet) == []
    # the model went to full before the move: nothing to follow
    full = [framing.frame_token("full", 1.0)]
    assert framing.follow_tokens(dance, full, "bust", log=_quiet) == []
    # and back to bust before it: the camera follows again
    back = full + [framing.frame_token("bust", 2.0)]
    assert len(framing.follow_tokens(dance, back, "bust", log=_quiet)) == 2


def test_follow_model_markers_in_the_window_win():
    dance = [_move("dance", 3.0, 4.0)]            # window 2.7 .. 7.5
    for t, view in ((4.0, "full"), (4.0, "bust"), (7.4, "bust"),
                    (2.8, "full")):
        got = framing.follow_tokens(dance, [framing.frame_token(view, t)],
                                    "bust", log=_quiet)
        assert got == [], (t, view)
    # outside the window they do not
    after = [framing.frame_token("full", 7.6)]
    assert [k["t"] for k in framing.follow_tokens(dance, after, "bust",
                                                  log=_quiet)] == [2.7, 7.5]
    log = []
    framing.follow_tokens(dance, [framing.frame_token("full", 4.0)], "bust",
                          log=log.append)
    assert "skipped" in log[0] and "dance" in log[0]


def test_follow_merges_close_moves_and_shifts_by_t0():
    mvs = [_move("dance", 1.0, 2.0), _move("bounce", 3.5, 1.0),
           _move("cheer", 20.0, 1.0)]
    toks = framing.follow_tokens(mvs, [], "bust", t0=10.0, log=_quiet)
    # dance and bounce (0.5 s apart, under lead + tail) share one trip
    assert [(k["view"], round(k["t"], 3)) for k in toks] == [
        ("full", 10.7), ("bust", 15.0), ("full", 29.7), ("bust", 31.5)]
    # a move at the very start cannot lead before t = 0
    assert framing.follow_tokens([_move("dance", 0.1, 1.0)], [], "bust",
                                 log=_quiet)[0]["t"] == 0.0


def test_follow_logs_when_it_triggers():
    log = []
    framing.follow_tokens([_move("twirl", 2.0, 3.0)], [], "bust",
                          log=log.append)
    assert len(log) == 1 and "twirl" in log[0] and "full" in log[0]


def test_runtime_and_director_flags_for_follow_and_drift():
    import argparse
    from flow import director as dr
    ap = argparse.ArgumentParser()
    dr.add_args(ap)
    a = ap.parse_args([])
    assert a.camera_follow and a.head_drift and a.pose_ease
    a = ap.parse_args(["--no-camera-follow", "--no-head-drift",
                       "--no-pose-ease", "--frame", "bust"])
    a.rig = a.model_json = a.events = "x"
    a.breath_gain, a._breath = 1.0, (1.0, "test")
    out = dr.runtime_args(a, 5.0)
    assert "--no-camera-follow" in out and "--pose-ease" not in out
    assert "--head-drift-scale" not in out
    assert out[out.index("--frame") + 1] == "bust"
    a = ap.parse_args([])
    a.rig = a.model_json = a.events = "x"
    a.breath_gain, a._breath = 1.0, (1.0, "test")
    out = dr.runtime_args(a, 5.0)
    assert out[out.index("--head-drift-scale") + 1] == "1"
    assert "--pose-ease" in out and "--no-camera-follow" not in out


# ----------------------------------------------------------- the hat window

def _numbers(top_hu, size=512, above=framing.ABOVE, fit=True):
    """bust_math on a synthetic figure: 10 model units tall, 5 wide, 40 px a
    unit, the chin 4 units up (hu = 1 unit) and the alpha top `top_hu` hu
    over it."""
    y1 = 4.0 + top_hu
    k = 40.0
    box = (0.0, y1 * k, 100.0, 100.0 + 5 * k)
    return framing.bust_math(box, ((0.0, 0.0), (5.0, y1)), (2.5, 4.0), 1.0,
                             size, above=above, fit=fit)


def test_window_top_keeps_above_for_an_ordinary_figure():
    s, ox, oy, used, top = _numbers(1.3)
    assert abs(top - 1.3) < 1e-9 and used == framing.ABOVE
    # scale from the window: (1.5 + 1.0) hu of 40 px on a 512 canvas
    assert abs(s - 512 / (2.5 * 40)) < 1e-9
    assert _numbers(1.3, fit=False)[3] == framing.ABOVE


def test_window_top_reaches_a_tall_hat_plus_margin():
    s, ox, oy, used, top = _numbers(2.0)
    assert abs(top - 2.0) < 1e-9
    assert abs(used - (2.0 + framing.HAT_MARGIN)) < 1e-9
    assert abs(s - 512 / ((used + framing.BELOW) * 40)) < 1e-9
    # the chin stays centred horizontally, the window sits on chin + BELOW
    cx = 100.0 + 2.5 * 40
    assert abs(ox - (-s * (cx - 256) / 256)) < 1e-9
    top_px, bot_px = (top - used) * 40, (top + framing.BELOW) * 40
    assert abs(oy - s * ((top_px + bot_px) / 2 - 256) / 256) < 1e-9


def test_window_top_is_capped_for_a_very_tall_figure():
    s, ox, oy, used, top = _numbers(5.0)
    assert abs(used - (framing.ABOVE + framing.HAT_CAP)) < 1e-9
    assert top == 5.0
    # BELOW is untouched whatever the top does
    assert abs(s - 512 / ((framing.ABOVE + framing.HAT_CAP + framing.BELOW)
                          * 40)) < 1e-9
    assert framing.fit_top(1.5, None) == 1.5
    assert framing.fit_top(1.5, 2.0, margin=0.0, cap=0.25) == 1.75
