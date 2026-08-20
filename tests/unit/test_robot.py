"""Device discovery and robot-environment probing.

Both are hardware-adjacent but neither needs hardware: discovery reads a
directory, and the environment probe shells out to an interpreter. So the part
that decides *which camera is the wrist* is testable, which matters more than it
sounds -- getting it wrong produces no error at all.
"""

from __future__ import annotations

from polima.policies.registry import get_policy
from polima.robot import devices
from polima.robot.env import RobotEnv

C920 = "usb-046d_HD_Pro_Webcam_C920_EEFE8DCF-video-index0"
SONIX = "usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"


def _make_cameras(tmp_path, names):
    for name in names:
        (tmp_path / name).write_text("")
    return tmp_path


# ------------------------------------------------------------------ discovery


def test_only_video_index0_counts(tmp_path):
    """Higher indices are metadata streams: they open fine and produce nothing."""
    _make_cameras(tmp_path, [C920, C920.replace("index0", "index1"), "usb-thing-event-index0"])
    found = devices.list_cameras(tmp_path)
    assert [c.name for c in found] == [C920]


def test_vendor_model_is_readable():
    assert devices.Camera(path="/x", name=C920).vendor_model == "046d_HD_Pro_Webcam_C920_EEFE8DCF"


def test_missing_directory_is_empty_not_an_error(tmp_path):
    assert devices.list_cameras(tmp_path / "absent") == []


def test_hints_assign_roles(tmp_path):
    cameras = devices.list_cameras(_make_cameras(tmp_path, [C920, SONIX]))
    spec = get_policy("act")
    assigned, problems = devices.match_cameras(
        spec.robot.camera_roles, cameras, spec.robot.camera_hints
    )
    assert problems == []
    assert assigned["overhead"].endswith(C920)
    assert assigned["wrist"].endswith(SONIX)


def test_role_order_does_not_decide_assignment(tmp_path):
    """The directory listing is alphabetical, so Sonix sorts after C920 here.
    Reversing the roles must still match by hint, not by position."""
    cameras = devices.list_cameras(_make_cameras(tmp_path, [C920, SONIX]))
    assigned, _ = devices.match_cameras(
        (("wrist", "Wrist"), ("overhead", "Overhead")), cameras,
        {"overhead": "C920", "wrist": "Sonix"},
    )
    assert assigned["overhead"].endswith(C920)
    assert assigned["wrist"].endswith(SONIX)


def test_a_missing_camera_is_reported_not_guessed(tmp_path):
    """Never fall back on enumeration order. Two cameras assigned the wrong way
    round is the failure with no symptom: everything runs and the arm reaches
    for the wrong place."""
    cameras = devices.list_cameras(_make_cameras(tmp_path, [C920]))
    assigned, problems = devices.match_cameras(
        (("overhead", "O"), ("wrist", "W")), cameras, {"overhead": "C920", "wrist": "Sonix"}
    )
    assert "overhead" in assigned and "wrist" not in assigned
    assert any("Sonix" in p for p in problems)


def test_ambiguous_hint_refuses_and_says_how_to_choose(tmp_path):
    cameras = devices.list_cameras(_make_cameras(
        tmp_path, ["usb-Sonix_A-video-index0", "usb-Sonix_B-video-index0"]))
    assigned, problems = devices.match_cameras(
        (("wrist", "W"),), cameras, {"wrist": "Sonix"})
    assert "wrist" not in assigned
    assert any("--wrist-camera" in p for p in problems)


def test_override_wins_over_the_hint(tmp_path):
    cameras = devices.list_cameras(_make_cameras(tmp_path, [C920, SONIX]))
    assigned, problems = devices.match_cameras(
        (("overhead", "O"),), cameras, {"overhead": "C920"},
        overrides={"overhead": "/dev/v4l/by-id/whatever"},
    )
    assert assigned["overhead"] == "/dev/v4l/by-id/whatever"
    assert problems == []


def test_one_camera_cannot_fill_two_roles(tmp_path):
    cameras = devices.list_cameras(_make_cameras(tmp_path, [C920]))
    assigned, problems = devices.match_cameras(
        (("a", "A"), ("b", "B")), cameras, {"a": "C920", "b": "C920"})
    assert list(assigned) == ["a"]
    assert any("b:" in p for p in problems)


def test_a_role_with_no_hint_says_so(tmp_path):
    assigned, problems = devices.match_cameras((("side", "Side"),), [], {})
    assert not assigned
    assert any("--side-camera" in p for p in problems)


# ---------------------------------------------------------------- arm ports


def test_single_port_is_selected(tmp_path):
    (tmp_path / "ttyACM0").write_text("")
    port, problem = devices.pick_arm_port(devices.list_serial_ports(tmp_path))
    assert port.endswith("ttyACM0") and problem == ""


def test_two_ports_is_ambiguous(tmp_path):
    """A leader and a follower arm both enumerate as ttyACM*, so this is the
    normal teleop setup -- guessing would drive the wrong one."""
    for name in ("ttyACM0", "ttyACM1"):
        (tmp_path / name).write_text("")
    port, problem = devices.pick_arm_port(devices.list_serial_ports(tmp_path))
    assert port == "" and "--robot-port" in problem


def test_no_port_says_what_to_check(tmp_path):
    port, problem = devices.pick_arm_port(devices.list_serial_ports(tmp_path))
    assert port == "" and "powered" in problem


def test_override_skips_discovery_entirely():
    port, problem = devices.pick_arm_port([], override="/dev/ttyACM9")
    assert port == "/dev/ttyACM9" and problem == ""


# --------------------------------------------------------------- environment


def test_env_needs_lerobot_and_opencv():
    assert RobotEnv("p", "o", lerobot="0.6.1", cv2="4.13").usable
    assert not RobotEnv("p", "o", lerobot="0.6.1").usable
    assert not RobotEnv("p", "o", cv2="4.13").usable


def test_flask_is_optional():
    """It backs the browser view only; the control loop runs without it."""
    env = RobotEnv("p", "o", lerobot="0.6.1", cv2="4.13", flask="")
    assert env.usable and env.missing == []


def test_missing_names_what_to_install():
    assert RobotEnv("p", "o").missing == ["lerobot", "cv2"]


def test_probe_reads_a_real_interpreter():
    import sys

    from polima.robot.env import probe

    env = probe(sys.executable, "test")
    assert env is not None and env.version.startswith("3.")


def test_probe_survives_a_bad_interpreter():
    from polima.robot.env import probe

    assert probe("/nonexistent/python", "test") is None


# ------------------------------------------------------------------- install


def test_robot_install_is_advertised_exactly_as_implemented():
    """The top-level help drifted once: it listed `teleop` and `install`, which
    did not exist, and omitted `calibrate`, which did."""
    from polima.cli import main as cli_main
    from polima.cli import robot

    advertised = [
        word.strip()
        for line in (cli_main.__doc__ or "").splitlines() if "polima robot" in line
        for word in line.split("polima robot", 1)[1].split("|")
    ]
    parser_names = _robot_subcommands(robot)
    assert advertised, "the help line should still list the robot subcommands"
    assert set(advertised) == set(parser_names), (advertised, parser_names)


def _robot_subcommands(robot_module) -> list[str]:
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
        robot_module.run(["--help"])
    text = buffer.getvalue()
    inner = text.split("{", 1)[1].split("}", 1)[0]
    return [name.strip() for name in inner.split(",")]


def test_robot_install_refuses_to_provision_a_non_board(monkeypatch, capsys):
    """It must fail before running anything, not 300 lines into a provision."""
    from polima.cli import robot

    monkeypatch.setattr(robot.platform, "machine", lambda: "x86_64")
    assert robot.run(["install"]) == 2
    assert "must run on it" in capsys.readouterr().err


def test_robot_install_reports_every_path_it_searched(monkeypatch, capsys):
    from polima.cli import robot

    monkeypatch.setattr(robot.platform, "machine", lambda: "aarch64")
    assert robot.run(["install", "--script", "/nonexistent/installer.sh"]) == 2
    assert "/nonexistent/installer.sh" in capsys.readouterr().err


def test_robot_install_looks_in_the_deployed_board_layout(monkeypatch, tmp_path):
    """On the board polima is a pip install under the LeRobot venv, so walking
    up from __file__ finds no repository -- $POLIMA_ROOT/src must be tried."""
    from polima.cli import robot

    installer = tmp_path / "src" / "scripts" / "install_polima_modalix.sh"
    installer.parent.mkdir(parents=True)
    installer.write_text("#!/bin/sh\nexit 7\n")
    installer.chmod(0o755)

    monkeypatch.setenv("POLIMA_ROOT", str(tmp_path))
    monkeypatch.setattr(robot.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(robot.Path, "is_file",
                        lambda self: str(self) == str(installer))
    assert robot.run(["install"]) == 7
