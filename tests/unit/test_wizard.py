"""The interactive session a bare `polima-compile` opens.

Two things are load-bearing and both are pinned here:

  1. it triggers ONLY on a genuinely bare call at a TTY, so every script,
     cron job and explicit flag keeps the behaviour it had; and
  2. it only ever composes an argv. The command it prints is the command that
     runs, which is why there is still one code path through the compile stage.
"""

from __future__ import annotations

import builtins

import pytest

from polima.cli import compile as compile_cli
from polima.cli import deploy as deploy_cli
from polima.cli import wizard
from polima.config.base import BoardConfig
from polima.policies.registry import get_policy

ACT = get_policy("act")


@pytest.fixture(autouse=True)
def isolated_polima_outputs(tmp_path, monkeypatch):
    """Discovery tests must never depend on builds running in the live tree."""
    monkeypatch.setattr(wizard, "outputs_root", lambda: tmp_path / "polima" / "outputs")


@pytest.fixture
def answers(monkeypatch):
    """Feed scripted replies to every prompt, in order."""
    def scripted(queue):
        pending = list(queue)

        def fake_input(prompt=""):
            if not pending:
                raise EOFError
            return pending.pop(0)

        monkeypatch.setattr(builtins, "input", fake_input)
    return scripted


def _checkpoint_tree(root, run, steps, *, alias=True):
    """A lerobot run directory, with the `last -> <steps>` symlink it writes."""
    target = root / "ACT" / "outputs" / run / "checkpoints" / str(steps)
    (target / "pretrained_model").mkdir(parents=True)
    if alias:
        (target.parent / "last").symlink_to(target.name)
    return target / "pretrained_model"


def _build_tree(root, name, *, graphs=2, compiled=False):
    tree = root / "ACT" / "outputs" / name
    (tree / "onnx").mkdir(parents=True)
    for index in range(graphs):
        (tree / "onnx" / f"graph{index}.onnx").write_bytes(b"")
    if compiled:
        (tree / "models_uncompressed").mkdir()
    return tree


def _bundle(root, name, *, policy="act", graphs=2):
    import json

    path = root / "polima" / "outputs" / "bundles" / name
    path.mkdir(parents=True)
    (path / "bundle.json").write_text(json.dumps({
        "bundle_id": name,
        "policy": policy,
        "graphs": [
            {"name": f"graph{index}", "elf_bytes": 1048576}
            for index in range(graphs)
        ],
    }))
    return path


# ------------------------------------------------------------------ discovery


def test_checkpoint_alias_is_not_offered_twice(tmp_path, monkeypatch):
    """`checkpoints/last` is a symlink to a numbered directory, so the glob
    returns the same checkpoint twice."""
    _checkpoint_tree(tmp_path, "rcwb_f_t_act_20260805", 100000)
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)

    found = wizard.find_checkpoints(ACT)
    assert len(found) == 1
    assert found[0].steps == 100000


def test_steps_are_read_through_the_alias(tmp_path, monkeypatch):
    """If only `last` matched, the step count must still come from its target."""
    target = tmp_path / "ACT" / "outputs" / "run" / "checkpoints" / "40000"
    (target / "pretrained_model").mkdir(parents=True)
    (target.parent / "last").symlink_to(target.name)
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)

    assert [c.steps for c in wizard.find_checkpoints(ACT)] == [40000]


def test_build_trees_report_whether_they_are_already_built(tmp_path, monkeypatch):
    _build_tree(tmp_path, "fresh", graphs=3)
    _build_tree(tmp_path, "built", graphs=6, compiled=True)
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)

    trees = {t.path.name: t for t in wizard.find_build_trees(ACT)}
    assert trees["fresh"].graphs == 3 and not trees["fresh"].compiled
    assert trees["built"].graphs == 6 and trees["built"].compiled


def test_a_directory_without_onnx_is_not_a_build_tree(tmp_path, monkeypatch):
    (tmp_path / "ACT" / "outputs" / "logs_only").mkdir(parents=True)
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)
    assert wizard.find_build_trees(ACT) == []


def test_default_build_dir_matches_the_shell_script_convention(tmp_path, monkeypatch):
    """Compiler products belong to PoLiMa, not the training stack's outputs."""
    monkeypatch.setattr(wizard, "outputs_root", lambda: tmp_path / "polima" / "outputs")
    checkpoint = wizard.Checkpoint(
        path=tmp_path, run="gewb_2_final_act_20260806_195153", steps=100000, mtime=0.0
    )
    assert wizard.default_build_dir(ACT, checkpoint) == (
        tmp_path / "polima" / "outputs" / "build" / "polima_gewb_2_final_act_100000"
    )


def test_stack_outputs_comes_from_the_spec(tmp_path, monkeypatch):
    """Derived from repo_dir_hint, not a second policy->directory map."""
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)
    assert wizard.stack_outputs(ACT) == tmp_path / "ACT" / "outputs"
    assert wizard.stack_outputs(get_policy("smolvla")) == tmp_path / "SmolVLA" / "outputs"


# --------------------------------------------------------------- when it opens


@pytest.mark.parametrize("argv", [["--build-dir", "/x"], ["--json"], ["--policy", "act"]])
def test_arguments_mean_no_session(argv, monkeypatch):
    monkeypatch.setattr(wizard.sys.stdin, "isatty", lambda: True, raising=False)
    assert not wizard.bare_invocation_is_interactive(argv)


def test_a_pipe_means_no_session(monkeypatch):
    """The regression that would matter most: a cron job hanging on a prompt."""
    monkeypatch.setattr(wizard.sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(wizard.sys.stdout, "isatty", lambda: True, raising=False)
    assert not wizard.bare_invocation_is_interactive([])


def test_bare_call_at_a_tty_opens_the_session(monkeypatch):
    monkeypatch.setattr(wizard.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(wizard.sys.stdout, "isatty", lambda: True, raising=False)
    assert wizard.bare_invocation_is_interactive([])


def test_non_interactive_bare_call_keeps_its_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(wizard.sys.stdin, "isatty", lambda: False, raising=False)
    assert compile_cli.run([]) == 2
    assert "nothing to do" in capsys.readouterr().err


def test_bare_call_is_not_gated_on_the_compiler():
    """The session offers packing, which needs no compiler, so gating the bare
    call would exit 3 before it could say so."""
    assert compile_cli.needs_capability([]) is None
    assert compile_cli.needs_capability(["--build-dir", "/x"]) == "compile"


# --------------------------------------------------------------- what it emits


def test_adopting_a_tree_composes_import_legacy(tmp_path, monkeypatch, answers, capsys):
    tree = _build_tree(tmp_path, "already_built", compiled=True)
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)
    # Only two modes exist without a compiler: adopt, or type a path.
    answers(["1", "1", "y"])       # adopt -> first tree -> run it

    argv = wizard.compose({"act": ACT}, can_compile=False)
    assert argv == ["--policy", "act", "--import-legacy", str(tree)]
    # The printed command must be the one that runs, or the session teaches a lie.
    assert "--import-legacy" in capsys.readouterr().out


def test_compiling_a_tree_shows_elf_and_core_counts(
    tmp_path, monkeypatch, capsys,
):
    tree = _build_tree(tmp_path, "fresh", graphs=3)
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(wizard.os, "cpu_count", lambda: 24)
    pending = iter(["1", "1", "4", "1", "y"])

    def input_with_visible_prompt(prompt=""):
        print(prompt, end="")
        return next(pending)

    monkeypatch.setattr(builtins, "input", input_with_visible_prompt)
    # No checkpoints here, so "build tree" is the first mode offered.

    argv = wizard.compose({"act": ACT}, can_compile=True)
    assert argv == ["--policy", "act", "--build-dir", str(tree), "--jobs", "4"]
    prompt_output = capsys.readouterr().out
    assert "3 ELFs to compile" in prompt_output
    assert "24 logical CPU cores" in prompt_output


def test_stop_after_is_only_passed_when_it_is_not_the_default(tmp_path, monkeypatch, answers):
    _build_tree(tmp_path, "fresh")
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)
    answers(["1", "1", "1", "2", "y"])   # ... -> jobs 1 -> stop after compile

    argv = wizard.compose({"act": ACT}, can_compile=True)
    assert argv[-2:] == ["--stop-after", "compile"]
    assert "--jobs" not in argv


def test_declining_the_confirmation_runs_nothing(tmp_path, monkeypatch, answers):
    _build_tree(tmp_path, "already_built", compiled=True)
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)
    answers(["1", "1", "n"])
    assert wizard.compose({"act": ACT}, can_compile=False) is None


def test_eof_cancels_rather_than_crashing(tmp_path, monkeypatch, answers):
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)
    answers([])
    with pytest.raises(wizard.Cancelled):
        wizard.compose({"act": ACT}, can_compile=True)


# --------------------------------------------------------------- deploy setup


def test_deploy_bundle_discovery_ignores_incomplete_manifests(tmp_path):
    valid = _bundle(tmp_path, "act-valid", graphs=6)
    broken = valid.parent / "broken"
    broken.mkdir()
    (broken / "bundle.json").write_text("{")

    found = wizard.find_local_bundles(valid.parent)
    assert [bundle.bundle_id for bundle in found] == ["act-valid"]
    assert found[0].graphs == 6
    assert found[0].elf_bytes == 6 * 1048576


def test_interactive_deploy_composes_bundle_board_port_and_start(
    tmp_path, answers, capsys,
):
    bundle = _bundle(tmp_path, "act-test-100-aabbccdd", graphs=6)
    answers(["1", "", "", "2", "n", "y"])

    argv = wizard.compose_deploy(
        BoardConfig(host="sima@10.0.0.2"), bundle.parent
    )
    assert argv == [
        "--bundle", str(bundle),
        "--board", "sima@10.0.0.2",
        "--port", "8092",
        "--start",
    ]
    shown = capsys.readouterr().out
    assert "6 graphs" in shown
    assert "polima deploy" in shown


def test_interactive_deploy_advanced_flags(tmp_path, answers):
    bundle = _bundle(tmp_path, "smolvla-test-100-aabbccdd", policy="smolvla", graphs=4)
    answers(["1", "", "", "1", "y", "y", "y", "y", "y"])

    argv = wizard.compose_deploy(BoardConfig(), bundle.parent)
    assert argv[-3:] == ["--no-build", "--no-activate", "--force"]
    assert "--start" not in argv
    assert "--verbose-server" not in argv
    assert argv[argv.index("--port") + 1] == str(
        get_policy("smolvla").wire.default_port
    )


def test_declining_interactive_deploy_runs_nothing(tmp_path, answers):
    bundle = _bundle(tmp_path, "act-test")
    answers(["1", "", "", "1", "n", "n"])
    assert wizard.compose_deploy(BoardConfig(), bundle.parent) is None


def test_bare_interactive_deploy_cancellation_returns_130(monkeypatch):
    monkeypatch.setattr(wizard, "bare_invocation_is_interactive", lambda argv: True)
    monkeypatch.setattr(deploy_cli, "_session", lambda parent: None)
    assert deploy_cli.run([]) == 130


def test_explicit_deploy_arguments_do_not_open_session(monkeypatch, tmp_path):
    monkeypatch.setattr(
        deploy_cli, "_session",
        lambda parent: pytest.fail("explicit arguments must not open the wizard"),
    )
    monkeypatch.setattr(wizard, "bare_invocation_is_interactive", lambda argv: False)
    with pytest.raises(SystemExit) as error:
        deploy_cli.run(["--bundle", str(tmp_path / "missing")])
    assert error.value.code == 2


def test_without_a_compiler_only_packing_is_offered(tmp_path, monkeypatch, answers, capsys):
    _build_tree(tmp_path, "fresh")                       # compilable, not built
    _build_tree(tmp_path, "built", compiled=True)        # packable
    _checkpoint_tree(tmp_path, "run_act_1", 100)
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)
    answers(["1", "1", "y"])                             # first option must be adopt

    argv = wizard.compose({"act": ACT}, can_compile=False)
    output = capsys.readouterr().out
    assert "no model compiler" in output
    assert "--import-legacy" in argv
    assert "checkpoint" not in output.split("what do you want")[0].split("1)")[-1]


def test_a_bad_index_reprompts_instead_of_raising(tmp_path, monkeypatch, answers):
    tree = _build_tree(tmp_path, "built", compiled=True)
    monkeypatch.setattr(wizard, "repo_root", lambda: tmp_path)
    answers(["9", "1", "0", "1", "y"])   # out of range, then valid

    argv = wizard.compose({"act": ACT}, can_compile=False)
    assert argv == ["--policy", "act", "--import-legacy", str(tree)]
