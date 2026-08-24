from __future__ import annotations

from polima.cli import main as cli_main


def test_help_command_prints_top_level_help(capsys) -> None:
    assert cli_main.main(["help"]) == 0
    output = capsys.readouterr().out
    assert "usage: polima" in output
    assert "COMMANDS:" in output
    assert "help      show help for PoLiMa or a specific command" in output


def test_help_command_prints_subcommand_help(capsys) -> None:
    assert cli_main.main(["help", "studio"]) == 0
    output = capsys.readouterr().out
    assert "usage: polima studio" in output
    assert "{status,start,stop,restart,enable,disable,open,logs,serve}" in output
