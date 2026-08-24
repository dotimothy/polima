"""Show help for PoLiMa or one of its commands."""

from __future__ import annotations

import argparse
import importlib


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    # Import here to avoid a module-level cycle with the top-level dispatcher.
    from polima.cli import main as cli_main

    parser = argparse.ArgumentParser(
        prog="polima help",
        description="show help for PoLiMa or a specific command",
    )
    parser.add_argument("command", nargs="?", choices=tuple(cli_main.COMMANDS))
    args = parser.parse_args(argv)

    if args.command is None or args.command == "help":
        cli_main.build_parser().print_help()
        return 0

    module_name, _ = cli_main.COMMANDS[args.command]
    module = importlib.import_module(module_name)
    try:
        return module.run(["--help"], parent=parent)
    except SystemExit as exc:
        return int(exc.code or 0)


def needs_capability(argv: list[str]) -> None:
    return None
