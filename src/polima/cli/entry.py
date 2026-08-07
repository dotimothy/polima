"""Standalone console-script entry points.

PoLiMa installs the same shape as LLiMa -- separate binaries per stage rather
than one monolith:

    HOST                          BOARD
    ----------------------        ----------------------
    polima-compile                polima-run
    polima-deploy                 polima-robot

`polima <subcommand>` still works everywhere as an umbrella alias, so
`polima compile ...` and `polima-compile ...` are the same code path. The split
matters for two reasons: the board install pulls no torch/afe, and a command
invoked on the wrong machine fails with one clear sentence instead of an
ImportError from inside a vendored library.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Callable

from polima import role
from polima.util.logging import setup


def make_entry(command: str, module_name: str) -> Callable[[list[str] | None], int]:
    """Build a `main(argv)` for one console script."""

    def main(argv: list[str] | None = None) -> int:
        argv = list(argv if argv is not None else sys.argv[1:])

        globals_argv, rest = _split_globals(argv)
        parser = argparse.ArgumentParser(prog=f"polima-{command}", add_help=False)
        parser.add_argument("--config", metavar="FILE")
        parser.add_argument("--log-level", default=None)
        parser.add_argument("--dry-run", action="store_true")
        parent, _ = parser.parse_known_args(globals_argv)
        parent.command = command

        setup(parent.log_level)

        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            print(f"polima-{command}: {exc}", file=sys.stderr)
            return 2

        # Import first, then gate: the capability a command needs depends on
        # what it is being asked to do (see polima.role.check).
        denial = role.check(command, rest, module)
        if denial:
            print(f"polima-{command}: {denial}", file=sys.stderr)
            return 3

        return module.run(rest, parent=parent)

    main.__name__ = f"{command}_main"
    main.__doc__ = f"Console-script entry point for `polima-{command}`."
    return main


def _split_globals(argv: list[str]) -> tuple[list[str], list[str]]:
    """Pull the shared --config/--log-level/--dry-run flags off the front.

    Anything after the first unrecognised token belongs to the subcommand, so
    each stage owns its own flag namespace.
    """
    shared_with_value = {"--config", "--log-level"}
    shared_flags = {"--dry-run"}
    globals_argv: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in shared_with_value and index + 1 < len(argv):
            globals_argv += [token, argv[index + 1]]
            index += 2
        elif token in shared_flags:
            globals_argv.append(token)
            index += 1
        elif token.split("=", 1)[0] in shared_with_value:
            globals_argv.append(token)
            index += 1
        else:
            break
    return globals_argv, argv[index:]


# --- the console scripts declared in pyproject.toml -------------------------

compile_ = make_entry("compile", "polima.cli.compile")
deploy = make_entry("deploy", "polima.cli.deploy")
run_ = make_entry("run", "polima.cli.run")
robot = make_entry("robot", "polima.cli.robot")
doctor = make_entry("doctor", "polima.cli.doctor")
