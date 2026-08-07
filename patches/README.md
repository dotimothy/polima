# lerobot local modifications

Both vendored clones sit at `36b8face988669509272b00f4abe6592d0b17aa0` and carry
the same three source edits. Neither can be dropped: they enable things the
upstream async-inference stack does not support.

| File | Change | Why |
|---|---|---|
| `async_inference/configs.py` | adds `PolicyServerConfig.pretrained_name_or_path` | lets the *server* choose the checkpoint instead of trusting the client's placeholder |
| `async_inference/policy_server.py` | honours that field | without it the server loads whatever the client names |
| `robots/so_follower/so_follower.py` | `LEROBOT_FORCE_PROVIDED_CALIBRATION=1` escape | skips the interactive "press ENTER to use provided calibration" prompt, which blocks unattended startup on the board |

## The clones have drifted

`ACT/patches/lerobot-act-stack.patch` is canonical and applies to `ACT/lerobot`
exactly (`git apply --reverse --check` passes). It does **not** apply to
`SmolVLA/lerobot`, which implements the same three changes with different
formatting — multi-line wrapping in `policy_server.py`, and a log line that
interpolates `self.config.pretrained_name_or_path` rather than the resolved
local. The behaviour is equivalent; only the text differs.

So the clones are **feature-identical and textually divergent**. Checking them
by applying a patch is therefore too strict and produces a false alarm;
`polima doctor` checks for the *features* instead (see
`polima/cli/doctor.py::check_lerobot_patches`).

## Reading a diff in these clones

`SmolVLA/lerobot` showed 974 dirty paths, all `100644 -> 100755` mode churn from
a recursive chmod, burying the three real edits. Both clones are now configured
with:

    git -C <clone> config core.fileMode false

which drops it to the 3 files that actually changed. Run that first in any fresh
clone or the diff is unreadable.

## Reconciling

Deferred to Phase 4, when SmolVLA is ported. Doing it now would mean editing a
vendored tree that currently trains and serves correctly, for a cosmetic gain.
The plan is one host-side clone shared by all policies; the board keeps its own
pip-installed lerobot 0.4.4, which needs a different version anyway.
