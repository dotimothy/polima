# PoLiMa

**Po**licy + Si**Ma** — a unified framework for compiling, deploying and running robot policies (ACT, SmolVLA, GR00T) on SiMa.ai Modalix hardware.

**PoLiMa does not train.** Training stays where it is, with `lerobot-train` in the model stacks. PoLiMa picks up at the checkpoint and owns everything after it: export → compile → bundle → deploy → serve → drive the arm.

PoLiMa is the robot-policy analogue of **LLiMa**, and uses it under the hood: VLM backbones go through `sima_lmm.host.compile_lmm`, action experts go through `afe`. It mirrors LLiMa's shape deliberately — separate console scripts per stage, not one monolith.

Adding a policy should mean writing one `PolicySpec`, not copying 3,000 lines.

---

## The two-machine split

PoLiMa spans two machines with incompatible dependency sets. That split is structural, not a convention.

**The host builds; the board runs.** Nothing on the host drives a robot, and nothing on the board compiles.

| | **HOST** (x86_64 workstation) | **BOARD** (Modalix SoM, aarch64) |
|---|---|---|
| role | build it | run it |
| | `polima-compile` — export (torch) then compile (afe), two separate venvs | `polima-run` — serve the policy |
| | `polima-deploy` — ssh, rsync | `polima-robot` — drive the arm |
| deps | `polima[host]` | `polima` (numpy only); the robot client uses the board's `/media/nvme/lerobot` venv |

The robot client sits with the server rather than on the host because it sends two 640×480 frames per control step — ~110 MB/s at 30 Hz — which should be a loopback copy, not a network hop.

`polima-doctor` runs on both and is the first thing to run after any install.

`polima <stage>` also works everywhere as an umbrella alias, so `polima compile …` and `polima-compile …` are the same code path.

### Why not two packages

Both halves share one core — `polima.policies`, `polima.wire`, `polima.bundle`, `polima.config`, `polima.util` — because that core **is the contract between them**: the host writes `bundle.json` and `plan.json`, the board reads them. Splitting them into disjoint distributions would mean duplicating the contract, which is exactly the failure mode PoLiMa exists to remove.

So the core carries a hard dependency floor of **stdlib + numpy**, and it must import in all three interpreters:

- the self-contained PoLiMa `.venv` (py3.12 + torch + LeRobot)
- the SiMa model-compiler venv (py3.12 + `afe`, **no torch**)
- the board's py3.11

`polima doctor --imports` proves this by importing the core under each interpreter, and `tests/unit/test_role.py` asserts no heavy dependency leaks in. A bare `pip install polima` pulls exactly one package: numpy.

Invoking a stage on the wrong machine exits `3` with one actionable sentence, rather than an `ImportError` three frames deep in a vendored library.

---

## Prerequisites

**The SiMa Model Compiler (ModelSDK) is required and is not part of this
repository.** It is proprietary and licensed, so it cannot be vendored or
installed by `make venv`; you obtain it from the SiMa Developer Portal (via
`sima-cli`, or the Palette `modelsdk` container). Everything except
quantization and compilation works without it — export, packing, deploy, run,
the robot client and Studio — but `polima compile` cannot do its job, because
`afe` lives inside the ModelSDK and nowhere else.

Point PoLiMa at it once installed:

```bash
export MODEL_COMPILER_BIN=/path/to/model-compiler/bin   # the dir holding its python
polima doctor                                           # verifies afe imports
```

Developed and verified against **ModelSDK 2.1.0**. The version participates in
the compile resume key, so upgrading it invalidates cached ELFs and recompiles
— by design, since an AFE upgrade changes code generation while every input
stays byte-identical.

If it is missing or misconfigured, `polima compile` says so and names the
interpreter it tried, rather than failing later inside AFE:

```
polima-compile: /path/to/python cannot import afe.
  Sourced .../activate-model-compiler and it still fails.
  Set MODEL_COMPILER_BIN, or run from the modelsdk environment.
```

That activation step matters: without it `import afe` fails on a missing
`libLLVM`, which is why `bin/polima` sources `activate-model-compiler` when it
finds one.

---

## Install

**Host**
```bash
cd polima
make venv
source .venv/bin/activate
polima-doctor
```

The project-local `.venv` owns the ordinary PoLiMa CLI, dataset validation,
tests, CPU-only Torch/LeRobot export stack, and deployment dependencies. CUDA
is not required. It never uses a Conda environment. Only the proprietary
ModelSDK extension remains external; PoLiMa launches AFE through
`MODEL_COMPILER_BIN` while all checkpoint loading and ONNX export stays inside
`.venv`.

**Palette modelsdk container** — no install; the launcher resolves its own location, so it works at whatever path the workspace is mounted:

```bash
export PATH=/workspace/MLSandbox/polima/bin:$PATH
polima compile --build-dir <dir>
```

`bin/polima` picks an interpreter that has numpy (the model compiler's own will do) and sources `activate-model-compiler` when present — without it `import afe` fails on a missing `libLLVM`.

**Board**
```bash
ssh sima@192.168.91.211
cd /path/to/polima
./scripts/install_polima_modalix.sh
polima robot status
```

The installer builds the native `polima_server` and `polima_cli`, exposes the
device command as `polima`, and reuses or provisions `/media/nvme/lerobot` via
the sibling `lerobot_sima/install_lerobot_modalix.sh`. After deploying a bundle,
`polima robot run` starts the matching policy server and runs the bundled
LeRobot control client in the foreground.

Device lifecycle commands:

```bash
polima activate                    # interactively select an installed bundle
polima activate 2                  # or select by list number
polima activate <bundle-id>        # or exact bundle id
polima activate 2 --start          # activate and start it in one operation
polima server status             # inspect the background policy server
polima server start              # serve /media/nvme/polima/current
polima server stop
polima robot status              # LeRobot env, follower arm, camera assignments
polima robot run                 # start server, then control the arm in foreground
polima robot preview             # browser camera view only; no server or arm
```

Activation is performed entirely on the device and atomically updates
`/media/nvme/polima/current`. If a policy server is running, activation stops it
first so its reported bundle cannot disagree with the model it actually loaded;
start it again with `polima server start`.
The interactive command asks whether to start the selected bundle; scripts can
request the same combined handoff with `--start`.
If no managed bundle is active, interactive `polima server start` presents the
same bundle selector, activates the choice, and starts it. In a script, select
explicitly with `polima activate <bundle-id>` (or use `--bundle DIR`).

Running `polima server` or `polima robot` without an action opens a guided
prompt using the active bundle. An explicit `polima robot run` still confirms
before motion; unattended invocation must opt in with `--yes`.

`polima robot run` refuses to start unless exactly one follower arm and both
bundle-declared cameras are resolved. Ctrl-C stops robot control but intentionally
leaves the policy server loaded; use `polima server stop` to release it.

While robot control is running, the client prints a tokenized live-preview URL
such as `http://<board-ip>:5001/<token>/`. Open it from a browser on the same
network to view both camera feeds. The preview reuses the frames already read by
LeRobot and does not open the camera devices a second time.

---

## Layout

```
src/polima/
  policies/   THE PLUGIN LAYER -- one PolicySpec per policy
  config/     layered config (dataclasses + stdlib loader; no pydantic)
  data/       dataset contracts, discovery, episode specs, aggregation
  export/     checkpoint -> ONNX + calibration + fixtures
  compile/    afe wrapper, the per-graph compile driver
  bundle/     content-addressed bundles, MPK unpack, path rewriting
  deploy/     ssh/rsync, board bootstrap, health poll, smoke test
  wire/       the SoM TCP protocol   <- replaces 2 hand-written clients
  robot/      robot client, live view, teleop, lerobot monkeypatch layer
  util/       proc, hashing, jsonio, paths, logging, table

native/       ONE polima_server binary, driven by plan.json
board/        scripts that run on the SoM
tests/        unit (fast, no hardware) + hardware (opt-in)
```

### The policy contract

Everything policy-specific is data in one frozen dataclass — `DatasetContract`, `TrainSpec`, `CompilePlan`, `RuntimePlan`, `WireSpec`, `RobotSpec`. It is JSON round-trippable because a spec has to cross all three interpreters.

`PolicySpec.validate()` cross-checks the pieces at import time: graph outputs must match the plan buffers that receive them, and the wire's response size must equal `actions_per_chunk × action_dim`. A malformed spec fails at `polima doctor`, not mid-deploy.

---

## Safety net

The four legacy stacks (`ACT/`, `SmolVLA/`, `GR00T-N1.6/`, `lerobot_sima/`) are **not modified** by PoLiMa. Two mechanisms enforce it:

```bash
make backup-legacy        # tree copy + tarball -> ../.legacy-backups/<stamp>/
make check-legacy-intact  # sha256 all 129 first-party files
make restore-legacy       # undo, from the newest backup
```

Backups live outside `polima/` so removing the framework never takes the safety net with it. `make check-legacy-intact` predates version control and is now largely redundant with git; it stays as a fast on-disk check.

---

## Status

186 unit tests, no hardware required. Every milestone so far is proven against
the real board rather than asserted:

| Stage | State | Proof |
|---|---|---|
| `polima doctor` / `data` / `list` | working | |
| `polima-deploy` / `polima-run` | working | 20.8 ms on the SoM, cosine 0.999990 vs PyTorch — 24% faster than the hand-written `act_llima` at 27.0 ms |
| `polima-compile` | working | checkpoint → ONNX → ELF → bundle reproduces **byte for byte**, landing on the same content-addressed bundle id as the one deployed ([export](docs/export.md), [compile](docs/compile.md)) |
| `polima-robot` | Phase 1d | |
| SmolVLA | native checkpoint export + compile integrated; hardware validation pending | ONNX chain max abs `1.67e-6`; suffix and vision accepted by ModelSDK 2.1.0 |
| GR00T | Phase 5 | |

The compile stage replaces three divergent copies of
`sima_compile_onnx_tensors.py` (518 lines) plus two per-policy controllers with
one driver over `PolicySpec.compile.graphs`, so adding a policy needs no
controller at all. The export stage splits `export_act_modalix.py` so that only
the policy's own graph decomposition needs torch.

```bash
polima compile --checkpoint <ckpt> --build-dir <dir>   # export, compile, pack
polima compile --build-dir <dir>                       # recompile what changed
polima compile --import-legacy <dir>                   # adopt an existing tree
polima deploy --bundle <bundle>                        # deploy only; does not serve
polima deploy --bundle <bundle> --start                # explicitly start serving
polima deploy                                          # interactive bundle/board setup
```

### On the board

`polima-cli` with no arguments opens a session over the board's model store,
the way `llima run` does — the model loads once and everything after is fast:

```
polima> use 1
  loaded act-rcwb_f_t-100000-a3573dae (6 graphs, 0.1s)
act-rcwb_f_t-100000-a3573dae> run
  600 floats in 22.5 ms
act-rcwb_f_t-100000-a3573dae> stages
  total 22.3 ms
    run_elf:vision_backbone=4.977770
    run_elf:vision_backbone=4.911727
    ...
```

`models` / `use` / `run` / `bench` / `stages` / `check` / `info` / `save`, with
history, cursor keys and tab completion. `--bundle` still runs once and exits.
Deploy links `polima-cli` and `polima-server` onto `PATH`, next to `llima`.

### PoLiMa Studio

The board installer registers a SOM-resident web cockpit, disabled and stopped
by default. Studio provides camera-only MJPG preview, bundle and
policy-server management, hardware assignment, guarded robot runs, guided
calibration, diagnostics, logs, and bounded run history. One browser holds the
controller lease while all observers retain access to the emergency halt.
It also provides isolated, hardware-free fixture benchmarks with warm-up,
latency percentiles, throughput, per-stage timings, and output verification.

```bash
polima studio status             # service state and URL
polima studio start              # start for this boot
polima studio enable             # opt in to starting at boot
polima studio stop               # stop without changing boot preference
polima studio disable            # stop and disable at boot
polima studio logs [-f]          # recent or following service logs
polima studio serve --port 8080  # foreground, without systemd
```

Robot start requires a successful preflight followed by confirmation within
15 seconds. Normal stop returns to the dataset rest pose; **HALT NOW** skips
that motion and immediately disconnects the follower. See
[docs/polima-studio.md](docs/polima-studio.md) for deployment and API details.
