# PoLiMa

**Po**licy + Si**Ma** — a unified framework for training, compiling and deploying robot policies (ACT, SmolVLA, GR00T) onto SiMa.ai Modalix hardware.

PoLiMa is the robot-policy analogue of **LLiMa**, and uses it under the hood: VLM backbones go through `sima_lmm.host.compile_lmm`, action experts go through `afe`. It mirrors LLiMa's shape deliberately — separate console scripts per stage, not one monolith.

Adding a policy should mean writing one `PolicySpec`, not copying 3,000 lines.

---

## The two-machine split

PoLiMa spans two machines with incompatible dependency sets. That split is structural, not a convention.

| | **HOST** (x86_64 workstation) | **BOARD** (Modalix SoM, aarch64) |
|---|---|---|
| | `polima-train` — torch, lerobot | `polima` — umbrella |
| | `polima-compile` — afe, onnx, onnxsim | `polima-run` — inference |
| | `polima-deploy` — ssh, rsync | `polima-robot` — teleop, live view |
| deps | `polima[host]` | `polima` (numpy only) or `polima[robot]` |

`polima-doctor` runs on both and is the first thing to run after any install.

`polima <stage>` also works everywhere as an umbrella alias, so `polima compile …` and `polima-compile …` are the same code path.

### Why not two packages

Both halves share one core — `polima.policies`, `polima.wire`, `polima.bundle`, `polima.config`, `polima.util` — because that core **is the contract between them**: the host writes `bundle.json` and `plan.json`, the board reads them. Splitting them into disjoint distributions would mean duplicating the contract, which is exactly the failure mode PoLiMa exists to remove.

So the core carries a hard dependency floor of **stdlib + numpy**, and it must import in all three interpreters:

- the `act` conda env (py3.12 + torch)
- the SiMa model-compiler venv (py3.12 + `afe`, **no torch**)
- the board's py3.11

`polima doctor --imports` proves this by importing the core under each interpreter, and `tests/unit/test_role.py` asserts no heavy dependency leaks in. A bare `pip install polima` pulls exactly one package: numpy.

Invoking a stage on the wrong machine exits `3` with one actionable sentence, rather than an `ImportError` three frames deep in a vendored library.

---

## Install

**Host**
```bash
conda activate act
pip install -e "polima[host]"
polima-doctor
```

**Board**
```bash
ssh sima@192.168.91.211
pip install -e "/media/nvme/polima/src[robot]"
polima-doctor
```

---

## Layout

```
src/polima/
  policies/   THE PLUGIN LAYER -- one PolicySpec per policy
  config/     layered config (dataclasses + stdlib loader; no pydantic)
  data/       dataset contracts, discovery, episode specs, aggregation
  train/      TrainRunner            <- replaces 3 bash scripts (~1,485 lines)
  export/     torch -> ONNX helpers
  compile/    afe + llima wrappers, the Stage state machine
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

Backups live outside `polima/` so removing the framework never takes the safety net with it. `make check-legacy-intact` gates every phase boundary until `git init` lands in Phase 2.

---

## Status

169 unit tests, no hardware required. Both milestones so far are proven against
the real board rather than asserted:

| Stage | State | Proof |
|---|---|---|
| `polima doctor` / `data` / `list` | working | |
| `polima-deploy` / `polima-run` | working | 20.8 ms on the SoM, cosine 0.999990 vs PyTorch — 24% faster than the hand-written `act_llima` at 27.0 ms |
| `polima-compile` | working | recompiles all six ACT ELFs **byte for byte** from the checkpoint's ONNX, yielding the same content-addressed bundle id as the one deployed ([docs/compile.md](docs/compile.md)) |
| `polima-train` | Phase 1c | |
| `polima-robot` | Phase 1d | |
| SmolVLA / GR00T | Phases 4–5 | |

The compile stage replaces three divergent copies of
`sima_compile_onnx_tensors.py` (518 lines) plus two per-policy controllers with
one driver over `PolicySpec.compile.graphs`, so adding a policy needs no
controller at all.
