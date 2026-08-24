# PoLiMa Studio

PoLiMa Studio is the SOM-resident operations cockpit for installed PoLiMa robot policies. It
does not compile or deploy models; those remain host-side workflows. Studio manages the active
bundle, policy server, cameras, follower arm, calibration, controlled runs, logs, and run history.

## Manage it

The Modalix installer registers `polima-studio.service` but deliberately leaves it disabled and
stopped. Start it only when needed; enabling it at boot is a separate, explicit choice:

```bash
polima studio status
polima studio start
polima studio enable
polima studio stop
polima studio disable
polima studio logs --follow
polima studio open
polima studio serve --host 0.0.0.0 --port 8080
```

`start` and `stop` affect the current boot only. `enable` opts into automatic startup; `disable`
also stops the service. With Studio running, open the URL printed by `polima studio status` from
the same LAN.

The service intentionally has no login or TLS and should only be exposed on an isolated robot
LAN. Mutations use same-origin CSRF protection. One browser owns the renewable controller lease;
other browsers are read-only, except that every browser may use **HALT NOW**.

## Safety model

Robot start is a two-step transaction. Preflight verifies the exact bundle and device paths and
issues a token valid for 15 seconds. The token is cryptographically bound to that configuration,
so changing a camera, task, or runtime setting requires another preflight. Graceful stop returns
the arm to its dataset rest pose. Emergency halt sends `SIGUSR1`, skips reset motion, disconnects
the follower immediately, and relies on `disable_torque_on_disconnect` to release torque.

The native CLI holds `/media/nvme/polima/var/run/control.lock` for every camera preview or robot
run. This prevents Studio and a terminal command from controlling the same devices concurrently.
Closing a browser does not stop an active run. If the Studio service itself terminates, its signal
handler triggers the emergency halt path before systemd restarts it.

Run summaries are stored in `var/lib/studio.sqlite3`; process logs are under `var/log/studio`.
The newest 100 summaries are retained and logs are bounded to 500 MiB. Camera video is not stored.

## Hardware-free benchmarks

The **MLA performance benchmark** panel runs the selected bundle against its packaged fixture
inputs. It never opens the follower arm or either camera. Studio loads the model once, discards
the requested warm-up runs, then reports minimum, mean, median, p95, p99, maximum, throughput,
per-stage timing, and fixture-output agreement. Benchmark history and raw logs are retained with
the other Studio records.

For an isolated measurement, Studio refuses to begin while the policy server, robot control,
camera preview, or calibration is active. The native benchmark holds the same controller lock as
robot operation, preventing a terminal process from starting control during the measurement.

## API

The versioned API is rooted at `/api/v1`. Important endpoints are `session`, `snapshot`,
`lease`, `robot/arm`, `robot/start`, `robot/stop`, `robot/halt`, `preview/start`, `server/start`,
`bundles/<id>/activate`, `calibration/*`, `history`, `logs/*`, and the `events` SSE stream.
Benchmark endpoints are `benchmarks/start`, `benchmarks/stop`, `benchmarks`, and
`logs/benchmarks/<id>`.
