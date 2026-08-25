#!/usr/bin/env bash
# Install the two native PoLiMa binaries and the LeRobot control environment on Modalix.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${POLIMA_NATIVE_SOURCE:-$(cd -- "$SCRIPT_DIR/../native" && pwd)}"
POLIMA_SOURCE="${POLIMA_SOURCE:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ROOT="${POLIMA_ROOT:-/media/nvme/polima}"
LEROBOT_VENV="${LEROBOT_VENV:-/media/nvme/lerobot}"
LEROBOT_INSTALLER="${LEROBOT_INSTALLER:-$POLIMA_SOURCE/lerobot_sima/install_lerobot_modalix.sh}"
JOBS="${JOBS:-$(nproc)}"

fail() { echo "[fail] $*" >&2; exit 1; }
log() { echo "==> $*"; }

[[ "$(uname -m)" == aarch64 ]] || fail "this installer must run on the Modalix devkit"
[[ -f "$SOURCE/CMakeLists.txt" ]] || fail "native source not found at $SOURCE"
mkdir -p "$ROOT/bin" "$ROOT/build" "$ROOT/var/log" "$ROOT/var/run"

if [[ ! -x "$LEROBOT_VENV/bin/python" ]]; then
    [[ "${SKIP_LEROBOT:-0}" != 1 ]] || fail "LeRobot venv is absent and SKIP_LEROBOT=1"
    [[ -x "$LEROBOT_INSTALLER" ]] || fail "LeRobot installer not found at $LEROBOT_INSTALLER; initialize a complete PoLiMa checkout or set LEROBOT_INSTALLER"
    log "Installing the standalone LeRobot environment"
    VENV="$LEROBOT_VENV" "$LEROBOT_INSTALLER"
else
    log "Reusing LeRobot environment at $LEROBOT_VENV"
fi

if ! "$LEROBOT_VENV/bin/python" -c '
import importlib.metadata as metadata
import flask, grpc, waitress, google.protobuf.runtime_version
assert metadata.version("protobuf") == "6.31.1"
assert metadata.version("wandb") == "0.24.2"
assert metadata.version("grpcio") == "1.80.0"
' >/dev/null 2>&1; then
    log "Installing Studio dependencies and compatible protobuf and gRPC runtimes"
    "$LEROBOT_VENV/bin/python" -m pip install \
        flask waitress "wandb==0.24.2" "protobuf==6.31.1" "grpcio==1.80.0"
fi

log "Installing the PoLiMa board package and Studio assets"
"$LEROBOT_VENV/bin/python" -m pip install --no-deps --upgrade "$POLIMA_SOURCE"

# Build in a staging tree, then atomically replace the one active binary pair.
# CMake intermediates are not an installed runtime and can be removed afterward.
STAGE="$ROOT/build/install-staging"
log "Building polima_server and polima_cli with $JOBS jobs"
cmake -E remove_directory "$STAGE"
cmake -S "$SOURCE" -B "$STAGE" -DCMAKE_BUILD_TYPE=Release
cmake --build "$STAGE" -j"$JOBS"
install -m 0755 "$STAGE/polima_server" "$ROOT/bin/polima_server.new"
install -m 0755 "$STAGE/polima_cli" "$ROOT/bin/polima_cli.new"
mv -f "$ROOT/bin/polima_server.new" "$ROOT/bin/polima_server"
mv -f "$ROOT/bin/polima_cli.new" "$ROOT/bin/polima_cli"

link_dir=""
for candidate in /usr/local/bin /usr/bin; do
    if [[ -d "$candidate" && -w "$candidate" ]]; then
        link_dir="$candidate"
        break
    fi
    if [[ -d "$candidate" ]] && sudo -n true 2>/dev/null; then
        sudo -n ln -sfn "$ROOT/bin/polima_cli" "$candidate/polima"
        sudo -n ln -sfn "$ROOT/bin/polima_cli" "$candidate/polima-cli"
        sudo -n ln -sfn "$ROOT/bin/polima_server" "$candidate/polima-server"
        link_dir="$candidate"
        break
    fi
done
if [[ -n "$link_dir" && -w "$link_dir" ]]; then
    ln -sfn "$ROOT/bin/polima_cli" "$link_dir/polima"
    ln -sfn "$ROOT/bin/polima_cli" "$link_dir/polima-cli"
    ln -sfn "$ROOT/bin/polima_server" "$link_dir/polima-server"
fi

if command -v systemctl >/dev/null 2>&1; then
    systemd_dir=/etc/systemd/system
    default_dir=/etc/default
    studio_service_was_installed=0
    [[ -e "$systemd_dir/polima-studio.service" ]] && studio_service_was_installed=1
    systemctl_cmd=(systemctl)
    install_cmd=(install)
    if [[ ! -w "$systemd_dir" ]]; then
        sudo -n true 2>/dev/null || fail "sudo is required to install polima-studio.service"
        systemctl_cmd=(sudo -n systemctl)
        install_cmd=(sudo -n install)
    fi
    log "Installing polima-studio.service"
    "${install_cmd[@]}" -m 0644 "$POLIMA_SOURCE/scripts/systemd/polima-studio.service" \
        "$systemd_dir/polima-studio.service"
    if [[ ! -e "$default_dir/polima-studio" ]]; then
        "${install_cmd[@]}" -m 0644 "$POLIMA_SOURCE/scripts/systemd/polima-studio.default" \
            "$default_dir/polima-studio"
    fi
    "${systemctl_cmd[@]}" daemon-reload
    if [[ "$studio_service_was_installed" == 0 ]]; then
        log "Leaving PoLiMa Studio disabled and stopped by default"
        "${systemctl_cmd[@]}" disable --now polima-studio.service
    else
        log "Preserving the existing PoLiMa Studio enable/running state"
    fi
fi

log "Verifying"
"$ROOT/bin/polima_cli" --help >/dev/null
"$ROOT/bin/polima_server" --help >/dev/null
"$LEROBOT_VENV/bin/python" -c 'import importlib.metadata as m; import lerobot, cv2, flask, waitress, torch, polima.studio; print("lerobot=%s torch=%s cv2=%s flask=%s waitress=%s" % (lerobot.__version__, torch.__version__, cv2.__version__, m.version("flask"), m.version("waitress")))'

cat <<EOF

Installed:
  $ROOT/bin/polima_cli
  $ROOT/bin/polima_server
  $LEROBOT_VENV

Next:
  polima activate        # select and optionally start an installed bundle
  polima server          # interactive start/stop
  polima robot           # inspect hardware and confirm startup
  polima studio start    # start Studio for this boot
  polima studio enable   # optionally enable Studio at boot
  polima studio status   # print state and URL
EOF
