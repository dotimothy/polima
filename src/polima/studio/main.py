from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PoLiMa Studio SOM control cockpit")
    parser.add_argument("--root", default=os.environ.get("POLIMA_ROOT", "/media/nvme/polima"))
    parser.add_argument("--host", default=os.environ.get("POLIMA_STUDIO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POLIMA_STUDIO_PORT", "8080")))
    args = parser.parse_args(argv)

    from .app import create_app

    app = create_app(Path(args.root))
    runtime = app.extensions["polima_runtime"]

    def shutdown(signum, _frame):
        runtime.emit("halt", reason=f"Studio received signal {signum}")
        runtime.close()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        from waitress import serve
    except ImportError:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    else:
        serve(app, host=args.host, port=args.port, threads=8, channel_timeout=60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
