"""Top-level Web Workbench launcher for MelodyDesk 2.0 (http://127.0.0.1:7865)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uvicorn

MELODY_ROOT = Path(__file__).resolve().parent
if str(MELODY_ROOT) not in sys.path:
    sys.path.insert(0, str(MELODY_ROOT))

from subpkgs.score_web.server import create_app

app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="MelodyDesk 2.0 Web Workbench Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7865, help="Bind port (default: 7865)")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
