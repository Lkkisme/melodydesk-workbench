"""Top-level Web Workbench launcher for MelodyDesk 2.0 (http://127.0.0.1:7865)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import uvicorn

MELODY_ROOT = Path(__file__).resolve().parent
if str(MELODY_ROOT) not in sys.path:
    sys.path.insert(0, str(MELODY_ROOT))

from runtime_paths import get_model_cache_root, get_writable_work_root  # noqa: E402
from subpkgs.score_web.server import create_app  # noqa: E402


def _configure_writable_caches() -> None:
    writable_work = get_writable_work_root()
    numba_dir = writable_work / "numba-cache"
    model_cache = get_model_cache_root(require_writable=True)
    for d in (numba_dir, model_cache):
        d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(model_cache))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_dir))
    os.environ.setdefault("HF_HOME", str(model_cache / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


_DEFAULT_APP = None
_DEFAULT_APP_LOCK = threading.Lock()


def __getattr__(name: str):
    global _DEFAULT_APP
    if name == "app":
        with _DEFAULT_APP_LOCK:
            if _DEFAULT_APP is None:
                _configure_writable_caches()
                _DEFAULT_APP = create_app()
            return _DEFAULT_APP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    _configure_writable_caches()
    parser = argparse.ArgumentParser(description="MelodyDesk 2.0 Web Workbench Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7865, help="Bind port (default: 7865)")
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
