"""Unified portable & frozen-executable filesystem path resolution for MelodyDesk 2.0.

Supports three layouts identically without unguarded `parents[N]` out-of-bounds indexing:
1. Workspace layout (`<workspace>/outputs/MelodyDesk` + `<workspace>/work`);
2. Standalone / cloned repository layout (`<repo>` + `<repo>/work`, even at drive root `D:\\repo`);
3. PyInstaller frozen `--onedir` layout (`<install>/MelodyDesk.exe` + `<install>/_internal` + `<install>/work`),
   with automatic fallback to `%LOCALAPPDATA%\\MelodyDesk\\work` (or system temp) for writable directories
   when installed under a read-only directory such as `C:\\Program Files`.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


def is_frozen() -> bool:
    """Return True when running inside a frozen PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def get_app_root() -> Path:
    """Return the root directory containing bundled source/static/data/third_party assets."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_install_root() -> Path:
    """Return the outer installation / workspace root directory."""
    env_home = os.environ.get("MELODYDESK_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()
    if is_frozen():
        return Path(sys.executable).resolve().parent
    app_root = get_app_root()
    for candidate in (app_root, *app_root.parents[:2]):
        if (candidate / "work").is_dir():
            return candidate
    return app_root


def _is_dir_writable(target: Path) -> bool:
    """Probe whether `target` (or its nearest existing parent) accepts file creation."""
    probe = target
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return False
        probe = parent
    if not probe.is_dir():
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=probe, prefix=".md_write_probe_", delete=True):
            return True
    except OSError:
        return False


def get_work_root() -> Path:
    """Return the primary `work` directory (used for bundled samples, repertoire, and caches)."""
    env_work = os.environ.get("MELODYDESK_WORK_DIR", "").strip()
    if env_work:
        return Path(env_work).expanduser().resolve()
    return get_install_root() / "work"


def get_writable_work_root() -> Path:
    """Return a probed writable `work` directory for uploads, logs, and JIT caches."""
    preferred = get_work_root()
    if _is_dir_writable(preferred):
        return preferred
    local_app = os.environ.get("LOCALAPPDATA", "").strip()
    candidates = []
    if local_app:
        candidates.append((Path(local_app) / "MelodyDesk" / "work").resolve())
    candidates.append((Path.home() / ".melodydesk" / "work").resolve())
    candidates.append((Path(tempfile.gettempdir()) / "MelodyDesk" / "work").resolve())
    for cand in candidates:
        if _is_dir_writable(cand):
            return cand
    raise PermissionError("Unable to locate any writable directory for MelodyDesk runtime state.")


def get_model_cache_root(*, require_writable: bool = False) -> Path:
    """Return the `model-cache` root directory."""
    env_cache = os.environ.get("MELODYDESK_MODEL_CACHE", "").strip()
    if env_cache:
        return Path(env_cache).expanduser().resolve()
    bundled = get_work_root() / "model-cache"
    if not require_writable and (bundled.is_dir() or _is_dir_writable(bundled)):
        return bundled
    if require_writable and _is_dir_writable(bundled):
        return bundled
    return get_writable_work_root() / "model-cache"


def candidate_model_cache_roots() -> list[Path]:
    """Return ordered unique `model-cache` directories to search for pre-installed weights."""
    roots: list[Path] = []
    env_cache = os.environ.get("MELODYDESK_MODEL_CACHE", "").strip()
    raw_candidates = []
    if env_cache:
        raw_candidates.append(Path(env_cache).expanduser())
    raw_candidates.extend([
        get_work_root() / "model-cache",
        get_writable_work_root() / "model-cache",
        get_app_root() / "model-cache",
        Path.home() / ".cache" / "melodydesk",
    ])
    for cand in raw_candidates:
        resolved = cand.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def resolve_model_path(*parts: str, require_writable_parent: bool = False) -> Path:
    """Resolve an existing model file/directory across candidate caches, or return the default target path."""
    for root in candidate_model_cache_roots():
        cand = root.joinpath(*parts)
        if cand.exists():
            if not require_writable_parent or _is_dir_writable(cand if cand.is_dir() else cand.parent):
                return cand
    return get_model_cache_root(require_writable=require_writable_parent).joinpath(*parts)
