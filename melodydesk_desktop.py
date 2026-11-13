"""Standalone Windows Desktop & Portable Executable Launcher for MelodyDesk 2.0.

Provides:
1. Frozen-safe `multiprocessing.freeze_support()` and re-entrancy-safe UTF-8 rotating file
   logging when packaged with PyInstaller `--noconsole` (`sys.stdout` / `sys.stderr` are `None`);
2. Single-instance port reuse: if MelodyDesk is already serving on the target port,
   attaches the Studio Console window cleanly and opens the existing workbench URL;
3. Process-isolated Quiet-Luxury Windows Studio Controller card (`tkinter` UI process +
   isolated `--headless` worker subprocess with Win32 `SYNCHRONIZE` parent-exit watchdog),
   eliminating CPython GIL contention between Tcl/Tk and Uvicorn/PyTorch while terminating
   the worker process tree deterministically on window close or forced termination.
"""
from __future__ import annotations

import argparse
import ctypes
import http.client
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser

MELODY_ROOT = Path(__file__).resolve().parent
if str(MELODY_ROOT) not in sys.path:
    sys.path.insert(0, str(MELODY_ROOT))

from runtime_paths import (  # noqa: E402
    get_app_root,
    get_install_root,
    get_model_cache_root,
    get_work_root,
    get_writable_work_root,
    is_frozen,
)

APP_VERSION = "2.0.2"

_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _SafeRotatingFileHandler(RotatingFileHandler):
    """Windows-safe RotatingFileHandler that tolerates concurrent file-lock contention during rollover."""

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except OSError:
            if self.stream is None:
                try:
                    self.stream = self._open()
                except OSError:
                    pass


class _LogStreamAdapter:
    """Thread-safe, re-entrancy-safe UTF-8 stream adapter for `--noconsole` builds."""

    encoding = "utf-8"

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._local = threading.local()

    def write(self, text: str) -> int:
        if not text:
            return 0
        raw = str(text)
        if getattr(self._local, "in_write", False):
            return len(raw)
        self._local.in_write = True
        try:
            buf = getattr(self._local, "buf", "") + raw
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                stripped = line.rstrip("\r")
                if stripped:
                    self._logger.log(self._level, "%s", stripped)
            self._local.buf = buf
            return len(raw)
        finally:
            self._local.in_write = False

    def flush(self) -> None:
        if getattr(self._local, "in_write", False):
            return
        self._local.in_write = True
        try:
            buf = getattr(self._local, "buf", "")
            if buf.strip():
                self._logger.log(self._level, "%s", buf.rstrip("\r\n"))
            self._local.buf = ""
        finally:
            self._local.in_write = False

    def isatty(self) -> bool:
        return False


def _show_error_dialog(message: str, title: str = "弦语 MelodyDesk 启动提示", icon_flags: int = 0x00000010) -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.MessageBoxW(0, str(message), str(title), int(icon_flags))
    except Exception as dlg_exc:  # noqa: BLE001
        logging.getLogger("melodydesk.launcher").warning("MessageBoxW failed: %s", dlg_exc)


def configure_runtime_environment(log_filename: str = "melodydesk.log") -> Path:
    """Initialize writable cache/log paths and ensure `sys.stdout`/`sys.stderr` exist."""
    writable_work = get_writable_work_root()
    log_dir = writable_work / "logs"
    numba_dir = writable_work / "numba-cache"
    writable_model_cache = get_model_cache_root(require_writable=True)
    hf_cache = writable_model_cache / "hf"

    for d in (log_dir, numba_dir, writable_model_cache):
        d.mkdir(parents=True, exist_ok=True)

    # Set default TORCH_HOME and HF_HOME to the probed writable cache directory when not overridden.
    os.environ.setdefault("TORCH_HOME", str(writable_model_cache))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_dir))
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    extra_bins = [
        get_install_root() / "bin",
        get_install_root(),
        get_app_root() / "bin",
        get_app_root(),
    ]
    cur_path = os.environ.get("PATH", "")
    path_parts = cur_path.split(os.pathsep) if cur_path else []
    for b in reversed(extra_bins):
        b_str = str(b)
        if b.is_dir() and b_str not in path_parts:
            path_parts.insert(0, b_str)
    os.environ["PATH"] = os.pathsep.join(path_parts)

    log_file = log_dir / log_filename
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler = _SafeRotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)
        if sys.stderr is not None and not isinstance(sys.stderr, _LogStreamAdapter):
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setFormatter(fmt)
            root_logger.addHandler(stream_handler)

    if sys.stdout is None:
        sys.stdout = _LogStreamAdapter(logging.getLogger("melodydesk.stdout"), logging.INFO)
    if sys.stderr is None:
        sys.stderr = _LogStreamAdapter(logging.getLogger("melodydesk.stderr"), logging.ERROR)

    return log_file


def verify_portable_bundle_integrity() -> None:
    """Ensure a frozen portable `.exe` has its companion `_internal` and model/sample directories."""
    if not is_frozen():
        return
    app_root = get_app_root()
    work_root = get_work_root()
    env_samples = os.environ.get("MELODYDESK_GUITAR_SAMPLES", "").strip()
    samples_root = Path(env_samples).expanduser().resolve() if env_samples else (work_root / "guitar_samples")
    env_cache = os.environ.get("MELODYDESK_MODEL_CACHE", "").strip()
    model_cache = Path(env_cache).expanduser().resolve() if env_cache else (work_root / "model-cache")
    has_models = (
        model_cache.is_dir()
        and (
            (model_cache / "mdx").is_dir()
            or (model_cache / "btc-chord").is_dir()
            or (model_cache / "panns-cnn6.pth").is_file()
        )
    )
    missing: list[str] = []
    if not (app_root / "static" / "index.html").is_file():
        missing.append("_internal/static")
    if not samples_root.is_dir():
        missing.append("work/guitar_samples")
    if not has_models:
        missing.append("work/model-cache")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"检测到便携版目录资源不完整（缺少 {joined}）。\n\n"
            "请先将整个 MelodyDesk 压缩包完整解压到普通文件夹后再双击 MelodyDesk.exe，"
            "切勿直接在压缩包预览窗口内运行或单独将 MelodyDesk.exe 拖出目录。"
        )


def _normalize_bind_host(bind_host: str) -> str:
    stripped = str(bind_host or "").strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped[1:-1]
    return stripped or "127.0.0.1"


def _connect_host(bind_host: str) -> str:
    norm = _normalize_bind_host(bind_host)
    if norm in ("0.0.0.0", ""):
        return "127.0.0.1"
    if norm == "::":
        return "::1"
    return norm


def _format_url_host(bind_host: str) -> str:
    conn = _connect_host(bind_host)
    return f"[{conn}]" if ":" in conn else conn


def _is_melodydesk_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("ok") is True and payload.get("app") == "melodydesk":
        return True
    return ("publication_html" in payload or "score_html" in payload) and "arrangement" in payload


def probe_melodydesk_server(bind_host: str, port: int, timeout: float = 1.5) -> bool:
    """Return True if a live MelodyDesk 2.0 server is responding on `(bind_host, port)` without using system proxies."""
    base = f"http://{_format_url_host(bind_host)}:{int(port)}"
    health_req = urllib.request.Request(f"{base}/api/health", headers={"Accept": "application/json"})
    try:
        with _NO_PROXY_OPENER.open(health_req, timeout=max(0.5, float(timeout))) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            return _is_melodydesk_payload(payload)
    except urllib.error.HTTPError as http_exc:
        code = http_exc.code
        http_exc.close()
        if code != 404:
            return False
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
        return False

    # Fallback for older servers that expose `/api/state` without `/api/health`.
    state_req = urllib.request.Request(f"{base}/api/state", headers={"Accept": "application/json"})
    try:
        with _NO_PROXY_OPENER.open(state_req, timeout=max(5.0, float(timeout))) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            return _is_melodydesk_payload(payload)
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
        return False


def _is_tcp_port_free(bind_host: str, port: int) -> bool:
    norm_host = _normalize_bind_host(bind_host)
    conn_host = _connect_host(norm_host)
    family = socket.AF_INET6 if ":" in norm_host else socket.AF_INET

    # First check whether an active listener already accepts connections on `(conn_host, port)`
    # (catches wildcard 0.0.0.0 / [::] or SO_REUSEADDR listeners on Windows).
    conn_family = socket.AF_INET6 if ":" in conn_host else socket.AF_INET
    with socket.socket(conn_family, socket.SOCK_STREAM) as probe_sock:
        probe_sock.settimeout(0.15)
        try:
            if probe_sock.connect_ex((conn_host, int(port))) == 0:
                return False
        except OSError:
            pass

    with socket.socket(family, socket.SOCK_STREAM) as sock:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        elif os.name != "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((norm_host, int(port)))
            return True
        except OSError:
            return False


def select_listen_port(bind_host: str, preferred_port: int, explicit_port: bool) -> int:
    """Select `preferred_port` if available, or scan `preferred_port + 1 .. +30` when auto-selecting."""
    if _is_tcp_port_free(bind_host, preferred_port):
        return preferred_port
    if explicit_port:
        raise OSError(f"端口 {preferred_port} 已被占用，请关闭占用进程或使用 --port 指定其他端口。")
    for candidate in range(preferred_port + 1, preferred_port + 31):
        if _is_tcp_port_free(bind_host, candidate):
            return candidate
    raise OSError(f"端口区间 {preferred_port}–{preferred_port + 30} 均被占用。")


class EmbeddedWorkbenchRunner:
    """Manages the in-process FastAPI + Uvicorn lifecycle for `--headless` mode."""

    is_attached = False

    def __init__(self, host: str, port: int) -> None:
        self.host = _normalize_bind_host(host)
        self.port = int(port)
        self.base_url = f"http://{_format_url_host(self.host)}:{self.port}"
        self._lock = threading.Lock()
        self._stop_requested = False
        self.server = None
        self.error: BaseException | None = None
        self.ready_event = threading.Event()
        self._stopped_event = threading.Event()
        self._thread = threading.Thread(target=self._run_worker, name="melodydesk-uvicorn", daemon=False)

    def start(self) -> None:
        with self._lock:
            if self._stop_requested:
                self._stopped_event.set()
                self.ready_event.set()
                return
        self._thread.start()

    def _run_worker(self) -> None:
        try:
            with self._lock:
                if self._stop_requested:
                    return
            import uvicorn
            from subpkgs.score_web.server import create_app

            app = create_app()
            config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                workers=1,
                log_level="info",
                log_config=None,
                access_log=False,
            )
            srv = uvicorn.Server(config)
            with self._lock:
                if self._stop_requested:
                    return
                self.server = srv

            def _watch_ready() -> None:
                deadline = time.monotonic() + 90.0
                while time.monotonic() < deadline and not self._stopped_event.is_set():
                    if getattr(srv, "started", False) and probe_melodydesk_server(self.host, self.port, timeout=1.5):
                        self.ready_event.set()
                        return
                    if self._stopped_event.wait(timeout=0.25):
                        return
                if not self._stopped_event.is_set() and not self.ready_event.is_set():
                    self.error = TimeoutError("本地曲谱服务在 90 秒内未就绪")
                    srv.should_exit = True
                    self.ready_event.set()

            threading.Thread(target=_watch_ready, name="melodydesk-readiness", daemon=True).start()
            srv.run()
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
            logging.getLogger("melodydesk.launcher").exception("MelodyDesk server failed to start")
        finally:
            self._stopped_event.set()
            self.ready_event.set()

    def stop(self, timeout: float = 6.0) -> None:
        with self._lock:
            self._stop_requested = True
            self._stopped_event.set()
            srv = self.server
            if srv is not None:
                srv.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive() and srv is not None:
                srv.force_exit = True
                self._thread.join(timeout=2.0)


def _read_server_log_error_hint(log_dir: Path) -> str:
    """Return the most recent error/exception line from `melodydesk.log`, or the final non-empty log line."""
    server_log = log_dir / "melodydesk.log"
    if not server_log.is_file():
        return ""
    try:
        lines = [
            ln.strip()
            for ln in server_log.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
        for ln in reversed(lines[-40:]):
            if any(tok in ln for tok in ("[ERROR]", "[CRITICAL]", "Error:", "Exception:", "Traceback")):
                return f"（{ln[:160]}）"
        if lines:
            return f"（{lines[-1][:160]}）"
    except OSError:
        pass
    return ""


class SubprocessWorkbenchRunner:
    """Runs the FastAPI + Uvicorn server in an isolated `--headless` child process for smooth GUI responsiveness."""

    def __init__(self, host: str, port: int, log_dir: Path) -> None:
        self.is_attached = False
        self.host = _normalize_bind_host(host)
        self.port = int(port)
        self.base_url = f"http://{_format_url_host(self.host)}:{self.port}"
        self._log_dir = log_dir
        self._lock = threading.Lock()
        self._stop_requested = False
        self._proc: subprocess.Popen | None = None
        self.error: BaseException | None = None
        self.ready_event = threading.Event()
        self._stopped_event = threading.Event()

    def _build_worker_command(self) -> list[str]:
        common = [
            "--headless",
            "--no-browser",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--parent-pid",
            str(os.getpid()),
        ]
        if is_frozen():
            return [sys.executable, *common]
        return [sys.executable, str(Path(__file__).resolve()), *common]

    def start(self) -> None:
        with self._lock:
            if self._stop_requested:
                self._stopped_event.set()
                self.ready_event.set()
                return
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0
            try:
                proc = subprocess.Popen(
                    self._build_worker_command(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
            except BaseException as exc:  # noqa: BLE001
                self.error = exc
                self._stopped_event.set()
                self.ready_event.set()
                return
            self._proc = proc

        threading.Thread(target=self._watch_worker, args=(proc,), name="melodydesk-worker-watch", daemon=True).start()

    def _watch_worker(self, proc: subprocess.Popen) -> None:
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline and not self._stopped_event.is_set():
            rc = proc.poll()
            if rc is not None:
                if not self._stop_requested:
                    # Handle simultaneous double-click race: if another MelodyDesk instance won the same port, attach cleanly.
                    if probe_melodydesk_server(self.host, self.port, timeout=1.5):
                        self.is_attached = True
                        self.ready_event.set()
                        return
                    hint = _read_server_log_error_hint(self._log_dir)
                    self.error = RuntimeError(f"后台服务异常退出 (code={rc}){hint}")
                self._stopped_event.set()
                self.ready_event.set()
                return
            if probe_melodydesk_server(self.host, self.port, timeout=1.5):
                self.ready_event.set()
                while not self._stopped_event.wait(timeout=0.5):
                    rc_live = proc.poll()
                    if rc_live is not None:
                        if not self._stop_requested:
                            if probe_melodydesk_server(self.host, self.port, timeout=1.5):
                                self.is_attached = True
                                return
                            hint = _read_server_log_error_hint(self._log_dir)
                            self.error = RuntimeError(f"后台服务意外中断 (code={rc_live}){hint}")
                        self._stopped_event.set()
                        return
                return
            if self._stopped_event.wait(timeout=0.25):
                return

        if not self._stop_requested and not self.ready_event.is_set():
            self.error = TimeoutError("后台曲谱服务在 90 秒内未响应")
            self.stop(timeout=3.0)
            self.ready_event.set()

    def stop(self, timeout: float = 6.0) -> None:
        with self._lock:
            self._stop_requested = True
            self._stopped_event.set()
            self.ready_event.set()
            proc = self._proc
        if proc is None or self.is_attached:
            return
        if proc.poll() is None:
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                    check=False,
                )
            else:
                proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired as kill_exc:
                    logging.getLogger("melodydesk.launcher").error(
                        "Worker PID %s did not exit after kill: %s", proc.pid, kill_exc
                    )


class AttachedWorkbenchRunner:
    """Lightweight handle attaching the Studio Console UI to an already-running MelodyDesk server."""

    is_attached = True

    def __init__(self, host: str, port: int) -> None:
        self.host = _normalize_bind_host(host)
        self.port = int(port)
        self.base_url = f"http://{_format_url_host(self.host)}:{self.port}"
        self.error: BaseException | None = None
        self.ready_event = threading.Event()
        self.ready_event.set()
        self._stopped_event = threading.Event()

    def start(self) -> None:
        return

    def stop(self, timeout: float = 6.0) -> None:
        del timeout
        self._stopped_event.set()


def _start_parent_watchdog(parent_pid: int, runner: EmbeddedWorkbenchRunner) -> None:
    """Stop `runner` automatically if the parent Studio Console process exits or is killed."""
    if parent_pid <= 0:
        return

    def _watch() -> None:
        if os.name == "nt":
            synchronize_access = 0x00100000
            wait_timeout = 0x00000102
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.WaitForSingleObject.restype = ctypes.c_uint32
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int

            handle = kernel32.OpenProcess(synchronize_access, 0, int(parent_pid))
            if not handle:
                runner.stop(timeout=2.0)
                return
            try:
                while not runner._stopped_event.is_set():
                    res = kernel32.WaitForSingleObject(handle, 500)
                    if res != wait_timeout:
                        logging.getLogger("melodydesk.launcher").info(
                            "Parent process %s exited; shutting down headless worker.", parent_pid
                        )
                        runner.stop(timeout=2.0)
                        return
            finally:
                kernel32.CloseHandle(handle)
        else:
            while not runner._stopped_event.wait(timeout=0.5):
                if os.getppid() != parent_pid:
                    runner.stop(timeout=2.0)
                    return

    threading.Thread(target=_watch, name="melodydesk-parent-watchdog", daemon=True).start()


def _enable_windows_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _resolve_icon_path() -> Path | None:
    for base in (get_app_root(), get_install_root()):
        cand = base / "assets" / "melodydesk.ico"
        if cand.is_file():
            return cand
    return None


def _run_frozen_self_check() -> int:
    """Verify dynamic PyInstaller subsystems (FFmpeg, audio_separator, ROSVOT/RMVPE, BTC, unidecode)."""
    import importlib
    import shutil
    import singing_model
    import vocal_activity
    from subpkgs.stem_mir import chord_recognizer

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("Bundled ffmpeg executable not found on PATH")

    importlib.import_module("unidecode")
    importlib.import_module("pretty_midi")
    importlib.import_module("audio_separator.separator.architectures.mdx_separator")
    importlib.import_module("audio_separator.separator.architectures.mdxc_separator")

    from faster_whisper import WhisperModel
    from runtime_paths import resolve_model_path
    from subpkgs.stem_mir.separator_engine import _instantiate_separator, _resolve_separator_cache_dir

    mdx_dir = _resolve_separator_cache_dir(
        "mdx",
        "UVR_MDXNET_KARA_2.onnx",
        extra_required=("Kim_Vocal_2.onnx",),
    )
    with tempfile.TemporaryDirectory() as tmp:
        sep = _instantiate_separator(output_dir=tmp, model_file_dir=mdx_dir, use_soundfile=True)
        if sep is None:
            raise RuntimeError("audio_separator failed to initialize")

    if singing_model.ready():
        singing_model.load_models("cpu")
    if vocal_activity.ready():
        vocal_activity.load_model()
    btc = chord_recognizer._get_btc_runtime()
    if btc is None:
        raise RuntimeError("BTC 170-chord model failed to load")
    local_w = resolve_model_path("whisper-small-int8", require_writable_parent=False)
    if (local_w / "model.bin").is_file():
        WhisperModel(str(local_w), device="cpu", compute_type="int8", local_files_only=True)

    logging.getLogger("melodydesk.selfcheck").info("SELF_CHECK_OK ffmpeg=%s", ffmpeg_bin)
    return 0


def _run_headless(runner: EmbeddedWorkbenchRunner, open_browser: bool, parent_pid: int | None = None) -> int:
    def _handle_sig(_signum, _frame) -> None:
        runner.stop(timeout=2.0)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _handle_sig)

    if parent_pid is not None and parent_pid > 0:
        _start_parent_watchdog(parent_pid, runner)

    runner.start()
    runner.ready_event.wait(timeout=95.0)
    if runner._stopped_event.is_set() and runner.error is None:
        return 0
    if runner.error is not None:
        logging.getLogger("melodydesk.launcher").error("Startup failed: %s", runner.error)
        runner.stop(timeout=3.0)
        return 1
    if not probe_melodydesk_server(runner.host, runner.port, timeout=2.0):
        if runner._stopped_event.is_set() and runner.error is None:
            return 0
        logging.getLogger("melodydesk.launcher").error("Server did not become healthy within timeout.")
        runner.stop(timeout=3.0)
        return 1

    logging.getLogger("melodydesk.launcher").info("MelodyDesk %s ready at %s", APP_VERSION, runner.base_url)
    if open_browser:
        webbrowser.open(runner.base_url)

    try:
        while not runner._stopped_event.wait(timeout=0.5):
            pass
    except KeyboardInterrupt:
        runner.stop(timeout=5.0)
    return 0 if runner.error is None else 1


def _run_studio_window(runner, open_browser: bool, log_file: Path) -> int:
    _enable_windows_dpi_awareness()
    try:
        import tkinter as tk
    except Exception as tk_exc:
        logging.getLogger("melodydesk.launcher").warning("tkinter unavailable (%s); falling back to headless", tk_exc)
        if isinstance(runner, EmbeddedWorkbenchRunner):
            return _run_headless(runner, open_browser=open_browser)
        embedded = EmbeddedWorkbenchRunner(host=runner.host, port=runner.port)
        return _run_headless(embedded, open_browser=open_browser)

    root = tk.Tk()
    root.title(f"弦语 MelodyDesk {APP_VERSION} · Studio Console")
    root.configure(bg="#F5F1EA")
    root.resizable(False, False)

    width, height = 472, 252
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    pos_x = max(40, (sw - width) // 2)
    pos_y = max(40, (sh - height) // 3)
    root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")

    icon_path = _resolve_icon_path()
    if icon_path is not None:
        try:
            root.iconbitmap(default=str(icon_path))
        except Exception:
            pass

    card = tk.Frame(root, bg="#FBF9F5", highlightbackground="#DDD5C7", highlightthickness=1, padx=24, pady=20)
    card.pack(fill="both", expand=True, padx=14, pady=14)

    header = tk.Frame(card, bg="#FBF9F5")
    header.pack(fill="x")
    tk.Label(
        header,
        text="弦语  MelodyDesk",
        font=("Microsoft YaHei UI", 15, "bold"),
        fg="#1C1917",
        bg="#FBF9F5",
    ).pack(side="left")
    tk.Label(
        header,
        text=f"v{APP_VERSION} · Urtext Studio",
        font=("Consolas", 9),
        fg="#78716C",
        bg="#FBF9F5",
    ).pack(side="right", pady=(6, 0))

    tk.Label(
        card,
        text="全链路人声扒谱 · 六大和声美学重配 · 原木吉他物理合成工作台",
        font=("Microsoft YaHei UI", 9),
        fg="#57534E",
        bg="#FBF9F5",
        anchor="w",
    ).pack(fill="x", pady=(6, 14))

    status_row = tk.Frame(card, bg="#F3EFE6", highlightbackground="#E4DDD0", highlightthickness=1, padx=12, pady=9)
    status_row.pack(fill="x")

    dot_label = tk.Label(status_row, text="●", font=("Segoe UI", 10), fg="#B45309", bg="#F3EFE6")
    dot_label.pack(side="left")

    status_var = tk.StringVar(value="正在预热 6 弦原木采样与曲谱引擎，请稍候…")
    status_label = tk.Label(
        status_row,
        textvariable=status_var,
        font=("Microsoft YaHei UI", 9),
        fg="#292524",
        bg="#F3EFE6",
        anchor="w",
    )
    status_label.pack(side="left", padx=(8, 0), fill="x", expand=True)

    btn_row = tk.Frame(card, bg="#FBF9F5")
    btn_row.pack(fill="x", pady=(18, 0))

    def _open_main() -> None:
        webbrowser.open(runner.base_url)

    def _open_atlas() -> None:
        webbrowser.open(f"{runner.base_url}/evolution_atlas")

    closing = False
    stop_done = threading.Event()

    def _wait_stop_done() -> None:
        if stop_done.is_set():
            root.destroy()
            return
        root.after(60, _wait_stop_done)

    def _shutdown() -> None:
        nonlocal closing
        if closing:
            return
        closing = True
        if bool(getattr(runner, "is_attached", False)):
            runner.stop(timeout=1.0)
            root.destroy()
            return
        status_var.set("正在安全关闭本地音频与曲谱服务…")
        dot_label.configure(fg="#78716C")
        root.update_idletasks()

        def _stop_bg() -> None:
            try:
                runner.stop(timeout=5.0)
            finally:
                stop_done.set()

        threading.Thread(target=_stop_bg, name="melodydesk-stop", daemon=True).start()
        root.after(60, _wait_stop_done)

    btn_open = tk.Button(
        btn_row,
        text="打开弦语工作台",
        command=_open_main,
        state="disabled",
        font=("Microsoft YaHei UI", 9, "bold"),
        bg="#B5442A",
        fg="#FFFFFF",
        activebackground="#99361F",
        activeforeground="#FFFFFF",
        disabledforeground="#E7E5E4",
        relief="flat",
        padx=16,
        pady=7,
        cursor="hand2",
    )
    btn_open.pack(side="left")

    btn_atlas = tk.Button(
        btn_row,
        text="进化图鉴",
        command=_open_atlas,
        state="disabled",
        font=("Microsoft YaHei UI", 9),
        bg="#EFEAE0",
        fg="#292524",
        activebackground="#E4DDD0",
        activeforeground="#1C1917",
        disabledforeground="#A8A29E",
        relief="flat",
        padx=14,
        pady=7,
        cursor="hand2",
    )
    btn_atlas.pack(side="left", padx=(10, 0))

    btn_quit = tk.Button(
        btn_row,
        text="关闭窗口" if bool(getattr(runner, "is_attached", False)) else "停止并退出",
        command=_shutdown,
        font=("Microsoft YaHei UI", 9),
        bg="#FBF9F5",
        fg="#78716C",
        activebackground="#EFEAE0",
        activeforeground="#1C1917",
        highlightbackground="#D6CFC2",
        highlightthickness=1,
        bd=1,
        relief="solid",
        padx=14,
        pady=6,
        cursor="hand2",
    )
    btn_quit.pack(side="right")

    root.protocol("WM_DELETE_WINDOW", _shutdown)

    # Bring Studio Console visibly to the foreground on Windows 11 launch without pinning topmost permanently.
    try:
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.after(280, lambda: root.attributes("-topmost", False))
    except Exception:
        pass

    browser_opened = False
    ready_announced = False

    def _poll_server() -> None:
        nonlocal browser_opened, ready_announced
        if closing:
            return
        if runner.error is not None:
            dot_label.configure(fg="#DC2626")
            status_label.configure(fg="#991B1B")
            status_var.set(f"启动失败：{runner.error}（日志：{log_file.parent / 'melodydesk.log'}）")
            btn_open.configure(state="disabled")
            btn_atlas.configure(state="disabled")
            return
        if runner.ready_event.is_set():
            if not ready_announced:
                ready_announced = True
                attached_now = bool(getattr(runner, "is_attached", False))
                dot_label.configure(fg="#15803D")
                status_label.configure(fg="#292524")
                if attached_now:
                    status_var.set(f"已连接现有服务 · {runner.base_url}（点击下方按钮打开工作台）")
                    btn_quit.configure(text="关闭窗口")
                else:
                    status_var.set(f"运行中 · {runner.base_url}（关闭本窗口即可停止服务）")
                btn_open.configure(state="normal")
                btn_atlas.configure(state="normal")
                if open_browser and not browser_opened:
                    browser_opened = True
                    webbrowser.open(runner.base_url)
            root.after(1000, _poll_server)
            return
        root.after(150, _poll_server)

    try:
        runner.start()
        root.after(60, _poll_server)
        root.mainloop()
    finally:
        if not bool(getattr(runner, "is_attached", False)):
            runner.stop(timeout=4.0)
    return 0 if runner.error is None else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MelodyDesk 2.0 Standalone Windows Workbench")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: auto from 7865)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically on readiness")
    parser.add_argument("--headless", action="store_true", help="Run server without desktop controller window")
    parser.add_argument("--parent-pid", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--self-check-frozen", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="store_true", help="Print MelodyDesk version and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    eff_argv = list(sys.argv[1:] if argv is None else argv)
    is_gui_controller = (
        "--headless" not in eff_argv
        and "--self-check-frozen" not in eff_argv
        and "--version" not in eff_argv
    )
    # Configure logging & stdio streams BEFORE argparse so `--version`, `--help`, or invalid
    # arguments never encounter `sys.stdout is None` / `sys.stderr is None` in windowed builds.
    log_file = configure_runtime_environment(
        "melodydesk-ui.log" if is_gui_controller else "melodydesk.log"
    )
    args = build_arg_parser().parse_args(eff_argv)
    if args.version:
        sys.stdout.write(f"MelodyDesk {APP_VERSION}\n")
        sys.stdout.flush()
        return 0

    try:
        verify_portable_bundle_integrity()
    except RuntimeError as bundle_exc:
        logging.getLogger("melodydesk.launcher").error("%s", bundle_exc)
        if not args.headless and not args.self_check_frozen:
            _show_error_dialog(str(bundle_exc), "弦语 MelodyDesk 便携版启动提示", 0x00000030)
        return 1

    if args.self_check_frozen:
        return _run_frozen_self_check()

    bind_host = _normalize_bind_host(args.host)
    explicit_port = args.port is not None or args.parent_pid is not None
    preferred_port = int(args.port) if args.port is not None else 7865

    if args.parent_pid is None and probe_melodydesk_server(bind_host, preferred_port):
        existing_url = f"http://{_format_url_host(bind_host)}:{preferred_port}"
        logging.getLogger("melodydesk.launcher").info("Reusing running MelodyDesk instance at %s", existing_url)
        if args.headless:
            if not args.no_browser:
                webbrowser.open(existing_url)
            return 0
        attached = AttachedWorkbenchRunner(host=bind_host, port=preferred_port)
        return _run_studio_window(attached, open_browser=not args.no_browser, log_file=log_file)

    try:
        listen_port = select_listen_port(bind_host, preferred_port, explicit_port=explicit_port)
    except OSError as exc:
        logging.getLogger("melodydesk.launcher").error("%s", exc)
        if not args.headless:
            _show_error_dialog(str(exc), "弦语 MelodyDesk 启动提示", 0x00000030)
        return 1

    if args.headless:
        embedded = EmbeddedWorkbenchRunner(host=bind_host, port=listen_port)
        return _run_headless(embedded, open_browser=not args.no_browser, parent_pid=args.parent_pid)

    worker_runner = SubprocessWorkbenchRunner(host=bind_host, port=listen_port, log_dir=log_file.parent)
    return _run_studio_window(worker_runner, open_browser=not args.no_browser, log_file=log_file)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        exit_code = main()
    except SystemExit as sys_exc:
        exit_code = sys_exc.code if isinstance(sys_exc.code, int) else (0 if sys_exc.code is None else 1)
    except KeyboardInterrupt:
        exit_code = 130
    except BaseException as fatal_exc:  # noqa: BLE001
        logging.getLogger("melodydesk.fatal").exception("Unhandled fatal startup error")
        if "--headless" not in sys.argv and "--self-check-frozen" not in sys.argv and "--version" not in sys.argv:
            tb_tail = "".join(traceback.format_exception_only(type(fatal_exc), fatal_exc)).strip()
            log_hint = get_writable_work_root() / "logs" / "melodydesk-ui.log"
            _show_error_dialog(
                f"弦语 MelodyDesk 启动时遇到未预期的错误：\n{tb_tail}\n\n详细日志路径：\n{log_hint}",
                "弦语 MelodyDesk 启动错误",
                0x00000010,
            )
        exit_code = 1
    sys.exit(exit_code)
