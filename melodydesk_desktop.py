"""Standalone Windows Desktop & Portable Executable Launcher for MelodyDesk 2.0.

Provides:
1. Frozen-safe `multiprocessing.freeze_support()` and UTF-8 rotating file logging when
   packaged with PyInstaller `--noconsole` (`sys.stdout` / `sys.stderr` are `None`);
2. Single-instance port reuse: if MelodyDesk is already serving on the target port,
   opens the existing workbench URL instead of failing with `EADDRINUSE`;
3. Quiet-Luxury native Windows Studio Controller card (`tkinter`) with live engine warm-up
   status, one-click Workbench / Evolution Atlas launchers, and thread-safe deterministic
   Uvicorn shutdown on window close (zero orphaned background processes), plus `--headless`
   and `--self-check-frozen` modes for CLI / automated smoke verification.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import threading
import time
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
    get_writable_work_root,
)

APP_VERSION = "2.0.0"


class _LogStreamAdapter:
    """Minimal UTF-8 text stream adapter for `--noconsole` builds where stdio is `None`."""

    encoding = "utf-8"

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._buf = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buf += str(text)
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.rstrip("\r")
            if stripped:
                self._logger.log(self._level, "%s", stripped)
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self._logger.log(self._level, "%s", self._buf.rstrip("\r\n"))
        self._buf = ""

    def isatty(self) -> bool:
        return False


def configure_runtime_environment() -> Path:
    """Initialize writable cache/log paths and ensure `sys.stdout`/`sys.stderr` exist."""
    writable_work = get_writable_work_root()
    log_dir = writable_work / "logs"
    numba_dir = writable_work / "numba-cache"
    writable_model_cache = get_model_cache_root(require_writable=True)
    hf_cache = writable_model_cache / "hf"

    for d in (log_dir, numba_dir, writable_model_cache):
        d.mkdir(parents=True, exist_ok=True)

    # Always point TORCH_HOME and HF_HOME to a probed writable cache directory so
    # on-demand downloads (e.g. MMS_FA) succeed even when installed under C:\Program Files.
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

    log_file = log_dir / "melodydesk.log"
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)
        if sys.stderr is not None:
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setFormatter(fmt)
            root_logger.addHandler(stream_handler)

    if sys.stdout is None:
        sys.stdout = _LogStreamAdapter(logging.getLogger("melodydesk.stdout"), logging.INFO)
    if sys.stderr is None:
        sys.stderr = _LogStreamAdapter(logging.getLogger("melodydesk.stderr"), logging.ERROR)

    return log_file


def _connect_host(bind_host: str) -> str:
    if bind_host in ("0.0.0.0", "::", ""):
        return "127.0.0.1"
    return bind_host


def probe_melodydesk_server(bind_host: str, port: int, timeout: float = 1.2) -> bool:
    """Return True if a live MelodyDesk 2.0 server is responding on `(bind_host, port)`."""
    url = f"http://{_connect_host(bind_host)}:{port}/api/state"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            return (
                isinstance(payload, dict)
                and ("publication_html" in payload or "score_html" in payload)
                and "arrangement" in payload
            )
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _is_tcp_port_free(bind_host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
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
    """Manages the in-process FastAPI + Uvicorn lifecycle on a clean background thread."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{_connect_host(host)}:{port}"
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
                deadline = time.monotonic() + 60.0
                while time.monotonic() < deadline and not self._stopped_event.is_set():
                    if getattr(srv, "started", False) and probe_melodydesk_server(self.host, self.port, timeout=1.0):
                        self.ready_event.set()
                        return
                    time.sleep(0.15)

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
    importlib.import_module("audio_separator.separator.architectures.mdxc_separator")

    from audio_separator.separator import Separator

    with tempfile.TemporaryDirectory() as tmp:
        sep = Separator(output_dir=tmp, output_format="WAV")
        if sep is None:
            raise RuntimeError("audio_separator failed to initialize")

    if singing_model.ready():
        singing_model.load_models("cpu")
    if vocal_activity.ready():
        vocal_activity.load_model()
    btc = chord_recognizer._get_btc_runtime()
    if btc is None:
        raise RuntimeError("BTC 170-chord model failed to load")

    logging.getLogger("melodydesk.selfcheck").info("SELF_CHECK_OK ffmpeg=%s", ffmpeg_bin)
    return 0


def _run_headless(runner: EmbeddedWorkbenchRunner, open_browser: bool) -> int:
    def _handle_sig(_signum, _frame) -> None:
        runner.stop(timeout=2.0)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _handle_sig)

    runner.start()
    runner.ready_event.wait(timeout=90.0)
    if runner.error is not None:
        logging.getLogger("melodydesk.launcher").error("Startup failed: %s", runner.error)
        return 1
    if not probe_melodydesk_server(runner.host, runner.port, timeout=2.0):
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


def _run_studio_window(runner: EmbeddedWorkbenchRunner, open_browser: bool, log_file: Path) -> int:
    _enable_windows_dpi_awareness()
    try:
        import tkinter as tk
    except Exception:
        return _run_headless(runner, open_browser=open_browser)

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
        root.after(80, _wait_stop_done)

    def _shutdown() -> None:
        nonlocal closing
        if closing:
            return
        closing = True
        status_var.set("正在安全关闭本地音频与曲谱服务…")
        dot_label.configure(fg="#78716C")
        root.update_idletasks()

        def _stop_bg() -> None:
            try:
                runner.stop(timeout=5.0)
            finally:
                stop_done.set()

        threading.Thread(target=_stop_bg, name="melodydesk-stop", daemon=True).start()
        root.after(80, _wait_stop_done)

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
        text="停止并退出",
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

    browser_opened = False

    def _poll_server() -> None:
        nonlocal browser_opened
        if closing:
            return
        if runner.error is not None:
            dot_label.configure(fg="#DC2626")
            status_label.configure(fg="#991B1B")
            status_var.set(f"启动失败：{runner.error}（详见 {log_file.name}）")
            return
        if runner.ready_event.is_set() and probe_melodydesk_server(runner.host, runner.port, timeout=0.8):
            dot_label.configure(fg="#15803D")
            status_var.set(f"运行中 · {runner.base_url}（关闭本窗口即可停止服务）")
            btn_open.configure(state="normal")
            btn_atlas.configure(state="normal")
            if open_browser and not browser_opened:
                browser_opened = True
                webbrowser.open(runner.base_url)
            return
        root.after(200, _poll_server)

    runner.start()
    root.after(200, _poll_server)
    root.mainloop()
    return 0 if runner.error is None else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MelodyDesk 2.0 Standalone Windows Workbench")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: auto from 7865)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically on readiness")
    parser.add_argument("--headless", action="store_true", help="Run server without desktop controller window")
    parser.add_argument("--self-check-frozen", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="store_true", help="Print MelodyDesk version and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Configure logging & stdio streams BEFORE argparse so `--version`, `--help`, or invalid
    # arguments never encounter `sys.stdout is None` / `sys.stderr is None` in windowed builds.
    log_file = configure_runtime_environment()
    args = build_arg_parser().parse_args(argv)
    if args.version:
        sys.stdout.write(f"MelodyDesk {APP_VERSION}\n")
        sys.stdout.flush()
        return 0
    if args.self_check_frozen:
        return _run_frozen_self_check()

    explicit_port = args.port is not None
    preferred_port = int(args.port) if explicit_port else 7865

    if probe_melodydesk_server(args.host, preferred_port):
        existing_url = f"http://{_connect_host(args.host)}:{preferred_port}"
        logging.getLogger("melodydesk.launcher").info("Reusing running MelodyDesk instance at %s", existing_url)
        if not args.no_browser:
            webbrowser.open(existing_url)
        return 0

    try:
        listen_port = select_listen_port(args.host, preferred_port, explicit_port=explicit_port)
    except OSError as exc:
        logging.getLogger("melodydesk.launcher").error("%s", exc)
        if not args.headless and os.name == "nt":
            ctypes.windll.user32.MessageBoxW(0, str(exc), "MelodyDesk 启动提示", 0x00000030)
        return 1

    runner = EmbeddedWorkbenchRunner(host=args.host, port=listen_port)
    if args.headless:
        return _run_headless(runner, open_browser=not args.no_browser)
    return _run_studio_window(runner, open_browser=not args.no_browser, log_file=log_file)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        exit_code = main()
    except SystemExit as sys_exc:
        exit_code = sys_exc.code if isinstance(sys_exc.code, int) else (0 if sys_exc.code is None else 1)
    except BaseException:  # noqa: BLE001
        logging.getLogger("melodydesk.fatal").exception("Unhandled fatal startup error")
        exit_code = 1
    sys.exit(exit_code)
