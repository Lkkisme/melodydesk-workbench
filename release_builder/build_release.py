"""End-to-end Windows `.exe` builder, asset bundler, isolated smoke verifier, and `.zip` packager for MelodyDesk 2.0."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from melodydesk_desktop import probe_melodydesk_server  # noqa: E402
from release_builder.generate_icon import build_rosette_icon  # noqa: E402
from runtime_paths import get_work_root  # noqa: E402

_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

GITHUB_MAX_ASSET_BYTES = 2 * 1024 * 1024 * 1024 - (32 * 1024 * 1024)  # 1.96875 GiB ceiling (< 2.0 GiB limit)


def _copy_guitar_samples(src_samples: Path, dst_samples: Path) -> tuple[int, int]:
    """Copy acoustic sample files while skipping `.git` repositories and git pack artifacts."""
    if not src_samples.is_dir():
        raise FileNotFoundError(f"Guitar sample directory not found: {src_samples}")
    skip_dirs = {".git", "__pycache__", ".pytest_cache"}
    skip_exts = {".pack", ".idx", ".rev", ".promisor", ".sample", ".worktree"}
    count, total_bytes = 0, 0
    for root, dirs, files in os.walk(src_samples):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        rel_dir = Path(root).relative_to(src_samples)
        target_dir = dst_samples / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src_f = Path(root) / fname
            if src_f.suffix.lower() in skip_exts:
                continue
            dst_f = target_dir / fname
            shutil.copy2(src_f, dst_f)
            count += 1
            total_bytes += dst_f.stat().st_size
    if count == 0:
        raise RuntimeError(f"No acoustic guitar sample files were copied from {src_samples}")
    return count, total_bytes


def _copy_repertoire_stems(src_work: Path, dst_work: Path) -> tuple[int, int]:
    """Copy repertoire manifest and every lead-vocal WAV stem referenced by the built-in repertoire."""
    manifest_src = REPO_ROOT / "data" / "repertoire_manifest.json"
    manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
    rep_dst = dst_work / "repertoire_audio"
    rep_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_src, rep_dst / "repertoire_manifest.json")

    stems_copied, total_bytes = 0, (rep_dst / "repertoire_manifest.json").stat().st_size
    work_parent = src_work.parent
    for key, entry in manifest.items():
        rel_voc = entry.get("vocals_wav")
        if not rel_voc:
            raise ValueError(f"Repertoire manifest entry '{key}' is missing required 'vocals_wav' field.")
        rel_path = Path(str(rel_voc).replace("\\", "/").lstrip("/"))
        src_file = work_parent / rel_path
        if not src_file.is_file():
            raise FileNotFoundError(f"Missing repertoire vocal stem for '{key}': {src_file}")
        dst_file = dst_work.parent / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        stems_copied += 1
        total_bytes += dst_file.stat().st_size
    if stems_copied != len(manifest):
        raise RuntimeError(f"Expected {len(manifest)} vocal stems, copied {stems_copied}")
    return stems_copied + 1, total_bytes


def _copy_core_models(src_cache: Path, dst_cache: Path, include_rosvot: bool = True) -> int:
    """Copy mandatory offline MIR/chord/beat/singing checkpoints into the portable bundle."""
    total_bytes = 0
    items = [
        ("btc-chord", True),
        ("panns-cnn6.pth", False),
        ("hub/checkpoints/beat_this-final0.ckpt", False),
        ("mdx", True),
        ("whisper-small-int8", True),
    ]
    if include_rosvot:
        items.append(("rosvot/checkpoints", True))

    for rel, is_dir in items:
        src = src_cache / rel
        dst = dst_cache / rel
        if not src.exists():
            raise FileNotFoundError(f"Required core model asset missing during packaging: {src}")
        if is_dir:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".cache", "btc_model.pt"),
            )
            total_bytes += sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            total_bytes += dst.stat().st_size
    return total_bytes


def _copy_ffmpeg_binaries(bundle_dir: Path) -> int:
    """Bundle `ffmpeg.exe` and `ffprobe.exe` into `<bundle>/bin/` so clean Windows PCs need zero setup."""
    bin_dir = bundle_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        if not found:
            raise FileNotFoundError(f"Required audio tool '{tool}' not found on host PATH during packaging.")
        src = Path(found).resolve()
        dst = bin_dir / src.name
        shutil.copy2(src, dst)
        total_bytes += dst.stat().st_size
    return total_bytes


def _write_portable_readme(bundle_root: Path, version: str, sample_count: int, repertoire_count: int) -> None:
    text = f"""========================================================================
  弦语 MelodyDesk v{version} · Windows 免安装便携版 (64-bit)
========================================================================

【快速启动】
1. 请先将整个压缩包完整解压到任意目录（例如 D:\\MelodyDesk-v{version}），
   请勿直接在压缩包预览窗口内双击运行。
2. 双击目录根目录下的 `MelodyDesk.exe`：
   - 会弹出「弦语 Studio Console」控制卡片并自动预热 6 弦原木吉他采样库；
   - 约 2–4 秒就绪后，系统默认浏览器将自动打开工作台（默认 http://127.0.0.1:7865）；
   - 使用完毕后，点击控制卡片上的「停止并退出」（或直接关闭控制卡片窗口）即可彻底释放后台服务。

【内置完整免装资产】
- 内置 Python 3.12 运行时、FFmpeg / FFprobe 音频解码器（无需手动配置系统环境变量）
- {sample_count} 个真实录音棚木吉他多力度/多轮循采样（Meta Steel + FSS + Shinyguitar + Emilyguitar）
- {repertoire_count} 首全曲真唱与 2026 SOTA 编配曲谱（《成都》《普通朋友》《Country Roads》《Tears in Heaven》
  《Scarborough Fair》《Lady Greensleeves》《Let's Go Fishin'》）
- 全离线全链路 AI 扒谱模型权重（零首次下载等待、断网可用）：
  * 人声伴奏极速分离：MDX-Net ONNX（UVR_MDXNET_KARA_2 + Kim_Vocal_2，支持 Windows DirectML 显卡加速与 AVX2 多核推理）
  * 词级时间戳对齐：Faster-Whisper Small (int8 离线量化版)
  * 神经节拍/和声/转谱：Beat This! + BTC 170-Chord + PANNs + RMVPE/ROSVOT

【常见问题】
1. 若双击后 Windows 弹出蓝色「Windows 已保护你的电脑」提示框，这是未购买商业 EV 证书的开源软件正常提示：
   请点击弹窗中的「更多信息」->「仍要运行」即可正常打开。
2. 请务必将整个压缩包完整解压后再运行 `MelodyDesk.exe`，请勿在压缩软件预览窗口内直接双击或单独拖出 exe 文件。

【命令行参数（可选）】
  MelodyDesk.exe --port 7865          指定监听端口
  MelodyDesk.exe --no-browser         启动但不自动弹浏览器
  MelodyDesk.exe --headless           无桌面控制卡片纯后台服务模式
  MelodyDesk.exe --version            输出版本号并退出
"""
    (bundle_root / "README-Windows便携版说明.txt").write_text(text, encoding="utf-8-sig")


def _find_visible_tk_window(pid: int) -> int | None:
    """Return the visible `TkTopLevel` HWND belonging to `pid` on Windows, if present."""
    if os.name != "nt":
        return None
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_proc_t = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    user32.EnumWindows.argtypes = [enum_proc_t, ctypes.c_void_p]
    user32.EnumWindows.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    user32.GetWindowThreadProcessId.restype = ctypes.c_uint32
    user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    user32.IsWindowVisible.restype = ctypes.c_int
    user32.GetClassNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int

    matched: list[int] = []

    def _cb(hwnd: int | None, _lparam: int | None) -> int:
        if not hwnd or not user32.IsWindowVisible(hwnd):
            return 1
        win_pid = ctypes.c_uint32(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(win_pid))
        if int(win_pid.value) != int(pid):
            return 1
        cls_buf = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls_buf, 64)
        if cls_buf.value == "TkTopLevel":
            matched.append(int(hwnd))
            return 0
        return 1

    cb_ref = enum_proc_t(_cb)
    user32.EnumWindows(cb_ref, 0)
    return matched[0] if matched else None


def _post_wm_close(hwnd: int) -> None:
    if os.name != "nt":
        return
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_ssize_t]
    user32.PostMessageW.restype = ctypes.c_int
    user32.PostMessageW(int(hwnd), 0x0010, 0, 0)


def _kill_port_listeners_windows(port: int) -> None:
    if os.name != "nt":
        return
    net_res = subprocess.run(
        ["netstat", "-ano", "-p", "TCP"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    target_suffix = f":{int(port)}"
    pids: set[str] = set()
    for line in net_res.stdout.splitlines():
        cols = line.split()
        if len(cols) >= 5 and cols[1].endswith(target_suffix) and cols[3].upper() == "LISTENING" and cols[4].isdigit():
            if int(cols[4]) > 0:
                pids.add(cols[4])
    for pid_str in pids:
        subprocess.run(["taskkill", "/F", "/T", "/PID", pid_str], capture_output=True, check=False)


def _clean_smoke_artifacts(bundle_root: Path) -> None:
    for temp_dir in (
        bundle_root / "work" / "logs",
        bundle_root / "work" / "uploads",
        bundle_root / "work" / "numba-cache",
        bundle_root / "work" / "model-cache" / "hf",
    ):
        for attempt in range(6):
            if not temp_dir.exists():
                break
            if attempt < 5:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if temp_dir.exists():
                    time.sleep(0.35)
            else:
                shutil.rmtree(temp_dir)
        if temp_dir.exists():
            raise RuntimeError(f"Smoke artifact directory could not be cleaned prior to zip packaging: {temp_dir}")


def _read_smoke_log_tail(bundle_root: Path) -> str:
    candidates = [
        bundle_root / "work" / "logs" / "melodydesk.log",
        bundle_root / "work" / "logs" / "melodydesk-ui.log",
    ]
    local_app = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app:
        candidates.append(Path(local_app) / "MelodyDesk" / "work" / "logs" / "melodydesk.log")
        candidates.append(Path(local_app) / "MelodyDesk" / "work" / "logs" / "melodydesk-ui.log")
    parts: list[str] = []
    for log_path in candidates:
        if log_path.is_file():
            parts.append(f"--- {log_path.name} ---\n" + log_path.read_text(encoding="utf-8", errors="replace")[-1500:])
    return "\n".join(parts)


def smoke_verify_frozen_exe(bundle_root: Path, port: int | None = None) -> dict:
    """Launch `MelodyDesk.exe` in a sanitized environment and verify CLI, MIR self-check, HTTP, SVG, Reharm, and WAV."""
    exe_path = bundle_root / "MelodyDesk.exe"
    if not exe_path.is_file():
        raise FileNotFoundError(f"Compiled executable not found: {exe_path}")

    if port is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])

    env = os.environ.copy()
    for var in (
        "PYTHONPATH",
        "PYTHONHOME",
        "MELODYDESK_HOME",
        "MELODYDESK_WORK_DIR",
        "MELODYDESK_MODEL_CACHE",
        "MELODYDESK_GUITAR_SAMPLES",
        "CUDA_PATH",
    ):
        env.pop(var, None)
    for key in [k for k in env if k.upper().startswith("CUDA_PATH_")]:
        env.pop(key, None)
    # Strip host FFmpeg & CUDA directories from PATH so `--self-check-frozen` proves `<bundle>/bin/ffmpeg.exe` works!
    win_sys = os.environ.get("SystemRoot", r"C:\Windows")
    env["PATH"] = os.pathsep.join([str(Path(win_sys) / "System32"), str(Path(win_sys))])
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["MELODYDESK_OFFLINE"] = "1"

    gui_proc: subprocess.Popen | None = None
    gui_port: int | None = None
    proc: subprocess.Popen | None = None
    try:
        # 1. Verify `--version` under `console=False` (`sys.stdout` / `sys.stderr` safety)
        ver_res = subprocess.run([str(exe_path), "--version"], cwd=tempfile.gettempdir(), env=env, timeout=20)
        if ver_res.returncode != 0:
            raise RuntimeError(f"MelodyDesk.exe --version failed with rc={ver_res.returncode}")

        # 2. Verify `--self-check-frozen` (ROSVOT/RMVPE, pretty_midi, unidecode, audio_separator, bundled ffmpeg, BTC)
        chk_res = subprocess.run([str(exe_path), "--self-check-frozen"], cwd=tempfile.gettempdir(), env=env, timeout=90)
        if chk_res.returncode != 0:
            log_tail = _read_smoke_log_tail(bundle_root)
            raise RuntimeError(f"MelodyDesk.exe --self-check-frozen failed (rc={chk_res.returncode}):\n{log_tail}")

        # 2.5 Verify REAL default desktop GUI mode (double-click launch WITHOUT `--headless`!)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as gui_sock:
            gui_sock.bind(("127.0.0.1", 0))
            gui_port = int(gui_sock.getsockname()[1])

        t_gui0 = time.monotonic()
        gui_proc = subprocess.Popen(
            [str(exe_path), "--no-browser", "--host", "127.0.0.1", "--port", str(gui_port)],
            cwd=tempfile.gettempdir(),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        gui_base_url = f"http://127.0.0.1:{gui_port}"
        gui_hwnd: int | None = None
        try:
            gui_deadline = time.monotonic() + 45.0
            gui_ready = False
            while time.monotonic() < gui_deadline:
                if gui_proc.poll() is not None:
                    log_tail = _read_smoke_log_tail(bundle_root)
                    raise RuntimeError(f"GUI MelodyDesk.exe exited prematurely (rc={gui_proc.returncode}):\n{log_tail}")
                if gui_hwnd is None:
                    gui_hwnd = _find_visible_tk_window(gui_proc.pid)
                if probe_melodydesk_server("127.0.0.1", gui_port, timeout=1.5) and (gui_hwnd is not None or os.name != "nt"):
                    gui_ready = True
                    break
                time.sleep(0.25)

            if not gui_ready:
                log_tail = _read_smoke_log_tail(bundle_root)
                raise TimeoutError(
                    f"Default GUI mode failed readiness (hwnd={gui_hwnd}) within 45s:\n{log_tail}"
                )
            gui_startup_sec = round(time.monotonic() - t_gui0, 2)

            with _NO_PROXY_OPENER.open(f"{gui_base_url}/api/health", timeout=3.0) as r_gh:
                gh_data = json.loads(r_gh.read().decode("utf-8"))
                if r_gh.status != 200 or gh_data.get("ok") is not True or gh_data.get("app") != "melodydesk":
                    raise RuntimeError(f"GUI /api/health returned unexpected payload: {gh_data}")

            with _NO_PROXY_OPENER.open(f"{gui_base_url}/api/state", timeout=20.0) as r_gs:
                gs_data = json.loads(r_gs.read().decode("utf-8"))
                if "<svg" not in gs_data.get("publication_html", ""):
                    raise RuntimeError("GUI mode /api/state did not return valid SVG score")

            if gui_hwnd is not None:
                _post_wm_close(gui_hwnd)
                gui_rc = gui_proc.wait(timeout=12)
                if gui_rc != 0:
                    raise RuntimeError(f"GUI MelodyDesk.exe exited with non-zero code after WM_CLOSE: {gui_rc}")
                port_freed = False
                free_deadline = time.monotonic() + 6.0
                while time.monotonic() < free_deadline:
                    if not probe_melodydesk_server("127.0.0.1", gui_port, timeout=0.6):
                        port_freed = True
                        break
                    time.sleep(0.2)
                if not port_freed:
                    raise RuntimeError("Child headless server remained alive after Studio Console WM_CLOSE!")
        finally:
            if gui_proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(gui_proc.pid)], capture_output=True, check=False)
                else:
                    gui_proc.kill()
                gui_proc.wait(timeout=5)
            if gui_port is not None and probe_melodydesk_server("127.0.0.1", gui_port, timeout=0.4):
                _kill_port_listeners_windows(gui_port)

        # 3. Verify live `--headless` server + `probe_melodydesk_server()` + HTTP/SVG/Audio endpoints
        proc = subprocess.Popen(
            [str(exe_path), "--headless", "--no-browser", "--host", "127.0.0.1", "--port", str(port)],
            cwd=tempfile.gettempdir(),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 75.0
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log_tail = _read_smoke_log_tail(bundle_root)
                raise RuntimeError(f"MelodyDesk.exe exited prematurely (rc={proc.returncode}):\n{log_tail}")
            if probe_melodydesk_server("127.0.0.1", port, timeout=1.5):
                ready = True
                break
            time.sleep(0.4)

        if not ready:
            raise TimeoutError("Timed out waiting for frozen MelodyDesk.exe probe_melodydesk_server() readiness.")

        with _NO_PROXY_OPENER.open(f"{base_url}/api/state", timeout=20.0) as r_state:
            state_data = json.loads(r_state.read().decode("utf-8"))
            pub_html = state_data.get("publication_html", "")
            if not isinstance(pub_html, str) or "<svg" not in pub_html:
                raise RuntimeError("Rendered publication_html did not contain expected <svg> score markup.")

        with _NO_PROXY_OPENER.open(f"{base_url}/", timeout=15.0) as r_index:
            index_html = r_index.read().decode("utf-8")
            if r_index.status != 200 or 'class="dock"' not in index_html:
                raise RuntimeError("Index HTML missing studio dock")

        with _NO_PROXY_OPENER.open(f"{base_url}/evolution_atlas", timeout=15.0) as r_atlas:
            atlas_html = r_atlas.read().decode("utf-8")
            if r_atlas.status != 200 or "MelodyDesk 2.0" not in atlas_html:
                raise RuntimeError("/evolution_atlas returned unexpected HTML payload")

        bar_count = int(state_data["arrangement"]["bar_count"])
        if bar_count < 100:
            raise RuntimeError(f"Expected full studio repertoire (>=100 bars), got {bar_count} bars")

        req = urllib.request.Request(
            f"{base_url}/api/reharmonize",
            data=json.dumps({"preset": "emotional_pop"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _NO_PROXY_OPENER.open(req, timeout=15.0) as r_reharm:
            reharm_data = json.loads(r_reharm.read().decode("utf-8"))
            if r_reharm.status != 200 or reharm_data.get("preset") != "emotional_pop":
                raise RuntimeError("POST /api/reharmonize did not apply emotional_pop preset")

        with _NO_PROXY_OPENER.open(f"{base_url}/api/audio/guitar-only.wav", timeout=45.0) as r_gtr:
            gtr_bytes = r_gtr.read()
            if r_gtr.status != 200 or gtr_bytes[:4] != b"RIFF" or len(gtr_bytes) <= 500_000:
                raise RuntimeError("GET /api/audio/guitar-only.wav failed validation")

        with _NO_PROXY_OPENER.open(f"{base_url}/api/audio/vocal-guitar-mix.wav", timeout=45.0) as r_mix:
            mix_bytes = r_mix.read()
            if r_mix.status != 200 or mix_bytes[:4] != b"RIFF" or len(mix_bytes) <= 500_000:
                raise RuntimeError("GET /api/audio/vocal-guitar-mix.wav failed validation")

        # 4. End-to-end offline custom .mp3 upload verification (POST /api/analyze_audio -> /api/job/<id>)
        import math
        import wave

        bundled_ffmpeg = bundle_root / "bin" / "ffmpeg.exe"
        with tempfile.TemporaryDirectory() as up_tmp:
            sr_u = 22050
            dur_u = 6.0
            n_u = int(sr_u * dur_u)
            pcm_buf = bytearray()
            for idx_s in range(n_u):
                t_s = idx_s / sr_u
                f0 = 261.63 if t_s < 3.0 else 329.63
                val = 0.45 * math.sin(2.0 * math.pi * f0 * t_s) + 0.2 * math.sin(4.0 * math.pi * f0 * t_s)
                samp = max(-32767, min(32767, int(val * 32767.0)))
                pcm_buf.extend(samp.to_bytes(2, byteorder="little", signed=True))
                pcm_buf.extend(samp.to_bytes(2, byteorder="little", signed=True))
            raw_wav = Path(up_tmp) / "smoke_song.wav"
            with wave.open(str(raw_wav), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(sr_u)
                wf.writeframes(bytes(pcm_buf))
            smoke_mp3 = Path(up_tmp) / "自定义上传测试曲.mp3"
            subprocess.run(
                [str(bundled_ffmpeg), "-nostdin", "-y", "-i", str(raw_wav), "-b:a", "128k", str(smoke_mp3)],
                check=True,
                capture_output=True,
                timeout=30,
            )
            mp3_bytes = smoke_mp3.read_bytes()

        boundary = "----MelodyDeskSmokeBoundary7MA4YWxkTrZu0gW"
        body_parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            'Content-Disposition: form-data; name="file"; filename="自定义上传测试曲.mp3"\r\nContent-Type: audio/mpeg\r\n\r\n'.encode("utf-8"),
            mp3_bytes,
            f"\r\n--{boundary}\r\n".encode("utf-8"),
            'Content-Disposition: form-data; name="lyrics"\r\n\r\n今夜微风轻吹\r\n'.encode("utf-8"),
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        t_up0 = time.monotonic()
        up_req = urllib.request.Request(
            f"{base_url}/api/analyze_audio",
            data=b"".join(body_parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with _NO_PROXY_OPENER.open(up_req, timeout=15.0) as r_up:
            job_id = json.loads(r_up.read().decode("utf-8"))["job_id"]

        up_deadline = time.monotonic() + 90.0
        job_final = None
        while time.monotonic() < up_deadline:
            with _NO_PROXY_OPENER.open(f"{base_url}/api/job/{job_id}", timeout=15.0) as r_job:
                job_state = json.loads(r_job.read().decode("utf-8"))
            if job_state.get("status") in ("done", "error", "superseded"):
                job_final = job_state
                break
            time.sleep(0.5)
        if not job_final or job_final.get("status") != "done":
            raise RuntimeError(f"Offline MP3 upload smoke test failed: {job_final}")
        upload_elapsed_s = round(time.monotonic() - t_up0, 2)

        with _NO_PROXY_OPENER.open(f"{base_url}/api/state", timeout=15.0) as r_after:
            after_state = json.loads(r_after.read().decode("utf-8"))
            if after_state.get("title") != "自定义上传测试曲" or "<svg" not in after_state.get("publication_html", ""):
                raise RuntimeError("Uploaded MP3 state did not render expected title and SVG score")

        unbundled_mms = bundle_root / "work" / "model-cache" / "hub" / "checkpoints" / "model.pt"
        if unbundled_mms.exists():
            raise RuntimeError(f"Unexpected live download into bundle during offline smoke test: {unbundled_mms}")

        return {
            "bar_count": bar_count,
            "svg_verified": True,
            "guitar_wav_bytes": len(gtr_bytes),
            "vocal_mix_wav_bytes": len(mix_bytes),
            "self_check_frozen": True,
            "gui_startup_sec": gui_startup_sec,
            "offline_mp3_upload_sec": upload_elapsed_s,
        }
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
                else:
                    proc.kill()
                proc.wait(timeout=3)
        if port is not None and probe_melodydesk_server("127.0.0.1", port, timeout=0.4):
            _kill_port_listeners_windows(port)
        _clean_smoke_artifacts(bundle_root)


def create_zip_archive(source_dir: Path, zip_path: Path) -> tuple[int, str]:
    """Compress `source_dir` (preserving top-level folder name) into `zip_path` and compute SHA-256."""
    if zip_path.exists():
        zip_path.unlink()
    parent_dir = source_dir.parent
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(parent_dir).as_posix()
                zf.write(file_path, arcname)

    size_bytes = zip_path.stat().st_size
    h = hashlib.sha256()
    with zip_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return size_bytes, h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build standalone MelodyDesk Windows .exe release")
    parser.add_argument("--version", default="2.0.2", help="Semantic version string (default: 2.0.2)")
    parser.add_argument("--stage-dir", default=None, help="Output staging directory")
    parser.add_argument("--skip-rosvot", action="store_true", help="Exclude 603 MB ROSVOT weights from main zip")
    args = parser.parse_args()

    src_work = get_work_root()
    stage_root = Path(args.stage_dir).resolve() if args.stage_dir else (src_work / "release_stage")
    pyi_dist = stage_root / "pyinstaller_dist"
    pyi_build = stage_root / "pyinstaller_build"
    bundle_name = f"MelodyDesk-v{args.version}-win-x64"
    bundle_dir = stage_root / bundle_name

    icon_path = REPO_ROOT / "assets" / "melodydesk.ico"
    if not icon_path.is_file():
        build_rosette_icon(icon_path)
    print(f"[1/5] Icon ready: {icon_path}")

    for clean_target in (pyi_dist, pyi_build, bundle_dir):
        if clean_target.exists():
            shutil.rmtree(clean_target)
    stage_root.mkdir(parents=True, exist_ok=True)

    spec_path = REPO_ROOT / "release_builder" / "MelodyDesk.spec"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec_path),
        "--noconfirm",
        "--clean",
        "--distpath",
        str(pyi_dist),
        "--workpath",
        str(pyi_build),
    ]
    print(f"[2/5] Running PyInstaller: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    shutil.move(str(pyi_dist / "MelodyDesk"), str(bundle_dir))
    bundle_work = bundle_dir / "work"
    bundle_work.mkdir(parents=True, exist_ok=True)

    print("[3/5] Staging FFmpeg, studio guitar samples, repertoire vocal stems, and MIR weights...")
    ff_bytes = _copy_ffmpeg_binaries(bundle_dir)
    gs_cnt, gs_bytes = _copy_guitar_samples(src_work / "guitar_samples", bundle_work / "guitar_samples")
    rep_cnt, rep_bytes = _copy_repertoire_stems(src_work, bundle_work)
    mdl_bytes = _copy_core_models(src_work / "model-cache", bundle_work / "model-cache", include_rosvot=not args.skip_rosvot)
    _write_portable_readme(bundle_dir, args.version, sample_count=gs_cnt, repertoire_count=rep_cnt - 1)
    print(
        f"      FFmpeg: {ff_bytes / 1e6:.1f} MB, "
        f"Guitar samples: {gs_cnt} files ({gs_bytes / 1e6:.1f} MB), "
        f"Repertoire files: {rep_cnt} ({rep_bytes / 1e6:.1f} MB), "
        f"Core models: {mdl_bytes / 1e6:.1f} MB"
    )

    print("[4/5] Running isolated frozen MelodyDesk.exe end-to-end smoke verification...")
    receipt = smoke_verify_frozen_exe(bundle_dir)
    print(f"      Smoke test PASSED: {receipt}")

    zip_path = stage_root / f"{bundle_name}.zip"
    print(f"[5/5] Creating release archive: {zip_path} ...")
    zip_bytes, sha256_hex = create_zip_archive(bundle_dir, zip_path)
    if zip_bytes >= GITHUB_MAX_ASSET_BYTES:
        raise RuntimeError(
            f"Archive {zip_path.name} ({zip_bytes / (1024**3):.2f} GiB) exceeds GitHub's 2.0 GiB per-asset limit!"
        )

    sha_file = stage_root / "SHA256SUMS.txt"
    sha_file.write_text(f"{sha256_hex}  {zip_path.name}\n", encoding="utf-8")
    print(f"SUCCESS: {zip_path} ({zip_bytes / (1024**2):.2f} MiB) | SHA256={sha256_hex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
