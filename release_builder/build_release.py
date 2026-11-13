"""End-to-end Windows `.exe` builder, asset bundler, isolated smoke verifier, and `.zip` packager for MelodyDesk 2.0."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
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
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
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
- 离线节拍、和声与人声转简谱模型权重（Beat This! + BTC 170-Chord + PANNs + RMVPE/ROSVOT）
- 首次上传自定义歌曲分离人声或对齐歌词时，若本地尚未缓存 RoFormer / Whisper / MMS_FA 权重，
  程序会自动下载并缓存在可写 `work\\model-cache` 目录（即使解压在只读目录亦自动回退至 AppData）。

【命令行参数（可选）】
  MelodyDesk.exe --port 7865          指定监听端口
  MelodyDesk.exe --no-browser         启动但不自动弹浏览器
  MelodyDesk.exe --headless           无桌面控制卡片纯后台服务模式
  MelodyDesk.exe --version            输出版本号并退出
"""
    (bundle_root / "README-Windows便携版说明.txt").write_text(text, encoding="utf-8-sig")


def smoke_verify_frozen_exe(bundle_root: Path, port: int = 17891) -> dict:
    """Launch `MelodyDesk.exe` in a sanitized environment and verify CLI, MIR self-check, HTTP, SVG, Reharm, and WAV."""
    exe_path = bundle_root / "MelodyDesk.exe"
    if not exe_path.is_file():
        raise FileNotFoundError(f"Compiled executable not found: {exe_path}")

    env = os.environ.copy()
    for var in ("PYTHONPATH", "PYTHONHOME", "MELODYDESK_HOME", "MELODYDESK_WORK_DIR", "MELODYDESK_MODEL_CACHE", "MELODYDESK_GUITAR_SAMPLES"):
        env.pop(var, None)
    # Strip host FFmpeg directories from PATH so `--self-check-frozen` proves `<bundle>/bin/ffmpeg.exe` works!
    win_sys = os.environ.get("SystemRoot", r"C:\Windows")
    env["PATH"] = os.pathsep.join([str(Path(win_sys) / "System32"), str(Path(win_sys))])

    # 1. Verify `--version` under `console=False` (`sys.stdout` / `sys.stderr` safety)
    ver_res = subprocess.run([str(exe_path), "--version"], cwd=tempfile.gettempdir(), env=env, timeout=20)
    if ver_res.returncode != 0:
        raise RuntimeError(f"MelodyDesk.exe --version failed with rc={ver_res.returncode}")

    # 2. Verify `--self-check-frozen` (ROSVOT/RMVPE, pretty_midi, unidecode, audio_separator, bundled ffmpeg, BTC)
    chk_res = subprocess.run([str(exe_path), "--self-check-frozen"], cwd=tempfile.gettempdir(), env=env, timeout=90)
    if chk_res.returncode != 0:
        log_tail = ""
        log_path = bundle_root / "work" / "logs" / "melodydesk.log"
        if log_path.is_file():
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"MelodyDesk.exe --self-check-frozen failed (rc={chk_res.returncode}):\n{log_tail}")

    # 3. Verify live `--headless` server + `probe_melodydesk_server()` + HTTP/SVG/Audio endpoints
    proc = subprocess.Popen(
        [str(exe_path), "--headless", "--no-browser", "--host", "127.0.0.1", "--port", str(port)],
        cwd=tempfile.gettempdir(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 75.0
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log_path = bundle_root / "work" / "logs" / "melodydesk.log"
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:] if log_path.is_file() else ""
                raise RuntimeError(f"MelodyDesk.exe exited prematurely (rc={proc.returncode}):\n{log_tail}")
            if probe_melodydesk_server("127.0.0.1", port, timeout=1.5):
                ready = True
                break
            time.sleep(0.4)

        if not ready:
            raise TimeoutError("Timed out waiting for frozen MelodyDesk.exe probe_melodydesk_server() readiness.")

        with urllib.request.urlopen(f"{base_url}/api/state", timeout=5.0) as r_state:
            state_data = json.loads(r_state.read().decode("utf-8"))
            pub_html = state_data.get("publication_html", "")
            if not isinstance(pub_html, str) or "<svg" not in pub_html:
                raise RuntimeError("Rendered publication_html did not contain expected <svg> score markup.")

        with urllib.request.urlopen(f"{base_url}/", timeout=5.0) as r_index:
            index_html = r_index.read().decode("utf-8")
            if r_index.status != 200 or 'class="dock"' not in index_html:
                raise RuntimeError("Index HTML missing studio dock")

        with urllib.request.urlopen(f"{base_url}/evolution_atlas", timeout=5.0) as r_atlas:
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
        with urllib.request.urlopen(req, timeout=15.0) as r_reharm:
            reharm_data = json.loads(r_reharm.read().decode("utf-8"))
            if r_reharm.status != 200 or reharm_data.get("preset") != "emotional_pop":
                raise RuntimeError("POST /api/reharmonize did not apply emotional_pop preset")

        with urllib.request.urlopen(f"{base_url}/api/audio/guitar-only.wav", timeout=45.0) as r_gtr:
            gtr_bytes = r_gtr.read()
            if r_gtr.status != 200 or gtr_bytes[:4] != b"RIFF" or len(gtr_bytes) <= 500_000:
                raise RuntimeError("GET /api/audio/guitar-only.wav failed validation")

        with urllib.request.urlopen(f"{base_url}/api/audio/vocal-guitar-mix.wav", timeout=45.0) as r_mix:
            mix_bytes = r_mix.read()
            if r_mix.status != 200 or mix_bytes[:4] != b"RIFF" or len(mix_bytes) <= 500_000:
                raise RuntimeError("GET /api/audio/vocal-guitar-mix.wav failed validation")

        return {
            "bar_count": bar_count,
            "svg_verified": True,
            "guitar_wav_bytes": len(gtr_bytes),
            "vocal_mix_wav_bytes": len(mix_bytes),
            "self_check_frozen": True,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        for temp_dir in (bundle_root / "work" / "logs", bundle_root / "work" / "numba-cache", bundle_root / "work" / "model-cache" / "hf"):
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


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
    parser.add_argument("--version", default="2.0.0", help="Semantic version string (default: 2.0.0)")
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
