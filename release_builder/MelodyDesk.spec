# -*- mode: python ; coding: utf-8 -*-
import importlib.util
from pathlib import Path
import sys
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

SPEC_DIR = Path(SPECPATH).resolve()
REPO_ROOT = SPEC_DIR.parent

datas = [
    (str(REPO_ROOT / "static"), "static"),
    (str(REPO_ROOT / "data"), "data"),
    (str(REPO_ROOT / "third_party"), "third_party"),
    (str(REPO_ROOT / "assets"), "assets"),
]
binaries = []
hiddenimports = [
    "runtime_paths",
    "music",
    "guitar",
    "score_cleanup",
    "score_layout",
    "score_pdf",
    "pitch_notes",
    "singing_model",
    "vocal_activity",
    "lyrics",
    "engine",
    "web_server",
    "pretty_midi",
    "unidecode",
    "ml_collections",
    "beartype",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "multipart",
    "python_multipart",
]

for pkg in (
    "subpkgs",
    "pypinyin",
    "soundfile",
    "librosa",
    "reportlab",
    "svglib",
    "audio_separator.separator",
    "pretty_midi",
    "unidecode",
):
    try:
        hiddenimports.extend(collect_submodules(pkg))
    except Exception:
        pass

for meta_pkg in ("audio-separator", "onnxruntime", "onnxruntime-directml", "faster-whisper", "transformers", "huggingface-hub", "pretty_midi"):
    try:
        datas.extend(copy_metadata(meta_pkg))
    except Exception:
        pass

for data_pkg in ("_soundfile_data", "librosa", "pypinyin", "certifi", "faster_whisper", "beat_this", "audio_separator", "pretty_midi"):
    try:
        datas.extend(collect_data_files(data_pkg))
    except Exception:
        pass

for dyn_pkg in ("ctranslate2", "onnxruntime", "av"):
    try:
        binaries.extend(collect_dynamic_libs(dyn_pkg))
    except Exception:
        pass

# Bundle Stable-ABI / torch.ops.load_library native extensions (.pyd/.dll) for torchvision & torchaudio
for ext_pkg in ("torchvision", "torchaudio"):
    spec_pkg = importlib.util.find_spec(ext_pkg)
    if spec_pkg and spec_pkg.submodule_search_locations:
        pkg_root = Path(list(spec_pkg.submodule_search_locations)[0]).resolve()
        for bin_file in pkg_root.rglob("*"):
            if bin_file.is_file() and bin_file.suffix.lower() in (".pyd", ".dll"):
                rel_parent = bin_file.parent.relative_to(pkg_root.parent).as_posix()
                binaries.append((str(bin_file), rel_parent))

# Bundle Tcl/Tk 8.6 runtime data when building from Conda/CPython on Windows
base_prefix = Path(sys.base_prefix)
for tcl_cand in (base_prefix / "Library" / "lib", base_prefix / "tcl"):
    if (tcl_cand / "tcl8.6").is_dir() and (tcl_cand / "tk8.6").is_dir():
        datas.append((str(tcl_cand / "tcl8.6"), "_tcl_data"))
        datas.append((str(tcl_cand / "tk8.6"), "_tk_data"))
        break
conda_lib_bin = base_prefix / "Library" / "bin"
if conda_lib_bin.is_dir():
    for dll_pattern in ("tcl86t.dll", "tk86t.dll", "ffi*.dll", "libexpat*.dll", "liblzma*.dll", "libbz2*.dll", "sqlite3.dll", "zlib.dll"):
        for dll_path in conda_lib_bin.glob(dll_pattern):
            binaries.append((str(dll_path), "."))

excludes = [
    "pytest",
    "_pytest",
    "IPython",
    "ipykernel",
    "jupyter",
    "notebook",
    "matplotlib.tests",
    "numpy.tests",
    "scipy.tests",
    "torch.test",
    "tkinter.test",
]

a = Analysis(
    [str(REPO_ROOT / "melodydesk_desktop.py")],
    pathex=[str(REPO_ROOT), str(REPO_ROOT / "third_party" / "rosvot")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MelodyDesk",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(REPO_ROOT / "assets" / "melodydesk.ico"),
    version=str(SPEC_DIR / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MelodyDesk",
)
