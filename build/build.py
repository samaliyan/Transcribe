"""Build the Windows package of Transcribe: dist/Transcribe/ and dist/Transcribe-<version>-windows.zip.

    python build/build.py [--version 1.2.0]

Run it on Windows in a virtual environment with app/transcribe/requirements.txt and
pyinstaller installed. The GitHub workflow .github/workflows/release.yml uses the same script.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "transcribe"
BUILD = ROOT / "build"
DIST = ROOT / "dist"
NAME = "Transcribe"


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_pyinstaller():
    import PyInstaller.__main__  # pylint: disable=import-outside-toplevel
    PyInstaller.__main__.run([
        str(BUILD / "launcher.py"),
        "--name", NAME,
        "--noconfirm", "--clean",
        "--distpath", str(DIST),
        "--workpath", str(BUILD / "pyinstaller-work"),
        "--specpath", str(BUILD / "pyinstaller-work"),
        "--paths", str(ROOT),
        # A window program: no console window next to it (launcher.py routes its output).
        "--windowed",
        # The whole application, also the parts that are imported only when chosen.
        "--collect-submodules", "app",
        "--collect-submodules", "sdk",
        "--collect-submodules", "tsutils",
        "--collect-submodules", "custom_speech_recognition",
        # Packages that need their data files or DLLs.
        "--collect-data", "customtkinter",      # themes
        "--collect-data", "wordcloud",          # font and stop words
        "--collect-data", "faster_whisper",     # voice activity model
        "--hidden-import", "onnxruntime",       # runs the voice activity model
        "--collect-binaries", "onnxruntime",
        "--collect-binaries", "ctranslate2",    # the speech to text engine
        "--collect-all", "soundfile",           # libsndfile
        "--collect-all", "_soundfile_data",
        # Optional engines that need PyTorch (several GB) stay out, even if installed here.
        "--exclude-module", "torch",
        "--exclude-module", "whisper",
        "--exclude-module", "funasr",
        "--exclude-module", "torchaudio",
    ])


BUNDLED_MODEL = "base"


def add_model(folder: Path):
    """Put the default speech to text model in the package, so the program works straight after
    unzipping, without a download (and where Hugging Face cannot be reached)."""
    from faster_whisper import download_model  # pylint: disable=import-outside-toplevel
    target = folder / "models" / f"faster-whisper-{BUNDLED_MODEL}"
    target.mkdir(parents=True, exist_ok=True)
    download_model(BUNDLED_MODEL, output_dir=str(target))
    # (only the model files: no download cache or links next to them)
    for leftover in (target / ".cache", target / ".gitattributes", target / "README.md"):
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)
        elif leftover.exists():
            leftover.unlink()
    if not (target / "model.bin").is_file():
        raise SystemExit(f"The speech to text model was not downloaded into {target}")


def add_files(folder: Path, version: str):
    shutil.copy2(APP / "parameters.yaml", folder / "parameters.yaml")
    shutil.copy2(APP / "override.yaml", folder / "override.yaml")
    (folder / "version.txt").write_text(f"{version} ({git_hash()})", encoding="utf-8")
    (folder / "bin").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "bin" / "README.md", folder / "bin" / "README.md")
    shutil.copy2(ROOT / "LICENSE", folder / "LICENSE.txt")
    (folder / "README.txt").write_text(
        "Transcribe\n"
        "==========\n\n"
        "1. Double-click Transcribe.exe. It starts writing down what you say (microphone) and what\n"
        "   the others say (the sound of this computer) at once. No internet is needed for that.\n"
        "2. For suggested replies and summaries: press Settings, enter the API key of your AI\n"
        "   provider (and its address and model if it is not OpenAI), press Test, then Save.\n"
        "3. F2 asks for a reply at any moment. F4 pauses. 'More' has 'Keep the window on top'.\n\n"
        "The transcript of every session is saved when the program closes\n"
        "(More > Open the folder with transcripts and logs).\n"
        "Your settings are in override.yaml in this folder: keep that file when you update.\n\n"
        "More: https://github.com/samaliyan/transcribe\n",
        encoding="utf-8")


def make_zip(folder: Path, version: str) -> Path:
    target = DIST / f"{NAME}-{version}-windows.zip"
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, Path(NAME) / path.relative_to(folder))
    return target


def folder_size_mb(folder: Path) -> float:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="dev", help="version written into the package, e.g. 1.2.0")
    args = parser.parse_args()
    version = args.version.lstrip("vV") or "dev"

    if sys.platform != "win32":
        print("Warning: the package only runs on Windows; build it on Windows.")
    run_pyinstaller()
    folder = DIST / NAME
    add_files(folder, version)
    add_model(folder)
    archive = make_zip(folder, version)
    print(f"Package folder: {folder} ({folder_size_mb(folder):.0f} MB)")
    print(f"Zip: {archive} ({archive.stat().st_size / 1e6:.0f} MB)")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"zip={archive}\n")


if __name__ == "__main__":
    main()
