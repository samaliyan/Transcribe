import os
import sys
import time
import atexit
from pathlib import Path

_T0 = float(os.environ.get("TRANSCRIBE_T0") or time.time())


def _mark(stage: str):
    """Start-up timing, printed to the console (or logs/console.txt for the packaged program)."""
    print(f"[START] {time.time() - _T0:6.1f} s  {stage}", flush=True)

if __package__ in (None, ""):
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    __package__ = "app.transcribe"

from tsutils import app_logging as al
from tsutils import configuration

from .appui import AppUI
from .cli.arguments import create_args, update_args_config
from .cli.batch import handle_batch_tasks
from .core.conversation import APP_DIR
from .core.state import create_app_runtime
from .desktop import DesktopController
from .desktop.runtime import (
    initialize_desktop_runtime,
    initiate_app_threads,
    shutdown,
    start_audio_capture,
)


def read_version() -> str:
    """'1.2.0' for a released package, 'dev' when run from the source code."""
    try:
        text = (APP_DIR / "version.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return "dev"
    first = text.split()[0] if text else ""
    return f"v{first}" if first[:1].isdigit() and "." in first else "dev"


def apply_proxy(config: dict):
    """A proxy from the settings is used for every connection the program makes."""
    proxy = str(config["General"].get("proxy") or "").strip()
    if proxy:
        for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
            os.environ[name] = proxy


def main():
    """Primary method to run transcribe
    """
    _mark("program code loaded")
    args = create_args()

    # The settings files are next to the program, wherever it is started from.
    config = configuration.Config(
        default_config_filename=str(APP_DIR / "parameters.yaml"),
        override_config_filename=str(APP_DIR / "override.yaml"),
    ).data
    apply_proxy(config)
    runtime = create_app_runtime()

    update_args_config(args, config)
    if handle_batch_tasks(args, config):
        return

    initialize_desktop_runtime(runtime, config)
    _mark("audio devices and speech to text ready to start")

    # Save the transcript and close devices when the program exits
    atexit.register(shutdown, runtime)

    start_audio_capture(runtime, config)
    _mark("audio capture started")

    # Initiate logging
    log_listener = al.initiate_log(config=config)

    controller = DesktopController(config=config, runtime=runtime)
    aui = AppUI(config=config, runtime=runtime, controller=controller)
    initiate_app_threads(runtime=runtime, config=config)
    aui.set_version(read_version())
    _mark("window built")

    print("READY")

    # Set the response lang in STT Model.
    runtime.transcriber.set_language(config['OpenAI']['audio_lang'])
    aui.update_initial_transcripts()
    aui.start()
    log_listener.stop()


if __name__ == "__main__":
    main()
