import atexit
import subprocess
import time

from .settings import settings


def start_server(exe_path: str, model_path: str, port: str | None = None):
    print("Booting up Koboldcpp server in the background...")
    cmd = [
        exe_path,
        "--model", model_path,
        "--host", "127.0.0.1",
        "--port", port or settings.kobold_port,
        "--gpulayers", "99",
        "--quiet"
    ]

    # Launch as a background process
    process = subprocess.Popen(cmd)

    # Ensure the background server dies when you stop the Python script
    atexit.register(lambda: process.terminate())

    # Give the server a few seconds to load the model into VRAM
    time.sleep(5)

    return process
