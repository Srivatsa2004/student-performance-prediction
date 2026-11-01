"""One-command launcher: starts the FastAPI backend, waits until it is ready,
then starts the Streamlit frontend. Ctrl+C stops both.

    python run.py
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8000"))
UI_PORT = int(os.environ.get("UI_PORT", "8501"))


def wait_for_api(url: str, proc: subprocess.Popen, timeout: int = 180) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def stop(proc: subprocess.Popen) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    env = os.environ.copy()
    env["API_URL"] = f"http://{API_HOST}:{API_PORT}"

    print(f"Starting backend on {env['API_URL']} ...")
    print("(first launch trains the neural network, which takes a few seconds)")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", API_HOST, "--port", str(API_PORT)],
        cwd=ROOT, env=env,
    )
    ui = None
    try:
        if not wait_for_api(f"{env['API_URL']}/health", api):
            print("Backend failed to start. See the error above.")
            return
        print(f"Backend ready. Starting UI on http://localhost:{UI_PORT}")
        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
             "--server.port", str(UI_PORT), "--browser.gatherUsageStats", "false"],
            cwd=ROOT, env=env,
        )
        while ui.poll() is None and api.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop(ui)
        stop(api)


if __name__ == "__main__":
    main()
