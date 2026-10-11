"""Run the API and Streamlit in one hosted service, with shared shutdown."""

import os
import signal
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen


REQUIRED_ENV = (
    "GROQ_API_KEY",
    "PORTKEY_API_KEY",
    "GEMINI_API_KEY",
    "QDRANT_API_KEY",
    "QDRANT_CLUSTER_ENDPOINT",
)


def main():
    missing = [key for key in REQUIRED_ENV if not os.getenv(key, "").strip()]
    if missing:
        print("Missing required environment variables: " + ", ".join(missing), flush=True)
        return 1

    stopped = threading.Event()
    children = []

    def stop(signum, frame):
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child_env = os.environ.copy()
    child_env["BACKEND_URL"] = "http://127.0.0.1:8000"
    child_env.setdefault("LANGSMITH_TRACING", "false")

    try:
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", "8000"],
            env=child_env,
        )
        children.append(api)
        deadline = time.monotonic() + 180
        while not stopped.is_set():
            if api.poll() is not None:
                raise RuntimeError("API exited before becoming ready")
            try:
                with urlopen(child_env["BACKEND_URL"] + "/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("API did not become ready within 180 seconds")
            stopped.wait(0.5)

        if stopped.is_set():
            return 0

        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "ui/app.py",
             "--server.address=0.0.0.0", "--server.port=" + os.getenv("PORT", "10000"),
             "--server.headless=true", "--browser.gatherUsageStats=false"],
            env=child_env,
        )
        children.append(ui)
        while not stopped.wait(0.5):
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A server exited; stopping the service")
        return 0
    except Exception as exc:
        print("Service startup failed: " + str(exc), flush=True)
        return 1
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    sys.exit(main())
