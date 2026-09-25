"""
SupportIQ Unified Entrypoint Script.
Allows running the API, Streamlit UI, test suite, or both services easily.

Usage:
    python run.py api        # Runs FastAPI REST server (port 8000)
    python run.py ui         # Runs Streamlit Web UI (port 8501)
    python run.py test       # Runs full test suite via pytest
    python run.py all        # Runs both API and UI concurrently
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent


def run_api():
    """Start Uvicorn FastAPI server."""
    print("[API] Starting SupportIQ REST API on http://localhost:8000 ...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--reload",
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR))


def run_ui():
    """Start Streamlit application."""
    print("[UI] Starting SupportIQ Streamlit Dashboard on http://localhost:8501 ...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/streamlit_app.py",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR))


def run_tests():
    """Run pytest suite."""
    print("[TEST] Running SupportIQ Test Suite ...")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-v",
        "--tb=short",
    ]
    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    sys.exit(res.returncode)


def run_all():
    """Run both API and UI simultaneously."""
    print("[ALL] Launching SupportIQ API and Streamlit UI ...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(BASE_DIR),
    )
    ui_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.port", "8501"],
        cwd=str(BASE_DIR),
    )
    try:
        api_proc.wait()
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        api_proc.terminate()
        ui_proc.terminate()


def main():
    parser = argparse.ArgumentParser(description="SupportIQ CLI Runner")
    parser.add_argument(
        "mode",
        nargs="?",
        default="api",
        choices=["api", "ui", "test", "all"],
        help="Service to run: 'api', 'ui', 'test', or 'all'",
    )
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    elif args.mode == "ui":
        run_ui()
    elif args.mode == "test":
        run_tests()
    elif args.mode == "all":
        run_all()


if __name__ == "__main__":
    main()
