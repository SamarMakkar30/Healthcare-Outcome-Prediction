
"""Unified launcher for the current Healthcare Prediction application stack."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")


def check_dependencies() -> bool:
    """Check core dependencies for backend API and frontend tooling."""
    print("Checking dependencies...")
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        import pandas  # noqa: F401
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        print(f"Missing Python dependency: {exc}")
        print("Install with: pip install -r requirements-production.txt")
        return False

    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    try:
        subprocess.run([npm_cmd, "--version"], cwd=ROOT, check=True, capture_output=True)
    except Exception:
        print("Node/NPM is not available. Frontend commands will fail.")
        print("Install Node.js from https://nodejs.org/")
        return False

    print("All required dependencies are available")
    return True


def run_backend(host: str = "127.0.0.1", port: int = 8000) -> int:
    print_header("STARTING BACKEND API")
    cmd = [
        sys.executable,
        "run_production.py",
        "--mode",
        "api",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.run(cmd, cwd=ROOT).returncode


def run_frontend(host: str = "127.0.0.1", port: int = 5173) -> int:
    print_header("STARTING FRONTEND")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    cmd = [npm_cmd, "run", "dev", "--", "--host", host, "--port", str(port)]
    return subprocess.run(cmd, cwd=ROOT / "frontend").returncode


def run_fullstack() -> int:
    print_header("STARTING FULL STACK")
    print("Backend:  http://127.0.0.1:8000")
    print("Frontend: http://127.0.0.1:5173")
    print("Press Ctrl+C to stop both processes\n")

    backend_cmd = [
        sys.executable,
        "run_production.py",
        "--mode",
        "api",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]

    backend_proc = subprocess.Popen(backend_cmd, cwd=ROOT)
    try:
        return subprocess.run(frontend_cmd, cwd=ROOT / "frontend").returncode
    finally:
        if backend_proc.poll() is None:
            backend_proc.terminate()


def interactive_menu() -> int:
    print_header("HEALTHCARE PREDICTION SYSTEM")
    print("1. Start backend API only")
    print("2. Start frontend only")
    print("3. Start full stack (backend + frontend)")
    print("4. Exit")

    choice = input("\nEnter your choice (1-4): ").strip()
    if choice == "1":
        return run_backend()
    if choice == "2":
        return run_frontend()
    if choice == "3":
        return run_fullstack()
    if choice == "4":
        print("Exiting")
        return 0

    print("Invalid choice")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Healthcare Prediction launcher")
    parser.add_argument("--mode", choices=["menu", "api", "frontend", "fullstack"], default="menu")
    args = parser.parse_args()

    if not check_dependencies():
        return 1

    if args.mode == "api":
        return run_backend()
    if args.mode == "frontend":
        return run_frontend()
    if args.mode == "fullstack":
        return run_fullstack()
    return interactive_menu()


if __name__ == "__main__":
    raise SystemExit(main())
