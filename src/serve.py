"""Run the dashboard + JSON API (uvicorn).

Requires a running Postgres (``docker compose up -d``). Open http://127.0.0.1:8000

Usage:
    python src/serve.py
    python src/serve.py --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn  # noqa: E402

from market_intel.api.app import create_app  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Serve the market-intelligence dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
