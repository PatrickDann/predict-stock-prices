"""HTTP API + dashboard.

``create_app`` builds a FastAPI app that exposes the ingested data (prices,
macro, news + search, filings) as JSON and serves a self-contained dark
dashboard. The JSON API is frontend-agnostic — reusable by the bundled
dashboard, OpenBB widgets, or a future React app, and later by model serving.
"""

from market_intel.api.app import create_app

__all__ = ["create_app"]
