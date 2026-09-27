import os

from supportbot.api import app

if not os.environ.get("VERCEL"):
    # Local development: serve the static frontend too (on Vercel, public/ is served by the CDN).
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "public"), html=True), name="web")

__all__ = ["app"]
