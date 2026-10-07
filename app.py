"""Deployment entrypoint. Vercel serves the FastAPI instance named `app` in this file.

Run locally with `uv run uvicorn app:app --reload`.
"""

import logging

from rag_me.api import create_production_app

# Vercel collects stdout/stderr as function logs; INFO carries the one-line
# request records written by rag_me.api.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

app = create_production_app()

# The chat page. API routes always take priority over these files. On Vercel the
# directory is promoted to the CDN at build time, so page loads never invoke the
# Python function. The path is relative to the project root, which is the
# working directory both on Vercel and for `uvicorn app:app`.
app.frontend("/", directory="web")
