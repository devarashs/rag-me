"""Deployment entrypoint. Vercel serves the FastAPI instance named `app` in this file.

Run locally with `uv run uvicorn app:app --reload`.
"""

import logging

from rag_me.api import create_production_app

# Vercel collects stdout/stderr as function logs; INFO carries the one-line
# request records written by rag_me.api.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

app = create_production_app()
