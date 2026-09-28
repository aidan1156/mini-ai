"""Run the API server: python -m mini.api

Host and port come from MINI_API_HOST / MINI_API_PORT (default 127.0.0.1:8000).
"""

import logging
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

uvicorn.run(
    "mini.api.app:app",
    host=os.environ.get("MINI_API_HOST", "127.0.0.1"),
    port=int(os.environ.get("MINI_API_PORT", "8000")),
)
