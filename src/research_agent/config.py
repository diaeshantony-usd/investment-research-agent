"""
config.py
=========
Central configuration and logging setup for Investment Research Agent.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).resolve().parent
SRC_DIR = CONFIG_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MEMORY_DIR = DATA_DIR / "memory"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, MEMORY_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_LOG_FILE = LOGS_DIR / "research_agent.log"
DEFAULT_MEMORY_FILE = MEMORY_DIR / "graph_memory_store.json"

# ---------------------------------------------------------------------------
# LLM Providers Configuration
# ---------------------------------------------------------------------------
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")

# Ollama Configuration
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma4:31b")
DEFAULT_API_KEY = os.getenv(
    "OLLAMA_API_KEY", "29556581fc324fe4a0ceba6430989b2a._f3DUv-Ue3XziHJF0asqQDHg"
)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")

# External Data API Keys (Free tiers / placeholders)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# Execution Defaults
DEFAULT_MAX_ITERATIONS = int(os.getenv("DEFAULT_MAX_ITERATIONS", "10"))
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("DEFAULT_TIMEOUT_SECONDS", "60"))

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    console_output: bool = False,
) -> logging.Logger:
    """Configures project-wide logging to file. Console output is disabled by default."""
    target_file = Path(log_file) if log_file else DEFAULT_LOG_FILE
    target_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("research_agent")
    logger.setLevel(level)
    logger.propagate = False

    # Suppress verbose noisy third-party libraries from leaking to console
    for lib in ("httpx", "httpcore", "urllib3", "openai", "requests"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        if not console_output:
            logger.handlers = [
                h for h in logger.handlers if not isinstance(h, logging.StreamHandler)
            ]
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Rotating file handler (5 MB max, up to 3 backups)
    file_handler = RotatingFileHandler(
        target_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Global root logger instance (logs strictly to logs/research_agent.log)
logger = setup_logging(console_output=False)
