"""
config.py
=========
Central configuration and logging setup for the Investment Research Agent.

All settings can be overridden with environment variables (see ``.env.example``).
Importing this module creates the ``data/`` and ``logs/`` folders if they are missing.

Author: N L N Sai Krishna Akula
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

# Load a local, git-ignored ``.env`` (copied from ``.env.example``) so API keys never
# have to live in source code. Variables already set in the environment take precedence.
try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at runtime; plain env vars still work
    pass
else:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

# Ensure the runtime folders exist before anything tries to write to them.
for directory in (DATA_DIR, CACHE_DIR, MEMORY_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_LOG_FILE = LOGS_DIR / "research_agent.log"
DEFAULT_MEMORY_FILE = MEMORY_DIR / "graph_memory_store.json"

# ---------------------------------------------------------------------------
# LLM Providers Configuration
# ---------------------------------------------------------------------------
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")

# Ollama Configuration
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-oss:120b")
# An OLLAMA_API_KEY set in the environment or a local .env file overrides this default.
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
_LOG_MAX_BYTES = 5 * 1024 * 1024  # rotate the log file at 5 MB
_LOG_BACKUP_COUNT = 3  # keep research_agent.log.1 .. .3


def setup_logging(
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    console_output: bool = False,
) -> logging.Logger:
    """Configure the ``research_agent`` logger (rotating file, optional console).

    Safe to call more than once: existing handlers are reused, not duplicated.

    Args:
        log_file: Log file path. Defaults to ``logs/research_agent.log``.
        level: Logging level for the logger and its handlers.
        console_output: Also log to stdout (off by default to keep notebooks clean).

    Returns:
        The configured logger.
    """
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

    file_handler = RotatingFileHandler(
        target_file,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
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


# Project-wide logger; writes to logs/research_agent.log only.
logger = setup_logging(console_output=False)
