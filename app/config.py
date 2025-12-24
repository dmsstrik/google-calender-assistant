"""Simple configuration management for the calendar assistant.

This module exposes a ``settings`` object containing configuration
values derived from environment variables and an optional `.env`
file.  When a `.env` file exists in the project root it is parsed to
populate the environment before any values are read.  This approach
avoids the need for heavy dependencies such as Pydantic while still
providing a central place to manage configuration.

The ``Settings`` class reads specific environment variables defined in
the `.env.example` file.  If a variable is not set it falls back to a
sensible default where appropriate.  See the `.env.example` file for
the full list of available configuration keys.

Example usage:

>>> from app.config import settings
>>> print(settings.llm_model)

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    # Use python-dotenv to load environment variables from a .env file
    from dotenv import load_dotenv  # type: ignore[import]
except Exception:
    # If python-dotenv isn't installed, define a no-op fallback
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return None


class Settings:
    """Container for application configuration values."""

    def __init__(self) -> None:
        # Attempt to load variables from a .env file at project root
        # Determine the project root relative to this file.  The
        # configuration lives in ``project/google-calender-assistant``, so
        # stepping up one directory yields the directory containing
        # `.env`.
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if env_file.exists():
            # Attempt to use python-dotenv if available.  When installed
            # this will correctly handle quoted values and other edge
            # cases.  The ``override=False`` flag ensures that any
            # existing environment variables take precedence over the
            # values in the file.  If python-dotenv isn't installed
            # our fallback ``load_dotenv`` defined above will simply
            # return without error.
            try:
                load_dotenv(env_file, override=False)  # type: ignore[call-arg]
            except Exception:
                # The fallback does not raise, but we still wrap it in
                # a try/except block to future‑proof against errors.
                pass
            # Manual parse: regardless of whether python-dotenv ran, we
            # manually read key/value pairs from the file and set any
            # missing environment variables.  This ensures that
            # configuration values are available even when python-dotenv
            # is unavailable.  Variables already present in the
            # environment will not be overwritten.
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ.setdefault(key, value)
            except Exception:
                # If reading the file fails for any reason we silently
                # skip manual parsing.  Missing variables will fall back
                # to the defaults defined below.
                pass

        # Read configuration values from the environment with
        # appropriate defaults.  Environment variable names are
        # uppercase to match those in the provided `.env.example`.
        self.openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        # Default to Qwen 2.5 model if not specified.  The user can
        # override this via the LLM_MODEL environment variable.
        self.llm_model: Optional[str] = os.getenv(
            "LLM_MODEL", "qwen2.5-vl-7b-instruct"
        )

        # Interface selection (currently only 'cli' is implemented)
        self.interface: str = os.getenv("INTERFACE", "cli")

        # Google OAuth configuration
        self.google_client_secret_file: str = os.getenv(
            "GOOGLE_CLIENT_SECRET_FILE", "secrets/client_secret.json"
        )
        self.google_token_file: str = os.getenv(
            "GOOGLE_TOKEN_FILE", "secrets/token.json"
        )

        # Default calendar ID (e.g. 'primary')
        self.default_calendar_id: str = os.getenv(
            "DEFAULT_CALENDAR_ID", "primary"
        )

        # Optional Telegram bot token
        self.telegram_bot_token: Optional[str] = os.getenv(
            "TELEGRAM_BOT_TOKEN"
        )


# Instantiate a single settings object for consumers to import
settings = Settings()