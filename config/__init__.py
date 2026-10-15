"""Re-export the `config.config` module contents so `from config import settings`
keeps working after config.py was relocated into this package."""
from __future__ import annotations

from .config import BASE_DIR, Settings, settings

__all__ = ["BASE_DIR", "Settings", "settings"]
