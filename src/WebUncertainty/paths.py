"""Canonical project paths.

All runtime code uses this module instead of inferring paths from ``__file__``
independently.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def _discover_project_root() -> Path:
    configured = os.environ.get("WEBUNCERTAINTY_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (PACKAGE_ROOT, *PACKAGE_ROOT.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = _discover_project_root()
WORKSPACE_ROOT = PROJECT_ROOT.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
DATA_ROOT = PROJECT_ROOT / "data"


def environment_files() -> tuple[Path, ...]:
    """Return local dotenv candidates in precedence order."""
    return PROJECT_ROOT / ".env", WORKSPACE_ROOT / ".env"
