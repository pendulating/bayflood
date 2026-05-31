from logger import setup_logger

import os
import shutil
from pathlib import Path
from typing import Optional

logger = setup_logger("cache-refresh-subroutine")
logger.setLevel("INFO")

# Default to project directory to avoid home disk quota issues
_PROJECT_ROOT = Path(__file__).parent.resolve()
_DEFAULT_CACHE_BASE = _PROJECT_ROOT / ".cache"


def refresh_cache(base_dir: Optional[str] = None):
    """Clear and recreate PyStan/httpstan cache directory.

    If base_dir is None, uses the project .cache directory (not home).
    """
    if base_dir:
        cache_path = os.path.join(base_dir, "httpstan")
    else:
        cache_path = str(_DEFAULT_CACHE_BASE / "httpstan")

    logger.info(f"Refreshing cache at {cache_path}")
    if os.path.exists(cache_path):
        logger.info(f"Removing {cache_path}")
        shutil.rmtree(cache_path)
    logger.info(f"Creating {cache_path}")
    os.makedirs(cache_path, exist_ok=True)
    logger.success("Successfully refreshed cache.")
