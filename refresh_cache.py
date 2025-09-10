from logger import setup_logger

import os
import shutil
from typing import Optional

logger = setup_logger("cache-refresh-subroutine")
logger.setLevel("INFO")


def refresh_cache(base_dir: Optional[str] = None):
    """Clear and recreate PyStan/httpstan cache directory.

    If base_dir is None, uses the current user's home directory.
    """
    if base_dir:
        cache_path = os.path.join(base_dir, ".cache", "httpstan")
    else:
        cache_path = os.path.expanduser("~/.cache/httpstan")

    logger.info(f"Refreshing cache at {cache_path}")
    if os.path.exists(cache_path):
        logger.info(f"Removing {cache_path}")
        shutil.rmtree(cache_path)
    logger.info(f"Creating {cache_path}")
    os.makedirs(cache_path, exist_ok=True)
    logger.success("Successfully refreshed cache.")
