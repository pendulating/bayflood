from logger import setup_logger

import os
import shutil

logger = setup_logger("cache-refresh-subroutine")
logger.setLevel("INFO")

def refresh_cache(netid):
    CACHE_PATH = f'/home/{netid}/.cache/httpstan'
    logger.info(f"Refreshing cache at {CACHE_PATH}")
    if os.path.exists(CACHE_PATH):
        logger.info(f"Removing {CACHE_PATH}")
        shutil.rmtree(CACHE_PATH)
    logger.info(f"Creating {CACHE_PATH}")
    os.makedirs(CACHE_PATH)
    logger.success("Successfully refreshed cache.")
