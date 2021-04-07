import logging
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from requests import Session

logger = logging.getLogger(__name__)

# based on https://stackoverflow.com/a/22726782


@contextmanager
def TemporaryDirectory():
    name = Path(tempfile.mkdtemp())
    try:
        yield name
    finally:
        shutil.rmtree(name)


def inclusive_date_range(start_date, end_date):
    """Generator for a range of dates, includes end_date."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def retry_get(session: Session, url: str, max_retries: int) -> Any:
    for _i in range(max_retries):
        try:
            blob = session.get(url).json()
            break
        except Exception:
            logger.info(f"Error fetching from {url}. Retrying...")
            time.sleep(1)
    else:
        raise Exception(f"Too many retries for {url}")
    return blob
