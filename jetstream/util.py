import logging
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from requests import Session

logger = logging.getLogger(__name__)

# based on https://stackoverflow.com/a/22726782


class RetryLimitExceededException(Exception):
    pass


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


def retry_get(
    session: Session, url: str, max_retries: int, user_agent: Optional[str] = None
) -> Any:
    for _i in range(max_retries):
        try:
            if user_agent:
                session.headers.update({"user-agent": user_agent})

            blob = session.get(url).json()
            break
        except Exception as e:
            print(e)
            logger.info(f"Error fetching from {url}. Retrying...")
            time.sleep(1)
    else:
        exception = RetryLimitExceededException(f"Too many retries for {url}")

        logger.exception(exception.__str__(), exc_info=exception)
        raise exception

    return blob
