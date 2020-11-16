import shutil
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

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
