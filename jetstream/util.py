from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

# based on https://stackoverflow.com/a/22726782


@contextmanager
def TemporaryDirectory():
    name = Path(tempfile.mkdtemp())
    try:
        yield name
    finally:
        shutil.rmtree(name)
