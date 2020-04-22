import pytest


def pytest_runtest_setup(item):
    if "FLAKE8" in item.nodeid or "BLACK" in item.nodeid:
        return
    if not item.config.getoption("--integration", False):
        pytest.skip("Skipping integration test")
