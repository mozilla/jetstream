def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", help="Run integration tests",
    )
