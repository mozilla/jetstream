from setuptools import setup


def text_from_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


test_dependencies = [
    "coverage",
    "isort",
    "jsonschema",
    "pytest",
    "pytest-black",
    "pytest-cov",
    "pytest-flake8",
    "mypy",
    "types-futures",
    "types-pkg-resources",
    "types-protobuf",
    "types-pytz",
    "types-PyYAML",
    "types-requests",
    "types-six",
    "types-toml",
]

extras = {
    "testing": test_dependencies,
}

setup(
    name="mozilla-jetstream",
    author="Mozilla Corporation",
    author_email="fx-data-dev@mozilla.org",
    description="Runs a thing that analyzes experiments",
    url="https://github.com/mozilla/jetstream",
    packages=[
        "jetstream",
        "jetstream.tests",
        "jetstream.tests.integration",
        "jetstream.logging",
        "jetstream.workflows",
        "jetstream.diagnostics",
    ],
    package_data={
        "jetstream.tests": ["data/*"],
        "jetstream.workflows": ["*.yaml"],
        "jetstream": ["../*.toml"],
    },
    install_requires=[
        "attrs",
        "cattrs",
        "Click",
        "dask[distributed]",
        "db-dtypes",
        "GitPython",
        "google-cloud-artifact-registry",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "google-cloud-container",
        "google-cloud-storage",
        "grpcio",  # https://github.com/googleapis/google-cloud-python/issues/6259
        "jinja2",
        "mozanalysis",
        "mozilla-metric-config-parser",
        "mozilla-nimbus-schemas",
        "pyarrow",
        "pytz",
        "PyYAML",
        "requests",
        "smart_open[gcs]",
        "statsmodels",
        "toml",
    ],
    include_package_data=True,
    tests_require=test_dependencies,
    extras_require=extras,
    long_description=text_from_file("README.md"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    entry_points="""
        [console_scripts]
        pensieve=jetstream.cli:cli
        jetstream=jetstream.cli:cli
    """,
    # This project does not issue regular releases, only when there
    # are changes that would be meaningful to our (few) dependents.
    version="2023.8.1",
)
