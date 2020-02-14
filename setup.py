from setuptools import setup

test_dependencies = [
    "coverage",
    "pytest",
    "pytest-cov",
]

extras = {
    "testing": test_dependencies,
}

with open("README.md", "rt") as f:
    long_description = f.read()

setup(
    name="Pensieve",
    use_incremental=True,
    author="Mozilla Corporation",
    author_email="fx-data-dev@mozilla.org",
    description="Runs a thing that analyzes experiments",
    url="https://github.com/mozilla/pensieve",
    packages=["pensieve", "pensieve.tests"],
    install_requires=[
        "attrs",
        "cattrs",
        "incremental",
        "pytz",
        "requests",
    ],
    setup_requires=["incremental"],
    tests_require=test_dependencies,
    extras_require=extras,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
)
