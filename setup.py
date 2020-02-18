from setuptools import setup
import os


def text_from_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


setup(
    name="Pensieve",
    use_incremental=True,
    author="Mozilla Corporation",
    author_email="fx-data-dev@mozilla.org",
    description="Runs a thing that analyzes experiments",
    url="https://github.com/mozilla/pensieve",
    packages=["pensieve", "pensieve.tests"],
    install_requires=text_from_file("requirements.txt").strip().split("\n"),
    setup_requires=["incremental"],
    tests_require=text_from_file("requirements_tests.txt").strip().split("\n"),
    long_description=text_from_file("README.md"),
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
)
