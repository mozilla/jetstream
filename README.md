[![CircleCI](https://circleci.com/gh/mozilla/jetstream/tree/main.svg?style=shield)](https://circleci.com/gh/mozilla/jetstream/tree/main)

# jetstream

Automated experiment analysis.

Jetstream automatically calculates metrics and applies statistical treatments to collected experiment data for different analysis windows.

For more information, see [the documentation](https://experimenter.info/jetstream/jetstream/).

## Running tests

Make sure `tox` is installed globally (run `brew install tox` or `pip install tox`).

Then, run `tox` from wherever you cloned this repository. (You don't need to install jetstream first.)

To run integration tests, run `tox -e py310-integration`.


## Local installation

```bash
# Create and activate a python virtual environment.
python3 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dependencies

Jetstream uses pip-tools to manage dependencies, along with a script that runs the pip-tools commands. There are two requirements files:
- `requirements.in`: Listing of dependencies and versions. This is the file to edit if you want to change a dependency or its version.
- `requirements.txt`: Auto-generated by pip-tools (`pip-compile`) from the `requirements.in` file. Also contains the hashes of each package for verification by pip during installation, and comments showing lineage for each dependency.

### Update all dependencies

`./script/update_deps`

Be sure to run `pip install -r requirements.txt` and reinstall jetstream (`pip install -e .`) afterwards, and **test** functionality!

### Update a single dependency

1. Edit `requirements.in`
- `mypy==1.8.0` --> `mypy==1.9.0`

2. Regenerate `requirements.txt`
- `pip-compile --generate-hashes -o requirements.txt requirements.in`
  - (*Note*: this is the last line of `script/update_deps`)

3. Install dependencies
- `pip install -r requirements.txt`
- `pip install -e .`

4. Test!
