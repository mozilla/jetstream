[![CircleCI](https://circleci.com/gh/mozilla/pensieve/tree/master.svg?style=shield)](https://circleci.com/gh/mozilla/pensieve/tree/master)

# pensieve

Automated experiment analysis.

Pensieve automatically calculates metrics and applies statistical treatments to collected experiment data for different analysis windows.

For more information, see [the documentation](https://github.com/mozilla/pensieve/wiki).

## Running tests

Make sure `tox` is installed globally (run `brew install tox` or `pip install tox`).

Then, run `tox` from wherever you cloned this repository. (You don't need to install pensieve first.)

To run integration tests, run `tox -e py38-integration`.


## Local installation

```bash
# Create and activate a python virtual environment.
python3 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
```
