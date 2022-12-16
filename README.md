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
```
