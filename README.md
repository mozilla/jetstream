[![CircleCI](https://circleci.com/gh/mozilla/pensieve/tree/master.svg?style=shield)](https://circleci.com/gh/mozilla/pensieve/tree/master)

# pensieve

Automated experiment analysis.

Pensieve automatically calculates metrics and applies statistical treatments to collected experiment data for different analysis windows.

For more information, see [the documentation](https://github.com/mozilla/pensieve/wiki).


## Development and Testing

Install requirements:

```bash
# Create and activate a python virtual environment.
python3 -m venv venv/
source venv/bin/activate

pip install -r requirements.txt
```

Run local tests:

```bash
venv/bin/pytest --black --flake8

# or using tox

venv/bin/tox
```

Run integration tests:

```bash
venv/bin/pytest --black --flake8 --integration pensieve/tests/integration/
```
