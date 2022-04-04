import datetime as dt
from pathlib import Path
from textwrap import dedent

# from threading import activeCount
from typing import cast
from unittest.mock import Mock, patch

import pytest
import toml

from jetstream.config import AnalysisSpec, OutcomeSpec
from jetstream.external_config import (
    ExternalConfig,
    ExternalConfigCollection,
    ExternalDefaultConfig,
    ExternalOutcome,
    entity_from_path,
    validate_config_settings,
)


class TestExternalConfig:
    def test_from_github_repo(self):
        external_configs = ExternalConfigCollection.from_github_repo()
        assert external_configs

        assert external_configs.spec_for_experiment("not-existing-conf") is None

    class FakePath:
        def __init__(self, path, config):
            self._path = Path(path)
            self._config = config

        def __getattr__(self, key):
            return getattr(self._path, key)

        def read_text(self):
            return self._config

        def stat(self):
            m = Mock()
            m.st_mtime = 0
            return m

    @pytest.mark.parametrize(
        "path",
        [
            "outcomes/fenix/my_cool_outcome.toml",
            "/some/garbage/outcomes/fenix/my_cool_outcome.toml",
        ],
    )
    def test_entity_from_path_yields_outcome(self, path: str):
        config = dedent(
            """\
            friendly_name = "I'm an outcome!"
            description = "It's rad to be an outcome."
            """
        )
        fakepath = self.FakePath(path, config)
        entity = entity_from_path(cast(Path, fakepath))
        assert isinstance(entity, ExternalOutcome)
        assert entity.slug == "my_cool_outcome"
        assert entity.platform == "fenix"

    @pytest.mark.parametrize(
        "path",
        [
            "my_cool_experiment.toml",
            "/some/garbage/not_an_outcome/foo/my_cool_experiment.toml",
        ],
    )
    def test_entity_from_path_yields_config(self, path: str):
        fakepath = self.FakePath(path, "")
        entity = entity_from_path(cast(Path, fakepath))
        assert isinstance(entity, ExternalConfig)
        assert entity.slug == "my_cool_experiment"

    def test_validating_external_outcome(self, monkeypatch):
        Analysis = Mock()
        monkeypatch.setattr("jetstream.external_config.Analysis", Analysis)
        config = dedent(
            """\
            friendly_name = "I'm an outcome!"
            description = "It's rad to be an outcome."
            """
        )
        spec = OutcomeSpec.from_dict(toml.loads(config))
        extern = ExternalOutcome(
            slug="cool_outcome", spec=spec, platform="firefox_desktop", commit_hash="0000000"
        )
        extern.validate()
        assert Analysis.validate.called_once()

    def test_validating_external_config(self, monkeypatch, experiments):
        Analysis = Mock()
        monkeypatch.setattr("jetstream.external_config.Analysis", Analysis)
        spec = AnalysisSpec.from_dict({})
        extern = ExternalConfig(
            slug="cool_experiment",
            spec=spec,
            last_modified=dt.datetime.now(),
        )
        extern.validate(experiments[0])
        assert Analysis.validate.called_once()

    def test_validating_external_default_config(self, monkeypatch, experiments):
        Analysis = Mock()
        monkeypatch.setattr("jetstream.external_config.Analysis", Analysis)
        spec = AnalysisSpec.from_dict({})
        extern = ExternalDefaultConfig(
            slug="firefox_desktop",
            spec=spec,
            last_modified=dt.datetime.now(),
        )
        extern.validate()
        assert Analysis.validate.called_once()


@pytest.mark.parametrize(
    "test_input,expected",
    (
        (
            {"experiment": dict(), "segments": dict(), "metrics": dict()},
            None,
        ),
    ),
)
@patch("jetstream.external_config.toml.loads")
def test_validate_config_settings(mock_toml_loads, test_input, expected):
    mock_toml_loads.return_value = test_input

    config_file = "README.md"
    actual = validate_config_settings(Path(config_file))

    assert actual == expected
