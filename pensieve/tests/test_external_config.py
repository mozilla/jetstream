from pensieve.external_config import ExternalConfigCollection


class TestExternalConfig:
    def test_from_github_repo(self):
        external_configs = ExternalConfigCollection.from_github_repo()
        assert external_configs

        example_conf = external_configs.spec_for_experiment("example_config")
        assert example_conf is not None

        assert external_configs.spec_for_experiment("not-existing-conf") is None
