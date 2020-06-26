from pensieve.external_config import ExternalConfigCollection 

class TestExterinalConfig:
    def test_from_github_repo(self):
        assert ExternalConfigCollection.from_github_repo()

