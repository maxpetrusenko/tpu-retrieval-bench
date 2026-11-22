import pytest
from bench import run


@pytest.fixture(autouse=True)
def clean_git_tree(monkeypatch):
    """bench.run refuses dirty trees; tests run in a working copy that is usually dirty."""
    monkeypatch.setattr(run, "repo_git_info", lambda a: {"commit": "t" * 40, "dirty": False})
