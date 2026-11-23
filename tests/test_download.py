import io
from unittest import mock

from bench import download


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_download_sends_browser_user_agent(tmp_path):
    seen = {}

    def fake_urlopen(req):
        seen["ua"] = req.get_header("User-agent")
        seen["url"] = req.full_url
        return _Resp(b"abc")

    dest = tmp_path / "x.part"
    with mock.patch("urllib.request.urlopen", fake_urlopen):
        download.download("https://example.invalid/x.hdf5", dest)
    assert seen["ua"] == download.USER_AGENT
    assert "Python-urllib" not in seen["ua"] and seen["ua"].startswith("Mozilla/5.0")
    assert seen["url"] == "https://example.invalid/x.hdf5"
    assert dest.read_bytes() == b"abc"


def test_fetch_uses_download_and_checks_size(tmp_path, monkeypatch):
    monkeypatch.setitem(download.DATASETS, "tiny", 3)
    calls = []
    monkeypatch.setattr(download, "download", lambda url, dest: (calls.append(url),
                                                                  dest.write_bytes(b"abc")))
    p = download.fetch("tiny", tmp_path)
    assert p.read_bytes() == b"abc" and calls == [download.URL.format("tiny")]


def test_download_refuses_non_https_urls(tmp_path):
    import pytest

    for url in ("http://example.invalid/x", "file:///etc/passwd", "ftp://example.invalid/x", "x"):
        with pytest.raises(ValueError, match="non-https"):
            download.download(url, tmp_path / "x.part")
    assert not (tmp_path / "x.part").exists()
    assert download.URL.startswith("https://")
