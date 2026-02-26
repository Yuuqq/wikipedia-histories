import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src import wikipedia_histories
from src.wikipedia_histories.get_histories import (
    _get_revision_content,
    _get_users,
    get_comment,
    get_kind,
)
from src.wikipedia_histories.revision import Revision


# --- Pure unit tests (no network needed) ---


def test_simple_extract_lang_code_from_domain() -> None:
    domain = "en.wikipedia.org"
    lang_code = wikipedia_histories.extract_lang_code_from_domain(domain)
    assert lang_code == "en"


def test_invalid_lang_code_returns_empty_string() -> None:
    domain = "a11.wikipedia.org"
    lang_code = wikipedia_histories.extract_lang_code_from_domain(domain)
    assert lang_code == ""


def test_complex_extract_lang_code_from_domain() -> None:
    domain = "zh-min-nan.wikipedia.org"
    lang_code = wikipedia_histories.extract_lang_code_from_domain(domain)
    assert lang_code == "zh-min-nan"


def test_get_users_basic() -> None:
    metadata = [{"user": "Alice"}, {"user": "Bob"}]
    assert _get_users(metadata) == ["Alice", "Bob"]


def test_get_users_hidden_user() -> None:
    metadata = [{"user": "Alice"}, {"revid": 123}]
    assert _get_users(metadata) == ["Alice", None]


def test_get_kind_minor() -> None:
    metadata = [{"minor": ""}, {"revid": 123}]
    assert get_kind(metadata) == [True, False]


def test_get_comment_basic() -> None:
    metadata = [{"comment": "fixed typo"}, {"revid": 123}]
    assert get_comment(metadata) == ["fixed typo", ""]


def test_get_revision_content_old_format() -> None:
    rev = {"*": "some wikitext", "revid": 123}
    assert _get_revision_content(rev) == "some wikitext"


def test_get_revision_content_mcr_slots_format() -> None:
    rev = {"slots": {"main": {"*": "slot wikitext"}}, "revid": 123}
    assert _get_revision_content(rev) == "slot wikitext"


def test_get_revision_content_empty() -> None:
    rev = {"revid": 123}
    assert _get_revision_content(rev) is None


def test_to_df() -> None:
    changes = [
        Revision(0, "Test", "2021-01-01", 123, False, "Alice", "comment", "NA", "text")
    ]
    df = wikipedia_histories.to_df(changes)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Test"
    assert df.iloc[0]["user"] == "Alice"
    assert df.iloc[0]["text"] == "text"


def test_revision_str() -> None:
    rev = Revision(0, "Test", "2021-01-01", 12345, False, "Alice", "", "NA", "content")
    assert str(rev) == "12345"
    assert repr(rev) == "12345"


# --- Mocked network tests ---


def test_get_text_parses_html() -> None:
    mock_response = {
        "parse": {
            "text": {
                "*": "<div><p>Hello world.</p><p>Second paragraph.</p></div>"
            }
        }
    }

    async def mock_get_text():
        with patch("aiohttp.ClientSession") as MockSession:
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_context.__aexit__ = AsyncMock(return_value=False)
            mock_session.get = MagicMock(return_value=mock_context)
            session_context = AsyncMock()
            session_context.__aenter__ = AsyncMock(return_value=mock_session)
            session_context.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = session_context
            result = await wikipedia_histories.get_text(12345)
            return result

    text = asyncio.run(mock_get_text())
    assert text == "Hello world.Second paragraph."


def test_get_text_deleted_page_returns_none() -> None:
    mock_response = {"error": {"code": "nosuchrevid"}}

    async def mock_get_text():
        with patch("aiohttp.ClientSession") as MockSession:
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_context.__aexit__ = AsyncMock(return_value=False)
            mock_session.get = MagicMock(return_value=mock_context)
            session_context = AsyncMock()
            session_context.__aenter__ = AsyncMock(return_value=mock_session)
            session_context.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = session_context
            result = await wikipedia_histories.get_text(99999999)
            return result

    text = asyncio.run(mock_get_text())
    assert text is None


def test_get_history_returns_list_on_success() -> None:
    mock_metadata = [
        {
            "revid": 100,
            "user": "Alice",
            "comment": "initial",
            "timestamp": (2021, 1, 1, 0, 0, 0, 0, 0, 0),
        }
    ]
    mock_talk_revisions_ts = [
        {"timestamp": (2021, 1, 1, 0, 0, 0, 0, 0, 0)}
    ]
    mock_talk_revisions_content = [
        {"*": "{{WikiProject|class=stub}}", "revid": 1}
    ]

    with patch("src.wikipedia_histories.get_histories.Site") as MockSite:
        mock_site = MagicMock()
        mock_page = MagicMock()
        mock_talk = MagicMock()

        mock_page.revisions.return_value = iter(mock_metadata)
        mock_talk.revisions.side_effect = lambda **kwargs: (
            iter(mock_talk_revisions_content)
            if kwargs.get("prop") == "content"
            else iter(mock_talk_revisions_ts)
        )

        mock_site.pages.__getitem__ = MagicMock(
            side_effect=lambda key: mock_talk if key.startswith("Talk:") else mock_page
        )
        MockSite.return_value = mock_site

        data = wikipedia_histories.get_history(
            "Test Article", include_text=False
        )
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0].title == "Test Article"
        assert data[0].user == "Alice"
        assert data[0].revid == 100


def test_get_history_connection_error_returns_minus_one() -> None:
    with patch(
        "src.wikipedia_histories.get_histories.Site",
        side_effect=ConnectionError("test"),
    ):
        data = wikipedia_histories.get_history("Test", include_text=False)
        assert data == -1
