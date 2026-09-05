import asyncio
import hashlib
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import aiohttp
import mwparserfromhell as mw
import pandas as pd
from lxml import html
from mwclient import Site
from requests.exceptions import ConnectionError

from .revision import Revision


UA = "wikipedia-histories/1.2.0 (https://github.com/Yuuqq/wikipedia-histories)"
TEXT_UNAVAILABLE = None


def _coerce_datetime(value):
    """Convert MediaWiki timestamps and common serialized values to UTC-naive datetimes."""
    if value is None:
        return None

    if isinstance(value, datetime):
        result = value
    elif isinstance(value, (tuple, list)):
        if len(value) < 6:
            return None
        try:
            result = datetime(*(int(part) for part in value[:6]))
        except (TypeError, ValueError):
            return None
    elif isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        try:
            result = value.to_pydatetime()
        except AttributeError:
            return None
        if not isinstance(result, datetime):
            return None

    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def _get_users(metadata):
    """
    Pull users, handles hidden user errors
    Parameters:
        metadata: sheet of metadata from mwclient
    Returns:
        the list of users
    """
    users = []
    for rev in metadata:
        try:
            users.append(rev["user"])
        except (KeyError):
            users.append(None)
    return users


def get_kind(metadata):
    """
    Gather edit types (minor or not), handles untagged edits
    Parameters:
        metadata: sheet of metadata from mwclient
    Returns:
        list of True/False representing whether an edit is minor
    """
    kind = []
    for rev in metadata:
        if "minor" not in rev:
            kind.append(False)
            continue
        value = rev.get("minor")
        kind.append(value == "" or bool(value))
    return kind


def get_comment(metadata):
    """
    Check for comments
    Parameters:
        metadata: sheet of metadata from mwclient
    Returns:
        The comments as a list
    """
    comment = []
    for rev in metadata:
        try:
            comment.append(rev["comment"])
        except KeyError:
            comment.append("")
    return comment


def _get_revision_content(rev):
    """
    Extract content text from a revision, handling both old format and MCR (slots) format.
    Parameters:
        rev: revision dict from mwclient
    Returns:
        The wikitext content string, or None if not found
    """
    if not isinstance(rev, Mapping):
        return None
    if "slots" in rev:
        try:
            return rev["slots"]["main"]["*"]
        except (KeyError, TypeError):
            pass
    return rev.get("*")


def get_ratings(talk):
    """
    Output classes of a page to a list (FA, good, etc.) given a talk page
    Parameters:
        talk: set of talk pages from metadata
    Returns:
        The ratings and timestamps for a page
    """
    timestamp_revisions, content_revisions = _get_talk_revisions(talk)
    timestamps_by_revid = {
        rev.get("revid"): rev.get("timestamp")
        for rev in timestamp_revisions
        if isinstance(rev, Mapping) and rev.get("revid") is not None
    }
    ratings = []

    prev = None
    for index, cur in enumerate(content_revisions):
        if (
            isinstance(cur, Mapping)
            and len(cur) == 1
            and _get_revision_content(cur) is None
        ):
            version = prev
        else:
            version = cur

        prev = cur

        if isinstance(cur, Mapping):
            raw_timestamp = timestamps_by_revid.get(cur.get("revid"))
        else:
            raw_timestamp = None
        if raw_timestamp is None and index < len(timestamp_revisions):
            timestamp_revision = timestamp_revisions[index]
            if isinstance(timestamp_revision, Mapping):
                raw_timestamp = timestamp_revision.get("timestamp")
        if raw_timestamp is None and isinstance(version, Mapping):
            raw_timestamp = version.get("timestamp")

        try:
            text = _get_revision_content(version)
            if text is None:
                continue
            templates = mw.parse(text).filter_templates()
        except (IndexError, TypeError, ValueError):
            continue

        rate = "NA"
        for template in templates:
            try:
                parameter = template.get("class")
                value = str(parameter.value).strip()
                if value:
                    rate = value
                    break
            except (AttributeError, TypeError, ValueError):
                continue

        rating_time = _coerce_datetime(raw_timestamp)
        if rating_time is not None:
            ratings.append((rate, rating_time))

    return ratings


def _get_talk_revisions(talk):
    """Load talk revision metadata and content with stable revision identifiers."""
    try:
        revisions = list(
            talk.revisions(prop="ids|timestamp|content", slots="main")
        )
    except TypeError:
        revisions = list(talk.revisions())
        return revisions, list(talk.revisions(prop="content"))

    if revisions and not any(
        _get_revision_content(revision) is not None for revision in revisions
    ):
        legacy_content = list(talk.revisions(prop="content"))
        if any(
            _get_revision_content(revision) is not None
            for revision in legacy_content
        ):
            return revisions, legacy_content

    return revisions, revisions


async def get_text(revid, attempts=0, lang_code="en"):
    """
    Pull plain text representation of a revision from API
    Parameters:
        revid: revision id of a page
        attempts: The number of attempts at retrieving the id so far
    """
    try:
        # async implementation of requests get
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://{lang_code}.wikipedia.org/w/api.php",
                params={"action": "parse", "format": "json", "oldid": revid,},
                headers={"User-Agent": UA},
            ) as resp:
                status = getattr(resp, "status", 200)
                if isinstance(status, int) and status >= 400:
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        status=status,
                    )
                response = await resp.json()
    # request errors from server
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
        if attempts >= 10:
            return -1
        # If there's a server error, just re-send the request until the server complies
        return await get_text(revid, attempts=attempts + 1, lang_code=lang_code)
    # Check if page was deleted (deleted pages have no text and are therefore un-parsable)
    if not isinstance(response, Mapping):
        return -1
    if "error" in response:
        error_code = response["error"].get("code") if isinstance(response["error"], Mapping) else None
        if error_code in {"nosuchrevid", "missingtitle", "invalidrevid"}:
            return None
        return -1

    try:
        raw_html = response["parse"]["text"]["*"]
    except (KeyError, TypeError):
        return None
    # Parse raw html from response
    document = html.document_fromstring(raw_html)
    text = document.xpath("//p")
    paragraphs = []
    for paragraph in text:
        paragraphs.append(paragraph.text_content())

    # Put everything together
    cur = "\n".join(paragraphs)

    return cur


async def get_texts(revids, lang_code="en"):
    """
    Get the text of articles given the list of revision ids

    Parameters:
        revids: A list of revids (type int) correlating to article revisions
    Returns:
        The text for each revision id
    """
    if not revids:
        return []

    # Keep API fan-out bounded so a large history does not trigger rate limits.
    sema = asyncio.Semaphore(3)

    async def fetch(revid):
        async with sema:
            return await get_text(revid, lang_code=lang_code)

    return await asyncio.gather(*(fetch(revid) for revid in revids))


def _run_async(coroutine):
    """Run a coroutine from both ordinary synchronous code and an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()

def get_history(title, include_text=True, domain="en.wikipedia.org"):
    """
    Collects everything and returns a list of Change objects

    Parameters:
        title: article title
        include_text: Whether to unclude body text or not. Speed increases if False
    Returns:
        A list of Change objects representing each revision to the
    """

    # Load the article
    try:
        site = Site(domain, clients_useragent=UA)
        page = site.pages[title]
    except (ConnectionError, OSError):
        return -1
    try:
        talk = site.pages["Talk:" + title]
    except (ConnectionError, OSError):
        return -1
    try:
        ratings = get_ratings(talk)

        # Collect metadata information
        metadata = list(page.revisions())
    except (ConnectionError, OSError):
        return -1
    users = _get_users(metadata)
    kind = get_kind(metadata)
    comments = get_comment(metadata)

    # Collect revision ids using the metadata pull.
    revids = [revision.get("revid") for revision in metadata]

    # Get the text of the revisions. Performance is improved if this isn't done, but you lose the revisions
    if include_text:
        lang_code = extract_lang_code_from_domain(domain)
        texts = _run_async(get_texts(revids, lang_code))
    else:
        texts = [TEXT_UNAVAILABLE] * len(metadata)

    # Iterate backwards through our metadata and put together the list of change items
    history = []
    for i in range(len(metadata) - 1, -1, -1):
        # Iterate against talk page editions.
        time = _coerce_datetime(metadata[i].get("timestamp"))
        rating = "NA"

        for item in ratings:
            if time is not None and time >= item[1]:
                rating = item[0]
                break

        change = Revision(
            i,
            title,
            time,
            metadata[i]["revid"],
            kind[i],
            users[i],
            comments[i],
            rating,
            texts[i],
        )

        # Compile the list of changes
        history.append(change)

    return history

def to_df(changes):
    """
    Make a dataframe out of the change objects

    Parameters:
        changes: A list of changes
    Returns:
        A DataFrame representation of the changes
    """
    columns = ["title", "time", "revid", "kind", "user", "comment", "rating", "text"]
    df = []

    for change in changes:
        row = dict(
            title=change.title,
            time=change.time,
            revid=change.revid,
            kind=change.kind,
            user=change.user,
            comment=change.comment,
            rating=change.rating,
            text=change.content,
        )
        df.append(row)
    return pd.DataFrame(df, columns=columns)

def extract_lang_code_from_domain(domain: str) -> str:
    if not isinstance(domain, str):
        return ""
    match = re.fullmatch(
        r"([a-z]+(?:-[a-z]+)*)\.wikipedia\.org\.?", domain.strip().lower()
    )
    if match:
        return match.group(1)
    return ""

def sanitize_filename(title: str) -> str:
    """Sanitize a Wikipedia page title to a safe cross-platform filename.

    Replaces characters that are invalid or problematic on filesystems
    (slash, backslash, :, *, ?, ", <, >, |, #, [], {}) with underscores, collapses
    repeated underscores, and limits the result to ~200 characters.
    """
    if not title or not isinstance(title, str):
        return "untitled"
    # Replace filesystem-unsafe chars (Wikipedia titles can contain some of these)
    safe = re.sub(r'[\\/:*?"<>|#{}\[\]\x00-\x1f]', '_', title)
    # Collapse multiple underscores and remove path-only names.
    safe = re.sub(r'_+', '_', safe).strip()
    safe = safe.strip(".")
    # Limit length for filesystem safety (most FS have 255 byte limit)
    if len(safe) > 200:
        safe = safe[:200]
        if not safe.endswith("_"):
            safe = safe[:199] + "_"

    if safe.upper().split(".", 1)[0] in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        safe = "_" + safe

    return safe if safe else "untitled"


def filename_for_title(title: str) -> str:
    """Return a collision-resistant filename while preserving safe legacy names."""
    safe = sanitize_filename(title)
    if not isinstance(title, str) or not title or safe == title:
        return safe

    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]
    suffix = f"_{digest}"
    return f"{safe[:200 - len(suffix)]}{suffix}"
