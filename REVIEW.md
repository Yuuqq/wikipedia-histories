# Wikipedia Histories Code Review

This repository provides tools to pull Wikipedia revision histories and analyze their editors and content. The following code review outlines areas for improvement across architecture, security, reliability, performance, and developer experience.

## CRITICAL

### 1. `aiohttp.ClientSession` recreated per request
**File:** `src/wikipedia_histories/get_histories.py` (line ~89)
**Impact:** In `get_text()`, a new `aiohttp.ClientSession` is created for every single HTTP request. This defeats connection pooling, creates massive overhead, and can lead to socket exhaustion ("Too many open files" errors) when fetching hundreds of revisions concurrently.
**Fix Suggestion:** Pass a single `aiohttp.ClientSession` instance down from `get_texts()` and share it across all `get_text()` coroutines.

### 2. `asyncio.run()` inside synchronous function
**File:** `src/wikipedia_histories/get_histories.py` (line ~186)
**Impact:** `get_history()` calls `asyncio.run(get_texts(...))`. If a user attempts to call this function from an environment where an event loop is already running (such as Jupyter notebooks, FastAPI apps, or Discord bots), it will crash with `RuntimeError: asyncio.run() cannot be called from a running event loop`.
**Fix Suggestion:** Expose an asynchronous `get_history_async()` function natively, or recommend that users rely on async-compatible event loop tools like `nest_asyncio` if mixing synchronous and asynchronous environments.

## HIGH

### 1. Missing backoff/delay on network retries
**File:** `src/wikipedia_histories/get_histories.py` (line ~100)
**Impact:** When an exception occurs in `get_text()`, it immediately recurses to retry without any `await asyncio.sleep()`. This can cause request flooding on a struggling server, increasing the likelihood of IP bans or permanent rate limits.
**Fix Suggestion:** Implement exponential backoff before the recursive retry, e.g., `await asyncio.sleep(2 ** attempts)`.

### 2. Error handling returns `-1` instead of raising exceptions
**File:** `src/wikipedia_histories/get_histories.py` (lines ~101, ~163, ~167)
**Impact:** If a page is missing or a connection error occurs, functions arbitrarily return `-1`. This breaks type contracts (e.g. returning an `int` instead of a `list` of objects), forces callers to add manual type checks, and hides the underlying stack trace.
**Fix Suggestion:** Remove the bare `return -1` and instead raise a custom exception (e.g., `class WikipediaConnectionError(Exception)`) or let the original `ConnectionError` bubble up naturally.

### 3. Full metadata loaded entirely into memory
**File:** `src/wikipedia_histories/get_histories.py` (line ~172)
**Impact:** `metadata = list(page.revisions())` forces all revisions to be pulled into memory at once. For articles with massive edit histories, this will cause significant memory bloat and long pauses before processing starts.
**Fix Suggestion:** Process the iterator directly where possible, or page/chunk the data as it streams in from `mwclient`.

## MEDIUM

### 1. Iterating DataFrame rows (`iterrows()`) instead of vectorizing
**File:** `src/wikipedia_histories/retrieve_metadata.py` (line ~38)
**Impact:** In `get_metadata()`, `df.iterrows()` is used to iterate over texts to calculate word counts and string differences. `iterrows()` is notoriously slow in Pandas.
**Fix Suggestion:** Vectorize the operations. For example: `df['word_count'] = df['text'].astype(str).str.split().str.len()`, followed by `df['word_count'].diff()` for differences.

### 2. Random sampling crash if categories are insufficient
**File:** `src/wikipedia_histories/networks/network_builder.py` (lines ~28, ~33)
**Impact:** The code uses `random.sample(..., 2)` to select categories. If a domain possesses fewer than 2 categories, this will throw a `ValueError` at runtime.
**Fix Suggestion:** Verify the size of the population before sampling, and handle gracefully if there are insufficient elements (e.g., fallback to 1, or skip the domain).

### 3. Path Traversal Risk (Unsanitized domain parameter)
**File:** `src/wikipedia_histories/networks/network_builder.py` (line ~59)
**Impact:** The `domain` variable is formatted directly into a file path: `"{}/{}/{}.csv".format(path, domain, safe_name)`. While `safe_name` is sanitized, `domain` is not. A malicious domain string (e.g., `../..`) could traverse directories.
**Fix Suggestion:** Apply the existing `sanitize_filename` function to the `domain` variable before writing/reading paths.

## LOW

### 1. Use of `__len__()` dunder method explicitly
**File:** `src/wikipedia_histories/get_histories.py` (lines ~60, ~142, ~179)
**Impact:** Calling `metadata.__len__()` or `cur.__len__()` directly is unpythonic and goes against PEP8 conventions.
**Fix Suggestion:** Refactor to use the built-in `len()` function (e.g., `len(metadata)`).

### 2. Gaps in test coverage
**File:** `tests/test_domain_change.py`
**Impact:** The test suite covers basic behaviors in `get_histories.py` but skips the `networks/` submodule and `retrieve_metadata.py`. High-risk transformations remain untested.
**Fix Suggestion:** Expand the test suite using `pytest` fixtures and mock dataframes to cover the network builder and metadata aggregators.

### 3. Loop over metadata multiple times
**File:** `src/wikipedia_histories/get_histories.py` (lines ~173-178)
**Impact:** `_get_users`, `get_kind`, `get_comment`, and the revision IDs loop are all O(N) operations over the exact same `metadata` list, running sequentially in different functions.
**Fix Suggestion:** Combine these metadata extraction operations into a single loop pass for better performance and readability.