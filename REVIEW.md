# Wikipedia Histories Code Review

This repository provides tools to pull Wikipedia revision histories and analyze their editors and content. The following code review outlines areas for improvement across architecture, security, reliability, performance, and developer experience.

## CRITICAL

### 1. HTTP 429 (Too Many Requests) treated as deleted page
**File:** `src/wikipedia_histories/get_histories.py` (line ~149)
**Impact:** `get_text()` catches JSON parsing errors and assumes a `None` result means the page was deleted. However, if the server replies with a 429 Too Many Requests (or another non-200 HTML error page), it will fail to parse JSON and also return `None`. This falsely flags active revisions as deleted and permanently loses data during rate limits.
**Fix Suggestion:** Add `resp.raise_for_status()` immediately after fetching the response, so HTTP errors correctly raise `aiohttp.ClientResponseError` and trigger the retry block.

### 2. Missing HTTP timeout on requests
**File:** `src/wikipedia_histories/get_histories.py` (line ~145)
**Impact:** `session.get()` is called without a timeout. In asynchronous contexts with many concurrent requests, stalled network connections can hang indefinitely, locking up the coroutines and exhausting connections.
**Fix Suggestion:** Add an explicit timeout to the request, e.g., `timeout=30`, or set it globally on the `ClientSession`.

## HIGH

### 1. `aiohttp.ClientSession` recreated per request
**File:** `src/wikipedia_histories/get_histories.py` (line 144)
**Impact:** In `get_text()`, a new `aiohttp.ClientSession` is created for every single HTTP request. This defeats connection pooling, creates massive overhead, and can lead to socket exhaustion ("Too many open files" errors) when fetching hundreds of revisions concurrently.
**Fix Suggestion:** Pass a single `aiohttp.ClientSession` instance down from `get_texts()` and share it across all `get_text()` coroutines.

### 2. `asyncio.run()` inside synchronous function
**File:** `src/wikipedia_histories/get_histories.py` (line 233)
**Impact:** `get_history()` calls `asyncio.run(get_texts(...))`. If a user attempts to call this function from an environment where an event loop is already running (such as Jupyter notebooks, FastAPI apps, or Discord bots), it will crash with `RuntimeError: asyncio.run() cannot be called from a running event loop`.
**Fix Suggestion:** Expose an asynchronous `get_history_async()` function natively that avoids `asyncio.run()`, and keep `get_history()` as a synchronous wrapper for backwards compatibility.

### 3. Missing backoff/delay on network retries
**File:** `src/wikipedia_histories/get_histories.py` (line 156)
**Impact:** When an exception occurs in `get_text()`, it immediately recurses to retry without any `await asyncio.sleep()`. This can cause request flooding on a struggling server, increasing the likelihood of IP bans or permanent rate limits.
**Fix Suggestion:** Implement exponential backoff before the recursive retry, e.g., `await asyncio.sleep(2 ** attempts)`.

### 4. Error handling returns `-1` instead of raising exceptions
**File:** `src/wikipedia_histories/get_histories.py` (lines 154, 211, 215)
**Impact:** If a page is missing or a connection error occurs, functions arbitrarily return `-1`. This breaks type contracts (e.g. returning an `int` instead of a `list` of objects), forces callers to add manual type checks, and hides the underlying stack trace.
**Fix Suggestion:** Remove the bare `return -1` and instead raise a custom exception (e.g., `class WikipediaConnectionError(Exception)`) or let the original `ConnectionError` bubble up naturally.

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
**File:** `src/wikipedia_histories/networks/network_builder.py` (lines ~59, ~133)
**Impact:** The `domain` variable is formatted directly into a file path: `"{}/{}/{}.csv".format(path, domain, safe_name)`. A malicious domain string (e.g., `../..`) could traverse directories.
**Fix Suggestion:** Apply `os.path.basename(domain)` or similar strict sanitization to the `domain` variable before writing/reading paths. `sanitize_filename` is insufficient because it preserves `..`.

## LOW

### 1. Use of `__len__()` dunder method explicitly
**File:** `src/wikipedia_histories/get_histories.py` (lines 102, 192, 224, 238)
**Impact:** Calling `metadata.__len__()` or `cur.__len__()` directly is unpythonic and goes against PEP8 conventions.
**Fix Suggestion:** Refactor to use the built-in `len()` function (e.g., `len(metadata)`).

### 2. Gaps in test coverage
**File:** `tests/`
**Impact:** The test suite covers basic behaviors in `get_histories.py` and `retrieve_metadata.py` but completely skips the `networks/` submodule. High-risk transformations remain untested.
**Fix Suggestion:** Expand the test suite using `pytest` fixtures and mock dataframes to cover the network builder.

### 3. Loop over metadata multiple times
**File:** `src/wikipedia_histories/get_histories.py` (lines 220)
**Impact:** `_get_users`, `get_kind`, `get_comment`, and the revision IDs loop are all O(N) operations over the exact same `metadata` list, running sequentially in different functions.
**Fix Suggestion:** Combine these metadata extraction operations into a single loop pass for better performance and readability.

### 4. Full metadata loaded entirely into memory
**File:** `src/wikipedia_histories/get_histories.py` (line 219)
**Impact:** `metadata = list(page.revisions())` forces all revisions to be pulled into memory at once. For articles with massive edit histories, this will cause memory bloat.
**Fix Suggestion:** While streaming is difficult with the current architecture, consider paging or chunking the data as it streams in from `mwclient` in a future refactor.
