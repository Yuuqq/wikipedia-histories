# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-07-08

### Added
- `sanitize_filename()` function to safely handle Wikipedia page titles containing filesystem-unsafe characters (/, \, :, *, ?, ", <, >, |, #, [], {}).
  - Applied consistently when saving article CSVs (`examples/collect_articles.py`) and loading them for network building (`src/wikipedia_histories/networks/network_builder.py`).
- Expanded test coverage:
  - New `test_sanitize_filename_basic()` covering edge cases (special chars, long titles, empty/None input).
  - New `test_get_metadata_new_columns()` validating the fixed column names from `to_df`.
- Modernized test imports in `tests/test_domain_change.py` (removed `from src import` hacks) for clean compatibility with `pip install -e .`.
- Exposed `sanitize_filename`, `generate_networks`, `get_network_metadata`, and `find_articles` at the top-level package for easier imports.

### Changed
- **CI Optimization** (`.github/workflows/python-package.yml`):
  - Upgraded to `actions/checkout@v4` and `actions/setup-python@v5` with `cache: 'pip'`.
  - Expanded Python test matrix to 3.9, 3.10, 3.11, 3.12.
  - Use `pip install -e "[networks]"` for proper editable install + extras in CI.
  - Cleaner pytest execution (`pytest tests/ -q --tb=line`).
- Bumped package version to **1.2.0** in `setup.py` and updated repository URL to current fork.
- Improved `get_metadata()` robustness (handles both datetime objects and strings, added empty-list guards for addition/deletion stats).
- `aggregate_metadata()` in examples now prefers the real `title` from inside each CSV.
- Centralized `User-Agent` constant (`UA`) in `get_histories.py` and applied to both `mwclient.Site` and `aiohttp` requests for better Wikipedia API compliance.

### Fixed
- **Critical column name mismatches**: `retrieve_metadata.get_metadata()` and related functions now correctly use the lowercase columns produced by `to_df()` (`time`, `text`, `user`, `rating` instead of legacy `Time`/`Content`/`User`/`Rating`).
- Network functionality bugs:
  - `generate_networks(..., domain=None, write=True)` no longer crashes on `None` directory name (uses `"cross_domain"`).
  - Fixed parameter name (`output_path` vs `output_folder`) and import paths in `examples/collect_networks.py`.
  - `get_users()` now uses sanitized filenames consistently.
- Missing/incorrect User-Agent headers on Wikipedia API calls (could cause throttling or blocks per Wikimedia policy).
- Various import and path inconsistencies between `README.md`, examples, and the restructured `networks/` subpackage.
- Minor robustness improvements in `convert_to_datetime()` fallback and error handling in metadata aggregation.

### Notes
- This release focuses on making the full workflow (article collection → metadata → network generation) reliable and production-ready.
- The February 2026 sync PR (v1.1.x) already addressed core MediaWiki API compatibility and added initial tests; this release builds on that foundation with polish and missing features.

## [1.1.0] - 2026-02-26

- Sync with current MediaWiki APIs and bug fixes (MCR slots handling, etc.).
- Added unit tests and CI improvements (previous work).

[Unreleased]: https://github.com/Yuuqq/wikipedia-histories/compare/main...HEAD
[1.2.0]: https://github.com/Yuuqq/wikipedia-histories/releases/tag/v1.2.0
[1.1.0]: https://github.com/Yuuqq/wikipedia-histories/releases/tag/v1.1.0
