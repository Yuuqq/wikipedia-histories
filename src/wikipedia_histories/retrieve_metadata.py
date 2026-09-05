"""
Based on a set of downloaded articles represented as CSV's, generate a metadata for those articles.
"""

from datetime import datetime, timezone
from statistics import mean

import pandas as pd


def get_time_diff(prev_time, cur_time):
    try:
        if prev_time is not None and cur_time is not None:
            time_diff = cur_time - prev_time
            return time_diff.total_seconds() / 3600
        return None
    except (AttributeError, TypeError, ValueError):
        return None


def convert_to_datetime(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, datetime):
        result = value
    else:
        try:
            result = pd.to_datetime(value, errors="raise").to_pydatetime()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Unable to parse timestamp: {value!r}") from exc

    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def _word_count(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and value == -1:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    return len(str(value).split())


def get_metadata(df, title):
    result = {
        "title": title,
        "edit_count": len(df),
        "added_words_per_edit": 0,
        "deleted_words_per_edit": 0,
        "article_age_hours": None,
        "unique_editors": 0,
    }
    if df.empty:
        return result

    required_columns = {"time", "text", "user"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required metadata columns: {missing}")

    work = df.copy()
    work["time"] = work["time"].apply(convert_to_datetime)
    work = work.sort_values("time", kind="mergesort", na_position="last")

    addition_lengths = []
    deletion_lengths = []
    prev_count = 0
    content_complete = True

    for _, row in work.iterrows():

        word_count = _word_count(row["text"])
        if word_count is None:
            content_complete = False
            prev_count = None
            continue

        if prev_count is None:
            content_complete = False
            prev_count = word_count
            continue

        if word_count < prev_count:
            deletion_lengths.append(prev_count - word_count)
        else:
            addition_lengths.append(word_count - prev_count)

        prev_count = word_count

    valid_times = work["time"].dropna()
    age = get_time_diff(valid_times.iloc[0], valid_times.iloc[-1]) if not valid_times.empty else None

    if not content_complete:
        deletion_length = None
        addition_length = None
    elif not deletion_lengths:
        deletion_length = 0
    else:
        deletion_length = mean(deletion_lengths)

    if content_complete:
        if not addition_lengths:
            addition_length = 0
        else:
            addition_length = mean(addition_lengths)

    result.update(
        {
            "added_words_per_edit": addition_length,
            "deleted_words_per_edit": deletion_length,
            "article_age_hours": age,
            "unique_editors": int(work["user"].dropna().nunique()),
        }
    )
    return result

def rating_meta(df):
    if "rating" not in df.columns or df.empty:
        return {}

    ratings = {}
    cur_ratings = df["rating"].value_counts()

    for rating in cur_ratings.keys():
        normalized = str(rating).strip().lower()
        ratings[normalized] = ratings.get(normalized, 0) + int(cur_ratings[rating])

    return ratings
