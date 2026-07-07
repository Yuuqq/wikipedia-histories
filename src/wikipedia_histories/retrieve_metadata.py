"""
Based on a set of downloaded articles represented as CSV's, generate a metadata for those articles.
"""

import pandas as pd
import os
from statistics import mean
import time
from datetime import datetime


def get_time_diff(prev_time, cur_time):
    try:
        if prev_time is not None and cur_time is not None:
            time_diff = cur_time - prev_time
            return time_diff.total_seconds() / 3600
        return None
    except TypeError:
        return None


def convert_to_datetime(time):
    if isinstance(time, datetime):
        return time
    try:
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        # fallback for other formats or already dt
        return pd.to_datetime(time).to_pydatetime()


def get_metadata(df, title):
    addition_lengths = []
    deletion_lengths = []
    time_diffs = []
    prev_count = 0
    prev_time = None

    prev_quality = str(df.iloc[0]["rating"]).strip().lower() if "rating" in df.columns else "na"
    prev_quality_time = df.iloc[0]["time"]
    rating_change_times = []

    df["time"] = df["time"].apply(convert_to_datetime)

    for i, row in df.iterrows():

        word_count = len(str(row["text"]).split())

        if word_count < prev_count:
            deletion_lengths.append(prev_count - word_count)
        else:
            addition_lengths.append(word_count - prev_count)

        prev_count = word_count

        cur_time = row["time"]
        # time_diff = get_time_diff(prev_time, cur_time)
        # time_diffs.append(time_diff)
        prev_time = cur_time

        # if str(row["rating"]).strip().lower() != prev_quality or i == len(df) - 1:
        #     time_to_change = get_time_diff(prev_quality_time, cur_time)
        #     rating_change_times.append(time_to_change)
        #     prev_quality_time = cur_time
        #     prev_quality = str(row["rating"]).strip().lower()

    age = get_time_diff(df.iloc[0]["time"], df.iloc[len(df) - 1]["time"])

    if not deletion_lengths:
        deletion_length = 0
    else:
        deletion_length = mean(deletion_lengths)

    if not addition_lengths:
        addition_length = 0
    else:
        addition_length = mean(addition_lengths)

    row = {
        "title": title,
        "edit_count": len(df),
        "added_words_per_edit": addition_length,
        "deleted_words_per_edit": deletion_length,
        # "hours_between_edits": mean(time_diffs),
        # "rating_change_times": mean(rating_change_times),
        "article_age_hours": age,
        "unique_editors": len(df["user"].unique()),
    }

    return row

def rating_meta(df):
    ratings = {}
    cur_ratings = df["rating"].value_counts()

    for rating in cur_ratings.keys():
        try:
            ratings[str(rating).strip().lower()] += cur_ratings[rating]
        except Exception as e:
            ratings[str(rating).strip().lower()] = cur_ratings[rating]

    return ratings
