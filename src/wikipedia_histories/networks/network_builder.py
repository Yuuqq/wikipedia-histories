"""
Network builder
In the generated graph: Each node represents an article, and each edge represents whether
the connected nodes share an editor. The weight of the edges is the number of users who edited
both articles.

for instance, if three total users edited both "The Dark Knight" and "Game of Thrones", then there is an
edge between those two nodes with a weight of three.
"""
import os
import random
from itertools import combinations
from numbers import Integral

import pandas as pd
import networkx as nx

# Import sanitize from parent package (centralized)
from ..get_histories import filename_for_title, sanitize_filename


def get_documents(domain, size, metadata_path):
    """
    Sample an equal number of articles from two different domains
    :param domain: can be 'sciences', 'sports', 'politics', or 'culture'
    :param size: The number of documents to collect
    :param metadata_path: The path to the metadata sheet
    """

    if not isinstance(size, Integral) or size <= 0 or size % 2:
        raise ValueError("size must be a positive even integer")
    if metadata_path is None:
        raise ValueError("metadata_path is required")

    df = pd.read_csv(metadata_path)
    required_columns = {"Domain", "Category", "Pages"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Metadata is missing required columns: {missing}")

    per_category = size // 2

    # pick two random submediums from which to draw documents if we're picking from all
    # (e.g. ('democrat', 'biology') or ('republican', 'democrat'))
    if domain is not None:
        df = df.loc[df["Domain"] == domain]
        categories = df["Category"].dropna().unique().tolist()
        if len(categories) < 2:
            raise ValueError(f"Domain {domain!r} must contain at least two categories")
        selected_categories = random.sample(categories, 2)
        selections = [(domain, category) for category in selected_categories]

    else:
        # Select two categories from different domains
        available_domains = df["Domain"].dropna().unique().tolist()
        if len(available_domains) < 2:
            raise ValueError("Metadata must contain at least two domains")
        selected_domains = random.sample(available_domains, 2)

        selections = []
        for selected_domain in selected_domains:
            categories = (
                df.loc[df["Domain"] == selected_domain, "Category"]
                .dropna()
                .unique()
                .tolist()
            )
            if not categories:
                raise ValueError(f"Domain {selected_domain!r} has no categories")
            selections.append((selected_domain, random.choice(categories)))

    # clear out rows of the dataframe which don't match the selected types
    selected_frames = []
    for selected_domain, category in selections:
        cur = df.loc[
            (df["Domain"] == selected_domain) & (df["Category"] == category)
        ]
        if len(cur) < per_category:
            raise ValueError(
                f"Category {category!r} in domain {selected_domain!r} has only "
                f"{len(cur)} article(s); {per_category} required"
            )
        selected_frames.append(cur.sample(n=per_category))

    return pd.concat(selected_frames, ignore_index=True)

def get_users(name, domain, path):
    """
    Get the list of users for an article given the dataframe

    :param name: the name of an article
    :param domain: the domain the article is a member of
    """
    if path is None:
        return None

    safe_names = [filename_for_title(name), sanitize_filename(name)]
    for safe_name in dict.fromkeys(safe_names):
        fpath = os.path.join(path, str(domain), f"{safe_name}.csv")
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        if "user" not in df.columns:
            return None
        return [user for user in df["user"].tolist() if _is_valid_user(user)]

    # In case the data isn't there
    return None

def intersection(lst1, lst2):
    """
    Get the intersection of two lists, O(n) time
    """

    if not lst1 or not lst2:
        return []

    second = {value for value in lst2 if _is_valid_user(value)}
    result = []
    seen = set()
    for value in lst1:
        if _is_valid_user(value) and value in second and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _is_valid_user(value):
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True

def build_graph(df, path):
    """
    Get the list of users for every article selected by the document selector

    :param df: A dataframe of selected articles (equal numbers from each domain)
    """
    required_columns = {"Pages", "Domain", "Category"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Document data is missing required columns: {missing}")

    df = df.loc[:, ["Pages", "Domain", "Category"]].copy()
    df["Users"] = df.apply(
        lambda row: get_users(row["Pages"], row["Domain"], path), axis=1
    )  # get the user lists for each page

    df = df.dropna(subset=["Users"])
    df = df.drop_duplicates(subset=["Pages"], keep="first")

    g = nx.Graph()
    # one node for every novel/film/tv
    g.add_nodes_from(list(df["Pages"]))

    attrs = {}
    for i, row in df.iterrows():
        attrs[row["Pages"]] = {"domain": row["Domain"], "category": row["Category"]}

    nx.set_node_attributes(g, attrs)

    for (_, row1), (_, row2) in combinations(df.iterrows(), 2):
        node1 = row1["Pages"]
        users1 = row1["Users"]
        node2 = row2["Pages"]
        users2 = row2["Users"]
        common_users = intersection(users1, users2)
        if common_users:
            g.add_edge(node1, node2, weight=len(common_users))

    return g

def generate_networks(
    count=50,
    size=100,
    domain=None,
    write=False,
    output_path=None,
    metadata_path=None,
    articles_path=None,
):
    """
    Generate networks for a set of mediums

    :param count: The number of articles to be referenced
    """
    if not isinstance(count, Integral) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if count == 0:
        return []
    if metadata_path is None or articles_path is None:
        raise ValueError("metadata_path and articles_path are required")
    if write and output_path is None:
        raise ValueError("output_path is required when write=True")

    graphs = []
    for i in range(0, count):
        documents = get_documents(domain, size, metadata_path)
        g = build_graph(documents, articles_path)

        if write:
            domain_dir = domain if domain is not None else "cross_domain"
            dir_path = os.path.join(output_path, domain_dir)
            os.makedirs(dir_path, exist_ok=True)
            out = os.path.join(dir_path, f"{i}.GraphML")
            nx.write_graphml(g, out)
        graphs.append(g)

    return graphs
