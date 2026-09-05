"""
Given a set of categories representing domains, gather the articles emcompassed by those categories
"""

import pandas as pd
import wikipediaapi


def get_pages_of_cat(
    category, categorymembers, dict_of_cats, level=0, max_level=2, _visited=None
):
    if dict_of_cats is None:
        dict_of_cats = {}
    if _visited is None:
        _visited = set()
    if category in _visited:
        return dict_of_cats
    _visited.add(category)

    pages = []

    for c in categorymembers.values():
        if c.ns == wikipediaapi.Namespace.CATEGORY:
            if level < max_level:
                dict_of_cats = get_pages_of_cat(
                    c.title,
                    c.categorymembers,
                    dict_of_cats=dict_of_cats,
                    level=level + 1,
                    max_level=max_level,
                    _visited=_visited,
                )
            continue
        pages.append((c.title, level))

    dict_of_cats[category] = pages
    return dict_of_cats

def find_articles(domains, max_level=2):
    if max_level < 0:
        raise ValueError("max_level must be non-negative")

    columns = ["Pages", "Level", "Subcategory", "Category", "Domain"]
    if not domains or not any(domains.values()):
        return pd.DataFrame(columns=columns)

    wiki = wikipediaapi.Wikipedia(
        user_agent="wikipedia-histories/1.2.0 (https://github.com/Yuuqq/wikipedia-histories)",
        language="en",
    )

    dfs = []
    for domain in domains:
        for category in domains[domain]:
            cat = wiki.page(category)
            d = get_pages_of_cat(category, cat.categorymembers, {}, max_level=max_level)

            for subcat in d:
                cur_df = pd.DataFrame()
                cur_df["Pages"] = [val[0] for val in d[subcat]]
                cur_df["Level"] = [val[1] for val in d[subcat]]
                cur_df["Subcategory"] = subcat
                cur_df["Category"] = category
                cur_df["Domain"] = domain

                if not cur_df.empty:
                    dfs.append(cur_df)

    if not dfs:
        return pd.DataFrame(columns=columns)

    full_df = pd.concat(dfs, ignore_index=True)

    return full_df
