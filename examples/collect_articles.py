"""
Example workflow for downloading a set of articles associated with a set of domains, 
represented by categories
"""


import os
import pandas as pd
import wikipedia_histories


def find_articles(domains):
    """
    Download a list of articles titles associated with a domain
    """
    df = wikipedia_histories.networks.get_category_articles.find_articles(
        domains, max_level=2
    )
    return df


def get_article(title):
    """
    Given an article title, download the article and return it as a DataFrame
    """
    cur = wikipedia_histories.get_history(title, include_text=False)
    if cur == -1:
        return -1
    df = wikipedia_histories.to_df(cur)
    return df


def download_articles(df, output_path):
    """
    Download a list of articles based on the find_articles function
    """
    for page, domain in zip(df["Pages"], df["Domain"]):
        article_df = get_article(page)
        # If there was an error in collecting the DataFrame
        if isinstance(article_df, int) and article_df == -1:
            continue
        domain_output = "{}/{}".format(output_path, domain)
        if not os.path.isdir(domain_output):
            os.makedirs(domain_output)

        safe_name = wikipedia_histories.filename_for_title(page)
        article_df.to_csv("{}/{}.csv".format(domain_output, safe_name))

    return 1

def aggregate_metadata(mediums, files_path):
    """
    Aggregate metadata for the articles
    """
    df = []
    for medium in mediums:
        directory = "{}/{}/".format(files_path, medium)
        files = os.listdir(directory)

        for file in files:
            try:
                page_df = pd.read_csv(directory + file)
                # Prefer actual title from inside the CSV if available
                title = page_df.iloc[0]["title"] if not page_df.empty and "title" in page_df.columns else file.rsplit(".", 1)[0]
                row = wikipedia_histories.get_metadata(page_df, title)
            except Exception:
                continue
            row["medium"] = medium
            df.append(row)

    df = pd.DataFrame(df)
    return df
