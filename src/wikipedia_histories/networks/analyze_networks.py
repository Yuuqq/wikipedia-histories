"""
Get the purity score given a network representation of an article
"""
import os
from collections import Counter, defaultdict
from math import nan
from statistics import mean
import pandas as pd

import igraph
import networkx as nx


def get_louvain(g):
    """
    LOUVAIN ALGORITHM
    This is a bottom-up algorithm: initially every vertex belongs to a separate community,
    and vertices are moved between communities iteratively in a way that maximizes the
    vertices' local contribution to the overall modularity score. When a consensus is
    reached (i.e. no single move would increase the modularity score), every community in
    the original graph is shrank to a single vertex (while keeping the total weight of the
    adjacent edges) and the process continues on the next level. The algorithm stops when it
    is not possible to increase the modularity any more after shrinking the communities to vertices.
    """
    weights = g.es["weight"] if "weight" in g.es.attributes() else None
    louvain = g.community_multilevel(weights=weights)
    return louvain


def purity(attribute, louvain, graph):
    """
    Get the average purity score for the two largest networks in a graph given an attribute to check the purity of
    Usually this attribute will be 'category'
    """
    if attribute not in graph.vs.attributes():
        raise ValueError(f"Graph is missing vertex attribute {attribute!r}")

    louvain = sorted((list(group) for group in louvain), key=len, reverse=True)[:2]

    cur_purity = []
    for group in louvain:
        if not group:
            continue
        categories = [graph.vs[node][attribute] for node in group]
        cur_purity.append(max(Counter(categories).values()) / len(categories))

    return mean(cur_purity) if cur_purity else 0.0


def get_assortativity(file, attribute):
    """
    Get weighted categorical assortativity for a graph using NetworkX.
    """
    g = nx.read_graphml(file)

    mixing = defaultdict(float)
    total_weight = 0.0
    for source, target, data in g.edges(data=True):
        try:
            weight = float(data.get("weight", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Graph edge weights must be numeric") from exc
        if weight < 0:
            raise ValueError("Graph edge weights must be non-negative")

        source_type = g.nodes[source][attribute]
        target_type = g.nodes[target][attribute]
        mixing[(source_type, target_type)] += weight
        mixing[(target_type, source_type)] += weight
        total_weight += 2 * weight

    if total_weight == 0:
        return nan

    labels = set()
    row_totals = defaultdict(float)
    column_totals = defaultdict(float)
    for (source_type, target_type), weight in mixing.items():
        labels.update((source_type, target_type))
        row_totals[source_type] += weight
        column_totals[target_type] += weight

    trace = sum(mixing[(label, label)] for label in labels) / total_weight
    expected = sum(
        (row_totals[label] / total_weight)
        * (column_totals[label] / total_weight)
        for label in labels
    )
    denominator = 1 - expected
    return (trace - expected) / denominator if denominator else nan


def get_purity(file, attribute):
    """
    Get the purity of a graph using igraph
    """
    g = igraph.load(file)
    louv = get_louvain(g)
    p = purity(attribute, louv, g)
    return p


def get_network_metadata(
    network_path,
    attribute="category",
    mediums=None,
):

    if mediums is None:
        if not os.path.isdir(network_path):
            raise FileNotFoundError(network_path)
        mediums = sorted(
            name
            for name in os.listdir(network_path)
            if not name.startswith(".")
            and os.path.isdir(os.path.join(network_path, name))
        )
    elif isinstance(mediums, str):
        mediums = [mediums]

    rows = []
    for medium in mediums:
        directory = os.path.join(network_path, medium)
        if not os.path.isdir(directory):
            continue

        files = sorted(
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if not filename.startswith(".")
            and filename.lower().endswith((".graphml", ".graphmlz"))
        )

        for file in files:
            cur = {}
            p = get_purity(file, attribute)
            a = get_assortativity(file, attribute)
            cur["assortativity"] = a
            cur["purity"] = p
            cur["medium"] = medium
            rows.append(cur)

    df = pd.DataFrame(rows, columns=["assortativity", "purity", "medium"])

    return df
