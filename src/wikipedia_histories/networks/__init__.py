"""Optional network-analysis helpers.

The network builder and analysis modules depend on the ``networks`` extra.
Keep those imports lazy so the core history API remains usable without it.
"""

__all__ = ["generate_networks", "get_network_metadata", "find_articles"]


def __getattr__(name):
    if name == "generate_networks":
        from .network_builder import generate_networks

        return generate_networks
    if name == "get_network_metadata":
        from .analyze_networks import get_network_metadata

        return get_network_metadata
    if name == "find_articles":
        from .get_category_articles import find_articles

        return find_articles
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
