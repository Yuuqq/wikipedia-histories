from .get_histories import (
    get_history,
    to_df,
    get_text,
    get_texts,
    extract_lang_code_from_domain,
    sanitize_filename,
)

from .retrieve_metadata import get_metadata

from . import networks

# Also expose networks convenience functions at top level for ease of use
try:
    from .networks.network_builder import generate_networks
    from .networks.analyze_networks import get_network_metadata
    from .networks.get_category_articles import find_articles
except ImportError:
    pass  # networks extra not installed
