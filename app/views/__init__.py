"""One module per dashboard page.

Each module exposes a single ``render(context)`` function. ``main.py`` calls
the right one based on the sidebar selection.

The folder is called ``views`` rather than ``pages`` on purpose: Streamlit
reserves a ``pages`` directory next to the entry script for its own
multipage mechanism, and a custom sidebar navigation would end up fighting
with it.
"""

__all__ = [
    "overview",
    "pollution_explorer",
    "relationships",
    "data_quality",
    "anomaly_detection",
    "benzene_regression",
    "methodology",
]
