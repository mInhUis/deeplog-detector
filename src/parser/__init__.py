"""Parser package — Drain log-template extraction.

Public API:
    DrainParser — Drain algorithm that converts unstructured log messages
                  into integer log keys and text templates.
"""

from src.parser.drain_parser import DrainParser

__all__: list[str] = [
    "DrainParser",
]
