from typing import Protocol


class Citable(Protocol):
    """A Citable object provides a method to obtain a formatted citation string"""

    def how_to_cite(self, format="bibtex") -> str:
        ...
