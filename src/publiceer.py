"""Ingang voor `python -m src.publiceer` (deploy.yml). De code staat in src/serving/publiceer.py.
Blijft hier zolang deploy.yml naar src.publiceer verwijst; dat bestand hoort bij deel 3."""

import sys

from .serving.publiceer import main

if __name__ == "__main__":
    main(sys.argv[1:])
