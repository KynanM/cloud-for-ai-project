"""Ingang voor `uvicorn src.api:app` (het Dockerfile en de Space). De API zelf staat in
src/serving/api.py. Blijft hier zolang het Dockerfile en deploy.yml naar src.api verwijzen; die
horen bij deel 3 en vallen buiten deze herstructurering."""

from .serving.api import app  # noqa: F401
