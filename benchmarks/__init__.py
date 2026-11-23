"""De softwaremetriek: workloads, metingen, software-experimenten, foutgedrag en containers."""

import warnings

# Starlette meldt bij elke import van zijn TestClient dat het httpx2 verkiest boven httpx. Dat
# verandert pas met andere versies, en die liggen vast in requirements-dev.lock.
warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated")
