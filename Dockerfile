# De API (deel 3): multi-stage, slank, niet als root. Zelfde image lokaal (compose) en op de
# Hugging Face Space (poort 7860). Het model komt uit models/actief/ (python -m src.publiceer).
FROM python:3.13.14-slim AS bouw
RUN python -m venv /venv
COPY requirements.lock .
RUN /venv/bin/pip install --no-cache-dir --require-hashes -r requirements.lock

FROM python:3.13.14-slim
RUN useradd --create-home --uid 1000 app
COPY --from=bouw /venv /venv
WORKDIR /app
COPY src/ src/
COPY configs/ configs/
COPY models/actief/ models/actief/
RUN mkdir -p logs && chown -R app:app /app
USER app
ENV PATH=/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OVERBOEKING_MODELMAP=models/actief \
    OVERBOEKING_LOGFORMAAT=json
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
