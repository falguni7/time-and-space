# Time and Space - the demo web app. Small image: FastAPI, the page, the recorded demo (no key needed).
#   docker compose up --build      ->  http://localhost:8000
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src HOST=0.0.0.0 PORT=8000
WORKDIR /app

COPY docker/requirements-app.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/presslens/__init__.py src/presslens/paths.py src/presslens/
COPY src/presslens/web/__init__.py src/presslens/web/server.py src/presslens/web/
COPY src/presslens/web/static src/presslens/web/static
COPY demo/app.json demo/
COPY demo/clips demo/clips
COPY demo/live demo/live

USER nobody
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/demo')"
CMD ["python", "-m", "presslens.web.server"]
