# recommendation service (demo)

This Python service reproduces recommendation faults for diagnosis and regression
testing. Run `python recommendation_server.py` to start the HTTP server on port
8080 and a background thread that generates requests.

## Endpoints

- `GET /recommend` — top-5 recommendations (catalog match, excluding the input ids)
- `GET /metrics` — Prometheus text exposition
- `GET /healthz` — liveness

## Build / test
```bash
pip install -r requirements.txt
python -m py_compile $(git ls-files '*.py')
pytest -q
```
