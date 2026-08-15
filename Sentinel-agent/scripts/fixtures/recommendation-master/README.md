# recommendation service (demo)

This Python service reproduces recommendation faults for diagnosis and regression
testing. Run `python recommendation_server.py` to start the HTTP server on port
8080 and a background thread that generates requests.

## Endpoints

- `GET /recommend` — top-5 recommendations: catalog match + merchandising
  trending ids, deduped, ranked by popularity
- `GET /catalog/<page>/<page_size>` — paginated catalog browsing (1-based page)
- `GET /trending` — merchandising-curated trending ids (cached, 60s TTL)
- `GET /price/<product_id>/<discount_pct>` — list price & member price, in cents
- `GET /metrics` — Prometheus text exposition
- `GET /healthz` — liveness

## Build / test
```bash
pip install -r requirements.txt
python -m py_compile $(git ls-files '*.py')
pytest -q
```

Known open issues are tracked in [BUGS.md](BUGS.md) — please fix **one issue
per PR** and don't bundle unrelated changes.
