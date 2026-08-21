# Contributing to ExpoChat

Thanks for your interest in improving the project. This guide covers local setup, the
checks we expect to pass, and how to propose changes.

## Development setup

```bash
cp .env.example .env
chmod 600 .env
make setup      # Python + Node dev dependencies
make check      # fast local gate (lint + Python tests + JS tests)
```

For the full container stack:

```bash
make up         # docker compose up -d
make logs       # follow logs
```

Open `http://127.0.0.1:31001`.

## The local gate

Run this before every push — CI runs the same checks:

```bash
make lint       # Ruff
make test       # Python unit/integration tests (pytest)
make test-js    # Frontend unit tests (node --test)
make audit      # public-release secret/artifact scan
make check      # all fast checks above
make e2e        # optional: isolated fake services + containerized Playwright
```

- Python targets 3.12, Ruff line length 120 (config in `pyproject.toml`).
- Frontend is a no-build ES-module app. Keep pure logic in `public/js/*.mjs` (DOM-free and
  unit-tested); DOM wiring lives in `public/app.js`.
- SQLite migrations in `backend/migrations.py` are ordered and idempotent. **Add new
  migrations; never edit an already-applied one.**

## Pull requests

1. Branch from the default branch.
2. Keep PRs focused; separate unrelated changes.
3. Add or update tests for behavior changes.
4. Ensure `make check` passes.
5. Fill in the pull request template.

## Reporting bugs and requesting features

Open an issue using the provided templates. For security issues, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.

## License

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).
