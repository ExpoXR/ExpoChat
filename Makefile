PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
NODE ?= $(if $(wildcard .node/bin/node),.node/bin/node,node)
NPM ?= $(if $(wildcard .node/bin/npm),PATH="$(CURDIR)/.node/bin:$$PATH" npm,npm)
PLAYWRIGHT_IMAGE ?= mcr.microsoft.com/playwright:v1.61.1-noble
E2E_BASE_URL ?= http://127.0.0.1:31002

.PHONY: setup lint test test-js audit e2e check up logs smoke backup

setup:
	python3 -m venv .venv
	$(PIP) install -r requirements-dev.txt
	$(NPM) ci

lint:
	.venv/bin/ruff check app.py backend tests scripts/public_release_audit.py

test:
	$(PYTHON) -m pytest -q

test-js:
	$(NODE) --check public/app.js
	@for file in public/js/*.mjs; do $(NODE) --check "$$file"; done
	$(NPM) test

audit:
	$(PYTHON) scripts/public_release_audit.py

e2e:
	bash -c '$(PYTHON) tests/e2e_server.py & pid=$$!; trap "kill $$pid 2>/dev/null || true" EXIT; for i in $$(seq 1 50); do curl -fsS $(E2E_BASE_URL)/livez >/dev/null && break; sleep .1; done; docker run --rm --network host -e E2E_BASE_URL=$(E2E_BASE_URL) -e E2E_USER=tester -e E2E_PASSWORD=correct-horse-battery-staple -v "$(CURDIR):/work" -w /work $(PLAYWRIGHT_IMAGE) npm run test:e2e'

check: lint test test-js audit

up:
	docker compose up -d --build

logs:
	docker compose logs --tail=200 -f

smoke:
	curl -fsS http://127.0.0.1:31001/livez
	curl -fsS http://127.0.0.1:31001/readyz

backup:
	bash scripts/backup.sh
