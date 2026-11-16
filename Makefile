# The documentation site (docs/, Astro + Starlight, from @max-models/starlight-theme).
PYTHON ?= .venv/bin/python
DOCS_ENV = DOCS_PYTHON=$(shell $(PYTHON) -c "import sys; print(sys.executable)")

.PHONY: help docs-install docs-hero docs-dev docs-build docs-preview docs-clean

help:  ## list the targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

docs-install:  ## Python docs extra and the npm packages
	$(PYTHON) -m pip install -e ".[docs]"
	cd docs && npm ci

docs-hero:  ## redraw the home page figure (docs/src/assets/hero.svg)
	$(PYTHON) docs/hero.py

docs-dev:  ## live preview at http://localhost:4321/cunumpy/
	cd docs && $(DOCS_ENV) DOCS_VALIDATE_LINKS=false npm run dev

docs-build:  ## the static site in docs/dist, as in CI
	cd docs && $(DOCS_ENV) npm run build

docs-preview:  ## serve docs/dist
	cd docs && npm run preview

docs-clean:  ## remove the build output
	rm -rf docs/dist docs/.astro
