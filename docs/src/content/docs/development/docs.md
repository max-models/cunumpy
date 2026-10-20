---
title: Building the docs
description: How this documentation site is built from Markdown pages and docstrings.
---

The site in `docs/` is built with [Astro](https://astro.build/) and
[Starlight](https://starlight.astro.build/). It has two kinds of pages:

- **Hand-written pages** in `docs/src/content/docs/`, as Markdown or MDX.
- **API reference**, generated from the docstrings by
  [starlight-pydocs](https://ewels.github.io/starlight-pydocs/), one page per module, at
  `/cunumpy/api/cunumpy/`.

The configuration (`defineDocs` in `docs/astro.config.mjs`), the look, the header and the build
workflow come from the shared [starlight-theme](https://github.com/max-models/starlight-theme)
of the max-models projects; change them there, and update the version in `docs/package.json`
and `.github/workflows/docs.yml` to pick up a new release.

## Requirements

- Node 22 or newer.
- The package installed with the `docs` extra: `pip install -e ".[docs]"`.

## Commands

```bash
make docs-install     # npm packages and the Python docs extra
make docs-dev         # live preview at http://localhost:4321/cunumpy/
make docs-build       # the static site in docs/dist, as in CI
make docs-preview     # serve docs/dist
make docs-clean       # remove the build output
```

## Writing pages

Pages are Markdown in `docs/src/content/docs/`; the path is the URL. Link to other pages with
the site path, `[Quickstart](/cunumpy/getting-started/quickstart/)`, and to the API reference
with the anchor of the object, `/cunumpy/api/cunumpy/kernels/#cunumpy.kernels.CudaKernel`. Add
a new page to the sidebar in `docs/astro.config.mjs`.

The code on the [Particle codes](/cunumpy/guides/particle-codes/) page is read from
`docs/src/snippets/particle_recipes.py`, which `tests/unit/test_particle_recipes.py` runs. The
[guide for AI assistants](/cunumpy/ai-assistants/) renders `src/cunumpy/LLM_GUIDE.md`.

## CI

`.github/workflows/docs.yml` builds the site on every push and pull request with the shared
workflow of starlight-theme, checks that the main pages exist, and deploys `devel` to GitHub
Pages.
