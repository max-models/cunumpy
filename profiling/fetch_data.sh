#!/usr/bin/env bash
# Fetch cunumpy's runs from the profiling catalogue into profiling/data (only cunumpy's folder).
set -euo pipefail
cd "$(dirname "$0")"
rm -rf data
git clone -q --depth 1 --filter=blob:none --sparse https://github.com/max-models/profiling-catalog data
git -C data sparse-checkout set cunumpy
