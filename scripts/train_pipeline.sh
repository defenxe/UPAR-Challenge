#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" -B "$ROOT/scripts/train_pipeline.py" "$@"
