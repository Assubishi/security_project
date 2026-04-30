#!/usr/bin/env bash
set -euo pipefail

python -m ragstack.cli evaluate --config configs/demo.yaml
python -m ragstack.cli search --config configs/demo.yaml
