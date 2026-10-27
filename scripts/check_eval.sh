#!/usr/bin/env bash
# Validate a config using the installed WIA package; no simulator/model is started.
set -eu
python -m wia eval --config "$1" --dry-run
