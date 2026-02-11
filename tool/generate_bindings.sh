#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
dart run ffigen --config ffigen.yaml
