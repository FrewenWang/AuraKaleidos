#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for script in "${project_root}"/scripts/*.sh; do
    bash -n "${script}"
done

echo "FantasyShell syntax checks passed."
