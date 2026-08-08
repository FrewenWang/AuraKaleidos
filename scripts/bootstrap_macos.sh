#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This bootstrap script supports macOS only." >&2
    exit 2
fi

for command_name in cmake python3 node npm; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required command: ${command_name}" >&2
        exit 1
    fi
done

python3 -m venv "${repo_root}/.venv"
"${repo_root}/.venv/bin/python" -m pip install --upgrade pip
"${repo_root}/.venv/bin/python" -m pip install "numpy>=1.26" "matplotlib>=3.8"

node_projects=(
    "FantasyJS"
    "FantasyNodeJS/AuraNodeCli"
    "FantasyNodeJS/AuraNodeSpider"
    "FantasyNodeJS/MyMovieWeb"
    "FantasyNodeJS/MyPicSpider"
    "FantasyNodeJS/MySpiderDemo"
    "FantasyNodeJS/NodeJsSamples/ExpressDemo"
    "FantasyNodeJS/NyxTSExpress"
    "FantasyNodeJS/myExpressGenarator"
)

for project in "${node_projects[@]}"; do
    echo "Installing Node dependencies: ${project}"
    npm --prefix "${repo_root}/${project}" install --no-audit --no-fund
done

echo "macOS development dependencies are ready."
