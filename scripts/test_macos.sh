#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${repo_root}/build/macos-tests"
python_bin="${repo_root}/.venv/bin/python"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This test suite supports macOS only." >&2
    exit 2
fi

if [[ ! -x "${python_bin}" ]]; then
    echo "Run scripts/bootstrap_macos.sh first." >&2
    exit 1
fi

run_cmake_tests() {
    local name="$1"
    local source_dir="$2"
    shift 2
    local build_dir="${build_root}/${name}"
    cmake -S "${repo_root}/${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release "$@"
    cmake --build "${build_dir}" --parallel
    ctest --test-dir "${build_dir}" --output-on-failure
}

run_python_tests() {
    local project="$1"
    local source_dir="${2:-}"
    local python_path="${repo_root}/${project}"
    if [[ -n "${source_dir}" ]]; then
        python_path="${repo_root}/${project}/${source_dir}:${python_path}"
    fi
    PYTHONPATH="${python_path}" \
        "${python_bin}" -m unittest discover -s "${repo_root}/${project}/tests" -v
}

run_node_tests() {
    local project="$1"
    npm --prefix "${repo_root}/${project}" test
}

mkdir -p "${build_root}"

run_cmake_tests fantasy-cxx FantasyCXX \
    -DBUILD_AURA_CV=OFF \
    -DBUILD_AURA_VISION=OFF \
    -DBUILD_AURA_VISION_HPC=OFF \
    -DBUILD_PERFORMANCE_TESTS=OFF
run_cmake_tests algorithms-cxx FantasyAlgorithm/CXX
run_cmake_tests opencv-cxx FantasyAI/AliceOpenCV/OpenCVCXX

run_python_tests FantasyAlgorithm/python
run_python_tests FantasyPython/alice-auto-driving src
run_python_tests FantasyPython/aura-data-compare src
run_python_tests FantasyPython/aura-pyutils src
run_python_tests FantasyAutoDrive/kalman_filter_with_yolo11_objects_tracker

run_node_tests FantasyJS
run_node_tests FantasyNodeJS/AuraNodeCli
run_node_tests FantasyNodeJS/AuraNodeSpider
run_node_tests FantasyNodeJS/MyMovieWeb
run_node_tests FantasyNodeJS/MyPicSpider
run_node_tests FantasyNodeJS/MySpiderDemo
run_node_tests FantasyNodeJS/NodeJsSamples/ExpressDemo
npm --prefix "${repo_root}/FantasyNodeJS/NyxTSExpress" run build
run_node_tests FantasyNodeJS/NyxTSExpress
run_node_tests FantasyNodeJS/myExpressGenarator

bash -n "${repo_root}/FantasyShell/scripts/addr.sh"
bash -n "${repo_root}/FantasyShell/scripts/addr_parser_file.sh"

echo "All supported macOS tests passed."
