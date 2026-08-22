#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${workspace_dir}"

# colcon's parallel worker count limits packages, while Make can still use all
# CPU cores inside a single package. Limit both layers for memory-heavy C++.
export MAKEFLAGS="-j1"
export CMAKE_BUILD_PARALLEL_LEVEL=1

exec colcon build "$@" \
  --executor sequential \
  --parallel-workers 1 \
  --packages-select aqua_slam \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
