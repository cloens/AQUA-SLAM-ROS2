#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${workspace_dir}"

# colcon's package worker count does not limit Make/Ninja inside one package.
# Do not append inherited MAKEFLAGS: a trailing -j32 overrides the local cap.
build_jobs="${AQUA_BUILD_JOBS:-2}"
if ! [[ "${build_jobs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AQUA_BUILD_JOBS must be a positive integer, got: ${build_jobs}" >&2
  exit 2
fi
unset MAKEFLAGS
export MAKEFLAGS="-j${build_jobs} -l${build_jobs}"
export CMAKE_BUILD_PARALLEL_LEVEL="${build_jobs}"
export AQUA_COMPILE_JOBS="${build_jobs}"

exec colcon build "$@" \
  --executor sequential \
  --parallel-workers 1 \
  --packages-select aqua_slam \
  --cmake-args -DCMAKE_BUILD_TYPE=Release "-DAQUA_COMPILE_JOBS=${build_jobs}"
