#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_PATH="$REPO_ROOT/peract2_raw"
ZARR_PATH="$REPO_ROOT/zarr_datasets/peract2"
mkdir -p "$DATA_PATH" "$ZARR_PATH"

if [[ ! -d "$ZARR_PATH/train.zarr" ]]; then
    wget -O "$ZARR_PATH/peract2.zip" \
        https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2.zip
    unzip "$ZARR_PATH/peract2.zip" -d "$ZARR_PATH"
    rm "$ZARR_PATH/peract2.zip"
fi

if [[ ! -d "$DATA_PATH/test" ]]; then
    wget -O "$DATA_PATH/peract2_test.zip" \
        https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip
    unzip "$DATA_PATH/peract2_test.zip" -d "$DATA_PATH"
    rm "$DATA_PATH/peract2_test.zip"
fi
