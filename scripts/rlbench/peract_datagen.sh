#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_PATH="$REPO_ROOT/peract_raw"
ZARR_PATH="$REPO_ROOT/zarr_datasets/peract"
TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
    TASKS=(push_buttons stack_cups)
fi

mkdir -p "$DATA_PATH" "$ZARR_PATH"

if [[ ! -d "$DATA_PATH/Peract_packaged" ]]; then
    wget -O "$DATA_PATH/Peract_packaged.zip" \
        https://huggingface.co/katefgroup/3d_diffuser_actor/resolve/main/Peract_packaged.zip
    unzip "$DATA_PATH/Peract_packaged.zip" -d "$DATA_PATH"
    rm "$DATA_PATH/Peract_packaged.zip"
fi

if [[ ! -d "$DATA_PATH/test" ]]; then
    wget -O "$DATA_PATH/peract_test.zip" \
        https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract_test.zip
    unzip "$DATA_PATH/peract_test.zip" -d "$DATA_PATH"
    rm "$DATA_PATH/peract_test.zip"
    mv "$DATA_PATH/peract_test" "$DATA_PATH/test"
fi

cd "$REPO_ROOT"
python data_processing/peract_to_zarr.py \
    --root "$DATA_PATH/Peract_packaged/" \
    --tgt "$ZARR_PATH/" \
    --tasks "${TASKS[@]}"
