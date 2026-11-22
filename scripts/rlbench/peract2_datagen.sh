#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_PATH="$REPO_ROOT/peract2_raw"
ZARR_PATH="$REPO_ROOT/zarr_datasets/peract2"
mkdir -p "$DATA_PATH" "$ZARR_PATH"
cd "$REPO_ROOT"

# This can be slow: download and unsquash all data
python scripts/rlbench/download_peract2.py --root "$DATA_PATH"

# Download the test seeds
if [[ ! -d "$DATA_PATH/test" ]]; then
    wget -O "$DATA_PATH/peract2_test.zip" \
        https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip
    unzip "$DATA_PATH/peract2_test.zip" -d "$DATA_PATH"
    rm "$DATA_PATH/peract2_test.zip"
fi

# Now we just need to package them
python data_processing/peract2_to_zarr.py \
    --root "$DATA_PATH" \
    --tgt "$ZARR_PATH"
# You can safely delete the train and val raw data now, not test
