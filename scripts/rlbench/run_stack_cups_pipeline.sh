#!/usr/bin/env bash
# Run after activating the 3dfa environment:
# bash scripts/rlbench/run_stack_cups_pipeline.sh
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

dataset_dir=zarr_datasets/peract_stack_cups
test_dir=peract_raw/test
exp_name=PeractStackCups
run_name=denoise3d-PeractTwoCam-C120-B16-lr1e-4-constant-H3-rectified_flow
run_dir="train_logs/$exp_name/$run_name"
vis_dir=visualizations/peract_stack_cups
train_iters=${TRAIN_ITERS:-50000}

if [[ ! $train_iters =~ ^[1-9][0-9]*$ ]] || (( train_iters < 4000 )); then
    echo "TRAIN_ITERS must be an integer of at least 4000 (the first validation)." >&2
    exit 1
fi
for path in "$dataset_dir/train.zarr" "$dataset_dir/val.zarr" \
            "$test_dir/stack_cups/variation0/episodes/episode0" \
            "$test_dir/stack_cups/variation1/episodes/episode0"; do
    [[ -e $path ]] || { echo "Missing required data: $path" >&2; exit 1; }
done
for command in python torchrun xvfb-run ffmpeg; do
    command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
python - <<'PY'
import matplotlib, torch, zarr
assert torch.cuda.is_available(), "A CUDA GPU is required"
PY

export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$repo_root/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}
[[ -d $COPPELIASIM_ROOT ]] || { echo "Missing CoppeliaSim: $COPPELIASIM_ROOT" >&2; exit 1; }
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
cudnn_dir=$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent.parent / "nvidia/cudnn/lib")
PY
)
if [[ -d $cudnn_dir ]]; then
    export LD_LIBRARY_PATH="$cudnn_dir:$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
fi
export PYTHONUNBUFFERED=1

mkdir -p "$run_dir" "$vis_dir"
complete_marker="$run_dir/training_complete_${train_iters}"
if [[ ! -f $complete_marker ]]; then
    echo "Training stack_cups for $train_iters iterations; validation runs every 4000."
    DATASET_DIR="$dataset_dir" EXP_NAME="$exp_name" TRAIN_ITERS="$train_iters" \
        bash scripts/rlbench/train_peract.sh 2>&1 | tee -a "$run_dir/train_console.log"
    touch "$complete_marker"
else
    echo "Training already completed for $train_iters iterations; reusing its checkpoint."
fi

checkpoint="$repo_root/$run_dir/best.pth"
[[ -f $checkpoint ]] || { echo "Training produced no best checkpoint: $checkpoint" >&2; exit 1; }

echo "Testing the validation-selected checkpoint on held-out RLBench test seeds."
checkpoint_alias="stack_cups_best_$(date -u +%Y%m%dT%H%M%SZ)"
CHECKPOINT="$checkpoint" CHECKPOINT_ALIAS="$checkpoint_alias" DATA_DIR="$test_dir" \
    TASKS=stack_cups EXP_NAME="$exp_name" \
    bash online_evaluation_rlbench/eval_peract.sh 2>&1 | tee "$run_dir/test_console.log"
test_result="eval_logs/$exp_name/$checkpoint_alias/seed0/stack_cups/eval.json"
[[ -s $test_result ]] || { echo "Missing test result: $test_result" >&2; exit 1; }

python scripts/rlbench/plot_peract_metrics.py \
    --run-dir "$run_dir" --task stack_cups \
    --output "$vis_dir/validation_metrics.png"

for variation in 0 1; do
    COPPELIASIM_ROOT="$COPPELIASIM_ROOT" xvfb-run -a \
        python scripts/rlbench/record_peract_rollout.py \
        --task stack_cups --checkpoint "$checkpoint" --data-dir "$test_dir" \
        --variation "$variation" --episode 0 \
        --output-dir "$vis_dir/test_variation${variation}_episode0"
done

echo "Test results: $test_result"
echo "Validation plot: $vis_dir/validation_metrics.png"
echo "Rollout videos and GIFs: $vis_dir/test_variation{0,1}_episode0/"
