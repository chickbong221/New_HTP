#!/bin/bash
#SBATCH --job-name=htp-eval-pullcubetool
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# The evaluation half of slurm_maniskill_pullcubetool.sh, on its own.
#
# Training is not repeated: this reads the checkpoint that run left behind and
# redoes only the manuscript block/prefix suite, writing the episode_*.npz
# into a tree the submitting user owns rather than the one the training job
# wrote to.
#
# It does NOT reproduce the failing run byte for byte any more. That run used
# the 16-warm-up/16-horizon protocol and 48-frame clips; corewm_eval/config.py
# now specifies no warm-up, a 64-step rollout and 80-frame clips, so the
# recorded episodes differ by construction. Sane numbers here therefore mean
# the pipeline is healthy under the new protocol, not that the old failure was
# explained; a second 0.167 matrix would be a live reproduction to dissect.
#
#   sbatch scripts/slurm_eval_pullcubetool.sh 20260917_194040

RUN=${1:-20260917_194040}
NAME=PullCubeTool-v1-corewm_full

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamer

export NVIDIA_USERSPACE_VERSION=570.133.20
export NVIDIA_USERSPACE_DIR=$HOME/nvidia-userspace/NVIDIA-Linux-x86_64-${NVIDIA_USERSPACE_VERSION}

cd "$NVIDIA_USERSPACE_DIR"
ln -sf libGLX_nvidia.so.${NVIDIA_USERSPACE_VERSION} libGLX_nvidia.so.0
ln -sf libEGL_nvidia.so.${NVIDIA_USERSPACE_VERSION} libEGL_nvidia.so.0
cat > "$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json" <<EOF
{
    "file_format_version": "1.0.1",
    "ICD": {
        "library_path": "$NVIDIA_USERSPACE_DIR/libEGL_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
EOF
export LD_LIBRARY_PATH=$NVIDIA_USERSPACE_DIR:${LD_LIBRARY_PATH:-}
export VK_DRIVER_FILES=$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json
export VK_ICD_FILENAMES=$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json

cd $HOME/projects/New_HTP

export MS_ASSET_DIR=/mnt/data/tuannl
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

CKPT_PATH=$MS_ASSET_DIR/mshab_transfer_checkpoint/${RUN}_${NAME}
# A tree this user owns, and deliberately not the original eval directory, so
# the clips from the failing run stay untouched if they turn up later.
EVAL_DIR=$HOME/logdir/New_HTP/$RUN/eval_rerun

echo "Checkpoint: $CKPT_PATH"
echo "Output:     $EVAL_DIR"
test -f "$CKPT_PATH/agent.pkl" || { echo "No agent.pkl at $CKPT_PATH"; exit 1; }
test -f "${CKPT_PATH}_config.yaml" || { echo "No run config beside it"; exit 1; }

python -m corewm_eval.manuscript_suite "$EVAL_DIR" \
  --game PullCubeTool-v1 \
  --checkpoint "$CKPT_PATH" \
  --config "${CKPT_PATH}_config.yaml" || exit 1

python -m corewm_eval.manuscript_suite "$EVAL_DIR" --report --maniskill

echo "=== clips written ==="
ls -la "$EVAL_DIR/PullCubeTool-v1/clips"
echo "=== figures and result.json ==="
ls -la "$EVAL_DIR/PullCubeTool-v1/seed_0"
echo "=== full MAE row from result.json ==="
python -c "import json,sys; d=json.load(open('$EVAL_DIR/PullCubeTool-v1/seed_0/result.json')); print('full_mae ', [round(v,4) for v in d['full_mae']]); print('horizons ', d['horizons']); print('checkpoint_sha256', d['checkpoint_sha256'])"
