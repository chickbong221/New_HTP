#!/bin/bash
#SBATCH --job-name=htp-ms-pullcubetool
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# CoRe-WM full (corewm_full) on ManiSkill PullCubeTool-v1: training, then the
# manuscript block/prefix evaluation on the checkpoint training leaves behind.
#
# maniskill_rgb carries the agreed settings -- size50m, 126 GPU envs, 128px
# RGB, pd_ee_delta_pos, nonprivileged_obs=false, shader_dir=minimal,
# batch_size 8, a 600k-transition replay -- so nothing here overrides them.
# PullCubeTool-v1 defaults to the plain panda, which has no wrist camera: one
# base_camera, a 3-channel image, and ~29.5 GB of pixels in the replay. The
# task registers a 100-step horizon, well above the 33 frames the evaluation
# protocol needs, so no horizon override is passed either.
#
# One checkpoint, overwritten in place. --run.save_every 3600 refreshes it
# hourly as crash insurance and the end of training rewrites it once more.
# It lands in the shared checkpoint folder rather than the log tree, under a
# name of its own, so clearing a logdir cannot take the checkpoint every later
# number is read from. The run config is copied beside it for the same reason:
# the evaluation needs it and would otherwise look for it inside the logdir.
#
# Deliberately no `set -e`. The evaluation is gated on training's exit status
# instead: a checkpoint from a run that died part-way would produce matrices
# that look final.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Arm: CoRe-WM full (corewm_full), ManiSkill PullCubeTool-v1"
echo "Budget: 4M steps, 50M model, one checkpoint overwritten hourly"
echo "Evaluation: manuscript block/prefix suite on the final checkpoint"
echo "================================="

# Activate conda
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

# Move to project directory
cd $HOME/projects/New_HTP

export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
export MS_ASSET_DIR=/mnt/data/tuannl

# The checkpoint lands outside the log tree, so clearing a
# logdir cannot take the checkpoint every later number is read from.
CKPT_DIR=$MS_ASSET_DIR/mshab_transfer_checkpoint

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

mkdir -p $HOME/output "$CKPT_DIR"

# The staged ManiSkill smoke on this task before the budget is spent: the env
# builds with its default cameras, the batched driver steps, a clip clears the
# protocol minimum, and the read-only evaluator runs on continuous actions.
# size1m keeps it to a few minutes; it checks plumbing, not the model.
python scripts/smoke_maniskill.py \
  --task maniskill_PullCubeTool-v1 \
  --configs "maniskill_rgb size1m corewm_full" \
  --logdir $HOME/logdir/New_HTP/preflight_${SLURM_JOB_ID}/PullCubeTool-v1 || exit 1

# Print initial GPU state
nvidia-smi

# Monitor GPU every 100 seconds in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

NAME=PullCubeTool-v1-corewm_full
LOGDIR=$HOME/logdir/New_HTP/$TIMESTAMP/$NAME
CKPT_PATH=$CKPT_DIR/${TIMESTAMP}_${NAME}
EVAL_DIR=$HOME/logdir/New_HTP/$TIMESTAMP/eval

echo "=== Training ==="
python -m dreamerv3.main_maniskill \
  --configs maniskill_rgb corewm_full \
  --task maniskill_PullCubeTool-v1 \
  --seed 42 \
  --run.steps 4e6 \
  --run.save_every 3600 \
  --run.checkpoint_dir $CKPT_PATH \
  --logger.wandb_group maniskill_PullCubeTool-v1 \
  --logger.wandb_name $NAME \
  --logdir $LOGDIR
TRAIN_STATUS=$?

# Written when training starts, so it exists even if the run died.
cp "$LOGDIR/config.yaml" "${CKPT_PATH}_config.yaml"

echo "=== Evaluation ==="
if [ $TRAIN_STATUS -eq 0 ]; then
  python -m corewm_eval.manuscript_suite $EVAL_DIR \
    --game PullCubeTool-v1 \
    --checkpoint $CKPT_PATH \
    --config ${CKPT_PATH}_config.yaml
  python -m corewm_eval.manuscript_suite $EVAL_DIR --report --maniskill
else
  echo "Training exited with status $TRAIN_STATUS; evaluation skipped."
fi

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
