#!/bin/bash
#SBATCH --job-name=htp-diagnose-pullcubetool
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Why the 20260917_194040 manuscript evaluation decoded noise.
#
# Read-only: it loads the checkpoint that run left behind and the clips that
# run wrote, retrains nothing and saves nothing back. The same Vulkan/NVIDIA
# preamble as the training job is still needed -- make_agent builds a
# num_envs=1 ManiSkill env just to discover the obs and act spaces, even when
# --clips means no episode has to be collected.
#
# Override RUN on the command line to point at a different run:
#   sbatch scripts/slurm_diagnose_pullcubetool.sh 20260917_194040

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
CLIPS=$HOME/logdir/New_HTP/$RUN/eval/PullCubeTool-v1/clips
OUT=$HOME/logdir/New_HTP/$RUN/diagnosis

echo "Checkpoint: $CKPT_PATH"
echo "Clips:      $CLIPS"
echo "Output:     $OUT"

# Falling back to collecting a fresh clip would still diagnose the decode, but
# it would no longer be the exact episode the bad figures came from.
CLIP_ARG=""
if [ -f "$CLIPS/episode_0.npz" ]; then
  CLIP_ARG="--clips $CLIPS"
else
  echo "No episode_0.npz under $CLIPS; collecting a fresh clip instead."
fi

python scripts/diagnose_maniskill_eval.py "$OUT" \
  --checkpoint "$CKPT_PATH" \
  --config "${CKPT_PATH}_config.yaml" \
  $CLIP_ARG

echo "Done. Figures in $OUT:"
ls -la "$OUT" 2>/dev/null
