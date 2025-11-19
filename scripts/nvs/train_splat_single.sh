#!/usr/bin/env bash
#SBATCH --account=<your_account>
#SBATCH --job-name=nvs_splat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


DATA="${1:?usage: $0 /abs/path/to/data_dir}"
if [ ! -d "$DATA" ]; then
  echo "ERROR: Data directory does not exist: $DATA" >&2
  exit 2
fi
if [ ! -d "$DATA/images_stride_10" ] || [ ! -d "$DATA/our_gt_to_user" ]; then
  echo "ERROR: Expected subdirs images_stride_10 and our_gt_to_user inside $DATA" >&2
  exit 3
fi
mkdir -p logs

# Conda without ~/.bashrc
if [ -f /path/to/mamba/etc/profile.d/conda.sh ]; then
  source /path/to/mamba/etc/profile.d/conda.sh
else
  eval "$(/path/to/mamba/bin/conda shell.bash hook)"
fi
conda activate nerfstudio

export PYTHONNOUSERSITE=1
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

printf "y\n" | srun ns-train splatfacto \
  --pipeline.model.camera-optimizer.mode off \
  --data "$DATA" \
  --vis viewer+tensorboard \
  --viewer.websocket-port 13315 \
  colmap \
  --load_3D_points False \
  --images_path images_stride_10 \
  --colmap_path our_gt_to_user \
  --downscale_factor 4