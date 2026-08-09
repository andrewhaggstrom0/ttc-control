#!/bin/bash
#SBATCH --job-name=ttc-audit
#SBATCH -A engr-acad-cse5100
#SBATCH -p condo-cse5100
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=experiments/logs/audit_%A_%a.out
#SBATCH --error=experiments/logs/audit_%A_%a.err

# No `set -u` and no ~/.bashrc: the system bashrc reads BASHRCSOURCED without a
# default, which kills the job at startup under -u.
set -eo pipefail
cd /home/compute/a.haggstrom/ttc-control
mkdir -p experiments/logs experiments/audit

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=.
export CONDA_PREFIX=/home/compute/a.haggstrom/.conda/envs/ttc
export PATH="$CONDA_PREFIX/bin:$PATH"

TASKS=(peg-insert-side-v3 basketball-v3 stick-pull-v3 \
       shelf-place-v3 disassemble-v3 bin-picking-v3)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "task=$TASK node=$(hostname) started=$(date -Is)"
echo "python: $(which python)"

# Per-task TSV, merged afterwards. Writing one file per array element avoids
# concurrent appends to a shared file from six jobs.
python scripts/audit_success.py \
  --task "$TASK" \
  --n-episodes 15 \
  --k-probe 16 \
  --decisions-per-episode 6 \
  --plan-horizon 4 \
  --out experiments/audit/${TASK}.tsv

echo "finished=$(date -Is)"
