#!/bin/bash
#
#SBATCH --workdir=.
#SBATCH --cores=6
#SBATCH --time 2:0:0
#SBATCH --mem 32GB
#SBATCH --output=runs/shared/fixed/video/ai_weights_individual__ci_weights_shared__lr_0.001__rewards_global/log.log
#SBATCH --job-name=ai_weights_individual__ci_weights_shared__lr_0.001__rewards_global
#!/bin/bash

module load python/3.7

source .venv/bin/activate

echo "Entered environment"

video runs/shared/fixed/video/ai_weights_individual__ci_weights_shared__lr_0.001__rewards_global runs/shared/fixed/train/rewards_global__ai_weights_individual__ci_weights_shared__lr_0.001
