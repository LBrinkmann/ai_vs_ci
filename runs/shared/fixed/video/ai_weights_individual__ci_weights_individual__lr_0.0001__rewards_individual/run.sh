#!/bin/bash
#
#SBATCH --workdir=.
#SBATCH --cores=6
#SBATCH --time 2:0:0
#SBATCH --mem 32GB
#SBATCH --output=runs/shared/fixed/video/ai_weights_individual__ci_weights_individual__lr_0.0001__rewards_individual/log.log
#SBATCH --job-name=ai_weights_individual__ci_weights_individual__lr_0.0001__rewards_individual
#!/bin/bash

module load python/3.7

source .venv/bin/activate

echo "Entered environment"

video runs/shared/fixed/video/ai_weights_individual__ci_weights_individual__lr_0.0001__rewards_individual runs/shared/fixed/train/rewards_individual__ai_weights_individual__ci_weights_individual__lr_0.0001
