#!/bin/bash
#
#SBATCH --workdir=.
#SBATCH --cores=6
#SBATCH --time 2:0:0
#SBATCH --mem 32GB
#SBATCH --output=runs/tests/on_off/video/batch_madqn_on_off__lr_0.0001__mix_freq_no_mixing__project_ai_vs_ci/log.log
#SBATCH --job-name=batch_madqn_on_off__lr_0.0001__mix_freq_no_mixing__project_ai_vs_ci
#!/bin/bash

module load python/3.7

source .venv/bin/activate

echo "Entered environment"

video runs/tests/on_off/video/batch_madqn_on_off__lr_0.0001__mix_freq_no_mixing__project_ai_vs_ci runs/tests/on_off/train/project_ai_vs_ci__batch_madqn_on_off__mix_freq_no_mixing__lr_0.0001
