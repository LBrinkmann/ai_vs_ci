#!/bin/bash
#
#SBATCH --workdir=.
#SBATCH --cores=1
#SBATCH --output=runs/djx_test_heu/project_ai_vs_ci__batch_test_djx_heuristic__self_weight_0.0__n_agents_2__networktype_nondense/log.log
#SBATCH --job-name=project_ai_vs_ci__batch_test_djx_heuristic__self_weight_0.0__n_agents_2__networktype_nondense

module load python/3.7

source .venv/bin/activate

echo "Entered environment"

python aci/adversial.py runs/djx_test_heu/project_ai_vs_ci__batch_test_djx_heuristic__self_weight_0.0__n_agents_2__networktype_nondense/params.yml runs/djx_test_heu/project_ai_vs_ci__batch_test_djx_heuristic__self_weight_0.0__n_agents_2__networktype_nondense/data 