#PBS -N project_ai_vs_ci__batch_test_djx_heuristic__self_weight_0.0__n_agents_10__networktype_dense
#PBS -l walltime=4:0:0
#PBS -l mem=2gb
#PBS -j oe
#PBS -o runs/djx_test_heu/project_ai_vs_ci__batch_test_djx_heuristic__self_weight_0.0__n_agents_10__networktype_dense/log.log
#PBS -m n
#PBS -d .

module load python/3.7

source .venv/bin/activate

echo "Entered environment"

python aci/adversial.py runs/djx_test_heu/project_ai_vs_ci__batch_test_djx_heuristic__self_weight_0.0__n_agents_10__networktype_dense/params.yml runs/djx_test_heu/project_ai_vs_ci__batch_test_djx_heuristic__self_weight_0.0__n_agents_10__networktype_dense/data 