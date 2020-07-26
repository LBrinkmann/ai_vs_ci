#PBS -N num_episodes_100__n_agents_10__networktype_nondense
#PBS -l walltime=4:0:0
#PBS -l mem=2gb
#PBS -j oe
#PBS -o runs/djx_cpu_test/num_episodes_100__n_agents_10__networktype_nondense/log.log
#PBS -m n
#PBS -d .

module load python/3.7

source .venv/bin/activate

echo "Entered environment"

python aci/multi_train.py runs/djx_cpu_test/num_episodes_100__n_agents_10__networktype_nondense/params.yml runs/djx_cpu_test/num_episodes_100__n_agents_10__networktype_nondense/data 