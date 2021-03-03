#PBS -l nodes=1:ppn=6
#PBS -N batch_learning_latency_symmetric__lr_0.001__project_ai_vs_ci__stabilizing_small
#PBS -l walltime=2:0:0
#PBS -l mem=32gb
#PBS -j oe
#PBS -o runs/technical/latency_sym/video/batch_learning_latency_symmetric__lr_0.001__project_ai_vs_ci__stabilizing_small/log.log
#PBS -m n
#PBS -d .

module load python/3.7

source .venv/bin/activate

echo "Entered environment"

video runs/technical/latency_sym/video/batch_learning_latency_symmetric__lr_0.001__project_ai_vs_ci__stabilizing_small runs/technical/latency_sym/train/project_ai_vs_ci__batch_learning_latency_symmetric__stabilizing_small__lr_0.001
