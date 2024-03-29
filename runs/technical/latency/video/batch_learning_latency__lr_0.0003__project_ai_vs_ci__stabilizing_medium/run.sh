#PBS -l nodes=1:ppn=6
#PBS -N batch_learning_latency__lr_0.0003__project_ai_vs_ci__stabilizing_medium
#PBS -l walltime=2:0:0
#PBS -l mem=32gb
#PBS -j oe
#PBS -o runs/technical/latency/video/batch_learning_latency__lr_0.0003__project_ai_vs_ci__stabilizing_medium/log.log
#PBS -m n
#PBS -d .

module load python/3.7

source .venv/bin/activate

echo "Entered environment"

video runs/technical/latency/video/batch_learning_latency__lr_0.0003__project_ai_vs_ci__stabilizing_medium runs/technical/latency/train/project_ai_vs_ci__batch_learning_latency__stabilizing_medium__lr_0.0003
