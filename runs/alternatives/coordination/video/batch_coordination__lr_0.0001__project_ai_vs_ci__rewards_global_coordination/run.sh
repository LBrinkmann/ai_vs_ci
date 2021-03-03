#PBS -l nodes=1:ppn=6
#PBS -N batch_coordination__lr_0.0001__project_ai_vs_ci__rewards_global_coordination
#PBS -l walltime=2:0:0
#PBS -l mem=32gb
#PBS -j oe
#PBS -o runs/alternatives/coordination/video/batch_coordination__lr_0.0001__project_ai_vs_ci__rewards_global_coordination/log.log
#PBS -m n
#PBS -d .

module load python/3.7

source .venv/bin/activate

echo "Entered environment"

video runs/alternatives/coordination/video/batch_coordination__lr_0.0001__project_ai_vs_ci__rewards_global_coordination runs/alternatives/coordination/train/project_ai_vs_ci__batch_coordination__rewards_global_coordination__lr_0.0001
