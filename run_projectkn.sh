#!/bin/bash
#SBATCH -n 1                     # Request 4 tasks
#SBATCH -c 3                     # Each task requires 3 processors
#SBATCH -N 1                     # Request a single computing node
#SBATCH -t 1-10:30:05            # Job duration of  1 day and 5 minutes
#SBATCH -p long             # Use the long queue
#SBATCH --gres=gpu:4             # Request 4 GPUs

# Load any necessary modules (if required)
# module load python/3.11

# export JAVA_HOME=~/java/jdk-21.0.2+13
# export PATH=$JAVA_HOME/bin:$PATH


# Create a virtual environment
# python3 -m venv .venvb

cd ..
cd ..

source .venvb/bin/activate

cd Modeling-GL/PKBdata/SCVdet/   # /Domain-Nk/CVEFixe

chmod +x ./run_code.sh




./run_code.sh









