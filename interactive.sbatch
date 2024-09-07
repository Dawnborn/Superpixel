#!/bin/bash
#SBATCH --job-name="superpxiel"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:7,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-user="ge32hij@mytum.de"
#SBATCH --mail-type=NONE
#SBATCH --output=/home/wiss/lhao/junpeng/ws_lightsphere/sbatchoutput/logs/slurm-%j.out
#SBATCH --error=/home/wiss/lhao/junpeng/ws_lightsphere/sbatchoutput/logs/slurm-%j.out

# Start tmux
tmux new -d -s "test"

# Run tmux commands to setup the session
tmux split-window -h # this will setup two panes, splitting the screen horizontally

# Finally, make job wait for user to connected
sleep 8h

# # outside: srun --jobid <JOBID> --pty tmux a -t test