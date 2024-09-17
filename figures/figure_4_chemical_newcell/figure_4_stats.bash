#!/bin/bash
#BSUB -J plot_figure_4_stats    # Job name
#BSUB -n 20                              # number of processors
#BSUB -q long                            # Select queue
#BSUB -o logs_shpc/output-figure3-%J.out    # Output file
#BSUB -e logs_shpc/output-figure3-%J.err    # Error file
#BSUB -M 5G                              # memory in MB
#BSUB -u gonzalez.guadalupe@gene.com                        # Email address for notifications
#BSUB -N                                 # send output by email 
#BSUB -W 72:00                            # Job timelimit


# Your command here

ml Anaconda3/2021.05 GCC CUDA/11.4.1  
export LD_LIBRARY_PATH=/apps/rocs/2020.08/sandybridge/software/CUDA/11.4.1/lib:/apps/rocs/2020.08/sandybridge/software/CUDA/11.4.1/lib:/lsf/10.1/linux3.10-glibc2.17-x86_64/lib
conda activate pdgrapher 

echo activated environment


python figure_4_stats.py 

