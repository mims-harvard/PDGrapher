
# Define the cell lines array
cell_lines=("A549_corrected_pos_emb" "MCF7_corrected_pos_emb" "MDAMB231_corrected_pos_emb" "BT20_corrected_pos_emb" "PC3_corrected_pos_emb" "VCAP_corrected_pos_emb")
# cell_lines=("PC3_corrected_pos_emb" )


# Iterate over each cell line
for cell_line in "${cell_lines[@]}"
do
  # Create a job submission script for each cell line
  echo "#!/bin/bash
#BSUB -J compute_usecases_${cell_line}    # Job name
#BSUB -n 20                              # number of processors
#BSUB -q long                            # Select queue
#BSUB -o logs_shpc/output-${cell_line}-%J.out    # Output file
#BSUB -e logs_shpc/output-${cell_line}-%J.err    # Error file
#BSUB -M 5G                              # memory in MB
#BSUB -u gonzalez.guadalupe@gene.com                        # Email address for notifications
#BSUB -N                                 # send output by email 
#BSUB -W 2:00                            # Job timelimit
#BSUB -gpu "num=1:j_exclusive=yes"     # Request 1 GPU exclusive


# Your command here

ml Anaconda3/2021.05 GCC CUDA/11.4.1  
export LD_LIBRARY_PATH=/apps/rocs/2020.08/sandybridge/software/CUDA/11.4.1/lib:$LD_LIBRARY_PATH
conda activate pdgrapher 

cd ../../
pip install -e .
cd figures/figure_5

echo "activated environment"


python compute_usecases.py ${cell_line}
" > job_scripts/job_compute_usecases_${cell_line}.bash

  # Submit the job
  bsub < job_scripts/job_compute_usecases_${cell_line}.bash
done