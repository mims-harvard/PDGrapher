#!/bin/bash

# Define the cell lines array
# cell_lines=("A549" "A375" "BT20" "HA1E" "HELA" "HT29" "MCF7" "MDAMB231" "PC3" "VCAP")
cell_lines=("A549" "BT20" "MCF7" "MDAMB231" "PC3" "VCAP")


# Iterate over each cell line
for cell_line in "${cell_lines[@]}"
do
  # Create a job submission script for each cell line
  echo "#!/bin/bash
#BSUB -J build_sciplex_1_${cell_line}    # Job name
#BSUB -n 20                              # number of processors
#BSUB -q long                            # Select queue
#BSUB -o logs_shpc/output-${cell_line}-%J.out    # Output file
#BSUB -e logs_shpc/output-${cell_line}-%J.err    # Error file
#BSUB -M 5G                              # memory in MB
#BSUB -N                                 # send output by email 
#BSUB -W 90:00                            # Job timelimit
#BSUB -gpu "num=1:j_exclusive=yes"     # Request 1 GPU exclusive


# Your command here

ml Anaconda3/2021.05 GCC CUDA/11.4.1  
export LD_LIBRARY_PATH=/apps/rocs/2020.08/sandybridge/software/CUDA/11.4.1/lib:$LD_LIBRARY_PATH
conda activate pdgrapher 

cd ../
pip install -e .
cd experiments_resubmission_bme

echo "activated environment"


python chemical.py ${cell_line}
" > job_chemical_${cell_line}.bash

  # Submit the job
  bsub < job_chemical_${cell_line}.bash
done
