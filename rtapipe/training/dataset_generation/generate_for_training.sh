#!/bin/bash

if [ $# -lt 2 ];
    then
      printf """
Not enought arguments supplied.
        \nArguments:
          - simulation dataset folder
          - integration time
          - lenght of the training time series sample
          - the output directory
          - the number of processes
          - integration type (t/te)
          - slurm (yes/no)
          - partition
          - simulation config file

       Examples:
       ./generate_for_training.sh /scratch/baroncelli/DATA/obs/backgrounds_10mln 1 5 /scratch/baroncelli/AP_DATA/ 30 te yes sim2 /scratch/baroncelli/DATA/obs/backgrounds_10mln/config.yml
"""
    else

      SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
      
      
      GEN_AP_DATA_SCRIPT_PATH=$SCRIPT_DIR/../../lib/dataset/generate_ap_data.py

      today=`date +%Y-%m-%d.%H:%M:%S`
      simdata=$1
      it=$2
      tsl=$3
      outdir="$4/ap_data_bkg_T_${it}_TSL_${tsl}"
      processes=$5
      integration_type=$6
      use_slurm=$7
      partition=$8
      simconfig=$9

      for i; do
         echo $i
      done

      if [ "$integration_type" = "t" ] || [ "$integration_type" = "te" ]; then

        command="python $GEN_AP_DATA_SCRIPT_PATH -c $simconfig -dd $simdata -itype $integration_type -itime $it -rr 0.2 -norm yes -tsl $tsl -out $outdir -proc $processes"

        echo "Command: $command"


        if [ "$use_slurm" = "yes" ]; then

          job_file="$SCRIPT_DIR/job_file.tmp"

          mkdir -p "slurm_out/gen_ap_data-t$it-tsl$tsl"

          echo "#!/bin/bash
#SBATCH --job-name=gen_ap_data-t$it-tsl$tsl.job
#SBATCH --cpus-per-task=$processes
#SBATCH --output=slurm_out/gen_ap_data-t$it-tsl$tsl/generate_ap_data.out
#SBATCH --error=slurm_out/gen_ap_data-t$it-tsl$tsl/generate_ap_data.err
#SBATCH --partition=$partition
$command" > $job_file

          printf "\n\t> Job file: $job_file created! Sbatch it!\n\n"

          #sbatch $job_file
          #sbatch "--job-name=generate_ap_data --cpus-per-task=$processes --output=.out/generate_ap_data.out --error=.out/generate_ap_data.err --qos=normal $command"
        else
          nohup "$command" > nohup.log 2>&1 &
        fi
      else
        printf "\nError! Integration type must be t or te\n\n"
      fi
fi
