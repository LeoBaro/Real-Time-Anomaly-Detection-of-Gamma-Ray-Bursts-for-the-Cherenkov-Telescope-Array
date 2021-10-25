#!/bin/bash

function findPackage {
    locationString=$(pip show RTAscience | grep Location)
    IFS=': '
    read -ra locArray <<< "$locationString"
    echo "${locArray[1]}"
}

function message {
    printf "${GREEN}\n${1}${NORMC}\n";
}


if [[ -z "${DATA}" ]]; then

    printf "DATA is undefined. Please, export DATA=..\n"

else

    if [[ -z "${1}" ]]; then
        printf """Usage:\n\t./run_simulation.sh
            - number of processes for MP or number of cores for SLURM
            - parallelization technology [mp, slurm]
            - simulation type [bkg, grb]
            - onset
            - tobs
            - trials
            - the starting seed (used for MP)
            - emin
            - emax
            - region of intereset (degrees)
            - scale
            - virtual environment name (used for SLURM)

          Examples:
            - ./run_simulation.sh 50 mp bkg 0   100 1000 0 0.03 1 2.5 1 bphd
            - ./run_simulation.sh 50 mp grb 900 100 1000 0 0.03 1 2.5 1 bphd
"""
    else

      cpus=$1
      parallelization=$2
      type=$3
      onset=$4
      tobs=$5
      trials=$6
      startcount=$7
      emin=$8
      emax=$9
      roi=${10}
      scale=${11}
      virtualenvname=${12}

      printf "cpus: $cpus\n"
      printf "parallelization: $parallelization\n"
      printf "type: $type\n"
      printf "onset: $onset\n"
      printf "tobs: $tobs\n"
      printf "trials: $trials\n"
      printf "startcount: $startcount\n"
      printf "emin: $emin\n"
      printf "emax: $emax\n"
      printf "roi: $roi\n"
      printf "scale: $scale\n"
      printf "virtualenvname: $virtualenvname\n"

      # Logs
      DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
      mkdir -p $DIR/logs
      current_time=$(date "+%Y.%m.%d-%H.%M.%S")

      # Configuration file
      message "Writing configuration file template.."
      configFileTemplate=$(source lib/configTemplate.sh $DIR)
      printf "Configuration file template=$configFileTemplate\n"

      message "Creating configuration file.."
      configFilePath=$(source lib/getConfiguration.sh $configFileTemplate $current_time $type $onset $tobs $trials $startcount $emin $emax $roi $scale)
      printf "Configuration file=$configFilePath\n"

      # RTAScience package
      scriptPath=$(findPackage)
      if [[ -z "${scriptPath}" ]]; then
          printf "Error: RTAscience was not found, simulations won't be started.\n"
      else

          # Script
          if [ "$type" = "bkg" ]; then
            script="${scriptPath}/RTAscience/simBkg.py"
          elif [ "$type" = "grb" ]; then
            script="${scriptPath}/RTAscience/simGRBcatalog.py"
          else
            printf "Error: '$type' is not supported. \$3=[bkg, grb]\n"
          fi


          if [ "$parallelization" = "mp" ]; then
            printf "Starting simulation with Multiprocessing..\n"
            logFile="$DIR/logs/simBKG_$current_time.log"
            # nohup python "$script" -f "$configFilePath" --mp-enabled false --mp-threads "$cpus" 2>&1 > "$logFile" &
            python "$script" -f "$configFilePath" --mp-enabled true --mp-threads "$cpus" 2>&1 > "$logFile"

          elif [ "$parallelization" = "slurm" ]; then
            printf "Starting simulation with Slurm..\n"
            trials_per_node=$(($trials/$cpus))
            printf "cpus: $cpus\ntrials_per_node: $trials_per_node\n"
            python "$scriptPath/RTAscience/makeConfig_runJobs.py" --infile "$configFilePath" --tt $trials --tn $trials_per_node --delay 0 --off 0 --flux 1 --env $virtualenvname --script $script --print false

          else
            printf "Error: '$parallelization' is not supported. \$2=[mp, slurm]\n"
          fi
      fi

      echo $(date "+%Y.%m.%d-%H.%M.%S") > "$DIR/logs/submitted_on.txt"


      rm $DIR/config.yaml
    fi
fi
