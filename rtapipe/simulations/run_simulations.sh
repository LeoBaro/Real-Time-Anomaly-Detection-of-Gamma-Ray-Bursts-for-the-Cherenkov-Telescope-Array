#!/bin/bash

function findPackage {
    locationString=$(pip show RTAscience | grep Location)
    IFS=': '
    read -ra locArray <<< "$locationString"
    echo "${locArray[1]}"
}

function startSimulation {
    if [[ -z "${1}" ]]; then
        printf "Error: no arguments have been passed to the function. Expected: the name of the simulation python script, path to the configuration file\n"
        echo ""
    else
        scriptName=$1
        configFilePath=$2
        threads=$3
        DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
        printf "Starting simulations with $scriptName..please wait..\n"
        current_time=$(date "+%Y.%m.%d-%H.%M.%S")
        logFile="$DIR/logs/simBKG_$current_time.log"
        scriptPath=$(findPackage)
        if [[ -z "${scriptPath}" ]]; then
          printf "RTAscience was not found, simulations won't be started.\n"
        else
          # nohup python "$scriptPath/RTAscience/$scriptName" -f "$configFilePath" --mp-enabled false --mp-threads "$threads" 2>&1 > "$logFile" &
          python "$scriptPath/RTAscience/$scriptName" -f "$configFilePath" --mp-enabled false --mp-threads "$threads" 2>&1 > "$logFile"
        fi
    fi
}


function message {
    printf "${GREEN}\n${1}${NORMC}\n";
}


if [[ -z "${DATA}" ]]; then

    printf "DATA is undefined. Please, export DATA=..\n"

else

    if [[ -z "${1}" ]]; then
        printf "Usage:\n\t./run_simulation.sh <number of threads>\n"
    else

      threads=$1

      DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
      mkdir -p $DIR/logs


      message "Creating configuration file template"
      configFileTemplate=$(source lib/configTemplate.sh $DIR)
      printf "Configuration file template=$configFileTemplate\n"

      message "BKG simulations"
      log_id=0
      # onset, tobs, trials, startcount, emin, emax, roi, scale
      configFile=$(source lib/getConfiguration.sh "$configFileTemplate" "$log_id" "bkg" 0   100 10000000 0   0.03 1 2.5 1)
      printf "Configuration file=$configFile\n"
      startSimulation "simBkg.py" $configFile $threads

      #message "GRB simulations"
      #log_id=$((log_id+1))
      #configFile=$(source lib/getConfiguration.sh "$configFileTemplate" "$log_id" "grb" 900 1800 50     0    0.03 1 2.5 1)
      #printf "Configuration file=$configFile\n"
      #startSimulation "simGRBcatalog.py" $configFile


      rm $DIR/config.yaml
    fi
fi
