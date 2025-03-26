#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
config_path=$1
experiment_name=$2

log_root=/home/whisper/code/hcc_project/logs/classification/ckpt/
log_path=${log_root}${experiment_name}

ROOT=../../..
export PYTHONPATH=$ROOT:$PYTHONPATH


if [ -e $log_path ]
then
   echo "log directory existing"
   echo "log diretory is ${log_path}" 
else
   echo "log directory not existing, now setting up log directory ...."
   mkdir -p ${log_path}
   echo "log diretory is ${log_path}"
fi

python ../../infer_on_dir.py -c ${config_path}
# python ../../infer.py -c ${config_path} 2>&1 | tee ${log_path}/infer_log_$now.log