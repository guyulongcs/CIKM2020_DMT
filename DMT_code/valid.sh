#!/usr/bin/env bash

python -c "print('*' * 210)"
echo "Valid..."

conf=$1
export TF_CONFIG=$(cat <<<'{"task":{"type":"evaluator"}}')

dt=`date +%s`
python -u ./run_dnn.py --conf_file=${conf} --is_test=false
dt_end=`date +%s`
echo "run duration $((dt_end-dt)) s"
