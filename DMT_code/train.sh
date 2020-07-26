#!/bin/sh
python -c "print('*' * 210)"
echo "Training..."

conf=$1
ckpt=$2

export TF_CONFIG=$(cat <<<'{"task":{"type":"chief"}}')

dt=`date +%s`
python -u ./run_dnn.py --conf_file=${conf} --is_test=false --model_ckpt=${ckpt}
dt_end=`date +%s`
echo "run duration $((dt_end-dt)) s"
