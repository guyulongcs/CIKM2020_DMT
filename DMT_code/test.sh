python -c "print('*' * 210)"
echo "Testing..."

conf=$1
ckpt=$2
test_tag=$3
test_score_method=$4

dt=`date +%s`
python -u ./run_dnn.py --conf_file=${conf} --is_test=true --model_ckpt=${ckpt} --test_tag=${test_tag} --test_score_method=${test_score_method}
dt_end=`date +%s`
echo "run duration $((dt_end-dt)) s"
