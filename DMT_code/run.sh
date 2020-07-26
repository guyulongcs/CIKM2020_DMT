#!/usr/bin/env bash

conf="dmt.conf"

checkpoint_train="model.ckpt-0"
checkpoint_test="model.ckpt-150000"


nohup sh train.sh ${conf} ${checkpoint_train} > log_train_${conf} &


#nohup sh valid.sh ${conf} > log_valid_${conf} &
#
#nohup sh test.sh ${conf} ${checkpoint_test} "clk" > log_test_clk_${conf} &
#nohup sh test.sh ${conf} ${checkpoint_test} "ord" "rel" > log_test_ord_rel_${conf} &
#
#nohup python -u rec_saved_model.py --conf_file=${conf} --model_ckpt=${checkpoint_test}  > log_save_${conf} &
