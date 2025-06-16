lr=1e-4
bs=32
epochs=100
lambda_v2v_loss=0.25
encoder_net=resnet18

# data file settings
real_train_file=real_1_80.txt
real_val_file=real_81_102.txt

# training settings
epochs_save=20 # save model every 20 epochs
epochs_metric=20 # calculate metrics every 20 epochs
exp_run=full-train-test # full-train-test | sim_only | real_only
mode=train # train test finetune
load_model_path=normal/sim_only/${encoder_net}_b${bs}_bodymap # use in finetune mode
name=${exp_run}/${real_train_file}/${encoder_net}_b${bs}_bodymap

python main_bodymap.py \
    --mode $mode \
    --exp_type normal \
    --exp_run $exp_run \
    --name $name \
    --lr $lr \
    --batch_size $bs \
    --epochs $epochs \
    --epochs_save $epochs_save \
    --epochs_metric $epochs_metric \
    --real_train_file $real_train_file \
    --real_val_file $real_val_file \
    --load_model_path $load_model_path \
    --viz_type image
