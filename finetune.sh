
lr=1e-4
bs=32
epochs=15
dropout=0.1
diffusion_steps=100
lambda_v2v_loss=1.0
lr_policy_diffusion=linear
predict_xstart=true

# model settings
diffusion_net=resblocks
encoder_net=none
channel_mult="1,2,2,4,4,8"
diffusion_model_channels=64
attention_resolutions="32,16"

# training settings
epochs_save=5
epochs_metric=5
exp_run=real_only
mode=finetune
name=${exp_run}/${encoder_net}_${diffusion_net}/s${diffusion_steps}_${lr_policy_diffusion}_b${bs}_d${dropout}
load_model_path=sim_only/${encoder_net}_${diffusion_net}/s${diffusion_steps}_none_b${bs}_d${dropout} 
viz_type=image # None | image | video

# finetune settings
real_train_file=real_1_80.txt

# eval settings
exp_type=normal # normal | hospital
real_val_file=real_81_102.txt 

python main.py \
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
    --dropout $dropout \
    --use_diffusion true \
    --predict_xstart $predict_xstart \
    --diffusion_steps $diffusion_steps \
    --lr_policy_diffusion $lr_policy_diffusion \
    --lambda_v2v_loss $lambda_v2v_loss \
    --encoder_net $encoder_net \
    --diffusion_net $diffusion_net \
    --diffusion_model_channels $diffusion_model_channels \
    --channel_mult $channel_mult \
    --attention_resolutions $attention_resolutions \
    --viz_type $viz_type
