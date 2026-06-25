# python train_CBDv2.py --max_epoch 800 --lr 1E-4 --sc_step 100 --batch_size 8 --num_cage_v 96 --in_type 1 --out_type 3 --last_activation 'relu' --data_toggle --use_data2 --log_dir ckpts_CBD3
# python train_CBDv2.py --max_epoch 800 --lr 1E-4 --sc_step 100 --batch_size 8 --num_cage_v 96 --in_type 1 --out_type 3 --last_activation 'sqrelu' --data_toggle --use_data2 --log_dir ckpts_CBD3
# python train_CBDv2.py --max_epoch 800 --lr 1E-4 --sc_step 100 --batch_size 8 --num_cage_v 128 --in_type 1 --out_type 3 --last_activation 'sqrelu' --data_toggle --use_data2 --log_dir ckpts_CBD3 --use_cage_normal_loss
# python train_CBDv2.py --max_epoch 800 --lr 1E-4 --sc_step 100 --batch_size 8 --num_cage_v 128 --in_type 1 --out_type 3 --last_activation 'sqrelu' --data_toggle --use_data2 --log_dir ckpts_CBD3    


## residual
# python train_CBDv2.py --max_epoch 800 --lr 1E-4 --sc_step 100 --batch_size 8 --num_cage_v 128 --in_type 3 --out_type 3 --last_activation 'sqrelu' --data_toggle --use_data1 --log_dir ckpts_CBD3 --version 2
python train_CBDv2.py --max_epoch 800 --lr 1E-4 --sc_step 100 --batch_size 8 --num_cage_v 128 --in_type 1 --out_type 3 --last_activation 'sqrelu' --data_toggle --use_data1 --log_dir ckpts_CBD3 --version 2