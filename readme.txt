Start :

Install pip env by requirements.txt

You can obtain the benchmark datasets from Google Drive 'https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy' provided in paper Autoformer, then create a folder named 'dataset' to put them in

For traffic(720), weather, electricity, solar, illness, etth2, PEMS03, PEMS04, PEMS07, PEMS08 --
running 'scripts\xxx.sh'

For traffic(96,192,336), exchange, etth1, ettm1, ettm2 --
running 'cd reconstruction'
running 'python rnn_pretrain.py --dset traffic --mask_ratio 0.4 --patch_len 16 --stride 8 --hidden_size 512 --context_points 96 --batch_size 32 --n_epochs_pretrain 100'
running 'python rnn_finetune.py --dset traffic --patch_len 16 --stride 8 --hidden_size 512 --context_points 96 --pretrained_model patchtst_pretrained_cw96_patch16_stride8_epochs-pretrain100_mask0.4_model1 --target_points 96 --batch_size 32 --n_epochs_finetune 100'
please replace '--dset traffic' with '--dset xxx', where xxx is the dataset you want to test; replacing '--target_points 96' with '--target_points xxx', where xxx is the prediction length you want to test. Note that for 96 prediction length of Traffic, the batch_size is 36; for ETTh1, the hidden_size is 64
if you want to test directly, please add '--is_finetune 0' in command. This will test the model through the prepared ckpts in '/saved_models' (ignore the patchtst in the name of ckpts, that is RNN in practice).

You can directly check the main results reported in the paper by the logs in 'logs/LongForecasting/' or the txt named 'result.txt'. 
