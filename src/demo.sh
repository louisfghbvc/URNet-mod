# URNet model (x2)
# python main.py --dir_data ../dataset --model urn --scale 2 --patch_size 128 --batch_size 32 --save urn_x2 --loss 1.0*L1+0.25*canny --lr 2e-4 --reset

# v2 + shuffle net
# python main.py --dir_data ../dataset --model urn2 --scale 2 --patch_size 128 --batch_size 32 --save urn2.1_x2 --loss 1.0*L1+0.25*canny --lr 5e-4 --reset

# V3 + MCA
# python main.py --dir_data ../dataset --model urn3 --scale 2 --patch_size 128 --batch_size 32 --save urn1x1_x2 --loss 1.0*L1+0.25*canny --lr 5e-4 --epochs 600 --resume 0 --pre_train ../experiment/urn1x1_x2/model/model_latest.pt --decay 400-600-800
python main.py --dir_data ../dataset --model urn3 --scale 2 --patch_size 128 --batch_size 32 --save urn1x1_x2 --loss 1.0*L1+0.25*canny --lr 5e-4 --epochs 600 --reset

  
# v5 + share weight
# python main.py --dir_data ../dataset --model urn5 --scale 2 --patch_size 128 --batch_size 32 --save urn5.1_x2 --loss 1.0*L1+0.25*canny --lr 5e-4 --reset

# v4 + mlp
# python main.py --dir_data ../dataset --model urn4 --scale 2 --patch_size 128 --batch_size 32 --save urn4.1_x2 --loss 1.0*L1+0.25*canny --lr 5e-4 --reset


# URNet model (x3) - from URNet (x2)
# python main.py --dir_data ../dataset --model urn3 --scale 3 --patch_size 192 --batch_size 32 --save urn1x1_x3 --loss 1.0*L1+0.25*canny --lr 2.5e-4 --reset --epochs 300 --pre_train ../experiment/urn1x1_x2/model/model_best.pt 
# resume
# python main.py --dir_data ../dataset --model urn --scale 3 --patch_size 192 --batch_size 32 --load urn_x3 --loss 1.0*L1+0.25*canny --lr 5e-4 --epochs 600 --resume 0 --pre_train ../experiment/urn_x3/model/model_latest.pt --decay 400-600-800



# URNet model (x4) - from URNet (x2)
#python main.py --model urn --scale 4 --patch_size 256 --batch_size 32 --save urn_x4 --loss 1.0*L1+0.25*canny --lr 5e-4 --reset --pre_train [pre-trained urn_x2 model dir]

# Test your own images
python main.py --dir_data ../dataset --model urn3 --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/urn1x1_x2/model/model_best.pt --test_only --save_results

# python main.py --dir_data ../dataset --model urn --data_test Set5+Set14+B100+Urban100--scale 2 --pre_train ../pre_trained/urn_x2.pt --test_only --save_results


# EDSR  in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR   in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]


# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
# python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results

# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt


# RFDN (paper + avg)
# python main.py --dir_data ../dataset --template RFDN --model RFDN --scale 2 --patch_size 128 --batch_size 32 --save RFDN_x2 --lr 5e-4 --reset
# python main.py --dir_data ../dataset --model RFDN --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/RFDN_x2/model/model_best.pt --test_only --save_results
