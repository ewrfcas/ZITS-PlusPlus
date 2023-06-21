# ZITS-PlusPlus
ZITS++: Image Inpainting by Improving the Incremental Transformer on Structural Priors (TPAMI2023)

[arxiv paper](https://arxiv.org/abs/2210.05950)

[ZITS: CVPR2022 Version](https://github.com/DQiaole/ZITS_inpainting)

## TODO

- [x] Releasing dataset and inference codes.
- [x] Releasing pre-trained weights.
- [ ] Releasing training codes.

## Dataset

Test data HR-Flickr: [Download](https://1drv.ms/u/s!AqmYPmoRZryegRz79ueT2gVqWR4T?e=LTZMZM).

Note that the HR-Flickr Dataset includes images obtained from [Flickr](https://www.flickr.com/). Use of the images must abide by the Flickr Terms of Use. 
We do not own the copyright of the images. 
They are solely provided for researchers and educators who wish to use the dataset for non-commercial research and/or educational purposes.

## Pre-trained Models

1. model_256: [Download](https://1drv.ms/u/s!AqmYPmoRZryegR1XjcmbjLV2OTk1?e=ToOT2d).

2. model_512: [Download](https://1drv.ms/u/s!AqmYPmoRZryegR9OPEgqq7LvgqJR?e=4Erzvr).

3. LSM-HAWP (line detector from MST): [Download](https://drive.google.com/drive/folders/1yg4Nc20D34sON0Ni_IOezjJCFHXKGWUW).

## Install

```
conda create -n zitspp python=3.8
conda activate zitspp
pip install -r requirements.txt
cd nms/src
source build.sh
```

## Test

Please use model_256 for images whose short sides are 256 or shorter. For larger images, using model_512 instead.

256 images
```
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/config_zitspp.yml \
                                      --exp_name <your model name> \
                                      --ckpt_resume ckpts/model_256/models/last.ckpt \
                                      --save_path ./outputs/model_256 \
                                      --img_dir <input image path> \
                                      --mask_dir <input mask path> \
                                      --wf_ckpt ckpts/best_lsm_hawp.pth \
                                      --use_ema
```

512 images
```
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/config_zitspp_finetune.yml \
                                      --exp_name <your model name> \
                                      --ckpt_resume ckpts/model_512/models/last.ckpt \
                                      --save_path ./outputs/model_512 \
                                      --img_dir <input image path> \
                                      --mask_dir <input mask path> \
                                      --wf_ckpt ckpts/best_lsm_hawp.pth \
                                      --use_ema 
```

## Acknowledgments

* This repo is built upon [MST](https://github.com/ewrfcas/MST_inpainting), [LaMa](https://github.com/saic-mdal/lama), and [ZITS](https://github.com/DQiaole/ZITS_inpainting).

## Cite

If you found our program helpful, please consider citing:

```
@article{cao2023zits++,
  title={ZITS++: Image Inpainting by Improving the Incremental Transformer on Structural Priors},
  author={Cao, Chenjie and Dong, Qiaole and Fu, Yanwei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```


