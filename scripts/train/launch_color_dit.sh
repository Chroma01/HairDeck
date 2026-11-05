#!/bin/bash
# Colour model color_model: the three runs of the released checkpoint (one GPU each, 183.5M parameters, batch 32) into
# ckpt/train_color_reg, ckpt/train_color_s1 and ckpt/train_color_model (its ema_040000.pt is the released checkpoint),
# from the image DiT of launch_img_dit.sh (train_color_dit.py has the recipe).
# Needs: the colour targets data/color/ (scripts/data.py --steps colors), the renders, data/cards_train.npz, the head
# frame (scripts/data.py), the head-frame latents of the card VAE (launch_vae.sh), ckpt/train_image_model/ema_100000.pt
# (launch_img_dit.sh; or --image_init ckpt/image_model/model.safetensors, the released image model) and DINOv3.
# usage: launch_color_dit.sh [GPU]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${1:-0} OMP_NUM_THREADS=2
$PY "$HERE/train_color_dit.py" --route reg --name train_color_reg --image_init ckpt/train_image_model/ema_100000.pt \
  --steps 20000 --lr 1e-4 --photo input --mask hair
$PY "$HERE/train_color_dit.py" --route reg1 --name train_color_s1 --init ckpt/train_color_reg/ema_020000.pt \
  --steps 20000 --lr 2e-4 --photo both --mask person --hard_rep 3
$PY "$HERE/train_color_dit.py" --route reg1 --name train_color_model --init ckpt/train_color_s1/ema_020000.pt \
  --steps 40000 --lr 2e-4 --photo both --mask person --hard_rep 3 --cosine --ckpt_every 10000
