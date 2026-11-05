"""Train the colour model (hairdeck.models.colormodel.ColorDiT): the image DiT's network on the clean card tokens of a
hairstyle (t = 0) with one rendered view of its character as the condition (hairdeck.color.data), regressing the colours
of the cards (hairdeck.color.targets); L1 in Lab / 100.

Released checkpoint color_model (one GPU, 183.5M parameters), three runs (scripts/train/launch_color_dit.sh):
  1. --route reg --name train_color_reg --image_init ckpt/train_image_model/ema_100000.pt --steps 20000 --lr 1e-4
     --photo input --mask hair   (per bin the lit and shade colour, shade weight 0.5)
  2. --route reg1 --name train_color_s1 --init ckpt/train_color_reg/ema_020000.pt --steps 20000 --lr 2e-4 --photo both
     --mask person --hard_rep 3   (per bin the colour of the image's view; measured bins weight 1, model-filled 0.5)
  3. as 2 with --name train_color_model --init ckpt/train_color_s1/ema_020000.pt --steps 40000 --cosine
     --ckpt_every 10000; the released checkpoint is its ema_040000.pt
--photo: brightness / contrast on the image only (input), on the image and the targets together (both) or none;
--mask: what the context's patch statistics are taken under: the hair label or the character (the render's alpha;
the person matte at inference); --hard_rep: the hairstyles whose lit and shade colours are apart (mean Delta E > 5)
this many times. Fixed settings: the train hairstyles with targets, a head fit, <= 192 cards and renders; batch 32,
AdamW (betas 0.9 / 0.99, no weight decay), warmup 1000 steps, EMA 0.9995, gradient clip 1.0, bf16 autocast, TF32;
hue rotation of the image and the targets (p 0.5, uniform angle); the train cards of the card VAE --vae (its latents
ckpt/<vae>/latents_train_head.npy). --init from a 'reg' checkpoint into 'reg1' takes the lit rows of its colour head.
Seed --seed: the data order, augmentation and dropout (torch.manual_seed for the rest).
Outputs: ckpt/<name>/{last.pt, ema_XXXXXX.pt}, logs/<name>.log.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RELEASE)
from hairdeck import paths  # noqa: E402
from hairdeck.ckpt import load as load_ckpt, training_dir  # noqa: E402
from hairdeck.color.data import Cards, RES, Views, color_stats, hard_items, items_of, lab2rgb, photo_aug, rgb2lab, rot_ab  # noqa: E402
from hairdeck.color.targets import BINS  # noqa: E402
from hairdeck.models.colormodel import ColorDiT  # noqa: E402
from hairdeck.models.imgmodel import DinoEncoder  # noqa: E402
from hairdeck.models.tokenizer import build_tokenizer  # noqa: E402

MODEL = dict(dim=768, depth=12, heads=12, dino='vitl16', res=RES, bins=BINS)  # the image DiT's network


def train(a):
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = 'cuda'
    tok = build_tokenizer(a.vae, head=True, device=dev)
    items = items_of('train')
    cards = Cards('train', tok, a.vae, dev, view=a.route == 'reg1', fill_w=a.fill_w)
    if a.route == 'reg1':  # hairstyles with the per-view targets
        items = items[cards.have[items]]
    if a.hard_rep > 1:
        items = np.concatenate([items] + [hard_items('train', items)] * (a.hard_rep - 1))
    ds = Views('train', items, seed=a.seed, photo=a.photo == 'input')
    enc = DinoEncoder(MODEL['dino'], RES).to(dev)
    mu, sd = color_stats()
    model = ColorDiT(tok.dim, argparse.Namespace(**MODEL), mu, sd, a.route).to(dev)
    if a.init:  # a colour model checkpoint (reg1 from a reg one: the lit rows of its colour head)
        sd0 = load_ckpt(paths.resolve(a.init))['ema']
        if a.route == 'reg1' and sd0['col_out.weight'].shape[0] == BINS * 6:
            sd0['col_out.weight'] = sd0['col_out.weight'].view(BINS, 6, -1)[:, :3].reshape(BINS * 3, -1)
            sd0['col_out.bias'] = sd0['col_out.bias'].view(BINS, 6)[:, :3].reshape(-1)
        model.load_state_dict(sd0)
    else:  # the image DiT's EMA weights
        model.dit.load_state_dict(load_ckpt(paths.resolve(a.image_init))['ema'], strict=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    cos = (lambda s: 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))) if a.cosine else (lambda s: 1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000) * cos(s))
    odir = training_dir(a.name)
    os.makedirs(odir, exist_ok=True)
    os.makedirs(paths.logs(), exist_ok=True)
    step = 0
    if os.path.isfile(f'{odir}/last.pt'):  # resume: weights, EMA, optimiser, step (the data order restarts at an epoch)
        st = torch.load(f'{odir}/last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(st['model'])
        ema.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        step, ds.epoch = st['step'], st['epoch']
        for _ in range(step):
            sched.step()
    args = dict(MODEL, **{k: v for k, v in vars(a).items() if k not in MODEL})  # with the card VAE (--vae)
    log = open(paths.logs(f'{a.name}.log'), 'a')
    print(json.dumps(args), 'items', len(ds), 'params %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6),
          'colour mu', np.round(mu, 2).tolist(), 'sd', np.round(sd, 2).tolist(), file=log, flush=True)
    gh = torch.Generator(device=dev).manual_seed(1 + a.seed)
    t0, acc = time.time(), []
    while step < a.steps:
        dl = DataLoader(ds, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                        persistent_workers=False, prefetch_factor=4,
                        generator=torch.Generator().manual_seed(ds.epoch + 1000 * a.seed))
        for b in dl:
            hid = b['hid'].to(dev)
            img = b['img'].to(dev, non_blocking=True).float()
            matte = (b['person'] if a.mask == 'person' else b['hair']).to(dev, non_blocking=True)
            c = cards(hid, b['v'] if a.route == 'reg1' else None)
            lab, lab_s = (c['lab_v'], c['lab_v']) if a.route == 'reg1' else (c['lab'], c['lab_s'])
            B = len(hid)
            if a.photo == 'both':
                img, (lab, lab_s) = photo_aug(img, (lab, lab_s), gh)
            sel = torch.rand(B, device=dev, generator=gh) < 0.5  # hue rotation of the image and the targets together
            th = (torch.rand(B, device=dev, generator=gh) * 2 - 1) * math.pi * sel
            img = lab2rgb(rot_ab(rgb2lab(img.permute(0, 2, 3, 1) / 255), th)).permute(0, 3, 1, 2) * 255
            lab, lab_s = rot_ab(lab.flatten(1, 2), th).view_as(lab), rot_ab(lab_s.flatten(1, 2), th).view_as(lab_s)
            w = (c['w_v'] if a.route == 'reg1' else c['w'])[..., None]  # (B, N, BINS, 1)
            tokens = enc(img)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ctx = model.encode_ctx(tokens, img, matte)
                pl, ps = model.unstd(model.raw(c['x'], ctx).float())
                den = w.sum().clamp(min=1) * 3
                ll = ((pl - lab).abs() * w).sum() / den / 100
                if a.route == 'reg1':
                    ls = ll.detach()
                    loss = ll
                else:
                    ls = ((ps - lab_s).abs() * w).sum() / den / 100
                    loss = ll + 0.5 * ls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    if pm.requires_grad:
                        pe.lerp_(pm, 1 - a.ema)
            acc.append([ll.item(), ls.item(), gn.item()])
            step += 1
            if step % 100 == 0:
                m = np.mean(acc, 0)
                print(f'step {step} L1 Lab/100 lit {m[0]:.4f} shade {m[1]:.4f} gn {m[2]:.3f} {(time.time() - t0) / 100:.3f}s/it',
                      file=log, flush=True)
                acc, t0 = [], time.time()
            if step % a.ckpt_every == 0 or step == a.steps:
                torch.save({'ema': ema.state_dict(), 'step': step, 'args': args}, f'{odir}/ema_{step:06d}.pt')
                torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'step': step,
                            'epoch': ds.epoch}, f'{odir}/.last.pt')
                os.replace(f'{odir}/.last.pt', f'{odir}/last.pt')
            if step >= a.steps:
                break
        ds.epoch += 1
    print('TRAIN_DONE', step, file=log, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--route', default='reg1', choices=['reg', 'reg1'])
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--ema', type=float, default=0.9995)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--ckpt_every', type=int, default=5000)
    ap.add_argument('--vae', default='train_card_vae')
    ap.add_argument('--image_init', default='ckpt/train_image_model/ema_100000.pt', help='the image DiT to start from '
                    '(without --init); the released ckpt/image_model/model.safetensors works too')
    ap.add_argument('--init', default='', help='a colour model checkpoint to start from')
    ap.add_argument('--photo', default='input', choices=['input', 'both', 'none'])
    ap.add_argument('--mask', default='hair', choices=['hair', 'person'])
    ap.add_argument('--fill_w', type=float, default=0.5, help='reg1: the weight of the model-filled bins')
    ap.add_argument('--hard_rep', type=int, default=1)
    ap.add_argument('--cosine', action='store_true', help='cosine decay of the lr to 0 at --steps')
    ap.add_argument('--seed', type=int, default=0)
    train(ap.parse_args())


if __name__ == '__main__':
    main()
