"""Pure Deep-Image-Prior baseline for bCARS denoising (no TV/SSTV, no ADMM).

Same network as bcars_denoising.py / grid_search.py, but the only loss is the DIP
data-fidelity fit  ||net(z) - y||^2  with early stopping. Use this as the reference
point: if it matches your best low-mu S2DIP run, the TV/SSTV+ADMM machinery isn't
buying anything for this data.

Saves best-PSNR + periodic checkpoints + final, all notebook-compatible
(same keys as bcars_denoising.py), so visualize_results.ipynb can point at any of them.

Run (DDS2M env, headless):
    MPLBACKEND=Agg \\
    LD_PRELOAD=/home/hxu256/miniconda3/envs/DDS2M/lib/libstdc++.so.6 \\
    /home/hxu256/miniconda3/envs/DDS2M/bin/python dip_baseline.py --num-iter 3000 --gpu 0
"""
from __future__ import print_function
import os, sys, argparse

# pick GPU before importing torch
if '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--gpu') + 1]
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import scipy.io
import torch
import torch.optim
import torch.nn as nn
from models.skip import skip
from utils.tv_utils import psnr3d

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

DATA_PATH = "/mnt/d/GaTech Dropbox/Haoyu Xu/workspace/DDS2M/exp/datasets/ood_msi/bcars_denoising.mat"
OUT_DIR   = "./results/bcars/dip"

# network / optimization defaults (match the S2DIP runs)
TT            = 4
REG_NOISE_STD = 0.01


def get_noise_2d(input_depth, spatial_size, var=1. / 10.):
    net_input = torch.zeros([1, input_depth, spatial_size[0], spatial_size[1]])
    net_input.uniform_()
    net_input *= var
    return net_input


def load_data():
    mat = scipy.io.loadmat(DATA_PATH)
    img_noisy = mat["y_0_real"].astype(np.float64)   # (320, 320, 695) noisy
    img       = mat["img_clean"].astype(np.float64)  # ground truth
    norm_min  = float(mat["norm_min"])
    norm_max  = float(mat["norm_max"])
    img_noisy_np = img_noisy.transpose(2, 0, 1).astype(np.float32)  # (bands, H, W)
    img_np       = img.transpose(2, 0, 1).astype(np.float32)
    img_noisy_var = torch.from_numpy(img_noisy_np).type(dtype)[None, :]  # [1, bands, H, W]
    return img_np, img_noisy_np, img_noisy_var, norm_min, norm_max


def save_cube(out_np, path, norm_min, norm_max, extra=None):
    cube = out_np.transpose(1, 2, 0)  # (H, W, bands)
    d = {
        'denoised':      cube.astype(np.float32),
        'denoised_phys': (cube * (norm_max - norm_min) + norm_min).astype(np.float32),
        'norm_min':      np.array([[norm_min]]),
        'norm_max':      np.array([[norm_max]]),
    }
    if extra:
        d.update(extra)
    scipy.io.savemat(path, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-iter', type=int, default=3000)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--save-every', type=int, default=500, help='dump a cube checkpoint every N iters (0=off)')
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--spec-tv', type=float, default=0.0,
                    help='weight of the pure spectral-TV penalty along the band axis (0 = pure DIP)')
    ap.add_argument('--spec-order', type=int, default=1, choices=[1, 2],
                    help='1 = |d/dlambda| (piecewise-const, may clip Raman peaks); '
                         '2 = |d2/dlambda2| (piecewise-linear, preserves peak shape)')
    ap.add_argument('--gpu', default=None)
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    img_np, img_noisy_np, img_noisy_var, norm_min, norm_max = load_data()
    B, H, W = img_noisy_var.shape[1], img_noisy_var.shape[2], img_noisy_var.shape[3]
    img_np_c = np.clip(img_np.astype(np.float32), 0, 1)
    print('noisy_PSNR:', psnr3d(img_np_c, np.clip(img_noisy_np, 0, 1)))

    net = skip(B, B, B,
               num_channels_down=[128] * TT, num_channels_up=[128] * TT,
               num_channels_skip=[4] * TT,
               filter_size_up=3, filter_size_down=3, filter_skip_size=1,
               upsample_mode='bilinear', need_sigmoid=False, need_bias=True,
               pad='reflection', act_fun='LeakyReLU').type(dtype)

    net_input = get_noise_2d(B, (H, W)).type(dtype).detach()
    net_input_saved = net_input.clone()
    noise = net_input.clone()
    net_input.requires_grad = True

    params = [x for x in net.parameters()] + [net_input]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    mse = nn.MSELoss()

    def spectral_tv(x):
        """Pure spectral TV along the band axis (dim=1); zero spatial derivatives."""
        if args.spec_order == 1:
            return (x[:, 1:] - x[:, :-1]).abs().mean()
        return (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).abs().mean()  # 2nd order (curvature)

    best = {'psnr': -1.0, 'iter': -1, 'out': None}
    history = []

    for it in range(args.num_iter):
        optimizer.zero_grad()
        ni = net_input_saved + (noise.normal_() * REG_NOISE_STD)
        out = net(ni)                       # [1, bands, H, W]
        data = mse(out, img_noisy_var)      # DIP data fidelity
        stv  = spectral_tv(out) if args.spec_tv > 0 else out.new_zeros(())
        loss = data + args.spec_tv * stv    # spatial path untouched; smoothing only along bands
        loss.backward()
        optimizer.step()

        if it % args.eval_every == 0 or it == args.num_iter - 1:
            out_np = out.detach().cpu().squeeze().numpy()
            psnr = psnr3d(img_np_c, np.clip(out_np, 0, 1))
            history.append((it, psnr))
            if psnr > best['psnr']:
                best = {'psnr': psnr, 'iter': it, 'out': out_np}
            print('iter %05d  data %.4e  specTV %.4e  PSNR_gt %.3f  (best %.3f @ %d)'
                  % (it, data.item(), stv.item(), psnr, best['psnr'], best['iter']), flush=True)

        if args.save_every and it % args.save_every == 0:
            out_np = out.detach().cpu().squeeze().numpy()
            save_cube(out_np, os.path.join(args.out_dir, 'dip_iter%05d.mat' % it), norm_min, norm_max)

    # best (early-stopping) cube
    save_cube(best['out'], os.path.join(args.out_dir, 'dip_best.mat'), norm_min, norm_max, extra={
        'img_clean':    img_np.transpose(1, 2, 0).astype(np.float32),
        'y_0_real':     img_noisy_np.transpose(1, 2, 0).astype(np.float32),
        'psnr_history': np.array(history, dtype=np.float32),
        'best_psnr':    np.array([[best['psnr']]]),
        'best_iter':    np.array([[best['iter']]]),
    })
    # final-iteration cube (since PSNR may not be representative, keep both)
    final_np = out.detach().cpu().squeeze().numpy()
    save_cube(final_np, os.path.join(args.out_dir, 'dip_final.mat'), norm_min, norm_max, extra={
        'img_clean':    img_np.transpose(1, 2, 0).astype(np.float32),
        'y_0_real':     img_noisy_np.transpose(1, 2, 0).astype(np.float32),
        'psnr_history': np.array(history, dtype=np.float32),
    })
    print('\nbest %.3f dB @ iter %d | final %.3f dB' % (best['psnr'], best['iter'], history[-1][1]))
    print('saved dip_best.mat / dip_final.mat (+ checkpoints) to', args.out_dir)


if __name__ == '__main__':
    main()
