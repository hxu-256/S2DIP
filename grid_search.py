"""Grid search over S2DIP hyperparameters for the bCARS denoising task.

Sweeps any combination of {mu, alpha_3, thres_tv, thres_sstv, lr, ...} (cartesian product of
the lists in GRID below) and runs the same S2DIP method as bcars_denoising.py once per config,
tracking the best PSNR-vs-GT (early stopping).

Parallelism: a dispatcher keeps BOTH GPUs busy, launching one worker subprocess per GPU
(each pinned via CUDA_VISIBLE_DEVICES) and feeding configs from a queue. Each run uses
~16.5 GB, so one config per 32 GB card at a time.

Run (DDS2M env, headless, both GPUs):
    MPLBACKEND=Agg \\
    LD_PRELOAD=/home/hxu256/miniconda3/envs/DDS2M/lib/libstdc++.so.6 \\
    /home/hxu256/miniconda3/envs/DDS2M/bin/python grid_search.py --num-iter 3000 --gpus 0,1

Outputs:
    results/bcars/gridsearch/<config_name>/bcars_best.mat   (notebook-compatible)
    results/bcars/gridsearch/<config_name>/run.log
    results/bcars/gridsearch/summary.csv
"""
from __future__ import print_function
import os, sys, csv, time, json, itertools, subprocess

os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import numpy as np
import scipy.io
import torch
import torch.optim
from models.skip import skip
from utils.tv_utils import psnr3d, TV_Loss, SSTV_Loss, soft

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# ── paths ────────────────────────────────────────────────────────────────
DATA_PATH = "/mnt/d/GaTech Dropbox/Haoyu Xu/workspace/DDS2M/exp/datasets/ood_msi/bcars_denoising.mat"
OUT_ROOT  = "./results/bcars/gridsearch"

# ── search space ───────────────────────────────────────────────────────────
# Every key listed here is swept; the cartesian product of the lists forms the configs.
# To search more parameters, just give a key more than one value (e.g. 'mu': [0.05, 0.12, 0.5]).
GRID = {
    'thres_tv':   [0.001],
    'thres_sstv': [0.005, 0.01,0.02,0.05],
    'mu':         [0.12,0.05,0.02],
    'alpha_3':    [0.01,0.005],
}

DEFAULTS = {
    'mu':            0.12,
    'alpha_3':       0.01,   # sparse-outlier soft threshold = 2*alpha_3
    'thres_tv':      0.1,    # spatial TV threshold
    'thres_sstv':    0.1,    # spatial-spectral TV threshold
    'lr':            0.005,
    'reg_noise_std': 0.01,
    'tt':            4,       # network depth (down/up levels)
}


def get_noise_2d(input_depth, spatial_size, var=1. / 10.):
    """Uniform-noise network input, shape [1, input_depth, H, W] (inlined from common_utils)."""
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

    # (H, W, bands) -> (bands, H, W); 320 is already a multiple of 32 (no crop)
    img_noisy_np = img_noisy.transpose(2, 0, 1).astype(np.float32)
    img_np       = img.transpose(2, 0, 1).astype(np.float32)
    img_noisy_var = torch.from_numpy(img_noisy_np).type(dtype)[None, None, :]  # [1,1,B,H,W]
    return img_np, img_noisy_np, img_noisy_var, norm_min, norm_max


def run_config(cfg, data, num_iter, eval_every, out_dir):
    """Run S2DIP once for one hyperparameter config; return (best_psnr, best_iter, final_psnr)."""
    img_np, img_noisy_np, img_noisy_var, norm_min, norm_max = data
    B, H, W = img_noisy_var.shape[2], img_noisy_var.shape[3], img_noisy_var.shape[4]
    input_depth = B

    mu          = cfg['mu']
    thres       = 2 * cfg['alpha_3']
    thres_tv    = cfg['thres_tv']
    thres_sstv  = cfg['thres_sstv']
    tt          = cfg['tt']

    net = skip(input_depth, input_depth, input_depth,
               num_channels_down=[128] * tt, num_channels_up=[128] * tt,
               num_channels_skip=[4] * tt,
               filter_size_up=3, filter_size_down=3, filter_skip_size=1,
               upsample_mode='bilinear', need_sigmoid=False, need_bias=True,
               pad='reflection', act_fun='LeakyReLU').type(dtype)

    net_input = get_noise_2d(input_depth, (H, W)).type(dtype).detach()
    net_input_saved = net_input.clone()
    noise = net_input.clone()
    net_input.requires_grad = True

    mask = torch.ones_like(img_noisy_var).type(dtype)
    TV, SSTV, soft_thres = TV_Loss(), SSTV_Loss(), soft()

    # ADMM dual variables (zeros), shapes follow the TV/SSTV gradient sizes
    D_2 = torch.zeros([1, 1, B,     H - 1, W    ]).type(dtype)
    D_3 = torch.zeros([1, 1, B,     H,     W - 1]).type(dtype)
    D_4 = torch.zeros([1, 1, B - 1, H - 1, W    ]).type(dtype)
    D_5 = torch.zeros([1, 1, B - 1, H,     W - 1]).type(dtype)

    params = [x for x in net.parameters()] + [net_input]
    optimizer = torch.optim.Adam(params, lr=cfg['lr'])

    best = {'psnr': -1.0, 'iter': -1, 'out': None}
    history = []
    img_np_c = np.clip(img_np.astype(np.float32), 0, 1)

    for it in range(num_iter):
        optimizer.zero_grad()
        ni = net_input_saved + (noise.normal_() * cfg['reg_noise_std'])

        out_ = net(ni)[None, :]
        D_x_,  D_y_  = TV(out_)
        D_xz_, D_yz_ = SSTV(out_)
        D_x, D_y     = D_x_.detach(),  D_y_.detach()
        D_xz, D_yz   = D_xz_.detach(), D_yz_.detach()
        out = out_.detach()

        S   = soft_thres(img_noisy_var - out, thres)
        V_2 = soft_thres(D_x  + D_2 / mu, thres_tv)
        V_3 = soft_thres(D_y  + D_3 / mu, thres_tv)
        V_4 = soft_thres(D_xz + D_4 / mu, thres_sstv)
        V_5 = soft_thres(D_yz + D_5 / mu, thres_sstv)

        total_loss  = mu / 2 * torch.norm(D_x_  - (V_2 - D_2 / mu), 2)
        total_loss += mu / 2 * torch.norm(D_y_  - (V_3 - D_3 / mu), 2)
        total_loss += 10 * mu / 2 * torch.norm(D_xz_ - (V_4 - D_4 / mu), 2)
        total_loss += 10 * mu / 2 * torch.norm(D_yz_ - (V_5 - D_5 / mu), 2)
        total_loss += torch.norm(img_noisy_var * mask - out_ * mask - S, 2)
        total_loss.backward()

        D_2 = (D_2 + mu * (D_x  - V_2)).detach()
        D_3 = (D_3 + mu * (D_y  - V_3)).detach()
        D_4 = (D_4 + mu * (D_xz - V_4)).detach()
        D_5 = (D_5 + mu * (D_yz - V_5)).detach()

        optimizer.step()

        if it % eval_every == 0 or it == num_iter - 1:
            out_np = out.detach().cpu().squeeze().numpy()
            psnr = psnr3d(img_np_c, np.clip(out_np, 0, 1))
            history.append((it, psnr))
            if psnr > best['psnr']:
                best = {'psnr': psnr, 'iter': it, 'out': out_np}
            print('  iter %05d  PSNR_gt %.3f  (best %.3f @ %d)' % (it, psnr, best['psnr'], best['iter']),
                  flush=True)

    # save best (notebook-compatible: same keys as bcars_denoising.py final save)
    os.makedirs(out_dir, exist_ok=True)
    best_cube = best['out'].transpose(1, 2, 0)
    scipy.io.savemat(os.path.join(out_dir, 'bcars_best.mat'), {
        'denoised':      best_cube.astype(np.float32),
        'denoised_phys': (best_cube * (norm_max - norm_min) + norm_min).astype(np.float32),
        'img_clean':     img_np.transpose(1, 2, 0).astype(np.float32),
        'y_0_real':      img_noisy_np.transpose(1, 2, 0).astype(np.float32),
        'norm_min':      np.array([[norm_min]]),
        'norm_max':      np.array([[norm_max]]),
        'psnr_history':  np.array(history, dtype=np.float32),
        'best_psnr':     np.array([[best['psnr']]]),
        'best_iter':     np.array([[best['iter']]]),
        'config':        repr(cfg),
    })

    final_psnr = history[-1][1]
    del net, net_input, optimizer, D_2, D_3, D_4, D_5
    torch.cuda.empty_cache()
    return best['psnr'], best['iter'], final_psnr


# ── config enumeration ───────────────────────────────────────────────────────
def build_configs():
    """Return list of (name, cfg) for the full cartesian product of GRID."""
    keys = list(GRID.keys())
    # name by ALL swept parameters so every folder is unique and self-describing
    # (avoids collisions when more parameters are added to GRID)
    configs = []
    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(DEFAULTS)
        cfg.update(dict(zip(keys, combo)))
        name = '_'.join('%s%s' % (k, cfg[k]) for k in keys)
        configs.append((name, cfg))
    return keys, configs


# ── single-config worker (one GPU) ───────────────────────────────────────────
def worker_main(args):
    cfg = json.loads(args.config_json)
    data = load_data()
    best_psnr, best_iter, final_psnr = run_config(cfg, data, args.num_iter, args.eval_every, args.out_dir)
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump({'best_psnr': best_psnr, 'best_iter': best_iter, 'final_psnr': final_psnr,
                   'config': cfg}, f)
    print('DONE best %.3f @ %d | final %.3f' % (best_psnr, best_iter, final_psnr), flush=True)


# ── dispatcher (multi-GPU) ────────────────────────────────────────────────────
def dispatch_main(args):
    gpus = [g.strip() for g in args.gpus.split(',') if g.strip() != '']
    os.makedirs(OUT_ROOT, exist_ok=True)
    keys, configs = build_configs()
    print('Grid: %d configs over %s | num_iter=%d | GPUs=%s\n' % (len(configs), keys, args.num_iter, gpus))

    queue = list(configs)            # (name, cfg)
    running = {}                     # gpu -> (proc, name, cfg, logfile, t0)
    done = {}                        # name -> metrics dict

    # Resume support: skip configs that already finished (metrics.json present).
    # Partially-run configs (folder exists but no metrics.json) are re-run.
    if not args.no_skip_existing:
        skipped = []
        kept = []
        for name, cfg in queue:
            mpath = os.path.join(OUT_ROOT, name, 'metrics.json')
            if os.path.exists(mpath):
                done[name] = json.load(open(mpath)); done[name].setdefault('seconds', None)
                skipped.append(name)
            else:
                kept.append((name, cfg))
        queue = kept
        if skipped:
            print('Skipping %d already-finished config(s); %d left to run.\n' % (len(skipped), len(queue)))

    def launch(gpu, name, cfg):
        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        cmd = [sys.executable, os.path.abspath(__file__), '--worker',
               '--config-json', json.dumps(cfg), '--out-dir', out_dir,
               '--num-iter', str(args.num_iter), '--eval-every', str(args.eval_every)]
        logf = open(os.path.join(out_dir, 'run.log'), 'w')
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        running[gpu] = (proc, name, cfg, logf, time.time())
        print('[gpu %s] start %s' % (gpu, name), flush=True)

    while queue or running:
        # fill idle GPUs
        for gpu in gpus:
            if gpu not in running and queue:
                launch(gpu, *queue.pop(0))
        # poll
        for gpu, (proc, name, cfg, logf, t0) in list(running.items()):
            if proc.poll() is not None:
                logf.close()
                dt = time.time() - t0
                mpath = os.path.join(OUT_ROOT, name, 'metrics.json')
                if proc.returncode == 0 and os.path.exists(mpath):
                    m = json.load(open(mpath)); m['seconds'] = round(dt, 1)
                    print('[gpu %s] done  %s  best %.3f @ %d  (%.0fs)' %
                          (gpu, name, m['best_psnr'], m['best_iter'], dt), flush=True)
                else:
                    m = {'best_psnr': float('nan'), 'best_iter': -1, 'final_psnr': float('nan'),
                         'config': cfg, 'seconds': round(dt, 1)}
                    print('[gpu %s] FAILED %s (rc=%s) — see run.log' % (gpu, name, proc.returncode), flush=True)
                done[name] = m
                del running[gpu]
        time.sleep(2)

    # summary CSV
    summary_path = os.path.join(OUT_ROOT, 'summary.csv')
    fieldnames = keys + ['config', 'best_psnr', 'best_iter', 'final_psnr', 'seconds']
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, cfg in configs:
            m = done.get(name, {})
            row = {k: cfg[k] for k in keys}
            row.update({'config': name, 'best_psnr': m.get('best_psnr'), 'best_iter': m.get('best_iter'),
                        'final_psnr': m.get('final_psnr'), 'seconds': m.get('seconds')})
            w.writerow(row)

    print('\n==== summary (sorted by best_psnr) ====')
    for name in sorted(done, key=lambda n: -(done[n]['best_psnr'] if done[n]['best_psnr'] == done[n]['best_psnr'] else -1)):
        m = done[name]
        print('  %-28s best %7.3f @ %5d   final %7.3f' % (name, m['best_psnr'], m['best_iter'], m['final_psnr']))
    print('\nSummary CSV: %s' % summary_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-iter', type=int, default=3000)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--gpus', default='0,1', help='comma-separated GPU indices for parallel search')
    ap.add_argument('--no-skip-existing', action='store_true',
                    help='re-run every config even if its metrics.json already exists')
    # worker-only args
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--config-json', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--out-dir', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker_main(args)
    else:
        dispatch_main(args)


if __name__ == '__main__':
    main()
