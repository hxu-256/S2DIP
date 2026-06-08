"""Grid search over pure-DIP (+ optional spectral-TV) hyperparameters for bCARS denoising.

Same DIP model as dip_baseline.py (loss = ||net(z) - y||^2 + spec_tv * spectral_TV(out)),
NO spatial TV / SSTV / ADMM. Sweeps any combination of the keys in GRID below.

Parallelism: a dispatcher keeps BOTH GPUs busy, one worker subprocess per GPU
(each pinned via CUDA_VISIBLE_DEVICES), fed from a config queue.

Run (DDS2M env, headless, both GPUs):
    MPLBACKEND=Agg \\
    LD_PRELOAD=/home/hxu256/miniconda3/envs/DDS2M/lib/libstdc++.so.6 \\
    /home/hxu256/miniconda3/envs/DDS2M/bin/python dip_grid_search.py --num-iter 3000 --gpus 0,1

Outputs (per config, named by ALL swept params to avoid collisions):
    results/bcars/dip_gridsearch/<config_name>/dip_best.mat   (notebook-compatible)
    results/bcars/dip_gridsearch/<config_name>/run.log
    results/bcars/dip_gridsearch/summary.csv
"""
from __future__ import print_function
import os, sys, csv, time, json, itertools, subprocess

os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
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

# ── paths ────────────────────────────────────────────────────────────────
DATA_PATH = "/mnt/d/GaTech Dropbox/Haoyu Xu/workspace/DDS2M/exp/datasets/ood_msi/bcars_denoising.mat"
OUT_ROOT  = "./results/bcars/dip_gridsearch_spectraltv_fine_10000"

# ── search space ───────────────────────────────────────────────────────────
# Every key listed here is swept; the cartesian product of the lists forms the configs.
GRID = {
    'spec_tv':    [0.001, 0.002, 0.005, 0.01],   # pure spectral-TV weight (0 = plain DIP)
    'spec_order': [1,2],                       # 1 = |d/dlambda|, 2 = |d2/dlambda2| (peak-preserving)
    'lr':         [0.005],
}

DEFAULTS = {
    'spec_tv':       0.0,
    'spec_order':    1,
    'lr':            0.005,
    'reg_noise_std': 0.01,
    'tt':            4,       # network depth (down/up levels)
}


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


def spectral_tv(x, order):
    """Pure spectral TV along the band axis (dim=1); zero spatial derivatives."""
    if order == 1:
        return (x[:, 1:] - x[:, :-1]).abs().mean()
    return (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).abs().mean()  # 2nd order (curvature)


def run_config(cfg, data, num_iter, eval_every, out_dir):
    """Run DIP (+ spectral TV) once for one config; return (best_psnr, best_iter, final_psnr)."""
    img_np, img_noisy_np, img_noisy_var, norm_min, norm_max = data
    B, H, W = img_noisy_var.shape[1], img_noisy_var.shape[2], img_noisy_var.shape[3]
    tt = cfg['tt']

    net = skip(B, B, B,
               num_channels_down=[128] * tt, num_channels_up=[128] * tt,
               num_channels_skip=[4] * tt,
               filter_size_up=3, filter_size_down=3, filter_skip_size=1,
               upsample_mode='bilinear', need_sigmoid=False, need_bias=True,
               pad='reflection', act_fun='LeakyReLU').type(dtype)

    net_input = get_noise_2d(B, (H, W)).type(dtype).detach()
    net_input_saved = net_input.clone()
    noise = net_input.clone()
    net_input.requires_grad = True

    params = [x for x in net.parameters()] + [net_input]
    optimizer = torch.optim.Adam(params, lr=cfg['lr'])
    mse = nn.MSELoss()

    best = {'psnr': -1.0, 'iter': -1, 'out': None}
    history = []
    img_np_c = np.clip(img_np.astype(np.float32), 0, 1)
    spec_w, spec_order, reg_std = cfg['spec_tv'], cfg['spec_order'], cfg['reg_noise_std']

    for it in range(num_iter):
        optimizer.zero_grad()
        ni = net_input_saved + (noise.normal_() * reg_std)
        out = net(ni)                       # [1, bands, H, W]
        data_loss = mse(out, img_noisy_var)
        stv = spectral_tv(out, spec_order) if spec_w > 0 else out.new_zeros(())
        loss = data_loss + spec_w * stv     # spatial path untouched; smoothing only along bands
        loss.backward()
        optimizer.step()

        if it % eval_every == 0 or it == num_iter - 1:
            out_np = out.detach().cpu().squeeze().numpy()
            psnr = psnr3d(img_np_c, np.clip(out_np, 0, 1))
            history.append((it, psnr))
            if psnr > best['psnr']:
                best = {'psnr': psnr, 'iter': it, 'out': out_np}
            print('  iter %05d  data %.4e  specTV %.4e  PSNR_gt %.3f  (best %.3f @ %d)'
                  % (it, data_loss.item(), stv.item(), psnr, best['psnr'], best['iter']), flush=True)

    # save best + final (notebook-compatible)
    os.makedirs(out_dir, exist_ok=True)

    def _save(out_np, fname, with_psnr=True):
        cube = out_np.transpose(1, 2, 0)
        d = {'denoised': cube.astype(np.float32),
             'denoised_phys': (cube * (norm_max - norm_min) + norm_min).astype(np.float32),
             'img_clean': img_np.transpose(1, 2, 0).astype(np.float32),
             'y_0_real':  img_noisy_np.transpose(1, 2, 0).astype(np.float32),
             'norm_min': np.array([[norm_min]]), 'norm_max': np.array([[norm_max]]),
             'psnr_history': np.array(history, dtype=np.float32), 'config': repr(cfg)}
        if with_psnr:
            d['best_psnr'] = np.array([[best['psnr']]]); d['best_iter'] = np.array([[best['iter']]])
        scipy.io.savemat(os.path.join(out_dir, fname), d)

    _save(best['out'], 'dip_best.mat')
    _save(out.detach().cpu().squeeze().numpy(), 'dip_final.mat', with_psnr=False)

    final_psnr = history[-1][1]
    del net, net_input, optimizer
    torch.cuda.empty_cache()
    return best['psnr'], best['iter'], final_psnr


# ── config enumeration ───────────────────────────────────────────────────────
def build_configs():
    keys = list(GRID.keys())
    configs = []
    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(DEFAULTS)
        cfg.update(dict(zip(keys, combo)))
        name = '_'.join('%s%s' % (k, cfg[k]) for k in keys)  # all swept params -> unique folder
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

    queue = list(configs)
    running = {}   # gpu -> (proc, name, cfg, logfile, t0)
    done = {}

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
        for gpu in gpus:
            if gpu not in running and queue:
                launch(gpu, *queue.pop(0))
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
        print('  %-40s best %7.3f @ %5d   final %7.3f' % (name, m['best_psnr'], m['best_iter'], m['final_psnr']))
    print('\nSummary CSV: %s' % summary_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-iter', type=int, default=3000)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--gpus', default='0,1', help='comma-separated GPU indices for parallel search')
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
