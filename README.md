# S2DIP — BCARS fork

Fork of [YisiLuo/S2DIP](https://github.com/YisiLuo/S2DIP) — *Hyperspectral Mixed Noise Removal
Via Spatial-Spectral Constrained Unsupervised Deep Image Prior*, IEEE JSTARS 2021 — carrying
the deep image prior (DIP) baseline used in *Denoising broadband CARS hyperspectral images on
variance-stabilized data: a unified benchmark* ([journal], [year]).

## What this fork adds

One file: **`dip_baseline.py`** (169 lines). Upstream solves a spatial-spectral TV model with
ADMM (`demo.py`, `S2DIP_main.py`); for BCARS that spatial prior over-smooths, so the paper uses
a plain DIP fit with a **spectral-only** TV penalty:

- the untrained encoder–decoder (`models/skip*`) and the training loop are upstream's,
- the loss is `‖f(z) − y‖² + λ · TV_spectral(f(z))`, where `TV_spectral` is
  `mean |Δ_λ x|` (`--spec-order 1`) or `mean |Δ²_λ x|` (`--spec-order 2`, curvature — preserves
  peak shape where first order clips it),
- nothing constrains the spatial axes, so sharp features such as bead edges survive,
- checkpoints are written along the whole trajectory, because DIP overfits and the useful
  iterate is an intermediate one.

Everything else — `models/`, `utils/` (including `utils/tv_utils.py`), `demo.py`, the sample
cubes — is upstream code, unmodified.

## Running it on a cube

Input is a MATLAB `.mat` with `y_0_real` (noisy cube, `(H, W, C)`, normalized to `[0, 1]`),
`img_clean` (ground truth, or zeros if you have none), `norm_min` / `norm_max` and `wn`. The
pipeline repository's `h5tomat_detrend.py` writes exactly this; DDS2M reads the same schema.

```bash
python dip_baseline.py \
    --data ./data/cube.mat --out-dir ./results/dip \
    --spec-tv 0.005 --spec-order 1 --lr 0.005 \
    --num-iter 10000 --eval-every 50 --save-every 1000 --gpu 0
```

Outputs in `--out-dir`: `dip_iter<N>.mat` along the trajectory, `dip_final.mat`, and
`dip_best.mat` (highest PSNR against `img_clean`), each holding the denoised cube in `denoised`
plus the de-normalization constants. `run.log` and `metrics.json` record the trajectory.

**Which iterate to take.** `dip_best.mat` is only meaningful when `img_clean` holds a real
reference. On experimental cubes `img_clean` is zeros, so its PSNR is computed against zeros
and the "best" iterate is noise-selected — take a checkpoint chosen by inspecting the
reconstruction, or score against a measurement-based pseudo-ground-truth. DIP overfits quickly
(on glycerol the useful iterate is around 50–100), so `--save-every` matters more than
`--num-iter`.

## Settings used in the paper

| cube | `--spec-tv` | `--spec-order` | `--lr` | iterate |
|---|---|---|---|---|
| simulated (5 FOVs) | 0.005 | 1 | 0.005 | `dip_best` (exact ground truth available) |
| glycerol | 0.05 | 2 | 0.003 | `dip_final` |
| 1 µm bead | 0 | 1 | 0.003 | checkpoint at iteration 1000 |
| *C. elegans* | 0 | 1 | 0.003 | checkpoint at iteration 2000 |

All on the noclip (unbounded) variant of the variance-stabilized cube; the grid searches behind
these choices are not part of this repository.

## Related

- Processing pipeline (VST, detrending, CCV, phase retrieval) and the paper's metrics:
  [pipeline repo URL]
- Denoised cubes and metric tables: [Zenodo DOI]

Please cite Luo et al. (JSTARS 2021) alongside our paper if you use this code.

---

## Upstream README

Official implementation of "Hyperspectral Mixed Noise Removal Via Spatial-Spectral Constrained
Unsupervised Deep Image Prior", IEEE JSTARS, 2021.
