# S2DIP — BCARS fork

Fork of [YisiLuo/S2DIP](https://github.com/YisiLuo/S2DIP) (Luo et al., *Hyperspectral Mixed
Noise Removal Via Spatial-Spectral Constrained Unsupervised Deep Image Prior*, IEEE JSTARS
2021), adapted to denoise broadband CARS (BCARS) hyperspectral cubes on variance-stabilized
data, as used in *Denoising broadband CARS hyperspectral images on variance-stabilized data: a
unified benchmark*.

The fork adds one file, **`dip_baseline.py`**. Upstream solves a spatial-spectral TV model with
ADMM (`demo.py`); on BCARS that spatial prior over-smooths, so this runner fits the untrained
network (upstream `models/skip*`) with a **spectral-only** TV penalty,
`‖f(z) − y‖² + λ·TV_λ(f(z))`, where `TV_λ` is `mean|Δ_λ x|` (`--spec-order 1`) or
`mean|Δ²_λ x|` (`--spec-order 2`, curvature — preserves peak shape where first order clips it).
Nothing constrains the spatial axes, so sharp features such as bead edges survive. Everything
else — `models/`, `utils/`, `demo.py`, the sample cubes — is upstream code, unmodified.

DIP overfits, so the useful reconstruction is an intermediate iterate: checkpoints are written
along the whole trajectory and `--save-every` matters more than `--num-iter`.

## Running it on your own cube

**1. Prepare the input.** A MATLAB `.mat` with

| key | meaning |
|---|---|
| `y_0_real` | the noisy cube, `(H, W, C)`, normalized to `[0, 1]` |
| `img_clean` | ground truth, same shape — **zeros** if you have none |
| `norm_min`, `norm_max` | de-normalization back to physical units |
| `wn` | wavenumber axis (cm⁻¹) |

The pipeline repository's `h5tomat_detrend.py` writes this schema; DDS2M reads the same file.

**2. Run.**

```bash
python dip_baseline.py \
    --data ./data/cube.mat --out-dir ./results/dip \
    --spec-tv 0.005 --spec-order 1 --lr 0.005 \
    --num-iter 10000 --eval-every 50 --save-every 1000 --gpu 0
```

**3. Collect the output.** Everything lands in `--out-dir`:

| file | meaning |
|---|---|
| `dip_iter<N>.mat` | checkpoint every `--save-every` iterations — **the trajectory to choose from** |
| `dip_final.mat` | the last iterate |
| `dip_best.mat` | highest PSNR *against `img_clean`* |
| `run.log`, `metrics.json` | loss and score history |

Each `.mat` holds the reconstruction in `denoised` plus the de-normalization constants;
`phys = x * (norm_max - norm_min) + norm_min`.

`dip_best.mat` is only meaningful when `img_clean` holds a real reference. With `img_clean = 0`
the PSNR is computed against zeros and the "best" iterate is noise-selected — pick a checkpoint
by inspecting the reconstruction, or score against a measurement-based pseudo-ground-truth. On
glycerol the useful iterate arrives around iteration 50–100; on the bead and worm cubes,
around 1000–2000.

Settings behind the paper's results: simulated cubes `--spec-tv 0.005 --spec-order 1 --lr 0.005`
(`dip_best`, exact ground truth available); glycerol `0.05 / 2 / 0.003` (`dip_final`); bead and
worm `0 / 1 / 0.003` at iterations 1000 and 2000. All on the noclip variance-stabilized cube.

<!--
## Related

- Processing pipeline (VST, detrending, CCV, phase retrieval) and the paper's metrics:
  [pipeline repo URL]
- Denoised cubes and metric tables: [Zenodo DOI] -->

Please cite Luo et al. (JSTARS 2021) alongside our paper if you use this code.
Upstream documentation: YisiLuo/S2DIP (https://github.com/YisiLuo/S2DIP) — the original
spatial-spectral TV model and its citation.
