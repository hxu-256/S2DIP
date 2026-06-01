The loss being minimized
The network net produces the clean estimate out. The total loss (the total_loss += ... block) is:


 mu/2 ‖∇x(out) − (V₂ − D₂/mu)‖²      ← spatial TV  (x)
+mu/2 ‖∇y(out) − (V₃ − D₃/mu)‖²      ← spatial TV  (y)
+10·mu/2 ‖∇xz(out) − (V₄ − D₄/mu)‖²  ← spatial-spectral TV (x)
+10·mu/2 ‖∇yz(out) − (V₅ − D₅/mu)‖²  ← spatial-spectral TV (y)
+ ‖y·mask − out·mask − S‖²           ← data fidelity, minus sparse outliers S
The V variables are soft-thresholded gradients (the soft_thres calls) — soft-thresholding is the proximal operator of the L1 norm, so this is what imposes sparsity on the gradients = TV regularization. The D₂…D₅ are ADMM dual variables that enforce "V ≈ gradient of out". S = soft_thres(y − out) is a sparse residual that absorbs impulse/stripe noise so it isn't forced into the image.

What each hyperparameter means
Param	Where	Meaning	Effect when increased
mu	=0.12	ADMM penalty / global regularization–vs–fidelity balance	Smoother result, more weight on TV/SSTV; also the ADMM step size (affects convergence)
thres_tv	=0.1	Soft-threshold on spatial gradients (≈ spatial-TV strength)	More spatial smoothing, more piecewise-flat / cartoon-like
thres_sstv	=0.1	Soft-threshold on spectral gradients (≈ spectral-TV strength)	More spectral smoothing along the 695 bands
10× factor	in loss	SSTV weighted 10× the spatial TV	Prioritizes spectral smoothness over spatial
alpha_3 → thres = 2·alpha_3	=0.01→0.02	Soft-threshold for the sparse outlier term S	More residual treated as sparse noise/stripes; out fits y less tightly
lr	=0.005	Adam learning rate of the DIP net	Faster fitting but more instability/artifacts; overfits sooner
num_iter	=7000	Iterations	DIP overfits eventually — PSNR rises then falls. The optimum is early stopping, not "more"
reg_noise_std	=0.01	Random perturbation added to net input each step	Mild regularization; larger delays overfitting
tt, num_channels_*	=4, 128	Network depth/width (capacity)	More expressive, fits faster, overfits sooner
exp_weight	=0.99	Declared but unused — intended for an EMA of the output (a standard DIP stabilizer)	(no effect currently)
show_every, save_every	logging only	—	—
How to optimize them
The big advantage here: you have ground truth (img_clean), and the script already logs psnr_history vs GT. So tuning is fully quantitative.

1. Early stopping first — biggest lever, free. Don't tune anything else until you've looked at the PSNR curve. DIP PSNR climbs, peaks, then drops as the net memorizes noise. Run once, plot psnr_history, and set num_iter to the peak (or save the checkpoint at the peak). This alone often beats hours of other tuning.

2. Then the regularization knobs, in order of impact for bCARS:

thres_sstv and the 10× SSTV weight — bCARS spectra are highly correlated band-to-band, so spectral TV is your strongest, safest denoiser. Sweep thres_sstv ∈ {0.05, 0.1, 0.2, 0.4}.
thres_tv — spatial smoothing. Raise it if spatial noise remains; lower it if fine spatial detail is being washed out. Sweep {0.05, 0.1, 0.2}.
mu — global balance; try {0.05, 0.1, 0.2, 0.5}. Interacts with the thresholds, so coarse-grid it together with one threshold.
alpha_3 — only matters if your noise has a sparse/stripe component. bCARS Z_noisy is largely broadband/Gaussian-ish, so keep this small (0.005–0.02); too large lets S eat real signal.
lr — usually leave at 0.005; only drop to 0.002 if you see high-frequency artifacts, or raise to 0.01 to converge faster for sweeps.
3. Practical workflow (each 7000-iter run is heavy):

Do sweeps at a shorter num_iter (e.g. 1500–2000) — relative ranking of settings shows up well before the peak.
You have two RTX 5090s — run parallel configs with CUDA_VISIBLE_DEVICES=0 and =1.
Score each run by peak PSNR vs GT from psnr_history. Optionally add SAM (spectral angle) since spectral fidelity is what matters for bCARS.
4. If you later run without GT, switch to blind criteria: watch the data-fidelity residual plateau and use the EMA output (wire up the unused exp_weight) plus a fixed early-stop iteration calibrated on this GT case.

Want me to add a small grid-search wrapper that loops over (mu, thres_tv, thres_sstv), runs short jobs across both GPUs, and reports peak PSNR per config? I can also wire in the exp_weight EMA and an automatic "save best-PSNR checkpoint" so early stopping is captured for you.