from __future__ import print_function
from models import pytorch_ssim
from utils.tv_utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import scipy.io
import numpy as np
from models.skip import skip
import torch
import torch.optim
import matplotlib.image as mp
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

# tune the hyperparameters to obtain good performance
# NOTE: these were tuned for the om1 (256x256x31) cube in demo.py; the bCARS cube
# (320x320x695) is much larger and likely needs its own tuning.
#########
mu = 0.12
alpha_3 = 0.01
#########

tt = 4
soft_thres = soft()
show = [100, 350, 600]  # display bands spread across the 695-band bCARS cube

# bCARS dataset prepared by DDS2M/npz2mat.py
DATA_PATH = "/mnt/d/GaTech Dropbox/Haoyu Xu/workspace/DDS2M/exp/datasets/ood_msi/bcars_denoising.mat"
RESULTS_DIR = "./results/bcars"
os.makedirs(RESULTS_DIR, exist_ok=True)

mat = scipy.io.loadmat(DATA_PATH)
img_noisy = mat["y_0_real"].astype(np.float64)   # noisy  (320, 320, 695)
img       = mat["img_clean"].astype(np.float64)  # ground truth (320, 320, 695)
norm_min  = float(mat["norm_min"])
norm_max  = float(mat["norm_max"])

# use ALL spectral bands; 320 is already a multiple of 32 so no spatial crop happens
img_noisy_np, img_noisy_var = prepare_noise_image(img_noisy, img_noisy.shape[2])  # (695, 320, 320)
img_np,       img_var       = prepare_image(img,       img.shape[2])

# pure denoising (no inpainting): mask is all ones
mask     = torch.ones_like(img_noisy_var).type(dtype)
mask_var = mask

print('noisy_PSNR:', psnr3d(img_np, img_noisy_np))

method = '2D'
pad = 'reflection'
OPT_OVER = 'net'
reg_noise_std = 0.01
OPTIMIZER='adam'
show_every = 50
save_every = 500
exp_weight=0.99
num_iter = 3000
input_depth = img_noisy_np.shape[0]
lr = 0.005

net = skip(input_depth, input_depth, img_noisy_np.shape[0],
       num_channels_down = [128]*tt,
       num_channels_up =   [128]*tt,
       num_channels_skip =    [4]*tt,
       filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
       upsample_mode='bilinear',
       need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

net_input = Variable(get_noise(input_depth, method, (img_noisy_np.shape[1], img_noisy_np.shape[2])).type(dtype).detach()).cuda()

print('Input_size: ',net_input.size())
s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
print ('Number of params: %d' % s)

img_noisy_var = img_noisy_var[None, None, :].cuda()

TV = TV_Loss()
SSTV = SSTV_Loss()

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
last_net = None
psrn_noisy_last = 0

thres = 2 * alpha_3
thres_tv = 0.1
thres_sstv = 0.1

psnr_history = []   # (iter, psnr_gt) collected every show_every
out_np_last = None  # most recent denoised cube, (bands, H, W)


def save_cube(out_np, path):
    """Save a denoised cube (bands, H, W) -> (H, W, bands) .mat with metadata."""
    denoised = out_np.transpose(1, 2, 0)
    scipy.io.savemat(path, {
        'denoised':      denoised.astype(np.float32),
        'denoised_phys': (denoised * (norm_max - norm_min) + norm_min).astype(np.float32),
        'norm_min':      np.array([[norm_min]]),
        'norm_max':      np.array([[norm_max]]),
    })


def closure(iter):
    global psnr_best, out_np_last

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    net_input = Variable(net_input).cuda()

    if iter == 0:
        psnr_best = 0

    out_ = net(net_input)
    out_ = out_[None, :]

    D_x_,D_y_ = TV(out_)
    D_xz_, D_yz_ = SSTV(out_)
    D_x = D_x_.clone().detach()
    D_y = D_y_.clone().detach()
    D_xz = D_xz_.clone().detach()
    D_yz = D_yz_.clone().detach()
    out = out_.clone().detach()

    if iter == 0:
        global D_1,D_2,D_3,D_4,D_5,V_1,V_2,V_3,V_4,V_5,S,mu,thres,thres_tv,thres_sstv
        D_2 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2],
                           img_noisy_var.shape[3]-1,img_noisy_var.shape[4]]).type(dtype)
        D_3 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2],
                           img_noisy_var.shape[3],img_noisy_var.shape[4]-1]).type(dtype)
        D_4 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2]-1,
                           img_noisy_var.shape[3]-1,img_noisy_var.shape[4]]).type(dtype)
        D_5 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2]-1,
                           img_noisy_var.shape[3],img_noisy_var.shape[4]-1]).type(dtype)

        V_2 = D_x.type(dtype)
        V_3 = D_y.type(dtype)
        V_4 = D_xz.type(dtype)
        V_5 = D_yz.type(dtype)

        S = (img_noisy_var-out).type(dtype)

    S = soft_thres(img_noisy_var-out, thres)

    V_2 = soft_thres(D_x + D_2 / mu, thres_tv)
    V_3 = soft_thres(D_y + D_3 / mu, thres_tv)

    V_4 = soft_thres(D_xz + D_4 / mu,thres_sstv)
    V_5 = soft_thres(D_yz + D_5 / mu,thres_sstv)

    total_loss = mu/2 * torch.norm(D_x_-(V_2-D_2/mu),2)
    total_loss += mu/2 * torch.norm(D_y_-(V_3-D_3/mu),2)
    total_loss += 10*mu/2 * torch.norm(D_xz_-(V_4-D_4/mu),2)
    total_loss += 10*mu/2 * torch.norm(D_yz_-(V_5-D_5/mu),2)
    total_loss += torch.norm(img_noisy_var*mask-out_*mask-S,2)

    total_loss.backward()

    D_2 = (D_2 + mu * (D_x  - V_2)).clone().detach()
    D_3 = (D_3 + mu * (D_y  - V_3)).clone().detach()
    D_4 = (D_4 + mu * (D_xz  - V_4)).clone().detach()
    D_5 = (D_5 + mu * (D_yz  - V_5)).clone().detach()

    psnr_gt = 0
    if iter % show_every == 0:
        out_np = out.detach().cpu().squeeze().numpy()
        out_np_last = out_np
        psnr_gt = psnr3d(np.clip(img_np.astype(np.float32),0,1), np.clip(out_np, 0, 1))
        psnr_history.append((iter, psnr_gt))
        print ('Iteration %05d    PSNR_gt: %f ' % (iter, psnr_gt), '\r', end='')

        plt.figure(figsize=(11,22))
        plt.subplot(121)
        plt.imshow(np.clip(np.stack((out_np[show[0],:,:],
                             out_np[show[1],:,:],
                             out_np[show[2],:,:]),2),0,1))
        plt.title('Recovered')

        plt.subplot(122)
        plt.imshow(np.clip(np.stack((img_noisy_np[show[0],:,:],
                             img_noisy_np[show[1],:,:],
                             img_noisy_np[show[2],:,:]),2),0,1))
        plt.title('Noisy')
        plt.draw()
        plt.pause(0.001)
        plt.close()

    if iter % save_every == 0:
        out_np = out.detach().cpu().squeeze().numpy()
        out_np_last = out_np
        save_cube(out_np, os.path.join(RESULTS_DIR, 'bcars_denoised_iter%05d.mat' % iter))

    return psnr_gt,0

p = get_params(OPT_OVER, net, net_input)
net_input.requires_grad = True
p += [net_input]
optimize(OPTIMIZER, p, closure, lr, num_iter, R = 0)

# final save
if out_np_last is None:
    out_np_last = net(net_input).detach().cpu().squeeze().numpy()
final_denoised = out_np_last.transpose(1, 2, 0)
scipy.io.savemat(os.path.join(RESULTS_DIR, 'bcars_denoised_final.mat'), {
    'denoised':      final_denoised.astype(np.float32),
    'denoised_phys': (final_denoised * (norm_max - norm_min) + norm_min).astype(np.float32),
    'img_clean':     img_np.transpose(1, 2, 0).astype(np.float32),
    'y_0_real':      img_noisy_np.transpose(1, 2, 0).astype(np.float32),
    'norm_min':      np.array([[norm_min]]),
    'norm_max':      np.array([[norm_max]]),
    'psnr_history':  np.array(psnr_history, dtype=np.float32),
})
print('\nSaved final results to', os.path.join(RESULTS_DIR, 'bcars_denoised_final.mat'))
