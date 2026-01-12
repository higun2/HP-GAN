# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
import dnnlib
import legacy

from pg_modules.blocks import Interpolate
import timm
from pg_modules.projector import get_backbone_normstats

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0,
                 train_head_only=False, style_mixing_prob=0.0, pl_weight=0.0,
                 cls_model='efficientnet_b1', cls_weight=0.0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.train_head_only = train_head_only

        # SG2 techniques
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = 2
        self.pl_decay = 0.01
        self.pl_no_weight_grad = True
        self.pl_mean = torch.zeros([], device=device)

        # classifier guidance
        if cls_weight > 0:
            cls = timm.create_model(cls_model, pretrained=True).eval()
            self.classifier = nn.Sequential(Interpolate(224), cls).to(device)
            normstats = get_backbone_normstats(cls_model)
            self.norm = Normalize(normstats['mean'], normstats['std'])
            self.cls_weight = cls_weight
            self.cls_guidance_loss = torch.nn.CrossEntropyLoss()

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]

        if c is None:
            img = self.G.synthesis(ws, update_emas=False)  # enabling emas leads to collapse with PG
        else:
            img = self.G.synthesis(ws, c, update_emas=False)

        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        return self. D(img, c)

    def run_ft(self, img, c, ft_head, img1=None):
        # Augmentation
        img0 = img
        img1 = img if img1 is None else img1

        # extract features for two views via feature networks
        _, out0 = self.D(img0, c, return_out0_sm=True)
        _, out1 = self.D(img1, c, return_out0_sm=True)

        # split feature maps obtained by two different feature networks, GAP and concat
        # out shape : [batch*2, 64, feature map h, feature map w]
        idx = len(out0) // 2
        out0_a = torch.stack(out0[:idx], dim=0).mean([2,3])
        out0_b = torch.stack(out0[idx:], dim=0).mean([2,3])
        out0 = torch.cat([out0_a, out0_b], dim=1)  # [batch, 64 * 2]

        out1_a = torch.stack(out1[:idx], dim=0).mean([2,3])
        out1_b = torch.stack(out1[idx:], dim=0).mean([2,3])
        out1 = torch.cat([out1_a, out1_b], dim=1)  # [batch, 64 * 2]

        # FT loss
        loss = ft_head(out0, out1)

        return loss

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg,
                             ft_phases=None, lw_ft=1.0, lw_dc=1.0, **kwargs):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg']: return  # no regularization

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        ########## L_G ##########
        if do_Gmain:
            # Maximize logits for generated images
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = sum([(-l).mean() for l in gen_logits])

                training_stats.report('Loss/G/GAN_loss', loss_Gmain)

                # apply DC to generator
                if lw_dc > 0:
                    idx = len(gen_logits) // 2
                    dc_g = nn.MSELoss()(torch.stack(gen_logits[:idx], 0).mean([1]), torch.stack(gen_logits[idx:], 0).mean([1]))
                    loss_Gmain = loss_Gmain + lw_dc * dc_g
                    training_stats.report('Loss/G/DC', dc_g * lw_dc)

                gen_logits = torch.cat(gen_logits)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                # FT loss to generator and head network
                if ft_phases.get('GHeadmain', None) is not None:
                    Gphase = ft_phases['GHeadmain']
                    Gphase.opt.zero_grad(set_to_none=True)

                    # noise perturbation
                    with torch.no_grad():
                        gen_z_mean = torch.abs(gen_z)
                        gen_z_mean = torch.clamp(gen_z_mean, min=0, max=3)
                        delta = gen_z_mean * 0.1 # l_1=0.1
                        delta_z = torch.randn(gen_z.shape, device=gen_z.device) * delta

                        # noise augmented image generation
                        noisy_gen_img, _ = self.run_G(gen_z + delta_z, gen_c)

                    ft_loss = self.run_ft(gen_img, gen_c, Gphase.module, img1=noisy_gen_img)
                    loss_Gmain = loss_Gmain + lw_ft * ft_loss
                    training_stats.report('Loss/G/FT', lw_ft * ft_loss)

                if self.cls_weight:
                    gen_img = self.norm(gen_img.add(1).div(2))
                    guidance_loss = self.cls_guidance_loss(self.classifier(gen_img), gen_c.argmax(1))
                    loss_Gmain += self.cls_weight * guidance_loss
                    training_stats.report('Loss/G/guidance_loss', self.cls_weight * guidance_loss)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

            # after backward of SSL loss together with the original loss,
            # manually call optim.step() to update the parameters of head network
            if ft_phases.get('GHeadmain', None) is not None:
                Gphase.opt.step()

        ########## L_D ##########
        if do_Dmain:
            # Minimize logits for generated images
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in gen_logits])

                loss_Dgen_ = loss_Dgen.detach().clone()

                # apply DC_f to discriminator
                if lw_dc > 0:
                    idx = len(gen_logits) // 2
                    dc_d_fake = nn.MSELoss()(torch.stack(gen_logits[:idx], 0).mean([1]), torch.stack(gen_logits[idx:], 0).mean([1]))
                    loss_Dgen = loss_Dgen + lw_dc * dc_d_fake
                    training_stats.report('Loss/D/DC_fake', lw_dc * dc_d_fake)

                gen_logits = torch.cat(gen_logits)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            name = 'Dreal'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = sum([(F.relu(torch.ones_like(l) - l)).mean() for l in real_logits])

                loss_Dreal_ = loss_Dreal.detach().clone()

                # apply DC_g to discriminator
                if lw_dc > 0:
                    idx = len(real_logits) // 2
                    dc_d_real = nn.MSELoss()(torch.stack(real_logits[:idx], 0).mean([1]), torch.stack(real_logits[idx:], 0).mean([1]))
                    loss_Dreal = loss_Dreal + lw_dc * dc_d_real
                    training_stats.report('Loss/D/DC_real', lw_dc * dc_d_real)

                real_logits = torch.cat(real_logits)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                training_stats.report('Loss/D/GAN_loss', loss_Dgen_ + loss_Dreal_)

            with torch.autograd.profiler.record_function(name + '_backward'):
                loss_Dreal.backward()
