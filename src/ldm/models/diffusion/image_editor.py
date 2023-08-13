from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import einops

import torch
import torch.nn as nn
from torchvision import transforms

from .cfg_denoiser import CFGDenoiser
from ...modules import k_diffusion as K
from ...utils.data_helpers import instantiate_from_config

class ImageEditor(nn.Module):
    def __init__(self, config_path, ckpt_path, vq_path="", half=False, device="cuda:0"):
        super().__init__()
        self.device = device

        config = OmegaConf.load(config_path)
        if vq_path != "":
            config.model.params.first_stage_config.params.ckpt_path = vq_path
        self.model_type = config.model.target.split(".")[-2]
        self.channels = config.model.params.channels

        self.model = instantiate_from_config(config.model).eval().to(device)
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        m, u = self.model.load_state_dict(pl_sd["state_dict"], strict=False)
        self.model.cond_stage_model = self.model.cond_stage_model.to(device)
        
        if self.model_type == "ddpm_edit":
            self.model.cond_stage_model.device = device
        if half:
            self.model = self.model.half()
        
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        # self.null_token = self.model.get_learned_conditioning([""])

    def forward(self, image, txt, negative_txt="", cfg=7.5, image_cfg=1.5, steps=100, seed=23):
        torch.manual_seed(seed)
        with torch.no_grad(), self.model.ema_scope():
            if self.model_type == "ddpm_edit":
                cond = {
                    "c_crossattn": [self.model.get_learned_conditioning([txt])],
                    "c_concat": [self.model.encode_first_stage(image[None]).mode()],
                }
                uncond = {
                    "c_crossattn": [self.model.get_learned_conditioning([negative_txt])],
                    "c_concat": [torch.zeros_like(cond["c_concat"][0])],
                }

            else:
                cond = {"c_concat" : [self.model.encode_first_stage(image[None])]}
                uncond = {"c_concat": [torch.zeros_like(cond["c_concat"][0])]}

            extra_args = {
                "uncond": uncond,
                "cond": cond,
                "image_cfg_scale": image_cfg,
                "text_cfg_scale": cfg,
            }          
            sigmas = self.model_wrap.get_sigmas(steps)
            
            x_shape = list(cond["c_concat"][0].shape)
            x_shape[1] = self.channels
            x = torch.randn(x_shape).to(self.device) * sigmas[0]

            x = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, x, sigmas, extra_args=extra_args)
            
            if self.channels > 4:
                single_ch = self.channels // 2
                x_front = self.model.decode_first_stage(x[:,:single_ch,:,:])
                x_back = self.model.decode_first_stage(x[:,single_ch:,:,:])
                x = torch.cat([x_front, x_back], dim=1)
            else:
                x = self.model.decode_first_stage(x)[0]
            return x
        
    def forward_wfront(self, i_normal_front, image, cfg=7.5, steps=100, seed=23, repeat_sample=5, jump_length=0.05):
        assert self.model_type == "ddpm", "Model should be dual model, not text"

        front_mask = (i_normal_front[2, :, :]>=0).float()
        alpha = 2 * front_mask - 1
        i_normal_front = torch.cat([i_normal_front, alpha[None]], dim=0)   
        i_normal_back = -torch.ones_like(i_normal_front)
        i_normal = torch.concat([i_normal_front, i_normal_back], dim=0).unsqueeze(0).float()
        mask = torch.concat([front_mask[None], 1-front_mask[None]], dim=0).unsqueeze(0).float()

        h_c = self.channels // 2
        torch.manual_seed(seed)
        with torch.no_grad(), self.model.ema_scope():
            normal = einops.rearrange(i_normal, 'b (c1 c2) h w -> (b c1) c2 h w', c1=2, c2=h_c)
            normal_enc = self.model.encode_first_stage(normal)
            normal_enc = einops.rearrange(normal_enc, '(b c1) c2 h w -> b (c1 c2) h w', c1=2, c2=h_c)
            mask_enc = transforms.Resize(normal_enc.shape[-2:])(mask)

            cond = {"c_concat" : [self.model.encode_first_stage(image[None])]}
            uncond = {"c_concat": [torch.zeros_like(cond["c_concat"][0])]}

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "image_cfg_scale": cfg,
                "text_cfg_scale": cfg,
            } 
            sigmas = self.model_wrap.get_sigmas(steps)
            jump_steps = int(steps * jump_length)

            input_shape = list(cond['c_concat'][0].shape)
            input_shape[1] = self.channels
            img = torch.randn(input_shape, device=self.device) * sigmas[0]
            
            s_in = img.new_ones([img.shape[0]])
            for i in tqdm(range(0, len(sigmas) - 1, jump_steps)):
                for u in range(1, repeat_sample+1):
                    for j in reversed(range(jump_steps)):
                        denoised = self.model_wrap_cfg(img, sigmas[i+j] * s_in, **extra_args)
                        sigma_down, sigma_up = K.sampling.get_ancestral_step(sigmas[i+j], sigmas[i+j+1], eta=1.)
                        d = K.sampling.to_d(img, sigmas[i+j], denoised)
                        # Euler method
                        dt = sigma_down - sigmas[i+j]
                        img = img + d * dt
                        img = img + torch.randn_like(img) * sigma_up

                        img_orig = normal_enc if i+j==0 else normal_enc + torch.randn_like(normal_enc) * sigmas[i+j-1]
                        img[:, :h_c] = mask_enc[:, 0:1] * img_orig[:, :h_c] + (1. - mask_enc[:, 0:1]) * img[:, :h_c]
                        img[:, h_c:] = mask_enc[:, 1:2] * img_orig[:, h_c:] + (1. - mask_enc[:, 1:2]) * img[:, h_c:]

                    if u < repeat_sample:
                        img = img + torch.randn_like(normal_enc) * (sigmas[i]-sigmas[i+jump_steps])

            masked_x_front = self.model.decode_first_stage(img[:,:h_c,:,:])
            masked_x_back = self.model.decode_first_stage(img[:,h_c:,:,:])
            masked_x_samples = torch.cat([masked_x_front, masked_x_back], dim=1)
            return masked_x_samples
        
    def resample(self, i_normal, image, cfg=7.5, steps=10, seed=23, repeat=1):
        assert self.model_type == "ddpm", "Model should be dual model, not text"

        h_c = self.channels // 2
        torch.manual_seed(seed)
        with torch.no_grad(), self.model.ema_scope():
            normal = einops.rearrange(i_normal, 'b (c1 c2) h w -> (b c1) c2 h w', c1=2, c2=h_c)
            normal_enc = self.model.encode_first_stage(normal)
            normal_enc = einops.rearrange(normal_enc, '(b c1) c2 h w -> b (c1 c2) h w', c1=2, c2=h_c)

            cond = {"c_concat" : [self.model.encode_first_stage(image[None])]}
            uncond = {"c_concat": [torch.zeros_like(cond["c_concat"][0])]}

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "image_cfg_scale": cfg,
                "text_cfg_scale": cfg,
            } 
            sigmas = self.model_wrap.get_sigmas(self.model.num_timesteps)
            sigmas = sigmas[len(sigmas) - (steps+1):]

            input_shape = list(cond['c_concat'][0].shape)
            input_shape[1] = self.channels
            img = torch.randn(input_shape, device=self.device) * sigmas[0]
            img = normal_enc + img

            s_in = img.new_ones([img.shape[0]])
            for k in range(1, repeat+1):
                for i in tqdm(range(0, len(sigmas) - 1)):
                    denoised = self.model_wrap_cfg(img, sigmas[i] * s_in, **extra_args)
                    sigma_down, sigma_up = K.sampling.get_ancestral_step(sigmas[i], sigmas[i+1], eta=1.)
                    d = K.sampling.to_d(img, sigmas[i], denoised)
                    # Euler method
                    dt = sigma_down - sigmas[i]
                    img = img + d * dt
                    img = img + torch.randn_like(img) * sigma_up

                if k < repeat:
                    img = img + torch.randn(input_shape, device=self.device) * sigmas[0]

            masked_x_front = self.model.decode_first_stage(img[:,:h_c,:,:])
            masked_x_back = self.model.decode_first_stage(img[:,h_c:,:,:])
            masked_x_samples = torch.cat([masked_x_front, masked_x_back], dim=1)
            return masked_x_samples
