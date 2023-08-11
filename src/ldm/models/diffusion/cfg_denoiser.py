import torch
import torch.nn as nn
import einops

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale=1.0):
        if "c_crossattn" in cond:
            cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
            cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
            cfg_cond = {
                "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
                "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
            }
            out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
            return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
        else:
            cfg_z = einops.repeat(z, "1 ... -> n ...", n=2)
            cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=2)
            cfg_cond = {
                "c_concat" : [torch.cat([cond["c_concat"][0], uncond["c_concat"][0]])]
            }
            out_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(2)
            return out_uncond + text_cfg_scale * (out_cond - out_uncond)