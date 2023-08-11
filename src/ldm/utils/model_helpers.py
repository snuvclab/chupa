import numpy as np
from tqdm.auto import tqdm
import torch

from ldm.utils.metrics.color_metrics import smooth_loss
from ldm.utils.data_helpers import postproc_mult, to_video

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.k_diffusion.external import CompVisDenoiser
from ldm.modules.k_diffusion.sampling import sample_euler_ancestral

def main_fn(x, model, seed, dual=False, cfg=True, cfg_weight=1.0, device="cuda:0", 
    use_cond=False, verbose=False, postfix={}, sampling='ddpm', **cond_kwargs):
    with torch.no_grad():#, torch.cuda.amp.autocast():
        with model.ema_scope():
            torch.manual_seed(seed)
            x_T = torch.randn((1, model.channels,) + (x.shape[2] // 4, x.shape[3] // 4), device=device)

            # xc = model.get_learned_conditioning(x)
            xc = model.encode_first_stage(x)
            img = x_T
            intermediates = []

            if sampling == 'ddpm':
                timesteps = np.flip(np.asarray(list(range(0, model.num_timesteps))))
                iterator = tqdm(timesteps) if verbose else timesteps
                for step in iterator:
                    ts = torch.full((1,), step, device=device, dtype=torch.long)
                    mean, var, log_var, pred_x0 = model.p_mean_variance(img, xc, ts, 
                        clip_denoised=False, quantize_denoised=False, return_x0=True, 
                        cfg=cfg, cfg_weight=cfg_weight
                    )

                    if use_cond:
                        x_in, grad = cond_fn(img, xc, ts, model, **cond_kwargs)
                        postfix["grad"] = grad.mean().item()
                        mean = mean + var * grad 
                    else:
                        x_in = pred_x0

                    if verbose:
                        iterator.set_postfix(postfix)
                    noise = torch.randn_like(img, device=device)
                    nonzero_mask = (1-(ts==0).float()).reshape(img.shape[0],1,1,1,)
                    img = mean + nonzero_mask * (0.5 * log_var).exp() * noise
                    
                    intermediates.append({"step" : step, "pred_x0" : pred_x0, "x_in" : x_in})
            elif sampling == 'ddim':
                sampler = DDIMSampler(model)
                sampler.make_schedule(50, ddim_eta=1.0, verbose=False)

                uc = model.encode_first_stage(-1 * torch.ones_like(x))
                img, _ = sampler.ddim_sampling(xc, img.shape, 
                                                    unconditional_guidance_scale=cfg_weight, 
                                                    unconditional_conditioning=uc)
            elif sampling == 'euler':
                sampler = CompVisDenoiser(model)
                sigmas = sampler.get_sigmas(50)
                img = img * sigmas[0]

                img = sample_euler_ancestral(sampler, img, sigmas, 
                                            extra_args={"cond" : {"c_concat": [xc]}}, 
                                            disable=False, eta=1.0) 

            # x_sample = model.cond_stage_model.decode(img)
            if dual:
                single_ch = model.channels // 2
                x_front = model.decode_first_stage(img[:,:single_ch,:,:])
                x_back = model.decode_first_stage(img[:,single_ch:,:,:])
                x_sample = torch.cat([x_front, x_back], dim=1)
            else:
                x_sample = model.decode_first_stage(img)
    return x_sample, intermediates

def cond_fn(img, xc, step, model, 
    tv_scale=0, range_scale=150, sat_scale=0, 
    lpips_scale=0, lpips_model=None, clamp_max=0):
    with torch.enable_grad():
        x_is_NaN = False
        img = img.detach().requires_grad_()

        _, _, _, pred_x0 = model.p_mean_variance(img, xc, step, 
            clip_denoised=False, quantize_denoised=False, return_x0=True, 
            )
        fac = model.sqrt_one_minus_alphas_cumprod[step]
        x_in = pred_x0 * fac + img * (1 - fac)
        
        x_in_grad = torch.zeros_like(x_in)
        loss = smooth_loss(x_in, pred_x0, tv_scale, range_scale, sat_scale)
        if lpips_scale > 0:
            l_loss = lpips_model(x_in, xc) * lpips_scale
            loss += l_loss.reshape(loss.shape)
        add = torch.autograd.grad(loss, x_in)[0]
        x_in_grad += add

        if torch.isnan(x_in_grad).any()==False:
            grad = -torch.autograd.grad(x_in, img, x_in_grad)[0]
        else:
            x_is_NaN = True
            grad = torch.zeros_like(img)

    if x_is_NaN == False:
        if clamp_max > 0:
            magnitude = grad.square().mean().sqrt()
            return x_in, grad * magnitude.clamp(max=clamp_max) / magnitude
        else:
            return x_in, grad
    else: 
        return x_in, grad

def intermediates_to_video(intermediates, model, video_name):
    img_list_0 = []
    with torch.no_grad():
        with model.ema_scope():
            for inter in tqdm(intermediates):
                x_sample = model.decode_first_stage(inter)
                img_list_0.append(postproc_mult([x_sample[0]]))
    to_video(video_name, img_list_0, fps=10)