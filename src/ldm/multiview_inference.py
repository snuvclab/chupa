import sys
sys.path.append("..")
import argparse
from tqdm.auto import tqdm, trange
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
from PIL import Image, ImageOps

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.utils.data_helpers import postproc_mult, postproc, load_model_from_config, get_x, instantiate_from_config
from ldm.utils.model_helpers import main_fn
torch.backends.cudnn.benchmark = True

def get_input_dirs(args):
    img_dirs = sorted(list(args.data.glob(f"**/{model_config.cond_stage_key}/*.png")))
    return img_dirs

def load_model(args, type="body"):
    root_dir = Path(args.root_path)
    main_dir = root_dir / getattr(args, f"{type}_task")
    if "checkpoints" in str(args.root_path):
        config_path = main_dir / "config.yaml"
        ckpt_path = main_dir / getattr(args, f"{type}_ckpt")
    else:
        config_path = list(main_dir.glob("configs/*-project.yaml"))[0]
        ckpt_path = main_dir / f"checkpoints/{args.ckpt}.ckpt"

    config = OmegaConf.load(config_path)
    if getattr(args, f"{type}_vq_path") is not None:
        config.model.params.first_stage_config.params.ckpt_path = str(root_dir / getattr(args, f"{type}_vq_path"))

    model = instantiate_from_config(config.model).eval().to(args.device)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    m, u = model.load_state_dict(pl_sd["state_dict"], strict=False)
    if args.half:
        model = model.half()

    data_config = config.data.params.train.params
    model_config = config.model.params
    return model, data_config, model_config

def get_output_dir(args):
    if args.output is None:
        output_dir = Path("outputs")
        output_dir = output_dir.joinpath(f"{args.task}/{args.ckpt}")
        output_dir = output_dir.joinpath(f"{args.data.stem}_{args.seed}_{args.batch_count}{args.suffix}")
    else:
        output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # if args.save_intermediates:
    #     output_dir.joinpath("intermediates").mkdir(parents=True, exist_ok=True)
    if args.save_video:
        output_dir.joinpath("video").mkdir(parents=True, exist_ok=True)

    return output_dir

def normal_diffusion(model, smpl_image_t, seed, device, cfg=True, cfg_weight=1.0, dual=False, sampling='ddpm'):
    with torch.no_grad():#, torch.cuda.amp.autocast():
        with model.ema_scope():
            x_sample, intermediates = main_fn(smpl_image_t, model, seed, 
                                              cfg=cfg, cfg_weight=cfg_weight, dual=dual, sampling=sampling, device=device)
    
    return x_sample

def repaint(model, normal, mask, cond, cfg=True, cfg_weight=1.0, resample=False, resample_T=10, n_resample=1, dual=False):
    with torch.no_grad():
        with model.ema_scope():
            if dual:
                model_single_ch = model.channels // 2
                input_single_ch = normal.shape[1] // 2
                normal_front = normal[:, :model_single_ch, :, :]
                normal_back = normal[:, input_single_ch:input_single_ch + model_single_ch, :, :]
                normal_enc_front = model.encode_first_stage(normal_front).repeat(1,1,1,1)
                normal_enc_back = model.encode_first_stage(normal_back).repeat(1,1,1,1)
                normal_enc = torch.cat([normal_enc_front, normal_enc_back], dim=1)
            else:
                model_single_ch = model.channels
                normal_enc = model.encode_first_stage(normal[:, :model_single_ch, :, :]).repeat(1,1,1,1)
            mask_resized = transforms.Resize([128, 128])(mask)
            cond_enc = model.encode_first_stage(cond[:, :model_single_ch]).repeat(1,1,1,1)
            input_shape = list(cond_enc.shape)
            if model.channels != input_shape[1]:
                input_shape[1] = model.channels
            masked_samples = model.p_sample_loop(cond=cond_enc, shape=input_shape, verbose=False, 
                                                 cfg=cfg, cfg_weight=cfg_weight,
                                                 mask=mask_resized, x0=normal_enc, 
                                                 repeat_sample=4, jump_length=4, dual=dual)
            if dual:
                single_ch = model.channels // 2
                masked_x_front = model.decode_first_stage(masked_samples[:,:single_ch,:,:])
                masked_x_back = model.decode_first_stage(masked_samples[:,single_ch:,:,:])
                masked_x_samples = torch.cat([masked_x_front, masked_x_back], dim=1)
                if resample:
                    masked_samples_resample = model.resample(x0=masked_samples, cond=cond_enc, shape=input_shape, 
                                                             resample_T=resample_T, n_resample=n_resample)
                    masked_x_front_resample = model.decode_first_stage(masked_samples_resample[:,:single_ch,:,:])
                    masked_x_back_resample = model.decode_first_stage(masked_samples_resample[:,single_ch:,:,:])
                    masked_x_samples_resample = torch.cat([masked_x_front_resample, masked_x_back_resample], dim=1)
                    return masked_x_samples, masked_x_samples_resample
            else:
                masked_x_samples = model.decode_first_stage(masked_samples)
                if resample:
                    masked_samples_resample = model.resample(x0=masked_samples, cond=cond_enc, shape=input_shape,
                                                             cfg=cfg, cfg_weight=cfg_weight, 
                                                             resample_T=resample_T, n_resample=n_resample)
                    masked_x_samples_resample = model.decode_first_stage(masked_samples_resample)
                    return masked_x_samples, masked_x_samples_resample
    
    return masked_x_samples, None

def repaint_by_resample(model, normal, mask, cond, cfg=True, cfg_weight=1.0, dual=False,
                        resample_T=10, n_resample=1, refine=False, refine_T=5, n_refine=3):
    with torch.no_grad():
        with model.ema_scope():
            if dual:
                model_single_ch = model.channels // 2
                input_single_ch = normal.shape[1] // 2
                normal_front = normal[:, :model_single_ch, :, :]
                normal_back = normal[:, input_single_ch:input_single_ch + model_single_ch, :, :]
                normal_enc_front = model.encode_first_stage(normal_front).repeat(1,1,1,1)
                normal_enc_back = model.encode_first_stage(normal_back).repeat(1,1,1,1)
                normal_enc = torch.cat([normal_enc_front, normal_enc_back], dim=1)
            else:
                model_single_ch = model.channels
                normal_enc = model.encode_first_stage(normal[:, :model_single_ch, :, :]).repeat(1,1,1,1)
            mask_resized = transforms.Resize([128, 128])(mask)
            cond_enc = model.encode_first_stage(cond[:, :model_single_ch]).repeat(1,1,1,1)
            input_shape = list(cond_enc.shape)
            if model.channels != input_shape[1]:
                input_shape[1] = model.channels

            if dual:
                masked_samples_repaint = model.resample(x0=normal_enc, cond=cond_enc, shape=input_shape, 
                                                         cfg=cfg, cfg_weight=cfg_weight, mask=mask_resized, dual=dual,
                                                         resample_T=resample_T, n_resample=n_resample)
                masked_x_front_repaint = model.decode_first_stage(masked_samples_repaint[:,:model_single_ch, :, :])
                masked_x_back_repaint = model.decode_first_stage(masked_samples_repaint[:,model_single_ch:, :, :])
                masked_x_samples_repaint = torch.cat([masked_x_front_repaint, masked_x_back_repaint], dim=1)
                if refine:
                    masked_samples_resample = model.resample(x0=masked_samples_repaint, cond=cond_enc, shape=input_shape, 
                                                             resample_T=refine_T, n_resample=n_refine)
                    masked_x_front_resample = model.decode_first_stage(masked_samples_resample[:,:model_single_ch,:,:])
                    masked_x_back_resample = model.decode_first_stage(masked_samples_resample[:,model_single_ch:,:,:])
                    masked_x_samples_resample = torch.cat([masked_x_front_resample, masked_x_back_resample], dim=1)
                    return masked_x_samples_repaint, masked_x_samples_resample
            else:
                masked_samples_repaint = model.resample(x0=normal_enc, cond=cond_enc, shape=input_shape,
                                                        cfg=cfg, cfg_weight=cfg_weight, mask=mask_resized, dual=dual,
                                                        resample_T=resample_T, n_resample=n_resample)
                                                        
                normal_enc = masked_samples_repaint
                masked_x_samples_repaint = model.decode_first_stage(masked_samples_repaint)
            return masked_x_samples_repaint, None

def resample_fn(model, normal, cond, cfg, cfg_weight, resample_T=5, n_resample=1, dual=False):
    with torch.no_grad():
        with model.ema_scope():
            if dual:
                model_single_ch = model.channels // 2
                input_single_ch = normal.shape[1] // 2
                normal_front = normal[:, :model_single_ch, :, :]
                normal_back = normal[:, input_single_ch:input_single_ch + model_single_ch, :, :]
                normal_enc_front = model.encode_first_stage(normal_front).repeat(1,1,1,1)
                normal_enc_back = model.encode_first_stage(normal_back).repeat(1,1,1,1)
                normal_enc = torch.cat([normal_enc_front, normal_enc_back], dim=1)
            else:
                model_single_ch = model.channels
                normal_enc = model.encode_first_stage(normal[:, :model_single_ch, :, :]).repeat(1,1,1,1)
            cond_enc = model.encode_first_stage(cond).repeat(1,1,1,1)
            input_shape = list(cond_enc.shape)
            resampled_enc = model.resample(x0=normal_enc, cond=cond_enc, shape=cond_enc.shape, 
                                           cfg=cfg, cfg_weight=cfg_weight, 
                                           resample_T=resample_T, n_resample=n_resample)
            if dual:
                normal_resampled_front = model.decode_first_stage(resampled_enc[:,:model_single_ch, :, :])
                normal_resampled_back = model.decode_first_stage(resampled_enc[:,model_single_ch:, :, :])
                normal_resampled = torch.cat([normal_resampled_front, normal_resampled_back], dim=1)
            else:
                normal_resampled = model.decode_first_stage(resampled_enc)
    return normal_resampled

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default="/datasets/thuman/thuman_result/0000")
        # description="Root data path of input images")
    parser.add_argument("--root_path", type=Path, default="checkpoints")
        # description="Root checkpoint path")
    parser.add_argument("--gt_path", type=Path, default=None)
    parser.add_argument("--task", type=str, default="sn2n")
    parser.add_argument("--ckpt", type=Path, default="epoch000243")
    parser.add_argument("--vq-path", type=str, default="configs/vq-f4.ckpt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument("--use-single", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--use-split", action="store_true")
    parser.add_argument("--use-cond", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    model, data_config, model_config = load_model(args)
    # img_dirs = get_input_dirs(args)
    img_dirs = Path(args.data)
    # output_dir = get_output_dir(args)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # gt_path = img_dirs[0].parent.parent.joinpath(model_config.first_stage_key)
    with torch.no_grad(), torch.cuda.amp.autocast():
        with model.ema_scope():
            for img_dir in tqdm(img_dirs.glob('*.png')):
                # gt_dir = gt_path.joinpath(img_dir.name)
                img1, xc1 = get_x(img_dir, use_invert_green=False, device=args.device)
                # img2, xc2 = get_x(gt_dir , use_invert_green=False, device=args.device)

                x_samples = []
                for i in range(args.batch_count):
                    x_sample, intermediates = main_fn(xc1, model, args.seed+i, device=args.device)
                    x_samples.extend([y for y in x_sample])
                    del x_sample
                result = make_grid(x_samples, nrow=args.batch_count)

                # result = postproc_mult([xc1[0], result]) 
                result = Image.fromarray(postproc(result))  # save only diffusion result
                result.save(output_dir.joinpath(f"{img_dir.stem}.png"))




