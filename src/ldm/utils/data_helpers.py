import os
import sys
sys.path.append("..")

from pathlib import Path, PosixPath
from tqdm.auto import tqdm, trange

from omegaconf import OmegaConf
import numpy as np
import cv2
from PIL import Image
import imageio
from imageio_ffmpeg import write_frames
from pygifsicle import optimize

import torch
from torchvision import transforms

from ldm.utils.util import instantiate_from_config

def read_img(img_path, to_tensor=True, crop=False, type="RGB", resize=None, device="cuda:0"):
    img = Image.open(img_path).convert(type)
    if crop == True:
        img = img.crop((img.size[0]//2, 0, img.size[0], img.size[1]))
    if resize:
        img = img.resize(resize, resample=Image.BICUBIC)
    if to_tensor:
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0).to(device)
    return img

def postproc(out, norm=True, limit=1):
    if norm:
        out = torch.clamp(out / limit, -1, 1)
        out = (out+1)/2
    else:
        out = torch.clamp(out / limit, 0, 1)
    out = out.permute(1,2,0).cpu().numpy()
    out = (out * 255.).astype(np.uint8) 
    return out

def postproc_mult(imgs, size=None, img_type="RGB"):
    if size is None:
        size = (imgs[0].shape[-1], imgs[0].shape[-2])
    out = Image.new(img_type, (size[0]*len(imgs), size[1]))
    for i, img in enumerate(imgs):
        img = img[-1] if img.ndim == 4 else img
        img = img[:2,:,:] if img_type=="LA" else img
        img = postproc(img)
        img = img[:,:,0] if img.shape[-1] == 1 else img
        pimg = Image.fromarray(img).resize((size[0], size[1]))
        out.paste(pimg, (size[0]*i, 0))
    return out

def invert_green(img):
    r,g,b,a = img.split()
    alpha = (np.asarray(a.convert("L")) / 255.).astype(bool)
    green = np.array(g)
    green[alpha] = 255 - green[alpha]
    final = Image.merge('RGB', (r,Image.fromarray(green),b))
    return final
    
def get_x(img_path, img_type="RGB", img_size=(512,512), 
    use_invert_green=False, device="cuda:0"):
    if type(img_path) == str or type(img_path) == PosixPath:
        img = Image.open(img_path)
    elif type(img_path) == Image.Image:
        img = img_path
    else:
        img = Image.fromarray(img_path)

    if use_invert_green:
        img = invert_green(img)
    img = img.convert(img_type)
    
    if img_type == "LA":
        img = np.asarray(img)
        img = np.concatenate([img, np.zeros(img.shape[:2] + (1,), dtype=np.uint8)], axis=-1)
        img = Image.fromarray(img)

    transf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    x = transf(img).unsqueeze(0).to(device, memory_format=torch.contiguous_format).float()    
    return img, x

def load_model_from_config(config_path, ckpt, vq_path=None,
    device="cuda:0", use_split=True, half=False):
    config = OmegaConf.load(config_path).model
    if vq_path is not None:
        config.params.first_stage_config.params.ckpt_path = vq_path
    model = instantiate_from_config(config).eval().to(device)

    pl_sd = torch.load(ckpt, map_location="cpu")
    m, u = model.load_state_dict(pl_sd["state_dict"], strict=False)
    if use_split:
        model.split_input_params = {
            "ks": (128,128), "stride": (64, 64), "vqf": 4,
            "patch_distributed_vq": True, "tie_braker": False,
            "clip_max_weight": 0.5, "clip_min_weight": 0.01,
            "clip_max_tie_weight": 0.5, "clip_min_tie_weight": 0.01
        }
        if "first_stage_config" in config.params:
            ch_mult = config.params.first_stage_config.params.ddconfig.ch_mult
        else:
            ch_mult = config.params.ddconfig.ch_mult
        model.split_input_params["vqf"] = 2**(len(ch_mult)-1)

    if half:
        model = model.half()
        # model.first_stage_model = model.first_stage_model.half()
        # model.cond_stage_model = model.cond_stage_model.half()
    return model, pl_sd["global_step"]

def to_video(output_dir, frames, fps=2, use_optimize=True):
    if output_dir.endswith(".gif"):
        imageio.mimsave(output_dir, frames, fps=fps)
        if use_optimize:
            optimize(output_dir)
    else:
        writer = write_frames(output_dir, frames[-1].size, fps=fps)  
        writer.send(None)
        for frame in frames:
            writer.send(frame.tobytes())
        writer.close()

def intermediate_to_video_npy(img_path, model, output_dir, device="cuda:0", fps=2, use_optimize=True):
    img_path = Path(img_path)
    npy_path = img_path.parent.joinpath(f"intermediates/{img_path.stem}.npy")

    _, x = get_x(img_path, device=device)
    x = x[:,:,:,:x.shape[3]//2]
    img_size = x.shape[2:]
    model.split_input_params["ks"] = (img_size[0] // 4, img_size[1] // 4)
    model.split_input_params["stride"] = (img_size[0] // 8, img_size[1] // 8)

    intermediates = np.load(npy_path, allow_pickle=True).item()

    debug_list = []
    with torch.no_grad():
        for i in trange(len(intermediates['pred_x0'])):
            x_sample = model.decode_first_stage(intermediates['pred_x0'][i].to(device))
            x_sample = postproc_mult([x, x_sample], size=(x.shape[-1] // 2, x.shape[-2] // 2))
            debug_list.append(x_sample)

    to_video(output_dir, debug_list, fps=fps, use_optimize=use_optimize)