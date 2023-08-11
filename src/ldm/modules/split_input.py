import torch
import torch.nn as nn

def get_weighting(h, w, Ly, Lx, device="cuda:0"):
    y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
    x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)
    arr = torch.cat([y, x], dim=-1)

    lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
    arr = arr / lower_right_corner

    dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
    dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
    edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]

    weighting = torch.clip(edge_dist, 0.01, 0.5)
    weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)
    return weighting

def get_ks_stride(x, factor=4):
    ks = (x.shape[2] // factor, x.shape[3] // factor)
    stride = (x.shape[2] // (factor*2), x.shape[3] // (factor*2))
    return ks, stride

def get_fold_unfold(x, ks, stride, type="down", factor=4, ks_lv=4, device="cuda:0"):
    assert type in ["up", "down", "same"]
    h,w = x.shape[2], x.shape[3]
    if type == "up":
        factor = 1/factor if factor > 1 else factor
    elif type == "down":
        factor = factor if factor > 1 else 1/factor
    else:
        factor = 1

    Ly = (h - ks[0]) // stride[0] + 1
    Lx = (w - ks[1]) // stride[1] + 1

    new_ks = (int(ks[0]//factor), int(ks[1]//factor))
    new_stride = (int(stride[0]//factor), int(stride[1]//factor))
    new_size = (int(h//factor), int(w//factor))

    unfold = nn.Unfold(ks, stride=stride)
    fold = nn.Fold(new_size, new_ks, stride=new_stride)

    weighting = get_weighting(new_ks[0], new_ks[1], Ly, Lx, device).to(x.dtype)
    normalization = fold(weighting).view((1,1) + new_size)  # normalizes the overlap
    weighting = weighting.view((1, 1) + new_ks + (Ly*Lx,))
    return fold, unfold, normalization, weighting

def encode(x, model, factor=4, device="cuda:0"):
    ks, stride = get_ks_stride(x, factor=factor)
    fold, unfold, norm, weight = get_fold_unfold(x, ks, stride, type="down", device=device)
    
    z = unfold(x)
    z = z.view((z.shape[0],-1,)+ks+(z.shape[-1],))
    
    output_list = []
    for i in range(z.shape[-1]):
        out = model.encoder(z[:,:,:,:,i])
        out = model.quant_conv(out)
        output_list.append(out)

    o = torch.stack(output_list, axis=-1) * weight
    o = o.view((o.shape[0], -1, o.shape[-1]))

    decoded = fold(o) / norm
    return decoded

def decode(z, model, factor=4, device="cuda:0"):
    ks, stride = get_ks_stride(z, factor=factor)
    fold, unfold, norm, weight = get_fold_unfold(z, ks, stride, type="up", device=device)
    z = unfold(z)
    z = z.view((z.shape[0],-1,)+ks+(z.shape[-1],))

    output_list = []
    for i in range(z.shape[-1]):
        out = model.post_quant_conv(z[:,:,:,:,i])
        out = model.decoder(out)
        output_list.append(out)

    o = torch.stack(output_list, axis=-1) * weight
    o = o.view((o.shape[0], -1, o.shape[-1]))

    decoded = fold(o) / norm
    return decoded