import torch
import torch.nn.functional as F

def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

def sat_loss(input):
    return torch.abs(input - input.clamp(-1,1)).mean()
    
def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

def smooth_loss(x_in, pred_xstart, tv_scale=0, range_scale=0, sat_scale=0):
    tv_losses = tv_loss(x_in).sum() * tv_scale 
    range_losses = range_loss(pred_xstart).sum() * range_scale
    sat_losses = sat_loss(x_in).sum() * sat_scale
    loss = tv_losses+ range_losses + sat_losses
    return loss    
