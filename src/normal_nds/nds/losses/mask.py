import torch
from typing import Dict, List

from normal_nds.nds.core import View
from torchvision import transforms

def mask_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], loss_function = torch.nn.MSELoss()):
    """ Compute the mask term as the mean difference between the original masks and the rendered masks.
    
    Args:
        views (List[View]): Views with masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """

    loss = 0.0
    for view, gbuffer in zip(views, gbuffers):
        loss += loss_function(view.mask, gbuffer["mask"])
    return loss / len(views)

def side_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], loss_function = torch.nn.MSELoss()):
    """ Compute the mask term as the mean difference between the original masks and the rendered masks.
    
    Args:
        views (List[View]): Views with masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """

    loss = 0.0
    for view, gbuffer in zip(views, gbuffers):
        smpl_mask = (view.mask + 1) * 0.5
        inside = (smpl_mask == 1)
        loss += loss_function(smpl_mask[inside], gbuffer["mask"][inside])
    return loss / len(views)