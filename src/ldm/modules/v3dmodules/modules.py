import numpy as np
import visu3d as v3d

def pose_enc_nerf(x, min_deg=0, max_deg=15):
    """
    Concatenate x and its positional encodings, following NeRF
    """
    if min_deg == max_deg:
        return x
    scales = np.array([2**i for i in range(min_deg, max_deg)])
    xb = np.reshape(
        x[..., None, :] * scales[..., None], 
        list(x.shape[:-1]) + [-1]
    )
    emb = np.sin(np.concatenate([xb, xb+np.pi/2.], axis=-1))
    conc = np.concatenate([x, emb], axis=-1)
    return conc

def pose_embedding(R, t, K, resolution=(256,256)):
    world_from_cam = v3d.Transform(R=R, t=t)
    cam_spec = v3d.PinholeCamera(resolution=resolution, K=K)
    rays = v3d.Camera(spec=cam_spec, world_from_cam=world_from_cam).rays()

    pose_emb_pos = pose_enc_nerf(rays.pos, min_deg=0, max_deg=15)
    pose_emb_dir = pose_enc_nerf(rays.dir, min_deg=0, max_deg=8)
    pose_emb = np.concatenate([pose_emb_pos, pose_emb_dir], axis=-1)
    return pose_emb