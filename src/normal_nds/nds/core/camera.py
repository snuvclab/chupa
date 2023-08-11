import torch
import numpy as np
import cv2

class Camera:
    """ Camera in OpenCV format.
        
    Args:
        K (tensor): Camera matrix with intrinsic parameters (3x3)
        R (tensor): Rotation matrix (3x3)
        t (tensor): translation vector (3)
        device (torch.device): Device where the matrices are stored
    """

    def __init__(self, K, R, t, orthographic=False, device='cpu'):
        self.K = K.to(device) if torch.is_tensor(K) else torch.FloatTensor(K).to(device)
        self.R = R.to(device) if torch.is_tensor(R) else torch.FloatTensor(R).to(device)
        self.t = t.to(device) if torch.is_tensor(t) else torch.FloatTensor(t).to(device)
        self.orthographic = orthographic
        self.device = device

    def to(self, device="cpu"):
        self.K = self.K.to(device)
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        self.device = device
        return self

    @property
    def center(self):
        return -self.R.t() @ self.t

    @property
    def P(self):
        return self.K @ torch.cat([self.R, self.t.unsqueeze(-1)], dim=-1)

    @classmethod
    def camera_with_angle(cls, scale, center, view_angle=0,
                          ortho_ratio=0.4, res=512,  camera_depth=1.6, orthographic=True, 
                          device='cpu'):
        if orthographic:
            ortho_ratio = ortho_ratio * (512 / res)
            y = np.deg2rad(int(view_angle))
            R = np.array([[np.cos(y), 0, np.sin(y)],
                        [0, 1, 0],
                        [-np.sin(y), 0, np.cos(y)]])

            t = -np.dot(R, center + np.array([0, 0, camera_depth]))

            K = np.identity(3)
            K[0, 0] = 2.0 * scale / (res * ortho_ratio)
            K[1, 1] = 2.0 * scale / (res * ortho_ratio)
            K[2, 2] = 0.001  # 2 / (zFar - zNear), following ICON rendering code : zFar=100, zNear=-100
        else:
            raise NotImplementedError

        camera = Camera(K, R, t, orthographic, device=device)
        return camera

    # from view.py - transform
    def normalize_camera(self, space_normalization, device='cpu'):
        """ Transform the view pose with an affine mapping.

        Args:
            A (tensor): Affine matrix (4x4)
            A_inv (tensor, optional): Inverse of the affine matrix A (4x4)
        """
        A = space_normalization.A
        A_inv = space_normalization.A_inv

        if not torch.is_tensor(A):
            A = torch.from_numpy(A)
        
        if A_inv is not None and not torch.is_tensor(A_inv):
            A_inv = torch.from_numpy(A_inv)

        A = A.to(device, dtype=torch.float32)
        if A_inv is not None:
            A_inv = A_inv.to(device, dtype=torch.float32)

        if A_inv is None:
            A_inv = torch.inverse(A)

        # Transform camera extrinsics according to  [R'|t'] = [R|t] * A_inv.
        # We compose the projection matrix and decompose it again, to correctly
        # propagate scale and shear related factors to the K matrix, 
        # and thus make sure that R is a rotation matrix.
        R = self.R @ A_inv[:3, :3]
        t = self.R @ A_inv[:3, 3] + self.t
        if self.orthographic:
            self.R = R
            self.t = t
            
        else:
            P = torch.zeros((3, 4), device=device)
            P[:3, :3] = self.K @ R
            P[:3, 3] = self.K @ t
            K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P.cpu().detach().numpy())
            c = c[:3, 0] / c[3]
            t = - R @ c

            # ensure unique scaling of K matrix
            K = K / K[2,2]
            
            self.K = torch.from_numpy(K).to(device)
            self.R = torch.from_numpy(R).to(device)
            self.t = torch.from_numpy(t).to(device)