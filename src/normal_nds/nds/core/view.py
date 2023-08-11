import cv2
import numpy as np
from pathlib import Path
from PIL import Image
if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
from rembg.bg import remove
import io
import torch
from torchvision import transforms

from normal_nds.nds.core import Camera

class View:
    """ A View is a combination of camera and image(s).

    Args:
        color (tensor): RGB color image (WxHx3)
        mask (tensor): Object mask (WxHx1)
        camera (Camera): Camera associated with this view
        device (torch.device): Device where the images and camera are stored
    """

    def __init__(self, color, mask, camera, view_angle=0, orthographic=True, device='cpu'):
        self.color = color.to(device)
        self.mask = mask.to(device)
        self.camera = camera.to(device)
        self.view_angle = view_angle
        self.orthographic = orthographic
        self.device = device

    def to_world(normal_cam, view_angle,):
        '''
            normal_cam : [4, h, w] tensor. normal map in camera view 
            view_angle: yaw angle of camera view in degree
        '''
        view_angle = torch.deg2rad(torch.tensor(view_angle))
        rot = torch.tensor([[torch.cos(view_angle), 0, torch.sin(view_angle)],
                            [0, 1, 0],
                            [-torch.sin(view_angle), 0, torch.cos(view_angle)]], dtype=normal_cam.dtype, device=normal_cam.device)
        mask = normal_cam[3, :, :] >= 0
        normal_world = normal_cam.clone()
        normal_world[:3, mask] = torch.einsum('ij, jk->ik', rot.T, normal_cam[:3, mask])

        return normal_world

    @classmethod
    def load_icon(cls, image_path, scale, center, ortho_ratio=0.4, res=512, view_angle=0, camera_depth=1.6, device='cpu'):
        """
            Set approximated orthographic cameras.
            We assume the mesh is
            (1) centered to the pelvis(0th joint), which means the camera axis goes through the pelvis,
            (2) scaled to the height of 180, which means the mesh occupies 450 out of 512 images with ortho_ratio=0.4
            reference: https://github.com/YuliangXiu/ICON/
        """

        camera = Camera.camera_with_angle(scale=scale, center=center,
                                          view_angle=view_angle, 
                                          device=device)

        # Load the color
        color = (Image.open(image_path))
        num_ch = len(color.split())
        if num_ch == 3:  # without alpha channel
            # clear background
            buf_front = io.BytesIO()
            color.save(buf_front, format='png')
            color = Image.open(io.BytesIO(remove(buf_front.getvalue()))).convert("RGBA")
        color_t = transforms.ToTensor()(color)
        color_world = (0.5 * (cls.to_world(2 * color_t - 1, int(view_angle)) + 1)).permute(1, 2, 0)
        
        # Extract the mask
        if color_world.shape[2] == 4:
            mask = color_world[:, :, 3:]
        else:
            mask = torch.ones_like(color_world[:, :, 0:1])

        color_world = color_world[:, :, :3]

        return cls(color_world, mask, camera, view_angle=view_angle, orthographic=True, device=device) 

    @classmethod
    def load_smpl_mask(cls, smpl_image, scale, center, ortho_ratio=0.4, res=512, view_angle=0, camera_depth=1.6, device='cpu'):
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
        camera = Camera(K, R, t, orthographic=True, device=device)
        
        # Extract the mask
        mask = smpl_image[:, :, 3:]

        return cls(smpl_image, mask, camera, view_angle=view_angle, orthographic=True, device=device) 

    @classmethod
    def load_dvr(cls, image_path, cameras_path=None, view_angle=0, device='cpu'):
        """ Load a view from a given image path.

        The paths of the camera matrices are deduced from the image path. 
        Given an image path `path/to/directory/foo.png`, the paths to the camera matrices
        in numpy readable text format are assumed to be `path/to/directory/foo_k.txt`, 
        `path/to/directory/foo_r.txt`, and `path/to/directory/foo_t.txt`.

        Args:
            image_path (Union[Path, str]): Path to the image file that contains the color and optionally the mask
            device (torch.device): Device where the images and camera are stored
        """

        image_path = Path(image_path)

        # Load the camera
        if cameras_path is None:
            cameras_path = image_path.parent.parent / "cameras.npz"
        if cameras_path.is_file():
            cam = np.load(cameras_path)
            frame_num = cam[image_path.name]
            pose = cam["pose_"+str(frame_num)]
            R = pose[:3, :3]
            t = pose[:3, 3]
            K = cam["intrinsic_"+str(frame_num)]
        else:
            K = np.loadtxt(image_path.parent / (image_path.stem + "_k.txt"))
            R = np.loadtxt(image_path.parent / (image_path.stem + "_r.txt"))
            t = np.loadtxt(image_path.parent / (image_path.stem + "_t.txt"))
        camera = Camera(K, R, t)
        
        # Load the color
        color = torch.FloatTensor(np.array(Image.open(image_path)))
        color /= 255.0
        
        # Extract the mask
        if color.shape[2] == 4:
            mask = color[:, :, 3:]
        else:
            mask = torch.ones_like(color[:, :, 0:1])

        color = color[:, :, :3]

        return cls(color, mask, camera, view_angle=view_angle, device=device)
    
    @classmethod
    def load_co3d(cls, image_path, mask_path, pose, intrinsic, mask_threshold, device='cpu'):
        """ Load co3d images.
        """

        image_path = Path(image_path)
        mask_path = Path(mask_path)

        # Load the camera
        K = intrinsic
        R = pose[:3, :3]
        t = pose[:3, 3]
        camera = Camera(K, R, t)
        
        # Load the color
        color = torch.FloatTensor(np.array(Image.open(image_path)))
        color /= 255.0
        
        # Extract the mask
        if color.shape[2] == 4:
            mask = color[:, :, 3:]
        else:
            mask = torch.ones_like(color[:, :, 0:1])

        color = color[:, :, :3]
        
        # Extract the mask
        mask = torch.FloatTensor(np.array(Image.open(mask_path)))
        mask /= 255.0
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(-1)
        else:
            mask = mask[:, :, 0].unsqueeze(-1)
        mask_type = mask.dtype
        mask = (mask > mask_threshold).type(mask_type)

        # Exclude black image
        y, x = np.where(mask[:, :, 0])
        if len(x) != 0 and len(y) != 0:
            return cls(color, mask, camera, device=device)
        else:
            None

    def to(self, device: str = "cpu"):
        self.color = self.color.to(device)
        self.mask = self.mask.to(device)
        self.camera = self.camera.to(device)
        self.device = device
        return self

    @property
    def resolution(self):
        return (self.color.shape[0], self.color.shape[1])
    
    def scale(self, inverse_factor):
        """ Scale the view by a factor.
        
        This operation is NOT differentiable in the current state as 
        we are using opencv.

        Args:
            inverse_factor (float): Inverse of the scale factor (e.g. to halve the image size, pass `2`)
        """
        
        scaled_height = self.color.shape[0] // inverse_factor
        scaled_width = self.color.shape[1] // inverse_factor

        scale_x = scaled_width / self.color.shape[1]
        scale_y = scaled_height / self.color.shape[0]
        
        self.color = torch.FloatTensor(cv2.resize(self.color.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)).to(self.device)
        self.mask = torch.FloatTensor(cv2.resize(self.mask.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)).to(self.device)
        self.mask = self.mask.unsqueeze(-1) # Make sure the mask is HxWx1

        self.camera.K = torch.FloatTensor(np.diag([scale_x, scale_y, 1])).to(self.device) @ self.camera.K  
    
    def transform(self, A, A_inv=None):
        """ Transform the view pose with an affine mapping.

        Args:
            A (tensor): Affine matrix (4x4)
            A_inv (tensor, optional): Inverse of the affine matrix A (4x4)
        """

        if not torch.is_tensor(A):
            A = torch.from_numpy(A)
        
        if A_inv is not None and not torch.is_tensor(A_inv):
            A_inv = torch.from_numpy(A_inv)

        A = A.to(self.device, dtype=torch.float32)
        if A_inv is not None:
            A_inv = A_inv.to(self.device, dtype=torch.float32)

        if A_inv is None:
            A_inv = torch.inverse(A)

        # Transform camera extrinsics according to  [R'|t'] = [R|t] * A_inv.
        # We compose the projection matrix and decompose it again, to correctly
        # propagate scale and shear related factors to the K matrix, 
        # and thus make sure that R is a rotation matrix.
        R = self.camera.R @ A_inv[:3, :3]
        t = self.camera.R @ A_inv[:3, 3] + self.camera.t

        if self.orthographic:
            self.camera.R = R
            self.camera.t = t
        else:
            P = torch.zeros((3, 4), device=self.device)
            P[:3, :3] = self.camera.K @ R
            P[:3, 3] = self.camera.K @ t
            K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P.cpu().detach().numpy())
            c = c[:3, 0] / c[3]
            t = - R @ c

            # ensure unique scaling of K matrix
            K = K / K[2,2]
            
            self.camera.K = torch.from_numpy(K).to(self.device)
            self.camera.R = torch.from_numpy(R).to(self.device)
            self.camera.t = torch.from_numpy(t).to(self.device)
        
    def project(self, points, depth_as_distance=False):
        """ Project points to the view's image plane according to the equation x = K*(R*X + t).

        Args:
            points (torch.tensor): 3D Points (A x ... x Z x 3)
            depth_as_distance (bool): Whether the depths in the result are the euclidean distances to the camera center
                                      or the Z coordinates of the points in camera space.
        
        Returns:
            pixels (torch.tensor): Pixel coordinates of the input points in the image space and 
                                   the points' depth relative to the view (A x ... x Z x 3).
        """

        # 
        points_c = points @ torch.transpose(self.camera.R, 0, 1) + self.camera.t
        pixels = points_c @ torch.transpose(self.camera.K, 0, 1)
        pixels = pixels[..., :2] / pixels[..., 2:]
        depths = points_c[..., 2:] if not depth_as_distance else torch.norm(points_c, p=2, dim=-1, keepdim=True)
        return torch.cat([pixels, depths], dim=-1)
