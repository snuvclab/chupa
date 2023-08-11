from pathlib import Path
from PIL import Image
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from nds.core import (
    Mesh, Renderer, View, Camera
)
from nds.modules import (
    SpaceNormalization, NeuralShader, ViewSampler
)
from nds.utils import (
    AABB, read_views, read_mesh, write_mesh, visualize_mesh_as_overlay, visualize_views, visualize_masks, generate_mesh, mesh_generator_names
)

import cv2

def main():
    parser = ArgumentParser(description='Multi-View Mesh Reconstruction with Neural Deferred Shading', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_type', type=str, default="dvr", help="dvr | co3d")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="a parameter that changes the mask values to binary when mask values are continuous")
    parser.add_argument('--image_scale', type=int, default=1, help="Scale applied to the input images. The factor is 1/image_scale, so image_scale=2 halves the image size")
    parser.add_argument('--device', type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="GPU to use; -1 is CPU")

    parser.add_argument('--input_dir', type=Path, default="./data", help="Path to the input data")
    parser.add_argument('--image_path', type=Path, help='relative path of images with respect to input dir')
    parser.add_argument('--masked_path', type=Path, help='relative path of generated masked novel view image with respect to input dir')
    parser.add_argument('--mesh_path', type=Path, help='relative path of mesh with respect to input dir')
    parser.add_argument('--target_view', type=int, help='specify target view angle')
    parser.add_argument('--shader_ckpt', type=Path)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    
    parser.add_argument('--hidden_features_layers', type=int, default=3, help="Number of hidden layers in the positional feature part of the neural shader")
    parser.add_argument('--hidden_features_size', type=int, default=256, help="Width of the hidden layers in the neural shader")
    parser.add_argument('--fourier_features', type=str, default='positional', choices=(['none', 'gfft', 'positional']), help="Input encoding used in the neural shader")
    parser.add_argument('--activation', type=str, default='relu', choices=(['relu', 'sine']), help="Activation function used in the neural shader")
    parser.add_argument('--fft_scale', type=int, default=4, help="Scale parameter of frequency-based input encodings in the neural shader")
    args = parser.parse_args()

    # set device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    input_dir = Path(args.input_dir)
    image_dir = input_dir / args.image_path
    masked_dir = input_dir / args.masked_path
    masked_dir.mkdir(exist_ok=True)
    mesh_path = input_dir / args.mesh_path
    mesh = read_mesh(str(mesh_path), device=device)
    mesh.compute_connectivity()
    views = read_views([image_dir], mask_threshold=args.mask_threshold, scale=args.image_scale, device=device, data_type=args.data_type)

    # create the bbox from the mesh vertices
    aabb = AABB(mesh.vertices.cpu().numpy())

    # Apply the normalizing affine transform, which maps the bounding box to 
    # a 2-cube centered at (0, 0, 0), to the views, the mesh, and the bounding box
    space_normalization = SpaceNormalization(aabb.corners)
    views = space_normalization.normalize_views(views)
    mesh = space_normalization.normalize_mesh(mesh)
    aabb = space_normalization.normalize_aabb(aabb)

    # Configure the renderer
    renderer = Renderer(device=device)
    renderer.set_near_far(views, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    # find faces visible from image_views
    vis_mask = renderer.get_face_visibility(views, mesh)

    # Load neural shader
    shader = NeuralShader(hidden_features_layers=args.hidden_features_layers,
                        hidden_features_size=args.hidden_features_size,
                        fourier_features=args.fourier_features,
                        activation=args.activation,
                        fft_scale=args.fft_scale,
                        last_activation=torch.nn.Sigmoid, 
                        device=device)
    ckpt_path = input_dir / args.shader_ckpt
    shader.load_state_dict(torch.load(str(ckpt_path))['state_dict'])

    # generate target view image with visible vertices
    target_view_camera = Camera.camera_with_angle(input_dir, args.target_view, device=device)
    target_view_R = target_view_camera.R
    target_view_camera.normalize_camera(space_normalization, device=device)
    resolution = (args.height, args.width)
    target_gbuffer = renderer.render_with_camera(target_view_camera, mesh, resolution, vis_mask=None, with_antialiasing=True)
    mask = target_gbuffer["mask"]
    position = target_gbuffer["position"]
    normal = target_gbuffer["normal"]
    view_direction = torch.nn.functional.normalize(target_view_camera.center - position, dim=-1)

    pix_to_face = target_gbuffer["pix_to_face"].long()
    out_of_fov = (pix_to_face == 0)
    target_view_mask = vis_mask[pix_to_face - 1].float() 
    target_view_mask[out_of_fov] = 0 
    target_view_normal_G = normal @ target_view_R  @ torch.tensor([[1., 0, 0], [0, -1, 0], [0, 0, -1]], device=device)
    target_view_normal_G_full = (target_view_normal_G + 1) * 0.5 * mask
    target_view_normal_G_masked = target_view_normal_G_full * target_view_mask #+ (1 - target_view_mask)
    
    target_view_normal_G_img = Image.fromarray((target_view_normal_G_full.detach().cpu().numpy() * 255).astype(np.uint8))
    target_view_mask_img = Image.fromarray(np.asarray(target_view_mask.repeat((1,1,3)).detach().cpu()).astype(np.uint8) * 255)

    target_view_normal_G_img.save(masked_dir / f'{args.target_view:03d}_full.png')
    target_view_mask_img.save(masked_dir / f'{args.target_view:03d}_mask.png')

if __name__ == '__main__':
    main()