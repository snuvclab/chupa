import cv2
import os
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict

from img_utils import pil_concat_h, pil_concat_v

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, required=True, help='specify obj file path')
    parser.add_argument('--output_dir', type=Path, default=None)
    parser.add_argument('--ext', type=str, default='obj')
    args = parser.parse_args()

    width = 512
    height = 1024

    # Load .obj files
    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir is not None else input_dir
    ext = args.ext
    mesh_list = []
    for mesh_file in sorted(input_dir.glob(f'*.{ext}')):
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        mesh.compute_vertex_normals()
        mesh_list.append(mesh)
    num_meshes = len(mesh_list)

    # initialize visualizer and add mesh
    mesh_image_dict = defaultdict(list)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    for mesh_idx, mesh in enumerate(mesh_list):
        vis.add_geometry(mesh) 
        # vis.run()
        # rotate mesh and save images
        for i in range(360):
            vis.poll_events()
            vis.update_renderer()
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            image = Image.fromarray((255 * np.array(vis.capture_screen_float_buffer())).astype(np.uint8))
            mesh_image_dict[i].append(image)
        vis.remove_geometry(mesh)

    # concatenate images
    for i in range(360):
        image_merged = pil_concat_h(mesh_image_dict[i], h=height, w=width)
        image_merged.save(output_dir / f'{input_dir.stem}_{i:04d}.png')

    # generate video
    cmd = 'ffmpeg -framerate 30 -i ' + str(output_dir / f'{input_dir.stem}_%04d.png') + \
    ' -vcodec libx264 -y -pix_fmt yuv420p -refs 16 ' + str(output_dir / f'{input_dir.stem}_rotate.mp4')
    os.system(cmd)
    cmd = f'rm {str(output_dir)}/{input_dir.stem}_*.png'

    os.system(cmd)
