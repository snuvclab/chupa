import argparse
import os
import glob
import cv2
import numpy as np
import random
import math
import time
import trimesh
import torch
import json
import colorsys
import shutil
import pickle
from matplotlib import cm as mpl_cm, colors as mpl_colors
from matplotlib import cm
from pathlib import Path
from PIL import Image

from smplx.body_models import SMPLX, SMPL
from smpl_related.utils import get_yaw_inverse_mat, load_smpl

def pil_concat_h(pil_list, h=512, w=512):
    dst = Image.new('RGBA', (w * len(pil_list), h), "black")
    for idx, image in enumerate(pil_list):
        dst.paste(image, (w * idx, 0))
    return dst

def pil_concat_v(pil_list, h=512, w=512):
    dst = Image.new('RGBA', (w, h * len(pil_list)), "black")
    for idx, image in enumerate(pil_list):
        dst.paste(image, (0, h * idx))
    return dst

def mat2vec(R):
    # reference: https://courses.cs.duke.edu/fall13/compsci527/notes/rodrigues.pdf
    A = 0.5 * (R - R.transpose(1, 2))
    rho = A.reshape(-1, 9)[:, [7, 2, 3]]
    s = torch.norm(rho, dim=1, keepdim=True)
    c = 0.5 * (R.reshape(-1, 9)[:, [0, 4, 8]].sum(dim=1, keepdim=True) - 1)
    u = rho / s
    theta = torch.arctan2(s, c)

    rotvec = u * theta
    return rotvec

# from https://gist.github.com/mkocabas
def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros(n_vertices)

    for part_idx, (k, v) in enumerate(part_segm.items()):
        vertex_labels[v] = part_idx

    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3] = alpha
    vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

    return vertex_colors

def get_yaw_inverse_mat(rot_vec):
    angle = np.linalg.norm(rot_vec + 1e-8)
    rot_dir = rot_vec[0] / angle

    cos = np.cos(angle)
    sin = np.sin(angle)

    # Bx1 arrays
    rx, ry, rz = rot_dir[0], rot_dir[1], rot_dir[2]
    K = np.array([0, -rz, ry, rz, 0, -rx, -ry, rx, 0]).reshape(3, 3)

    ident = np.eye(3)
    rot_mat = ident + sin * K + (1 - cos) * (K @ K)
    front_axis = np.array([0, 0, 1]) @ rot_mat.T
    yaw = np.arctan2(front_axis[0], front_axis[2])
    yaw_inverse_mat = np.array([[np.cos(-yaw), 0, np.sin(-yaw)], 
                                [0, 1, 0], 
                                [-np.sin(-yaw), 0, np.cos(-yaw)]])

    return yaw_inverse_mat

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=Path, help='input dir')
parser.add_argument('--dataset', type=str, default='renderpeople')
parser.add_argument('--split', type=str, default='train', help='train/test')
parser.add_argument('--subject', type=str, help='subject name')
parser.add_argument('--num_views', default=36, type=int, help='the number of views')
parser.add_argument('--res', default=512, type=int, help='render size')
parser.add_argument('--egl', action='store_true')

parser.add_argument('--body', action='store_true')
parser.add_argument('--face', action='store_true')
parser.add_argument('--gt', action='store_true')
parser.add_argument('--smpl', action='store_true')
parser.add_argument('--smpl_type', choices=['smpl', 'smplx'], default='smplx')
parser.add_argument('--color', action='store_true')
parser.add_argument('--normal', action='store_true')
parser.add_argument('--geo', action='store_true', help='render shaded image for eval FID_shade')
parser.add_argument('--depth', action='store_true')
parser.add_argument('--thumbnail', action='store_true')
parser.add_argument('--align_yaw', action='store_true')

args = parser.parse_args()

input_dir = args.input_dir
dataset = args.dataset
split = args.split
subject = args.subject

data_dir = input_dir / dataset
save_dir = data_dir / 'render' / split / f'{subject}'
save_dir.mkdir(parents=True, exist_ok=True)

num_views = args.num_views
yaw_step = 360 // num_views
res = args.res
dataset = args.dataset
thumbnail = args.thumbnail
align_yaw = args.align_yaw

render_body = args.body
render_face = args.face
render_gt = args.gt
render_smpl = args.smpl


# headless
if args.egl:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    egl = True
else:
    os.environ["PYOPENGL_PLATFORM"] = ""
    egl = False

import renderer.opengl_util as opengl_util
from renderer.mesh import load_fit_body, load_scan, compute_tangent
import renderer.prt_util as prt_util
from renderer.gl.init_gl import initialize_GL_context
from renderer.gl.prt_render import PRTRender
from renderer.gl.color_render import ColorRender
from renderer.camera import Camera

# render
initialize_GL_context(width=res, height=res, egl=egl)

scale = 1.0
up_axis = 1
smpl_type = args.smpl_type 
smpl_gender = 'male' if dataset == 'thuman' else 'neutral'

color = args.color
normal = args.normal
depth = args.depth
geo = args.geo

if dataset == 'renderpeople':
    mesh_file = data_dir / 'scans' / subject / f'{subject}_100k.obj'
    tex_file = data_dir / 'scans' / subject / f'{subject}_dif_2k.jpg'
elif dataset == 'thuman':
    mesh_file = data_dir / 'scans' / subject / f'{subject}.obj'
    tex_file = data_dir / 'scans' / subject / 'material0.jpeg'
else:
    raise NotImplementedError

smpl_file = data_dir / smpl_type / f'{subject}.pkl'
smpl_param = np.load(smpl_file, allow_pickle=True)

# load mesh
vertices, faces, vertex_normals, faces_normals, textures, face_textures = load_scan(
    str(mesh_file), with_normal=True, with_texture=True)

if align_yaw:
    yaw_inverse_mat = get_yaw_inverse_mat(smpl_param['global_orient'])
    vertices = vertices @ yaw_inverse_mat.T
    vertex_normals = vertex_normals @ yaw_inverse_mat.T

# load smpl mesh
smpl_info = {}
smpl_info['model_path'] = f"src/smpl_related/models/{smpl_type}"
smpl_info['param'] = smpl_param
smpl_info['type'] = smpl_type
smpl_info['gender'] = smpl_gender
smpl_mesh, joints, _ = load_smpl(smpl_info,
                            scale=scale,
                            align_yaw=align_yaw)
smpl_vertices = smpl_mesh.vertices
smpl_faces = smpl_mesh.faces
smpl_vertex_normals = smpl_mesh.vertex_normals

if smpl_type == 'smplx':
    smpl_segmentation_path = 'src/smpl_related/smpl_data/smplx_vert_segmentation.json' 
    with open(smpl_segmentation_path, 'r') as f:
        smpl_segmentation = json.load(f)
    # face_idxs = smpl_segmentation['face'] + smpl_segmentation['neck'] + [8811, 8812, 8813, 8814, 9161, 9165] +\
    #             smpl_segmentation['rightEye'] + smpl_segmentation['leftEye'] + smpl_segmentation['eyeballs']
    # flame version
    face_idxs = np.load('src/smpl_related/smpl_data/SMPL-X__FLAME_vertex_ids.npy')

    # smpl_vertex_colors = 2 * part_segm_to_vertex_colors(smpl_segmentation, smpl_vertices.shape[0])[:, :3] - 1

elif smpl_type == 'smpl':
    smpl_segmentation_path = 'src/smpl_related/smpl_data/smpl_vert_segmentation.json' 
    with open(smpl_segmentation_path, 'r') as f:
        smpl_segmentation = json.load(f)
    face_idxs = smpl_segmentation['face'] + smpl_segmentation['neck']

# find body scale and center
body_scale = 180.0 / (smpl_vertices.max(0)[up_axis] - smpl_vertices.min(0)[up_axis])
body_vmin = smpl_vertices.min(0)
body_vmax = smpl_vertices.max(0)
body_vmed = joints[0]
body_vmed[up_axis] = 0.5 * (body_vmax[up_axis] + body_vmin[up_axis])

if render_face:
    # find face scale and center
    face_vertices = smpl_vertices[face_idxs]
    face_scale = 150.0 / (face_vertices.max(0)[up_axis] - face_vertices.min(0)[up_axis])  # Use 150 for face meshes instead of 180 due to the horizontal/vertical ratio

    face_vmin = face_vertices.min(0)
    face_vmax = face_vertices.max(0)
    face_vmed = 0.5 * (face_vmax + face_vmin)

if render_gt:
    # Set detailed mesh renderer
    if geo:
        rndr = ColorRender(width=res, height=res, egl=egl, geo=geo)
        rndr.set_mesh(vertices, 
                    faces,
                    vertices,
                    vertex_normals)
    else:
        prt, face_prt = prt_util.computePRT(mesh_file, scale, 10, 2)
        rndr = PRTRender(width=res, height=res, egl=egl)
        tan, bitan = compute_tangent(vertex_normals)
        rndr.set_mesh(
            vertices,
            faces,
            vertex_normals,
            faces_normals,
            textures,
            face_textures,
            prt,
            face_prt,
            tan,
            bitan,
            np.zeros((vertices.shape[0], 3)),
        )
        # texture
        texture_image = cv2.imread(str(tex_file))
        texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
        rndr.set_albedo(texture_image)

if render_smpl:
    # set smpl mesh renderer
    rndr_smpl = ColorRender(width=res, height=res, egl=egl)
    rndr_smpl.set_mesh(smpl_vertices, smpl_faces,
                    smpl_vertices,
                    smpl_vertex_normals)

##################################################################################### 
################################     Render body     ################################ 
##################################################################################### 

if render_body:
    if render_gt:
        rndr.set_norm_mat(body_scale, body_vmed)
    if render_smpl:
        rndr_smpl.set_norm_mat(body_scale, body_vmed)

    # camera
    cam = Camera(width=res, height=res)
    cam.ortho_ratio = 0.4 * (512 / res)

    for y in range(0, 360, yaw_step):
        R = opengl_util.make_rotate(0, math.radians(y), 0)
        if render_gt:
            # detailed mesh render
            rndr.rot_matrix = R
            cam.near = -1000
            cam.far = 1000
            cam.sanity_check()

            rndr.set_camera(cam)
            rndr.display()
            if color:
                opengl_util.render_result(
                    rndr, 0,
                    os.path.join(save_dir, 'color_F', f'{y:03d}.png'))
            if normal:
                opengl_util.render_result(
                    rndr, 1,
                    os.path.join(save_dir, 'normal_F', f'{y:03d}.png'))

            if depth:
                opengl_util.render_result(
                    rndr, 2,
                    os.path.join(save_dir, 'depth_F', f'{y:03d}.png'))

        if render_smpl:
        # smpl mesh render
            rndr_smpl.rot_matrix = R
            cam.near = -1000
            cam.far = 1000
            cam.sanity_check()

            rndr_smpl.set_camera(cam)
            rndr_smpl.display()
            if color:
                opengl_util.render_result(
                    rndr_smpl, 0, 
                    os.path.join(save_dir, 'T_color_F', f'{y:03d}.png'))
            if normal:
                opengl_util.render_result(
                    rndr_smpl, 1,
                    os.path.join(save_dir, 'T_normal_F', f'{y:03d}.png'))
            if depth:
                opengl_util.render_result(
                    rndr_smpl, 2,
                    os.path.join(save_dir, 'T_depth_F',
                                f'{y:03d}.png'))

    if thumbnail:
        # save thumbnail images
        angle_list = [0, 90, 180, 270]
        thumbnail_dir = save_dir.parent.parent / f'{dataset}_body_thumbnails_4view'
        thumbnail_dir.mkdir(exist_ok=True)
        color_dir = save_dir / 'color_F'
        normal_dir = save_dir / 'normal_F'
        T_color_dir = save_dir / 'T_color_F'
        T_normal_dir = save_dir / 'T_normal_F'
        first_row = []
        second_row = []
        if render_gt:
            if color:
                for idx, angle in enumerate(angle_list):
                        if idx // 2 == 0:
                            first_row.append(Image.open(color_dir / f'{angle:03d}.png'))
                            first_row.append(Image.open(T_color_dir / f'{angle:03d}.png'))
                        else:
                            second_row.append(Image.open(color_dir / f'{angle:03d}.png'))
                            second_row.append(Image.open(T_color_dir / f'{angle:03d}.png'))
                thumbnail = pil_concat_v([pil_concat_h(first_row), pil_concat_h(second_row)], w=4*512).convert("RGB")
                thumbnail.save(thumbnail_dir / f'{subject}.jpg')
            elif normal:
                for idx, angle in enumerate(angle_list):
                        if idx // 2 == 0:
                            first_row.append(Image.open(normal_dir / f'{angle:03d}.png'))
                            first_row.append(Image.open(T_normal_dir / f'{angle:03d}.png'))
                        else:
                            second_row.append(Image.open(normal_dir / f'{angle:03d}.png'))
                            second_row.append(Image.open(T_normal_dir / f'{angle:03d}.png'))
                thumbnail = pil_concat_v([pil_concat_h(first_row), pil_concat_h(second_row)], w=4*512).convert("RGB")
                thumbnail.save(thumbnail_dir / f'{subject}.jpg')
        else:
            if normal:
                for idx, angle in enumerate(angle_list):
                    if idx // 2 == 0:
                        first_row.append(Image.open(T_normal_dir / f'{angle:03d}.png'))
                    else:
                        second_row.append(Image.open(T_normal_dir / f'{angle:03d}.png'))
                thumbnail = pil_concat_v([pil_concat_h(first_row), pil_concat_h(second_row)], w=2*512).convert("RGB")
                thumbnail.save(thumbnail_dir / f'{subject}.jpg')

##################################################################################### 
################################     Render face     ################################ 
##################################################################################### 

if render_face:
    ''' Render face '''
    if render_gt:
        rndr.set_norm_mat(face_scale, face_vmed)
    if render_smpl:
        rndr_smpl.set_norm_mat(face_scale, face_vmed)

    # remove vertices except flame vertices
    # https://github.com/mikedh/trimesh/issues/1344
    '''
    # Leave only face vertices
    face_mask = np.zeros(len(smpl_vertices), dtype=np.int32)
    face_mask[face_idxs] = 1
    face_with_first_vertex = (smpl_mesh.faces.prod(axis=1) == 0)  # find faces with the 0-th vertex
    smpl_mesh.update_vertices(face_mask==1)
    valid_faces = (smpl_mesh.faces.prod(axis=1) != 0) + face_with_first_vertex
    smpl_mesh.update_faces(valid_faces)

    face_vertices = smpl_mesh.vertices
    face_faces = smpl_mesh.faces
    face_vertex_normals = smpl_mesh.vertex_normals
    '''

    # camera
    cam = Camera(width=res, height=res)
    cam.ortho_ratio = 0.4 * (512 / res)

    for y in range(0, 360, yaw_step):
        R = opengl_util.make_rotate(0, math.radians(y), 0)
        if render_gt:
            rndr.rot_matrix = R
            # detailed mesh render
            cam.near = -1000
            cam.far = 1000
            cam.sanity_check()

            rndr.set_camera(cam)
            rndr.display()
            if color:
                opengl_util.render_result(
                    rndr, 0,
                    os.path.join(save_dir, 'color_face_F', f'{y:03d}.png'))
            if normal:
                opengl_util.render_result(
                    rndr, 1,
                    os.path.join(save_dir, 'normal_face_F', f'{y:03d}.png'))

            if depth:
                opengl_util.render_result(
                    rndr, 2,
                    os.path.join(save_dir, 'depth_face_F', f'{y:03d}.png'))

        # smpl mesh render
        if render_smpl:
            rndr_smpl.rot_matrix = R
            cam.near = -1000
            cam.far = 1000
            cam.sanity_check()

            rndr_smpl.set_camera(cam)
            rndr_smpl.display()
            if normal:
                opengl_util.render_result(
                    rndr_smpl, 1,
                    os.path.join(save_dir, 'T_normal_face_F', f'{y:03d}.png'))
            if depth:
                opengl_util.render_result(
                    rndr_smpl, 2,
                    os.path.join(save_dir, 'T_depth_face_F',
                                f'{y:03d}.png'))

    if thumbnail:
        angle_list = [0, 90, 180, 270]
        thumbnail_dir = save_dir.parent.parent / f'{dataset}_face_thumbnails_4view'
        thumbnail_dir.mkdir(exist_ok=True)
        normal_dir = save_dir / 'normal_face_F'
        T_normal_dir = save_dir / 'T_normal_face_F'
        first_row = []
        second_row = []
        if render_gt:
            if normal:
                for idx, angle in enumerate(angle_list):
                    if idx // 2 == 0:
                        first_row.append(Image.open(normal_dir / f'{angle:03d}.png'))
                        first_row.append(Image.open(T_normal_dir / f'{angle:03d}.png'))
                    else:
                        second_row.append(Image.open(normal_dir / f'{angle:03d}.png'))
                        second_row.append(Image.open(T_normal_dir / f'{angle:03d}.png'))
                thumbnail = pil_concat_v([pil_concat_h(first_row), pil_concat_h(second_row)], w=4*512).convert("RGB")
                thumbnail.save(thumbnail_dir / f'{subject}.jpg')
        else:
            if normal:
                for idx, angle in enumerate(angle_list):
                    if idx // 2 == 0:
                        first_row.append(Image.open(T_normal_dir / f'{angle:03d}.png'))
                    else:
                        second_row.append(Image.open(T_normal_dir / f'{angle:03d}.png'))
                thumbnail = pil_concat_v([pil_concat_h(first_row), pil_concat_h(second_row)], w=2*512).convert("RGB")
                thumbnail.save(thumbnail_dir / f'{subject}.jpg')
