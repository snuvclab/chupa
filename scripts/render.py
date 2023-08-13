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

os.environ['PYOPENGL_PLATFORM'] = "egl"

import renderer.opengl_util as opengl_util
from renderer.mesh import load_fit_body, load_scan, compute_tangent
import renderer.prt_util as prt_util
from renderer.gl.init_gl import initialize_GL_context
from renderer.gl.prt_render import PRTRender
from renderer.gl.color_render import ColorRender
from renderer.camera import Camera
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

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--in_dir', type=str, help='input dir')
parser.add_argument('-s', '--subject', type=str, help='subject name')
parser.add_argument('-o', '--out_dir', type=str, help='output dir')
parser.add_argument('-r', '--rotation', default=36, type=int, help='rotation num')
parser.add_argument('-w', '--size', default=512, type=int, help='render size')
parser.add_argument('--egl', action='store_true')
parser.add_argument('--body', action='store_true')
parser.add_argument('--head', action='store_true')
parser.add_argument('--gt', action='store_true')
parser.add_argument('--smpl', action='store_true')
parser.add_argument('--smpl_type', choices=['smpl', 'smplx'], default='smplx')
parser.add_argument('--color', action='store_true')
parser.add_argument('--offset', action='store_true')
parser.add_argument('--normal', action='store_true')
parser.add_argument('--geo', action='store_true', help='render shaded image for eval FID_shade')
parser.add_argument('--depth', action='store_true')
parser.add_argument('--front', action='store_true')
parser.add_argument('--back', action='store_true')
parser.add_argument('--dataset', type=str, default='renderpeople')
parser.add_argument('--format', type=str, default='obj', choices=(['obj', 'ply']))
parser.add_argument('--eval', action='store_true')
parser.add_argument('--exp', type=str, default=None)
parser.add_argument('--model', choices=['ours', 'gdna', 'pifuhd', 'icon'], default=None)
parser.add_argument('--fine', action='store_true')
parser.add_argument('--thumbnail', action='store_true')
parser.add_argument('--align_yaw', action='store_true')
parser.add_argument('--pifu', action='store_true')
parser.add_argument('--tpose', action='store_true')
parser.add_argument('--random_light', action='store_true')

parser.add_argument('--beta', default=None, type=float, help='custom beta for smpl')
parser.add_argument('--beta_idx', default=0, type=int)
args = parser.parse_args()

eval = args.eval
exp = args.exp
model = args.model
gdna_split = 'fine' if args.fine else 'coarse'

subject = args.subject
if subject.endswith('txt'):
    exit()
subject_prefix = f'{subject}_' if eval else ''
if model == 'ours':
    subfolder = Path('thumbnails') / exp
elif model == 'gdna':
    subfolder = Path('meshes') / gdna_split

input_dir = Path(args.in_dir) / args.dataset if model is None else Path(args.in_dir) / subfolder 
save_folder = Path(args.out_dir) / args.dataset / 'render' / f'{subject}' if not eval else Path(args.out_dir)

if model == 'gdna':
    save_folder = save_folder / f'{exp}_{gdna_split}'
elif model == 'ours':
    save_folder = save_folder / f'{exp}'
# if save_folder.exists():
#     exit()
save_folder.mkdir(parents=True, exist_ok=True)
rotation = int(args.rotation)
size = int(args.size)
dataset = args.dataset
thumbnail = args.thumbnail
align_yaw = args.align_yaw
pifu = args.pifu
tpose = args.tpose
random_light = args.random_light

beta = args.beta
beta_idx = args.beta_idx

# headless
if args.egl:
    egl = True
else:
    egl = False

render_body = args.body
render_head = args.head
render_gt = args.gt
render_smpl = args.smpl
render_front = args.front
render_back = args.back

# render
initialize_GL_context(width=size, height=size, egl=egl)

format = args.format
scale = 100.0
up_axis = 1
smpl_type = 'smpl' if model == 'gdna' else args.smpl_type 
thuman = not subject.startswith('rp_')
smpl_gender = 'male' if thuman else 'neutral'

with_light = False
color = args.color
offset = args.offset
depth = args.depth
normal = args.normal
geo = args.geo
suffix = '_100k' if dataset == 'renderpeople' else ''
scale = 100.0 if model == 'ours' else 1.0

if eval:
    if model == 'ours':
        mesh_file = input_dir / f'{subject}.obj'
    elif model == 'gdna':
        mesh_file = input_dir / f'{subject}.obj'
    else:
        mesh_file = input_dir  / subject / f'{subject}.{format}'
    smpl_dir = save_folder.parent.parent.parent / f'999_{smpl_type}_params'
    smpl_file = smpl_dir / f'{subject}_{smpl_type}.pkl'
else:
    if dataset == 'renderpeople' or dataset == 'shhq':
        mesh_file = input_dir / 'scans' / subject / f'{subject}{suffix}.{format}'
        smpl_file = input_dir / smpl_type / f'{subject}.pkl'
        tex_file = input_dir / 'scans' / subject / f'{subject}_dif_2k.jpg'
        # smpl_file = input_dir / subject / f'{smpl_type}' / f'{smpl_type}_param.pkl'
    elif dataset == 'thuman':
        mesh_file = input_dir / 'scans' / subject / f'{subject}{suffix}.{format}'
        smpl_file = input_dir / smpl_type / f'{subject}.pkl'
        tex_file = input_dir / 'scans' / subject / 'material0.jpeg'
    elif dataset == 'coco':
        smpl_file = input_dir / subject / f'{subject}.pkl'
    elif dataset == 'agora':
        smpl_file = input_dir / subject / f'{subject}.pkl'
    elif dataset == 'frankmocap':
        smpl_file = input_dir / f'{subject}_prediction_result.pkl'
    elif dataset.startswith('renderpeople_offset'):
        mesh_file = input_dir / 'scans' / f'{subject}.{format}'
        smpl_file = input_dir / smpl_type / f'{subject}.pkl'
    else:
        raise NotImplementedError

smpl_param = np.load(smpl_file, allow_pickle=True)
if beta is not None:
    smpl_param['betas'][0, beta_idx] = beta

if not eval:
    if not (save_folder / f'{smpl_type}_param.pkl').exists():
        # shutil.copyfile(smpl_file, save_folder / f'{smpl_type}_param.pkl')
        with open(save_folder / f'{smpl_type}_param.pkl', 'wb') as f:
            pickle.dump(smpl_param, f)

# mesh
if render_gt:
    mesh = trimesh.load(str(mesh_file),
                        skip_materials=True,
                        process=False,
                        maintain_order=True,
                        force='mesh')

    if format == 'obj':
        vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
            str(mesh_file), with_normal=True, with_texture=True)
    else:
        # remove floating outliers of scans
        mesh_lst = mesh.split(only_watertight=False)
        comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
        mesh = mesh_lst[comp_num.index(max(comp_num))]

        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.vertex_normals

        # if gdna:
        #     smpl_param_trans_scale = np.load(smpl_dir / f'{subject}_smpl.pkl', allow_pickle=True)
        #     vertices = vertices * smpl_param_trans_scale['scale'] + smpl_param_trans_scale['transl']

    if align_yaw and model is None:
        yaw_inverse_mat = get_yaw_inverse_mat(smpl_param['global_orient'])
        vertices = vertices @ yaw_inverse_mat.T
        normals = normals @ yaw_inverse_mat.T

# if not eval:
# smpl_mesh, joints = load_fit_body(smpl_file,
#                                   scale=1.0,
#                                   smpl_type=smpl_type,
#                                   smpl_gender=smpl_gender)
if model == 'gdna':
    gdna_smpl_dir = save_folder.parent.parent.parent / f'smpl_from_gdna_server'
    smpl_dict = np.load(gdna_smpl_dir / f'{subject}.npz', allow_pickle=True)['arr_0'].item()
    smpl_mesh = trimesh.Trimesh(vertices=smpl_dict['vertices'], faces=smpl_dict['faces'], process=False, maintains_order=True)
    joints = smpl_dict['joints']
elif dataset == 'frankmocap':
    # pkl file contatins vertices, faces, joints
    flip_yz = np.array([[1,0,0], [0, -1, 0], [0, 0, -1]])
    smpl_param = smpl_param['pred_output_list'][0]
    smpl_mesh = trimesh.Trimesh(vertices=smpl_param['pred_vertices_img'] @ flip_yz, faces=smpl_param['faces'], process=False, maintains_order=True)
    joints = smpl_param['pred_body_joints_img'] @ flip_yz
else:
    smpl_info = {}
    smpl_info['model_path'] = f"src/smpl_related/models/{smpl_type}"
    smpl_info['param'] = smpl_param
    smpl_info['type'] = smpl_type
    smpl_info['gender'] = smpl_gender
    smpl_mesh, joints = load_smpl(smpl_info,
                                scale=scale,
                                align_yaw=align_yaw,
                                tpose=tpose)
smpl_vertices = smpl_mesh.vertices
smpl_faces = smpl_mesh.faces
smpl_vertex_normals = smpl_mesh.vertex_normals

if smpl_type == 'smplx':
    smpl_segmentation_path = 'src/smpl_related/smpl_data/smplx_vert_segmentation.json' 
    with open(smpl_segmentation_path, 'r') as f:
        smpl_segmentation = json.load(f)
    # head_idxs = smpl_segmentation['head'] + smpl_segmentation['neck'] + [8811, 8812, 8813, 8814, 9161, 9165] +\
    #             smpl_segmentation['rightEye'] + smpl_segmentation['leftEye'] + smpl_segmentation['eyeballs']
    # flame version
    head_idxs = np.load('src/smpl_related/smpl_data/SMPL-X__FLAME_vertex_ids.npy')
    smpl_vertex_colors = 2 * part_segm_to_vertex_colors(smpl_segmentation, smpl_vertices.shape[0])[:, :3] - 1

elif smpl_type == 'smpl':
    smpl_segmentation_path = 'src/smpl_related/smpl_data/smpl_vert_segmentation.json' 
    with open(smpl_segmentation_path, 'r') as f:
        smpl_segmentation = json.load(f)
    head_idxs = smpl_segmentation['head'] + smpl_segmentation['neck']

# find body scale and center
if pifu:
    body_scale = 180.0 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])
    body_vmin = vertices.min(0)
    body_vmax = vertices.max(0)
    body_vmed = np.median(vertices, 0)
    body_vmed[up_axis] = 0.5 * (body_vmax[up_axis] + body_vmin[up_axis])
else:
    body_scale = 180.0 / (smpl_vertices.max(0)[up_axis] - smpl_vertices.min(0)[up_axis])
    body_vmin = smpl_vertices.min(0)
    body_vmax = smpl_vertices.max(0)
    body_vmed = joints[0]
    body_vmed[up_axis] = 0.5 * (body_vmax[up_axis] + body_vmin[up_axis])

if render_head:
    # find head scale and center
    head_vertices = smpl_vertices[head_idxs]
    head_scale = 150.0 / (head_vertices.max(0)[up_axis] - head_vertices.min(0)[up_axis])  # Use 150 for head meshes instead of 180 due to the horizontal/vertical ratio

    head_vmin = head_vertices.min(0)
    head_vmax = head_vertices.max(0)
    head_vmed = 0.5 * (head_vmax + head_vmin)

if render_gt:
# Set detailed mesh renderer
    if geo:
        rndr = ColorRender(width=size, height=size, egl=egl, geo=geo)
        rndr.set_mesh(vertices, 
                      faces,
                      vertices,
                      normals)
    else:
        if format == 'ply':
            # colors = mesh.visual.vertex_colors[:, :3] / 255.0
            offsets = 2 * (mesh.visual.vertex_colors[:, :3] / 255.0) - 1
            # offsets = mesh.visual.vertex_colors[:, :3]
            # rndr = ColorRender(width=size, height=size, egl=egl)
            # rndr.set_mesh(vertices, faces, colors, normals)
            rndr = ColorRender(width=size, height=size, egl=egl, offset=True)
            rndr.set_mesh(vertices, faces, offsets, normals)

        else:
            prt, face_prt = prt_util.computePRT(mesh_file, scale, 10, 2)
            rndr = PRTRender(width=size, height=size, egl=egl)
            tan, bitan = compute_tangent(normals)
            rndr.set_mesh(
                vertices,
                faces,
                normals,
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
    rndr_smpl = ColorRender(width=size, height=size, egl=egl)
    rndr_smpl.set_mesh(smpl_vertices, smpl_faces,
                       smpl_vertices,
                    # smpl_vertex_colors,
                    smpl_vertex_normals)

''' Render full body'''
if render_body:
    if render_gt:
        rndr.set_norm_mat(body_scale, body_vmed)
    if render_smpl:
        rndr_smpl.set_norm_mat(body_scale, body_vmed)

    # camera
    cam = Camera(width=size, height=size)
    cam.ortho_ratio = 0.4 * (512 / size)

    for y in range(0, 360, 360 // rotation):
        R = opengl_util.make_rotate(0, math.radians(y), 0)
        if render_gt:
            # detailed mesh render
            rndr.rot_matrix = R
            if render_front:
                cam.near = -1000
                cam.far = 1000
                cam.sanity_check()

                rndr.set_camera(cam)
                rndr.display()
                if color:
                    # random light
                    if random_light:
                        shs = np.load('./scripts/env_sh.npy')
                        sh_id = random.randint(0, shs.shape[0] - 1)
                        sh = shs[sh_id]
                        sh_angle = 0.2 * np.pi * (random.random() - 0.5)
                        sh = opengl_util.rotateSH(sh, opengl_util.make_rotate(0, sh_angle, 0).T)

                        rndr.set_sh(sh)
                        rndr.analytic = False
                        rndr.use_inverse_depth = False

                    opengl_util.render_result(
                        rndr, 0,
                        os.path.join(save_folder,'color_F', f'{y:03d}.png'))
                if normal:
                    if eval:
                        opengl_util.render_result(
                        rndr, 1,
                        os.path.join(save_folder, f'{subject_prefix}{y:03d}.png'))
                    else:
                        opengl_util.render_result(
                            rndr, 1,
                            os.path.join(save_folder, 'normal_F', f'{subject_prefix}{y:03d}.png'))

                if depth:
                    opengl_util.render_result(
                        rndr, 2,
                        os.path.join(save_folder, 'depth_F', f'{y:03d}.png'))
                    
                if offset:
                    opengl_util.render_result(
                        rndr, 0,
                        os.path.join(save_folder,'offset_F', f'{y:03d}.png'))
                    opengl_util.render_result(
                        rndr, 2,
                        os.path.join(save_folder,'T_offset_F', f'{y:03d}.png'))

            if render_back:
                cam.near = 1000
                cam.far = -1000
                cam.sanity_check()
                rndr.set_camera(cam)
                rndr.display()
                if color:
                    opengl_util.render_result(
                        rndr, 0, 
                        os.path.join(save_folder, 'color_B', f'{y:03d}.png'))
                if normal:
                    opengl_util.render_result(
                        rndr, 1,
                        os.path.join(save_folder, 'normal_B', f'{y:03d}.png'))

                if depth:
                    opengl_util.render_result(
                        rndr, 2,
                        os.path.join(save_folder, 'depth_B', f'{y:03d}.png'))

        if render_smpl:
        # smpl mesh render
            rndr_smpl.rot_matrix = R
            if render_front:
                cam.near = -1000
                cam.far = 1000
                cam.sanity_check()

                rndr_smpl.set_camera(cam)
                rndr_smpl.display()
                if color:
                    opengl_util.render_result(
                        rndr_smpl, 0, 
                        os.path.join(save_folder, 'T_color_F', f'{y:03d}.png'))
                if normal:
                    opengl_util.render_result(
                        rndr_smpl, 1,
                        os.path.join(save_folder, 'T_normal_F', f'{y:03d}.png'))
                if depth:
                    opengl_util.render_result(
                        rndr_smpl, 2,
                        os.path.join(save_folder, 'T_depth_F',
                                    f'{y:03d}.png'))

            if render_back:
                cam.near = 1000
                cam.far = -1000
                cam.sanity_check()

                rndr_smpl.set_camera(cam)
                rndr_smpl.display()
                # if color:
                #     opengl_util.render_result(
                #         rndr_smpl, 0, os.path.join(save_folder, subject, 'T_color_B', f'{y:03d}.png'))
                if normal:
                    opengl_util.render_result(
                    rndr_smpl, 1,
                    os.path.join(save_folder, 'T_normal_B', f'{y:03d}.png'))
                if depth:
                    opengl_util.render_result(
                        rndr_smpl, 2,
                        os.path.join(save_folder, 'T_depth_B',
                                    f'{y:03d}.png'))

    if thumbnail:
        # save thumbnail images
        angle_list = [0, 90, 180, 270]
        thumbnail_dir = save_folder.parent.parent / f'{dataset}_body_thumbnails_4view'
        thumbnail_dir.mkdir(exist_ok=True)
        color_dir = save_folder / 'color_F'
        normal_dir = save_folder / 'normal_F'
        T_color_dir = save_folder / 'T_color_F'
        T_normal_dir = save_folder / 'T_normal_F'
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

if render_head:
    ''' Render head '''
    if render_gt:
        rndr.set_norm_mat(head_scale, head_vmed)
    if render_smpl:
        rndr_smpl.set_norm_mat(head_scale, head_vmed)

    # remove vertices except flame vertices
    # https://github.com/mikedh/trimesh/issues/1344
    '''
    # Leave only head vertices
    head_mask = np.zeros(len(smpl_vertices), dtype=np.int32)
    head_mask[head_idxs] = 1
    face_with_first_vertex = (smpl_mesh.faces.prod(axis=1) == 0)  # find faces with the 0-th vertex
    smpl_mesh.update_vertices(head_mask==1)
    valid_faces = (smpl_mesh.faces.prod(axis=1) != 0) + face_with_first_vertex
    smpl_mesh.update_faces(valid_faces)

    head_vertices = smpl_mesh.vertices
    head_faces = smpl_mesh.faces
    head_vertex_normals = smpl_mesh.vertex_normals
    '''

    # camera
    cam = Camera(width=size, height=size)
    cam.ortho_ratio = 0.4 * (512 / size)

    for y in range(0, 360, 360 // rotation):
        R = opengl_util.make_rotate(0, math.radians(y), 0)
        if render_gt:
            rndr.rot_matrix = R
            # detailed mesh render
            if render_front:
                cam.near = -1000
                cam.far = 1000
                cam.sanity_check()

                rndr.set_camera(cam)
                rndr.display()
                if color:
                    # random light
                    if random_light:
                        shs = np.load('./scripts/env_sh.npy')
                        sh_id = random.randint(0, shs.shape[0] - 1)
                        sh = shs[sh_id]
                        sh_angle = 0.2 * np.pi * (random.random() - 0.5)
                        sh = opengl_util.rotateSH(sh, opengl_util.make_rotate(0, sh_angle, 0).T)

                        rndr.set_sh(sh)
                        rndr.analytic = False
                        rndr.use_inverse_depth = False

                    opengl_util.render_result(
                        rndr, 0,
                        os.path.join(save_folder,'color_face_F', f'{y:03d}.png'))
                if normal:
                    if eval:
                        opengl_util.render_result(
                        rndr, 1,
                        os.path.join(save_folder, f'{subject_prefix}{y:03d}.png'))
                    else:
                        opengl_util.render_result(
                            rndr, 1,
                            os.path.join(save_folder, 'normal_face_F', f'{y:03d}.png'))

                if depth:
                    opengl_util.render_result(
                        rndr, 2,
                        os.path.join(save_folder, 'depth_face_F', f'{y:03d}.png'))

            if render_back:
                cam.near = 1000
                cam.far = -1000
                cam.sanity_check()

                rndr.set_camera(cam)
                rndr.display()
                if normal:
                    opengl_util.render_result(
                        rndr, 1,
                        os.path.join(save_folder, 'normal_face_B', f'{y:03d}.png'))

                if depth:
                    opengl_util.render_result(
                        rndr, 2,
                        os.path.join(save_folder, 'depth_face_B', f'{y:03d}.png'))

        # smpl mesh render
        if render_smpl:
            rndr_smpl.rot_matrix = R
            if render_front:
                cam.near = -1000
                cam.far = 1000
                cam.sanity_check()

                rndr_smpl.set_camera(cam)
                rndr_smpl.display()
                if normal:
                    opengl_util.render_result(
                        rndr_smpl, 1,
                        os.path.join(save_folder, 'T_normal_face_F', f'{y:03d}.png'))
                if depth:
                    opengl_util.render_result(
                        rndr_smpl, 2,
                        os.path.join(save_folder, 'T_depth_face_F',
                                    f'{y:03d}.png'))

            # if render_back:
            #     cam.near = 1000
            #     cam.far = -1000
            #     cam.sanity_check()

            #     rndr_smpl.set_camera(cam)
            #     rndr_smpl.display()
            #     if normal:
            #         opengl_util.render_result(
            #         rndr_smpl, 1,
            #         os.path.join(save_folder, 'T_normal_head_B', f'{y:03d}.png'))
            #     if depth:
            #         opengl_util.render_result(
            #             rndr_smpl, 2,
            #             os.path.join(save_folder, 'T_depth_head_B',
            #                         f'{y:03d}.png'))

    if thumbnail:
        angle_list = [0, 90, 180, 270]
        thumbnail_dir = save_folder.parent.parent / f'{dataset}_face_thumbnails_4view'
        thumbnail_dir.mkdir(exist_ok=True)
        normal_dir = save_folder / 'normal_face_F'
        T_normal_dir = save_folder / 'T_normal_face_F'
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


