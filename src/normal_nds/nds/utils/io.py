import numpy as np
from pathlib import Path
import torch
import json
import trimesh

from normal_nds.nds.core import Mesh, View

def read_mesh(path, device='cpu'):
    mesh_ = trimesh.load_mesh(str(path), process=False)

    vertices = np.array(mesh_.vertices, dtype=np.float32)
    indices = None
    if hasattr(mesh_, 'faces'):
        indices = np.array(mesh_.faces, dtype=np.int32)

    return Mesh(vertices, indices, device)

def write_mesh(path, mesh, flip=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = mesh.vertices.numpy()
    indices = mesh.indices.numpy() if mesh.indices is not None else None

    mesh.compute_normals()
    vertex_normals = mesh.vertex_normals.numpy()
    face_normals = mesh.face_normals.numpy()

    mesh_ = trimesh.Trimesh(
        vertices=vertices, faces=indices, 
        face_normals=face_normals, 
        vertex_normals=vertex_normals,
        process=False)
    if flip:
        mesh_.apply_transform([[-1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    mesh_.export(path, include_normals=True)

def read_views(directory, mask_threshold, scale, device, data_type):
    if data_type == 'dvr':
        assert len(directory) == 1
        image_paths = sorted([path for path in directory[0].iterdir() if (path.is_file() and path.suffix == '.png')])
    
        views = []
        for image_path in image_paths:
            views.append(View.load_dvr(str(image_path), view_angle=(image_path.stem), device=device))
    elif data_type == 'co3d':
        assert len(directory) == 3 # directory[0] : images path, directory[1] : masks path, directory[2] : cameras path
        cam = np.load(directory[2])
        assert len(cam.keys()) % 3 == 0

        image_paths = sorted([path for path in directory[0].iterdir()])
        mask_paths = sorted([path for path in directory[1].iterdir()])

        cam_key = cam.keys()
        views = []
        '''
        for i in range(len(image_paths)):
            if image_paths[i].name in cam_key:
                #assert image_paths[i].name in mask_paths[i].name
                cam_id = str(cam[image_paths[i].name])
                pose = cam["pose_"+cam_id]
                intrinsic = cam["intrinsic_"+cam_id]
                views.append(View.load_co3d(image_paths[i], mask_paths[i], pose, intrinsic, device))
        '''
        mask_name_list = [path.name for path in mask_paths]
        for i in range(len(image_paths)):
            if image_paths[i].name in cam_key:
                if image_paths[i].name+".png" not in mask_name_list:
                    continue
                ind = mask_name_list.index(image_paths[i].name+".png")
                cam_id = str(cam[image_paths[i].name])
                pose = cam["pose_"+cam_id]
                intrinsic = cam["intrinsic_"+cam_id]
                view_co3d = View.load_co3d(image_paths[i], mask_paths[ind], pose, intrinsic, mask_threshold, device)
                if view_co3d is not None:
                    views.append(view_co3d)
    else:
        raise Exception("Invalid dataset type")
    
    print("Found {:d} views".format(len(views)))

    if scale > 1:
        for view in views:
            view.scale(scale)
        print("Scaled views to 1/{:d}th size".format(scale))

    return views

def load_smpl_info(smpl_param_path, smpl_type='smplx', gender='neutral', smpl_related_dir='./smpl_related/'):
    smpl_info = {}
    smpl_related_dir = Path(smpl_related_dir)
    smpl_info['model_path'] = smpl_related_dir / f"models/{smpl_type}"
    smpl_info['param'] = np.load(smpl_param_path, allow_pickle=True)
    smpl_info['type'] = smpl_type
    smpl_info['gender'] = gender
    smpl_seg_path = smpl_related_dir / 'smpl_data' / f'{smpl_type}_vert_segmentation.json'
    if smpl_type == 'smpl':
        with open(smpl_seg_path) as f:
            smpl_segmentation = json.load(f)
        smpl_info['head_idxs'] = smpl_segmentation['head'] + smpl_segmentation['neck']
    elif smpl_type == 'smplx':
        smpl_info['face_idxs'] = np.load(smpl_related_dir / 'smpl_data' / 'FLAME_SMPLX_vertex_ids.npy')
        smpl_info['eyeball_fid'] = np.load(smpl_related_dir / 'smpl_data' / 'eyeball_fid.npy')
        smpl_info['fill_mouth_fid'] = np.load(smpl_related_dir / 'smpl_data' / 'fill_mouth_fid.npy')

    return smpl_info