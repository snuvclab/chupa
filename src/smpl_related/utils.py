import torch
import numpy as np
import trimesh
from smpl_related.smplx.body_models import SMPL, SMPLX

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

def load_smpl(smpl_info, scale, align_yaw=True, tpose=False):
        # slight modification from src/renderer/mesh.py load_fit_body

        param = smpl_info['param']
        for key in param.keys():
            if key == 'left_hand_pose' or key == 'right_hand_pose':
                param[key] = torch.tensor(param[key][:, :6])
            else:
                if not type(param[key]) == str:
                    param[key] = torch.as_tensor(param[key])

        # TODO: remove this part and make smpl pkl file preprocessing code!
        # For pkl from MultiviewSMPLify-X 
        if param['body_pose'].shape[1] == 66:  # for smplx, num_body_joints=21 due to wrist joints are covered by MANO
            param['body_pose'] = param['body_pose'][:, 3:]
        if param.get('scale') is None:
            try:
                param['scale'] = param['body_scale']
            except:
                param['scale'] = 1.0
        if param.get('translation') is None:
            try: 
                param['translation'] = param['global_body_translation']
            except:
                param['translation'] = 0

        smpl_type = smpl_info['type']
        smpl_gender = smpl_info['gender']

        if smpl_type == 'smpl':
            smpl_model = SMPL(model_path=smpl_info['model_path'])
            model_forward_params = dict(betas=param['betas'],
                                        global_orient=param['global_orient'],
                                        body_pose=param['body_pose'],
                                        return_verts=True)
        elif smpl_type == 'smplx':
            smpl_model = SMPLX(model_path=smpl_info['model_path'])
            if tpose:
                model_forward_params = dict(betas=torch.zeros_like(param['betas']),
                                            global_orient=torch.zeros_like(param['global_orient']),
                                            body_pose=torch.zeros_like(param['body_pose']),
                                            left_hand_pose=torch.zeros_like(param['left_hand_pose']),
                                            right_hand_pose=torch.zeros_like(param['right_hand_pose']),
                                            jaw_pose=torch.zeros_like(param['jaw_pose']),
                                            leye_pose=torch.zeros_like(param['leye_pose']),
                                            reye_pose=torch.zeros_like(param['reye_pose']),
                                            expression=torch.zeros_like(param['expression']),
                                            return_verts=True)
            else:
                model_forward_params = dict(betas=param['betas'],
                                            global_orient=param['global_orient'],
                                            body_pose=param['body_pose'],
                                            left_hand_pose=param['left_hand_pose'],
                                            right_hand_pose=param['right_hand_pose'],
                                            jaw_pose=param['jaw_pose'],
                                            leye_pose=param['leye_pose'],
                                            reye_pose=param['reye_pose'],
                                            expression=param['expression'],
                                            return_verts=True)
        else:
            raise NotImplementedError

        smpl_out = smpl_model(**model_forward_params)

        smpl_verts = (
            (smpl_out.vertices[0] * param['scale'] + param['translation']) *
            scale).detach()
        smpl_joints = (
            (smpl_out.joints[0] * param['scale'] + param['translation']) *
            scale).detach()

        yaw_inverse_mat = get_yaw_inverse_mat(param['global_orient'])
        if align_yaw: 
            smpl_verts = smpl_verts @ yaw_inverse_mat.T
            smpl_joints = smpl_joints @ yaw_inverse_mat.T
            
        smpl_mesh = trimesh.Trimesh(smpl_verts,
                                    smpl_model.faces,
                                    process=False,
                                    maintain_order=True)

        return smpl_mesh, smpl_joints, yaw_inverse_mat