import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from PIL import Image
import pickle as pkl
import random
import pymeshlab
import shutil
import json
import omegaconf
from omegaconf import OmegaConf
from datetime import datetime
import io
import time
import os, sys
import cv2
from functools import partial
from scipy.spatial.transform import Rotation as R

from ldm.models.diffusion.image_editor import ImageEditor
from ldm.utils.data_helpers import instantiate_from_config
from normal_nds.reconstruct import NormalNDS 
from normal_nds.nds.modules import ViewSampler
from normal_nds.nds.utils import read_mesh, write_mesh, load_smpl_info

from img_utils import (to_world, to_cam, to_cam_B, fliplr_nml, 
                       to_rgba, to_pil, pil_concat_h, pil_concat_v,
                       frontback_img, merge_frontback)

if 'gradio_test.py' in sys.argv[0]:
    from bodymocap.body_mocap_api import BodyMocap
    sys.path.append('third_party/frankmocap/detectors/body_pose_estimator')
    from bodymocap.body_bbox_detector import BodyPoseEstimator
    from bodymocap import constants
    from bodymocap.utils.imutils import crop, crop_bboxInfo, process_image_bbox, process_image_keypoints, bbox_from_keypoints
    
class Chupa():
    def __init__(self, config : omegaconf.dictconfig.DictConfig):
        for key, value in config.chupa.items():
            setattr(self, key, value)
        self.smpl_gender = 'male' if config.dataset.dataset_type == 'thuman' else 'neutral'

        body_config_path = Path(config.diffusion.root_path) / config.diffusion.body_task / "config.yaml" 
        body_ckpt_path = Path(config.diffusion.root_path) / config.diffusion.body_task / config.diffusion.body_ckpt 
        self.model_body = ImageEditor(body_config_path, body_ckpt_path, device=self.device)
        self.ch_slice_body = self.model_body.channels // 2

        if self.use_closeup:
            face_config_path = Path(config.diffusion.root_path) / config.diffusion.face_task / "config.yaml" 
            face_ckpt_path = Path(config.diffusion.root_path) / config.diffusion.face_task / config.diffusion.face_ckpt 
            self.model_face = ImageEditor(face_config_path, face_ckpt_path, device=self.device)
            self.ch_slice_face = self.model_face.channels // 2

        if self.use_text or self.gradio:
            text_config_path = Path(config.diffusion.root_path) / config.diffusion.bodytext_task / "config.yaml" 
            text_ckpt_path = Path(config.diffusion.root_path) / config.diffusion.bodytext_task / config.diffusion.bodytext_ckpt 
            self.text_editor = ImageEditor(text_config_path, text_ckpt_path, device=self.device)

        self.normal_nds = NormalNDS(args=config.nds, device=self.device)

        if self.gradio:
            self.mesh_ext = '.glb'
            smpl_model_dir = str(Path(self.smpl_related_dir) / 'models' / 'smplx')
            self.body_bbox_detector = BodyPoseEstimator()
            self.body_mocap = BodyMocap(regressor_checkpoint=config.frankmocap.body_regressor_checkpoint, 
                                        smpl_dir=smpl_model_dir, 
                                        device = self.device, use_smplx= True)
        else:
            self.mesh_ext = '.obj'
            self.body_bbox_detector = None
            self.body_mocap = None
            
        self.output_dir = Path(self.output_root) / config.dataset.dataset_type / datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(config, self.output_dir / 'config.yaml')
    
    def run_frankmocap(self, input_image):
        # import from frankmocap
        from integration.copy_and_paste import transfer_rotation
        import mocap_utils.geometry_utils as gu

        input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        
        _, body_bbox_list = self.body_bbox_detector.detect_body_pose(input_image_bgr)
        body_bbox = body_bbox_list[np.argmax([(x[2] * x[3]) for x in body_bbox_list])]  

        pred_body_list = self.body_mocap.regress(input_image_bgr, [body_bbox])

        body_info = pred_body_list[0]
    
        img, norm_img, boxScale_o2n, bboxTopLeft, bbox = process_image_bbox(
                input_image_bgr, body_bbox, input_res=constants.IMG_RES)

        
        with torch.no_grad():
            # model forward
            pred_rotmat, pred_betas, pred_camera = self.body_mocap.model_regressor(norm_img.to(self.device))

            #Convert rot_mat to aa since hands are always in aa
            # pred_aa = rotmat3x3_to_angle_axis(pred_rotmat)
            pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
            pred_aa = pred_aa.reshape(pred_aa.shape[0], 72)
            global_orient = pred_aa[:,:3]
            global_orient_mat = R.from_rotvec(global_orient.cpu()).as_matrix()
            r_xpi = R.from_euler('x', 180, degrees=True).as_matrix()
            final = r_xpi @ global_orient_mat
            global_orient_flipped = torch.tensor(R.from_matrix(final).as_rotvec(),
                                                 dtype=global_orient.dtype, device=global_orient.device)
            smpl_output = self.body_mocap.smpl(
                betas=pred_betas, 
                body_pose=pred_aa[:,3:],
                global_orient=global_orient_flipped, 
                pose2rot=True)
            pred_vertices = smpl_output.vertices[0].cpu().numpy() 
            pred_joints_3d = smpl_output.joints[0].cpu().numpy()  # (1,49,3)
            global_orient = smpl_output.global_orient.cpu().numpy()

        smpl_info = dict()
        smpl_info['pred_vertices_smpl'] = pred_vertices
        smpl_info['faces'] = self.body_mocap.smpl.faces
        smpl_info['pred_body_joints_smpl'] = pred_joints_3d
        smpl_info['global_orient'] = global_orient

        smpl_related_dir = Path(self.smpl_related_dir) / 'smpl_data'
        smpl_info['type'] = 'smplx'
        smpl_info['face_idxs'] = np.load(smpl_related_dir / 'FLAME_SMPLX_vertex_ids.npy')
        smpl_info['eyeball_fid'] = np.load(smpl_related_dir / 'eyeball_fid.npy')
        smpl_info['fill_mouth_fid'] = np.load(smpl_related_dir / 'fill_mouth_fid.npy')

        return smpl_info
            
    def diff_sample(self, img_t, seed, steps, cfg_scale=2.0):
        initial_normal = self.model_body(
                img_t[:4, :, :], "", "",
                cfg=cfg_scale, image_cfg=cfg_scale, steps=steps, seed=seed)
            
        return initial_normal

    def difftext_sample(self, img_t, text, negative_text, 
                        seed, steps=100, cfg_scale=7.5, image_cfg_scale=1.5,
                        resample_steps=20, n_resample=5):
        assert self.use_text or self.gradio, "Text-based model not loaded. Please check your arguments."
        initial_normal_front = self.text_editor(
            img_t[:3, :, :], text, negative_text,
            cfg=cfg_scale, image_cfg=image_cfg_scale, steps=steps, seed=seed)

        initial_normal = self.model_body.forward_wfront(
            initial_normal_front, img_t[:4, :, :], 
            cfg=cfg_scale, steps=steps, seed=seed, 
            repeat_sample=5, jump_length=0.05
        )
        normal_resample = self.model_body.resample(
            initial_normal, img_t[:4, :, :], 
            cfg=cfg_scale, steps=resample_steps, seed=seed, repeat=n_resample
        )

        return normal_resample

    def diff_to_nds_input(self, normal_dual, closeup=False):
        ch_slice = self.ch_slice_face if closeup else self.ch_slice_body

        normal_front, normal_back = normal_dual[:ch_slice, :, :], normal_dual[ch_slice:, :, :]
        normal_front_cam = to_rgba(normal_front)
        normal_back_cam = torch.flip(to_rgba(normal_back), dims=[2])
        initial_normal_back_alpha = normal_back_cam[3, :, :]
        back_mask = (initial_normal_back_alpha >= 0)
        normal_back_cam[0, back_mask] = -normal_back_cam[0, back_mask]  # flip x

        return (normal_front_cam, normal_back_cam)

    def nds_dual(self, body_dual_normal_maps, output_dir):
        self.normal_nds.set_views_from_normal_maps(body_dual_normal_maps, use_closeup=False)

        if self.use_side_loss:
            side_view_angles = [(self.initial_angle + 90) % 360, (self.initial_angle - 90) % 360]
            self.normal_nds.set_side_views(side_view_angles)

        # Do optimization with dual normal maps
        nds_mesh = self.normal_nds(output_dir / 'nds_dual')

        return nds_mesh

    def nds_refine(self, body_refine_normal_maps, face_refine_normal_maps, use_closeup, output_dir):
        self.normal_nds.set_views_from_normal_maps(body_refine_normal_maps, 
                                                   face_refine_normal_maps, 
                                                   use_closeup=use_closeup)

        # Do optimization with refined normal maps
        nds_mesh = self.normal_nds(output_dir / 'nds_refine')

        return nds_mesh

    def render_dual(self, mesh, view_angles, closeup=False, ch_slice=4):
        if self.normal_nds.yaw_inverse_mat is not None:
            mesh.vertices = mesh.vertices @ self.normal_nds.yaw_inverse_mat.T

        body_rendered_normal_maps = []
        for view_angle in view_angles:
            # front
            normal_F_world, vis_mask_F  = \
                self.normal_nds.render_target_view(mesh, view_angle, closeup=closeup)
            normal_F_cam = to_cam(normal_F_world.permute(2, 0, 1), view_angle)

            # back
            normal_B_world, vis_mask_B  = \
                self.normal_nds.render_target_view(mesh, (view_angle + 180) % 360, closeup=closeup)
            normal_B_cam = to_cam(normal_B_world.permute(2, 0, 1), (view_angle + 180) % 360)
            normal_B_cam_flipped = fliplr_nml(normal_B_cam)
            
            normal_cam = torch.cat([normal_F_cam[:ch_slice], 
                                    normal_B_cam_flipped[:ch_slice]], dim=0)
            body_rendered_normal_maps.append(normal_cam)
        
        if self.normal_nds.yaw_inverse_mat is not None:
            mesh.vertices = mesh.vertices @ self.normal_nds.yaw_inverse_mat

        return body_rendered_normal_maps

    def forward_gradio(self, input_image, seed=None,
                    steps=20, cfg_scale=2.0, image_cfg_scale=1.5,
                    use_text=None, prompt="", negative_prompt="", 
                    use_resample=None, resample_T=0.02, n_resample=2, 
                    use_closeup=None, resample_T_face=0.02, n_resample_face=2):

        return self.forward(None, input_image, None, seed, steps, cfg_scale, image_cfg_scale,
                    use_resample, resample_T, n_resample, 
                    use_closeup, resample_T_face, n_resample_face,
                    use_text, prompt, negative_prompt )

    def forward(self, smpl_param_path=None, input_image=None, subject=None, seed=None,
                steps=20, cfg_scale=2.0, image_cfg_scale=1.5,
                use_resample=None, resample_T=0.02, n_resample=2, 
                use_closeup=None, resample_T_face=0.02, n_resample_face=2,
                use_text=None, prompt="", negative_prompt=""):

        assert not (smpl_param_path is None and input_image is None), \
                   'You need to specify smpl parameter or input image'        

        # In gradio inference, you can toggle/adjust the options below
        use_text = use_text if self.gradio else self.use_text
        use_resample = use_resample if self.gradio else self.use_resample
        resample_T = resample_T if self.gradio else self.resample_T
        n_resample = n_resample if self.gradio else self.n_resample
        use_closeup = use_closeup if self.gradio else self.use_closeup
        resample_T_face = resample_T_face if self.gradio else self.resample_T_face
        n_resample_face = n_resample_face if self.gradio else self.n_resample_face

        if seed is None:
            seed = int(self.seed)
        else:
            seed = int(seed)

        output_dir = Path(self.output_dir)
        if subject is not None:
            output_dir = output_dir / subject

        if use_text:
            prompt = self.prompt if prompt == "" else prompt
            output_dir = output_dir / ('_'.join(prompt.split()) + f'_{seed:04d}')
        else:
            output_dir = output_dir / f'random_{seed:04d}'
       
        if self.save_intermediate:
            (output_dir / "T_normal_F").mkdir(parents=True, exist_ok=True)
            (output_dir / "T_normal_F_face").mkdir(parents=True, exist_ok=True)
            (output_dir / "normal_F").mkdir(parents=True, exist_ok=True)
            (output_dir / "nds_dual").mkdir(parents=True, exist_ok=True)
            (output_dir / "normal_F_re").mkdir(parents=True, exist_ok=True)
            (output_dir / "normal_F_re_face").mkdir(parents=True, exist_ok=True)
            (output_dir / "nds_refine").mkdir(parents=True, exist_ok=True)
        
        # set initial mesh from smpl info for normal NDS
        if input_image is not None:  # gradio
            use_frankmocap = True
            smpl_info = self.run_frankmocap(input_image)
        else:
            use_frankmocap = False
            smpl_info = load_smpl_info(smpl_param_path, smpl_type=self.smpl_type, smpl_related_dir=self.smpl_related_dir)
        self.normal_nds.set_initial_mesh(smpl_info=smpl_info, frankmocap=use_frankmocap)

        # prepare SMPL-X normal maps for normal NDS
        view_angles = np.arange(-90 + self.initial_angle, 90 + self.initial_angle, self.angle_step) % 360
        angle_to_idx = dict(zip(view_angles, range(len(view_angles))))

        smpl_body_images = self.normal_nds.render_smpl(view_angles)
        smpl_body_images = [to_cam(smpl_body_image, view_angle) for (smpl_body_image, view_angle) 
                                                                in zip(smpl_body_images, view_angles)]
        if self.save_intermediate:
            for view_angle, smpl_body_image in zip(view_angles, smpl_body_images):
                to_pil(smpl_body_image).save(output_dir / 'T_normal_F' / f'{view_angle:03d}.png')

        if use_closeup:
            smpl_face_images = self.normal_nds.render_smpl(view_angles, closeup=True)
            smpl_face_images = [to_cam(smpl_face_image, view_angle) for (smpl_face_image, view_angle) 
                                                                    in zip(smpl_face_images, view_angles)]
            if self.save_intermediate:
                for view_angle, smpl_body_image in zip(view_angles, smpl_body_images):
                    to_pil(smpl_body_image).save(output_dir / 'T_normal_F_face' / f'{view_angle:03d}.png') 

        # Dual normal map generation
        smpl_body_front = smpl_body_images[angle_to_idx[self.initial_angle]]
        with torch.no_grad():
            if use_text:
                text_resample_steps = int(self.model_body.model.num_timesteps * self.resample_T_from_text)
                initial_normal = self.difftext_sample(smpl_body_front, prompt, negative_prompt, 
                                    seed, steps=steps, cfg_scale=cfg_scale, image_cfg_scale=image_cfg_scale,
                                    resample_steps=text_resample_steps, n_resample=self.n_resample_from_text)[0]
            else:
                initial_normal = self.diff_sample(smpl_body_front, seed=seed, steps=steps)[0]

        initial_normal_front, initial_normal_back = self.diff_to_nds_input(initial_normal, closeup=False)
        body_dual_normal_maps = {
                            self.initial_angle: initial_normal_front,
                            (self.initial_angle + 180) % 360: initial_normal_back
                           }

        if self.save_intermediate:
            for angle, normal_map in body_dual_normal_maps.items():
                to_pil(normal_map).save(output_dir / 'normal_F' / f'{angle:03d}.png')

        # NDS with dual normal map
        nds_mesh_dual = self.nds_dual(body_dual_normal_maps, output_dir)
        output_mesh_path = output_dir / f'mesh_dual{self.mesh_ext}'
        write_mesh(output_mesh_path, nds_mesh_dual, flip=self.gradio)

        # Resampling
        if use_resample:
            body_resample_steps = int(self.model_body.model.num_timesteps * resample_T)
            face_resample_steps = int(self.model_face.model.num_timesteps * resample_T_face)
            body_rendered_normal_maps = self.render_dual(nds_mesh_dual, view_angles,
                                                             closeup=False, ch_slice=self.ch_slice_body)
            if use_closeup:
                face_rendered_normal_maps = self.render_dual(nds_mesh_dual, view_angles, 
                                                                 closeup=True, ch_slice=self.ch_slice_face)
            body_refine_normal_maps = {}
            face_refine_normal_maps = {}
            for view_idx, view_angle in enumerate(view_angles):
                smpl_body = smpl_body_images[view_idx]
                body_normal_rendered = body_rendered_normal_maps[view_idx]
                # body_normal_rendered = self.render_dual(nds_mesh_dual, view_angle, closeup=False)
                with torch.no_grad():
                    body_normal_resample = self.model_body.resample(body_normal_rendered[None], 
                                                                    smpl_body[:self.ch_slice_body], 
                                                                    seed=seed, cfg=cfg_scale, 
                                                                    steps=body_resample_steps, 
                                                                    repeat=n_resample)[0]

                body_normal_resample_front, body_normal_resample_back = \
                                                self.diff_to_nds_input(body_normal_resample, closeup=False)
                body_refine_normal_maps.update({view_angle: body_normal_resample_front, 
                                               (view_angle + 180) % 360: body_normal_resample_back})

                if use_closeup:
                    smpl_face = smpl_face_images[view_idx]
                    face_normal_rendered = face_rendered_normal_maps[view_idx]
                    # face_normal_rendered = self.render_dual(nds_mesh_dual, view_angle, closeup=True)
                    with torch.no_grad():
                        face_normal_resample = self.model_face.resample(face_normal_rendered[None], 
                                                                        smpl_face[:self.ch_slice_face], 
                                                                        seed=seed, cfg=cfg_scale, 
                                                                        steps=face_resample_steps, 
                                                                        repeat=n_resample_face)[0]

                    face_normal_resample_front, face_normal_resample_back = \
                                                self.diff_to_nds_input(face_normal_resample, closeup=True)
                    face_refine_normal_maps.update({view_angle: face_normal_resample_front, 
                                                (view_angle + 180) % 360: face_normal_resample_back})

            if self.save_intermediate:
                for angle, normal_map in body_refine_normal_maps.items():
                    to_pil(normal_map).save(output_dir / 'normal_F_re' / f'{angle:03d}.png')

                if use_closeup:
                    for angle, normal_map in face_refine_normal_maps.items():
                        to_pil(normal_map).save(output_dir / 'normal_F_re_face' / f'{angle:03d}.png')

            # NDS with refined normal maps
            nds_mesh_refine = self.nds_refine(body_refine_normal_maps, 
                                              face_refine_normal_maps, 
                                              use_closeup,
                                              output_dir)
            output_mesh_path = output_dir / f'mesh_refine{self.mesh_ext}'
            write_mesh(output_mesh_path, nds_mesh_refine, flip=self.gradio)

        return output_mesh_path

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = OmegaConf.load(config_file)
    config_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(config, config_cli)  # update config from command line

    data_dir = Path(args.dataset.data_dir) / args.dataset.dataset_type
    smpl_type = args.chupa.smpl_type
    if args.dataset.subject is None:
        subjects = [subject.stem for subject in (data_dir / smpl_type).glob('*.pkl')]
    else:
        subjects = [args.dataset.subject]

    chupa = Chupa(args)

    for subject in sorted(subjects):
        print(subject)
        smpl_param_path = data_dir / smpl_type / f'{subject}.pkl'
        chupa.forward(smpl_param_path=smpl_param_path, subject=subject)