from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import meshzoo
import numpy as np
from pathlib import Path
from pyremesh import remesh_botsch
import pymeshlab
import torch
import torch.nn as nn
from tqdm import tqdm

from .nds.core import (
    Mesh, Renderer, View, Camera
)
from .nds.losses import (
    mask_loss, normal_consistency_loss, laplacian_loss, shading_loss, normal_map_loss, side_loss, offset_map_loss
)
from .nds.modules import (
    SpaceNormalization, NeuralShader, ViewSampler
)
from .nds.utils import (
    AABB, read_views, read_mesh, write_mesh, visualize_mesh_as_overlay, visualize_views, visualize_masks, generate_mesh, mesh_generator_names
)

from torchvision import transforms
from PIL import Image
import pymeshlab

from smplx.body_models import SMPL, SMPLX
from smpl_related.utils import get_yaw_inverse_mat, load_smpl
import trimesh

class NormalNDS(nn.Module):
    def __init__(self, args, orthographic=True, device='cpu'):
        super(NormalNDS, self).__init__()
        self.smpl_mesh = None
        self.initial_mesh = None
        self.angle_step = args.angle_step
        self.up_axis = args.up_axis  # 1 corresponds to y-axis
        self.with_closeup = args.with_closeup
        self.align_yaw = args.align_yaw
        self.yaw_inverse_mat = None
        self.tpose = args.tpose
        self.scale = 100.0
        self.device = device
    
        # optimization
        self.start_iteration = args.start_iteration
        self.iterations = args.iterations
        self.optim_only_visible = args.optim_only_visible
        self.upsample_interval = args.upsample_interval
        self.upsample_iterations = list(range(args.upsample_start, args.iterations, args.upsample_interval))
        self.initial_num_vertex = args.initial_num_vertex
        self.lr_vertices = args.lr_vertices
        self.loss_weights = {
            "mask": args.loss.weight_mask,
            "normal": args.loss.weight_normal,
            "laplacian": args.loss.weight_laplacian,
            "shading": args.loss.weight_shading,
            "side": args.loss.weight_side
        }
        self.visualization_frequency = args.visualization_frequency
        self.visualization_views = args.visualization_views
        self.save_frequency = args.save_frequency

        self.aabb = None
        self.space_normalization = None

        self.orthographic = (args.camera == 'orthographic')
        self.renderer = Renderer(orthographic=self.orthographic, device=self.device)

        self.views = []
        self.view_sampler_args = ViewSampler.get_parameters(args)
        self.side_views = []

        self.keep_hand = False
        self.not_hand_mask = None

    def set_initial_mesh(self, mesh=None, smpl_info=None, smpl=True, tpose=False, frankmocap=False, refine=False):
        if mesh is not None and smpl:
            raise ValueError("mesh should be not given if you want to set smpl mesh as initial mesh!")

        if smpl:
            if frankmocap:
                self.yaw_inverse_mat = get_yaw_inverse_mat(smpl_info['global_orient'])
                smpl_verts = smpl_info['pred_vertices_smpl'] * self.scale
                joints = smpl_info['pred_body_joints_smpl'] * self.scale 
                if self.align_yaw:
                    smpl_verts = smpl_verts @ self.yaw_inverse_mat.T
                    joints = joints @ self.yaw_inverse_mat.T
                smpl_verts = smpl_verts 
                joints = joints
                smpl_mesh = trimesh.Trimesh(vertices=smpl_verts,
                                            faces=smpl_info['faces'],
                                            process=False,
                                            maintain_order=True)
            else:
                smpl_mesh, joints, self.yaw_inverse_mat = load_smpl(smpl_info, scale=self.scale, 
                                                                    align_yaw=self.align_yaw, tpose=self.tpose)

            self.smpl_mesh = Mesh(vertices=torch.tensor(smpl_mesh.vertices / self.scale).contiguous(), 
                                indices=torch.tensor(smpl_mesh.faces).contiguous(), device=self.device)

            smpl_vertices = smpl_mesh.vertices
            body_vmax = smpl_vertices.max(0)
            body_vmin = smpl_vertices.min(0)
            body_vmed = joints[0]  # ICON convention
            # body_vmed = np.median(smpl_vertices, 0)  # PIFu convention
            body_vmed[self.up_axis] = 0.5 * (body_vmax[self.up_axis] + body_vmin[self.up_axis])
            self.body_vmed = body_vmed
            self.body_scale = 180.0 / (body_vmax[self.up_axis] - body_vmin[self.up_axis])

            face_idxs = smpl_info['face_idxs']
            face_vertices = smpl_vertices[face_idxs]
            face_vmax = face_vertices.max(0)
            face_vmin = face_vertices.min(0)
            self.face_vmed = 0.5 * (face_vmax + face_vmin)
            self.face_scale = 150.0 / (face_vmax[self.up_axis] - face_vmin[self.up_axis])

            vertices = smpl_vertices
            if smpl_info['type'] == 'smplx':
                # To make it watertight, remove the eyeballs and zip the mouth
                eyeball_fid = smpl_info['eyeball_fid']
                fill_mouth_fid = smpl_info['fill_mouth_fid']
                smpl_faces_not_watertight = smpl_mesh.faces
                smpl_faces_no_eybeballs = smpl_faces_not_watertight[~eyeball_fid]
                faces = np.concatenate([smpl_faces_no_eybeballs, fill_mouth_fid], axis=0)
            else:
                faces = smpl_mesh.faces
        else:
            if mesh is None:
                raise ValueError("initial mesh was not given!")
            vertices = mesh.vertices * self.scale
            faces = mesh.faces

            body_vmax = vertices.max(0)
            body_vmin = vertices.min(0)
            body_vmed = np.median(vertices, 0)  # PIFu convention
            body_vmed[self.up_axis] = 0.5 * (body_vmax[self.up_axis] + body_vmin[self.up_axis])
            self.body_vmed = body_vmed
            self.body_scale = 180.0 / (body_vmax[self.up_axis] - body_vmin[self.up_axis])

        if refine:
            # do not apply decimation
            self.initial_mesh = mesh.to(self.device)
        else:
            # decimate smpl mesh to 3000 vertices for mesh optimization
            ms = pymeshlab.MeshSet()
            smpl_mesh_ml = pymeshlab.Mesh(vertex_matrix=vertices,
                                        face_matrix=faces)
            ms.add_mesh(smpl_mesh_ml)
            ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=self.initial_num_vertex, 
                                    preserveboundary=True, preservenormal=True, preservetopology=True)
            initial_mesh = ms.current_mesh()
            initial_mesh = Mesh(vertices=torch.tensor(initial_mesh.vertex_matrix()).contiguous(), 
                                indices=torch.tensor(initial_mesh.face_matrix()).contiguous(), device=self.device)
            initial_mesh.compute_connectivity()

        # configure the space normalization
        aabb = AABB(initial_mesh.vertices.cpu().numpy())  # create bbox from the mesh vertices
        self.space_normalization = SpaceNormalization(aabb.corners)

        # Apply the normalizing affine transform, which maps the bounding box to 
        # a 2-cube centered at (0, 0, 0), to the views, the mesh, and the bounding box
        self.initial_mesh = self.space_normalization.normalize_mesh(initial_mesh)
        self.aabb = self.space_normalization.normalize_aabb(aabb)

    def set_views(self, view_angles, with_closeup=False):
        views = []
        # generate view objects
        body_image_paths = sorted([(self.body_input_dir / f'{view_angle:03d}.png') for view_angle in view_angles])
        print(f'Found {len(body_image_paths)} views')
        for body_image in body_image_paths:
            views.append(View.load_icon(str(body_image), scale=self.body_scale, center=self.body_vmed, 
                              view_angle=(body_image.stem), device=self.device))
        
        if with_closeup:
            face_image_paths = sorted([(self.face_input_dir / f'{view_angle:03d}.png') for view_angle in view_angles])
            for face_image in face_image_paths:
                if face_image.exists():
                    views.append(View.load_icon(str(face_image), scale=self.face_scale, center=self.face_vmed, 
                                    view_angle=(face_image.stem), device=self.device))
        self.views = self.space_normalization.normalize_views(views)
        self.renderer.set_near_far(self.views, torch.from_numpy(self.aabb.corners).to(self.device), epsilon=0.5)

    def reset_views(self):
        self.views = []

    def set_views_from_normal_maps(self, body_normal_maps:dict, face_normal_maps:dict=None, use_closeup=False):
        """
            body_normal_maps: dict with view angles as keys and body normal maps as values
            face_normal_maps: dict with view angles as keys and face normal maps as values
        """
        # generate view objects
        print(f'Found {len(body_normal_maps.keys())} views')
        views = []
        for view_angle, body_normal_map_cam in body_normal_maps.items():
            body_normal_map_world = View.to_world(body_normal_map_cam, view_angle)
            body_normal_map = 0.5 * (torch.clamp(body_normal_map_world.permute(1, 2, 0), -1, 1) + 1)  # [C, H, W] => [H, W, C]
            camera = Camera.camera_with_angle(scale=self.body_scale, center=self.body_vmed,
                                              view_angle=view_angle, 
                                              device=self.device)
            views.append(View(color=body_normal_map[:, :, :3], mask=body_normal_map[:, :, 3:],
                                   camera=camera, view_angle=view_angle, device=self.device))
        
        if use_closeup and face_normal_maps is not None:
            for view_angle, face_normal_map_cam in face_normal_maps.items():
                face_normal_map_world = View.to_world(face_normal_map_cam, view_angle)
                face_normal_map = 0.5 * (torch.clamp(face_normal_map_world.permute(1, 2, 0), -1, 1) + 1)  # [C, H, W] => [H, W, C]
                camera = Camera.camera_with_angle(scale=self.face_scale, center=self.face_vmed,
                                                view_angle=view_angle, 
                                                device=self.device)
                views.append(View(color=face_normal_map[:, :, :3], mask=face_normal_map[:, :, 3:],
                                    camera=camera, view_angle=view_angle, device=self.device))
        self.views = self.space_normalization.normalize_views(views)
        self.renderer.set_near_far(self.views, torch.from_numpy(self.aabb.corners).to(self.device), epsilon=0.5)

    def set_views_from_mesh(self, mesh_path, view_angles, use_closeup=False):
        views = []
        for view_angle in view_angles:
            mesh_tri = trimesh.load(mesh_path, process=False, maintain_order=True)
            mesh = Mesh(vertices=torch.tensor(mesh_tri.vertices), 
                        indices=torch.tensor(mesh_tri.faces), device=self.device)

            normal_map = (self.render_target_view(mesh, view_angle, return_vis_mask=False) + 1) * 0.5
            camera = Camera.camera_with_angle(scale=self.body_scale, center=self.body_vmed,
                                              view_angle=view_angle, 
                                              device=self.device)
            views.append(View(color=normal_map[:, :, :3], mask=normal_map[:, :, 3:],
                                   camera=camera, view_angle=view_angle, device=self.device))
        self.views = self.space_normalization.normalize_views(views)
        self.renderer.set_near_far(self.views, torch.from_numpy(self.aabb.corners).to(self.device), epsilon=0.5)

    def render_smpl(self, view_angles, closeup=False):
        smpl_normal_maps = []
        for view_angle in view_angles:
            normal_map = self.render_target_view(self.smpl_mesh, view_angle, return_vis_mask=False, closeup=closeup)
            smpl_normal_maps.append(normal_map.permute(2, 0, 1))  # [H, W, C] => [C, H, W]
        
        return smpl_normal_maps

    def set_side_views(self, side_view_angles):
        side_views = []
        side_smpl_normal_maps = self.render_smpl(side_view_angles)
        for view_angle, body_normal_map in zip(side_view_angles, side_smpl_normal_maps):
            camera = Camera.camera_with_angle(scale=self.body_scale, center=self.body_vmed,
                                            view_angle=view_angle, 
                                            device=self.device)
            body_normal_map = body_normal_map.permute(1, 2, 0)  # [C, H, W] => [H, W, C]
            side_views.append(View(color=body_normal_map[:, :, :3], mask=body_normal_map[:, :, 3:],
                                camera=camera, view_angle=view_angle, device=self.device))
        if len(side_views) != 0:
            self.side_views = self.space_normalization.normalize_views(side_views)

    def reset_side_views(self):
        self.side_views = []

    def render_target_view(self, mesh, target_view_angle, return_vis_mask=True, closeup=False, h=512, w=512):
        mesh_copy = Mesh(vertices=mesh.vertices.clone(), indices=mesh.indices, device=self.device)
        mesh_copy.vertices *= self.scale

        # # Apply the normalizing affine transform, which maps the bounding box to 
        # # a 2-cube centered at (0, 0, 0), to the views, the mesh, and the bounding box
        mesh_copy = self.space_normalization.normalize_mesh(mesh_copy)

        # generate next view image with visible vertices
        if closeup:
            scale = self.face_scale
            center = self.face_vmed
        else:
            scale = self.body_scale
            center = self.body_vmed

        target_view_camera = Camera.camera_with_angle(scale=scale, center=center, 
                                                      view_angle=target_view_angle, 
                                                      orthographic=self.orthographic, device=self.device)
        target_view_R = target_view_camera.R
        target_view_camera.normalize_camera(self.space_normalization, device=self.device)
        resolution = (h, w)
        target_gbuffer = self.renderer.render_with_camera(target_view_camera, mesh_copy, resolution, 
                                                          vis_mask=None, with_antialiasing=True)
        mask = target_gbuffer["mask"]
        position = target_gbuffer["position"]
        target_view_normal = target_gbuffer["normal"]

        target_view_normal = target_view_normal * mask - (1 - mask)  # set normals of masked region to [-1, -1, -1]
        target_view_normal_with_alpha = torch.concat([target_view_normal, 2 * mask - 1], dim=-1)

        if return_vis_mask:
            # find faces visible from image_views
            vis_mask = self.renderer.get_face_visibility(self.views, mesh_copy)
            
            pix_to_face = target_gbuffer["pix_to_face"].long()
            out_of_fov = (pix_to_face == 0)
            target_view_mask = vis_mask[pix_to_face - 1].float() 
            target_view_mask[out_of_fov] = 0 

            return target_view_normal_with_alpha, target_view_mask
        
        return target_view_normal_with_alpha

    def forward(self, output_dir):
        num_views = len(self.views)
        loss_weights = self.loss_weights.copy()
        lr_vertices = self.lr_vertices
        if self.start_iteration!= 1:
            for i in range(self.start_iteration // self.upsample_interval):
                loss_weights['laplacian'] *= 4
                loss_weights['normal'] *= 4
                lr_vertices *= 0.25
                loss_weights['side'] *= 0.25

        if (self.visualization_frequency > 0):
            save_normal_path = output_dir / 'normals'
            save_normal_path.mkdir(exist_ok=True)
        if (self.save_frequency > 0):
            save_mesh_path = output_dir / 'meshes'
            save_mesh_path.mkdir(exist_ok=True)

        # Configure the view sampler
        view_sampler = ViewSampler(views=self.views, **self.view_sampler_args)

        # Create the optimizer for the vertex positions 
        # (we optimize offsets from the initial vertex position)
        vertex_offsets = nn.Parameter(torch.zeros_like(self.initial_mesh.vertices))

        if not self.optim_only_visible:
            optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)

        # Initialize the loss weights and losses
        losses = {k: torch.tensor(0.0, device=self.device) for k in loss_weights}

        initial_mesh = self.initial_mesh
        progress_bar = tqdm(range(self.start_iteration, self.iterations))
        for iteration in progress_bar:
            progress_bar.set_description(desc=f'Iteration {iteration}')
            
            if iteration in self.upsample_iterations and (mesh.indices.shape[0] * 4) <= 250000:
                # Upsample the mesh by remeshing the surface with half the average edge length
                
                #NDS default : pyremesh
                e0, e1 = mesh.edges.unbind(1)
                average_edge_length = torch.linalg.norm(mesh.vertices[e0] - mesh.vertices[e1], dim=-1).mean()
                
                v_upsampled, f_upsampled = remesh_botsch(mesh.vertices.cpu().detach().numpy().astype(np.float64), mesh.indices.cpu().numpy().astype(np.int32), h=float(average_edge_length/2))
                v_upsampled = np.ascontiguousarray(v_upsampled)
                f_upsampled = np.ascontiguousarray(f_upsampled)

                initial_mesh = Mesh(v_upsampled, f_upsampled, device=self.device)
                initial_mesh.compute_connectivity()

                # Adjust weights and step size
                loss_weights['laplacian'] *= 4
                loss_weights['normal'] *= 4
                lr_vertices *= 0.25
                loss_weights['side'] *= 0.25

                # Create a new optimizer for the vertex offsets
                vertex_offsets = nn.Parameter(torch.zeros_like(initial_mesh.vertices))
                if not self.optim_only_visible:
                    optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)

                if (self.save_frequency > 0) and ((iteration == 0) or ((iteration + 1) % self.save_frequency == 0)):
                    with torch.no_grad():
                        mesh_for_writing = self.space_normalization.denormalize_mesh(initial_mesh.detach().to('cpu'))
                        if self.yaw_inverse_mat is not None:
                            mesh_for_writing.vertices = mesh_for_writing.vertices @ self.yaw_inverse_mat 
                        mesh_for_writing.vertices /= self.scale
                        write_mesh(save_mesh_path / f"{num_views}views_{iteration:04d}_upsample.obj", mesh_for_writing)

            # Deform the initial mesh
            mesh = initial_mesh.with_vertices(initial_mesh.vertices + vertex_offsets)

            # Sample a view subset
            views_subset = view_sampler(self.views)

            # find vertices visible from image views
            if self.optim_only_visible:
                vis_mask = self.renderer.get_vert_visibility(views_subset, mesh)
                if self.not_hand_mask is not None and iteration < self.upsample_interval:
                    vis_mask *= torch.tensor(self.not_hand_mask, device=self.device)
                target_vertices = nn.Parameter(vertex_offsets[vis_mask].clone())
                detach_vertices = vertex_offsets[~vis_mask].detach()
                optimizer_vertices = torch.optim.Adam([target_vertices], lr=lr_vertices)
                
                mesh_vertices = initial_mesh.vertices.detach().clone()
                mesh_vertices[vis_mask] += target_vertices  
                mesh_vertices[~vis_mask] += detach_vertices
                mesh = initial_mesh.with_vertices(mesh_vertices)

            # Render the mesh from the views
            # Perform antialiasing here because we cannot antialias after shading if we only shade a some of the pixels
            gbuffers = self.renderer.render(views_subset, mesh, channels=['mask', 'normal'], with_antialiasing=True) 

            # Combine losses and weights
            if loss_weights['mask'] > 0:
                losses['mask'] = mask_loss(views_subset, gbuffers)
            if loss_weights['normal'] > 0:
                losses['normal'] = normal_consistency_loss(mesh)
            if loss_weights['laplacian'] > 0:
                losses['laplacian'] = laplacian_loss(mesh)
            if loss_weights['shading'] > 0:
                losses['shading'] = normal_map_loss(views_subset, gbuffers)

            if len(self.side_views) != 0:
                side_gbuffers = self.renderer.render(self.side_views, mesh, channels=['mask'], with_antialiasing=True) 
                losses['side'] = side_loss(self.side_views, side_gbuffers)

            loss = torch.tensor(0., device=self.device)
            for k, v in losses.items():
                loss += v * loss_weights[k]

            # Optimize
            optimizer_vertices.zero_grad()
            loss.backward()
            optimizer_vertices.step()

            if self.optim_only_visible:
                vertex_offsets = torch.zeros_like(vertex_offsets)
                vertex_offsets[vis_mask] = target_vertices
                vertex_offsets[~vis_mask] = detach_vertices

            progress_bar.set_postfix({'loss': loss.detach().cpu()})

            # Visualizations
            if (self.visualization_frequency > 0) and ((iteration == 0) or ((iteration + 1) % self.visualization_frequency == 0)):
                import matplotlib.pyplot as plt
                with torch.no_grad():
                    use_fixed_views = len(self.visualization_views) > 0
                    view_indices = self.visualization_views if use_fixed_views else [np.random.choice(list(range(len(views_subset))))]
                    for vi in view_indices:
                        debug_view = self.views[vi] if use_fixed_views else views_subset[vi]
                        debug_gbuffer = self.renderer.render([debug_view], mesh, channels=['mask', 'position', 'normal'], with_antialiasing=True)[0]
                        position = debug_gbuffer["position"]
                        normal = debug_gbuffer["normal"]
                        view_direction = torch.nn.functional.normalize(debug_view.camera.center - position, dim=-1)

                        # Save a normal map in camera space
                        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=self.device, dtype=torch.float32)
                        # normal_image = (0.5*(normal @ debug_view.camera.R.T @ R.T + 1)) * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])  # camera-view normal
                        normal_image = (0.5*(normal + 1)) * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])  # global normal
                        plt.imsave(save_normal_path / f'{num_views}views_{(iteration + 1):04d}_{debug_view.view_angle}.png', normal_image.cpu().numpy())

            if (self.save_frequency > 0) and ((iteration == 0) or ((iteration + 1) % self.save_frequency == 0)):
                with torch.no_grad():
                    mesh_for_writing = self.space_normalization.denormalize_mesh(mesh.detach().to('cpu'))
                    if self.yaw_inverse_mat is not None:
                        mesh_for_writing.vertices = mesh_for_writing.vertices @ self.yaw_inverse_mat 
                    mesh_for_writing.vertices /= self.scale
                    write_mesh(save_mesh_path / f"{num_views}views_{(iteration + 1):04d}.obj", mesh_for_writing)
        
        nds_result = self.space_normalization.denormalize_mesh(mesh.detach().to('cpu'))  
        if self.yaw_inverse_mat is not None:
            nds_result.vertices = nds_result.vertices @ self.yaw_inverse_mat 
        nds_result.vertices /= self.scale
        return nds_result