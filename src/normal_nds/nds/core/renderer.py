import numpy as np
import nvdiffrast.torch as dr
import torch


class Renderer:
    """ Rasterization-based triangle mesh renderer that produces G-buffers for a set of views.

    Args:
        device (torch.device): Device used for rendering (must be a GPU)
        near (float): Near plane distance
        far (float): Far plane distance
    """

    def __init__(self, device, near=1, far=1000, orthographic=False):
        self.glctx = dr.RasterizeGLContext()
        self.device = device
        self.near = near
        self.far = far
        self.orthographic = orthographic

    def set_near_far(self, views, samples, epsilon=0.1):
        """ Automatically adjust the near and far plane distance
        """

        mins = []
        maxs = []
        for view in views:
            samples_projected = view.project(samples, depth_as_distance=True)
            mins.append(samples_projected[...,2].min())
            maxs.append(samples_projected[...,2].max())

        near, far = min(mins), max(maxs)
        self.near = near - (near * epsilon)
        self.far = far + (far * epsilon)

    @staticmethod
    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[:, 0:1])], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]

    @staticmethod
    def projection(fx, fy, cx, cy, n, f, width, height, device):
        """
        Returns a gl projection matrix
        The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up.
        Note that cy has been inverted 1 - cy!
        """
        return torch.tensor([[2.0*fx/width,           0,       1.0 - 2.0 * cx / width,                  0],
                            [         0, 2.0*fy/height,      1.0 - 2.0 * cy / height,                  0],
                            [         0,             0,                 -(f+n)/(f-n),     -(2*f*n)/(f-n)],
                            [         0,             0,                           -1,                  0.0]], device=device) 
    @staticmethod
    def to_gl_camera(camera, resolution, n=1000, f=5000, orthographic=False):
        if orthographic:
            projection_matrix = torch.eye(4, device=camera.device)
            projection_matrix[:3, :3] = camera.K
            gl_transform = torch.tensor([[1., 0,  0,  0],
                                        [0,  -1., 0,  0],
                                        [0,  0, -1., 0],
                                        [0,  0,  0,  1.]], device=camera.device)
        else:
            projection_matrix = Renderer.projection(fx=camera.K[0,0],
                                                    fy=camera.K[1,1],
                                                    cx=camera.K[0,2],
                                                    cy=camera.K[1,2],
                                                    n=n,
                                                    f=f,
                                                    width=resolution[1],
                                                    height=resolution[0],
                                                    device=camera.device)
            gl_transform = torch.tensor([[1., 0,  0,  0],
                                        [0,  1., 0,  0],
                                        [0,  0, -1., 0],
                                        [0,  0,  0,  1.]], device=camera.device)

        Rt = torch.eye(4, device=camera.device)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t

        Rt = gl_transform @ Rt
        return projection_matrix @ Rt

    def get_gl_camera(self, camera, resolution):
        P = self.to_gl_camera(camera, resolution, n=self.near, f=self.far, orthographic=self.orthographic)
        return P

    def render(self, views, mesh, channels, with_antialiasing=True):
        """ Render G-buffers from a set of views.

        Args:
            views (List[Views]): 
        """

        # TODO near far should be passed by view to get higher resolution in depth
        gbuffers = []
        for i, view in enumerate(views):
            gbuffer = {}

            # Rasterize only once
            P = self.to_gl_camera(view.camera, view.resolution, n=self.near, f=self.far, orthographic=self.orthographic)
            pos = self.transform_pos(P, mesh.vertices)
            idx = mesh.indices.int()
            rast, rast_out_db = dr.rasterize(self.glctx, pos, idx, resolution=view.resolution)

            # Collect arbitrary output variables (aovs)
            if "mask" in channels:
                mask = torch.clamp(rast[..., -1:], 0, 1)
                gbuffer["mask"] = dr.antialias(mask, rast, pos, idx)[0] if with_antialiasing else mask[0]

            if "position" in channels or "depth" in channels:
                position, _ = dr.interpolate(mesh.vertices[None, ...], rast, idx)
                gbuffer["position"] = dr.antialias(position, rast, pos, idx)[0] if with_antialiasing else position[0]

            if "normal" in channels:
                normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, idx)
                gbuffer["normal"] = dr.antialias(normal, rast, pos, idx)[0] if with_antialiasing else normal[0]
                # debug
                # Image.fromarray(((gbuffer['normal'] + 1) / 2 * 255).cpu().detach().numpy().astype(np.uint8)).show()

            if "depth" in channels:
                gbuffer["depth"] = view.project(gbuffer["position"], depth_as_distance=True)[..., 2:3]

            if "offset" in channels:
                initial_pos = self.transform_pos(P, mesh.initial_vertices)
                initial_rast, _ = dr.rasterize(self.glctx, initial_pos, idx, resolution=view.resolution)
                offset, _ = dr.interpolate((mesh.vertices - mesh.initial_vertices)[None, ...], initial_rast, idx)
                gbuffer["offset"] = dr.antialias(offset, initial_rast, initial_pos, idx)[0] if with_antialiasing else offset[0] 

            if "initial_mask" in channels:
                initial_mask = torch.clamp(initial_rast[..., -1:], 0, 1)
                gbuffer["initial_mask"] = dr.antialias(initial_mask, initial_rast, initial_pos, idx)[0] if with_antialiasing else initial_mask[0]

            gbuffers += [gbuffer]

        return gbuffers

    def render_with_camera(self, camera, mesh, resolution, vis_mask=None, with_antialiasing=False):

        gbuffer = {}

        # Rasterize only once
        P = self.to_gl_camera(camera, resolution, n=self.near, f=self.far, orthographic=self.orthographic)
        if vis_mask is not None:
            vertices = mesh.vertices[vis_mask]
        else:
            vertices = mesh.vertices
        pos = self.transform_pos(P, vertices)
        idx = mesh.indices.int()
        rast, rast_out_db = dr.rasterize(self.glctx, pos, idx, resolution=resolution)

        pix_to_face = rast[..., -1:]
        gbuffer["pix_to_face"] = pix_to_face[0]

        mask = torch.clamp(rast[..., -1:], 0, 1)
        gbuffer["mask"] = dr.antialias(mask, rast, pos, idx)[0] if with_antialiasing else mask[0]

        position, _ = dr.interpolate(vertices[None, ...], rast, idx)
        gbuffer["position"] = dr.antialias(position, rast, pos, idx)[0] if with_antialiasing else position[0]

        normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, idx)
        gbuffer["normal"] = dr.antialias(normal, rast, pos, idx)[0] if with_antialiasing else normal[0]

        return gbuffer

    def render_attributes_with_camera(self, camera, mesh, attribute_list, resolution, vis_mask=None, with_antialiasing=False):

        gbuffer = {}

        # Rasterize only once
        P = self.to_gl_camera(camera, resolution, n=self.near, f=self.far, orthographic=self.orthographic)
        if vis_mask is not None:
            vertices = mesh.vertices[vis_mask]
        else:
            vertices = mesh.vertices
        pos = self.transform_pos(P, vertices)
        idx = mesh.indices.int()
        rast, rast_out_db = dr.rasterize(self.glctx, pos, idx, resolution=resolution)

        mask = torch.clamp(rast[..., -1:], 0, 1)
        gbuffer["mask"] = dr.antialias(mask, rast, pos, idx)[0] if with_antialiasing else mask[0]

        for attribute in attribute_list:
            if vis_mask is not None:
                mesh_attribute = mesh.getattr(attribute)[vis_mask]
            else:
                mesh_attribute = getattr(mesh, attribute)
            attribute_render, _ = dr.interpolate(mesh_attribute[None, ...], rast, idx)
            gbuffer[attribute] = dr.antialias(attribute_render, rast, pos, idx)[0] if with_antialiasing else attribute_render[0]

        return gbuffer

    def get_vert_visibility(self, views, mesh, upsample=8):
        vertices = mesh.vertices
        idx = mesh.indices.int()
        num_verts = len(vertices)
        vis_mask = torch.zeros(size=(num_verts,), device=self.device).bool() # visibility mask

        with torch.no_grad():
            masked_face_idxs_list = []
            for i, view in enumerate(views):
                # Rasterize only once
                P = Renderer.to_gl_camera(view.camera, view.resolution, n=self.near, f=self.far, orthographic=self.orthographic)
                pos = Renderer.transform_pos(P, vertices)                
                rast, rast_out_db = dr.rasterize(self.glctx, pos, idx, resolution=np.array(view.resolution) * upsample)

                # Do not support batch operation yet
                face_ids = rast[..., -1].long()
                masked_face_idxs_list.append(face_ids[face_ids != 0] - 1)  # num_masked_face Tensor
            masked_face_idxs_all = torch.unique(torch.cat(masked_face_idxs_list, dim=0))
            masked_verts_idxs = torch.unique(idx[masked_face_idxs_all].long())
            vis_mask[masked_verts_idxs] = 1
            vis_mask = vis_mask.bool().to(self.device)

        return vis_mask

    def get_face_visibility(self, views, mesh, upsample=8):   
        num_faces = len(mesh.indices)
        vis_mask_all = torch.zeros(size=(num_faces,), device=self.device).bool() # visibility mask

        with torch.no_grad():
            for i, view in enumerate(views):
                # Rasterize only once
                P = Renderer.to_gl_camera(view.camera, view.resolution, n=self.near, f=self.far, orthographic=self.orthographic)
                pos = Renderer.transform_pos(P, mesh.vertices)
                idx = mesh.indices.int()
                rast, rast_out_db = dr.rasterize(self.glctx, pos, idx, resolution=np.array(view.resolution) * upsample)

                # Do not support batch operation yet
                face_ids = rast[..., -1]
                masked_face_idxs = torch.unique(face_ids[face_ids != 0].long() - 1)  # num_masked_face Tensor
                vis_mask = torch.zeros(size=(num_faces,), device=self.device) 
                vis_mask[masked_face_idxs] = 1
                vis_mask_all += vis_mask.bool()
            vis_mask_all = vis_mask_all.to(self.device)

        return vis_mask_all
