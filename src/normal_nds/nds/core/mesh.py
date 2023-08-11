import torch

class Mesh:
    """ Triangle mesh defined by an indexed vertex buffer.

    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        device (torch.device): Device where the mesh buffers are stored
    """

    def __init__(self, vertices, indices, device='cpu'):
        self.device = device

        self.vertices = vertices.to(device, dtype=torch.float32) if torch.is_tensor(vertices) else torch.tensor(vertices, dtype=torch.float32, device=device)
        self.indices = indices.to(device, dtype=torch.int64) if torch.is_tensor(indices) else torch.tensor(indices, dtype=torch.int64, device=device) if indices is not None else None

        if self.indices is not None:
            self.compute_normals()

        self._edges = None
        self._connected_faces = None
        self._laplacian = None

    def to(self, device):
        mesh = Mesh(self.vertices.to(device), self.indices.to(device), device=device)
        mesh._edges = self._edges.to(device) if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.to(device) if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.to(device) if self._laplacian is not None else None
        return mesh

    def detach(self):
        mesh = Mesh(self.vertices.detach(), self.indices.detach(), device=self.device)
        mesh.face_normals = self.face_normals.detach()
        mesh.vertex_normals = self.vertex_normals.detach()
        mesh._edges = self._edges.detach() if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.detach() if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        return mesh

    def with_vertices(self, vertices):
        """ Create a mesh with the same connectivity but with different vertex positions

        Args:
            vertices (tensor): New vertex positions (Vx3)
        """

        assert len(vertices) == len(self.vertices)

        mesh_new = Mesh(vertices, self.indices, self.device)
        mesh_new._edges = self._edges
        mesh_new._connected_faces = self._connected_faces
        mesh_new._laplacian = self._laplacian
        return mesh_new

    @property
    def edges(self):
        if self._edges is None:
            from normal_nds.nds.utils.geometry import find_edges
            self._edges = find_edges(self.indices)
        return self._edges

    @property
    def connected_faces(self):
        if self._connected_faces is None:
            from normal_nds.nds.utils.geometry import find_connected_faces
            self._connected_faces = find_connected_faces(self.indices)
        return self._connected_faces

    @property
    def laplacian(self):
        if self._laplacian is None:
            from normal_nds.nds.utils.geometry import compute_laplacian_uniform
            self._laplacian = compute_laplacian_uniform(self)
        return self._laplacian

    def compute_connectivity(self):
        self._edges = self.edges
        self._connected_faces = self.connected_faces
        self._laplacian = self.laplacian

    def compute_normals(self):
        # Compute the face normals
        a = self.vertices[self.indices][:, 0, :]
        b = self.vertices[self.indices][:, 1, :]
        c = self.vertices[self.indices][:, 2, :]
        self.face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1) 

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(self.vertices)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 0], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 1], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 2], self.face_normals)
        self.vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1) 

    # @profile
    def connected_faces_with_mask(self, mask):
        masked_vertex_indices = mask.nonzero(as_tuple=True)[0]
        num_faces = self.indices.shape[0]
        is_vertex_masked = torch.isin(self.indices.reshape(-1), masked_vertex_indices).reshape([num_faces, -1])
        self.faces_with_mask = self.indices[is_vertex_masked.prod(axis=1).nonzero(as_tuple=True)[0]]
        from nds.utils.geometry import find_connected_faces
        connected_faces_with_mask_ = find_connected_faces(self.faces_with_mask)
        return connected_faces_with_mask_

    def compute_normals_with_mask(self):
        # Compute the face normals
        vertices = self.vertices[self.faces_with_mask]
        a = vertices[:, 0, :]
        b = vertices[:, 1, :]
        c = vertices[:, 2, :]
        return torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1) 
