import numpy as np
import torch
import torch.nn as nn

from backprojection import *

"""
def compute_difference_vector_2(xyz):
    """"""compute difference vector using 2 neighbours (righ; bottom)
    surface normal map border (right & bottom) is set to 0
    Args:
        xyz (Nx3xHxW): 3d point cloud
    Returns:
        diff_vec_1 (Nx3xHxW) horizontal difference vector
        diff_vec_2 (Nx3xHxW) vectical difference vector
    """"""
    diff_vec_1 = xyz.clone().detach() * 0 # horizontal
    diff_vec_2 = xyz.clone().detach() * 0 # vectical
    diff_vec_1[:,:,:,:-1] = xyz[:,:,:,1:] - xyz[:,:,:,:-1]
    diff_vec_2[:,:,:-1,:] = xyz[:,:,1:,:] - xyz[:,:,:-1,:]
    return diff_vec_1, diff_vec_2


def compute_difference_vectors_8(xyz):
    """"""compute difference vectors using neighbouring 8 points. 
    Outermost pixels in the image are ignored
    Args:
        xyz: Nx3xHxW
    Returns:
        list of difference vectors: each element is Nx3x(H-2)x(W-2)
        the order is as following:
            5 6 7
            4 * 0
            3 2 1
    """"""
    diff_vec_0 = xyz[:,:,1:-1,2:] - xyz[:,:,1:-1,1:-1]
    diff_vec_1 = xyz[:,:,2:,2:] - xyz[:,:,1:-1,1:-1]
    diff_vec_2 = xyz[:,:,2:,1:-1] - xyz[:,:,1:-1,1:-1]
    diff_vec_3 = xyz[:,:,2:,:-2] - xyz[:,:,1:-1,1:-1]
    diff_vec_4 = xyz[:,:,1:-1,:-2] - xyz[:,:,1:-1,1:-1]
    diff_vec_5 = xyz[:,:,:-2,:-2] - xyz[:,:,1:-1,1:-1]
    diff_vec_6 = xyz[:,:,:-2,1:-1] - xyz[:,:,1:-1,1:-1]
    diff_vec_7 = xyz[:,:,:-2,2:] - xyz[:,:,1:-1,1:-1]
    diff_vecs = [diff_vec_0, diff_vec_1, diff_vec_2, diff_vec_3,
                diff_vec_4, diff_vec_5, diff_vec_6, diff_vec_7]
    return diff_vecs
"""

def img_grad_weight(img_grad, alpha=10):
    """compute a weighting map based on image gradient
    Args:
        img_grad (Nx2xHxW): image gradient map (horizontal & vertical)
    Returns
        weight (Nx1xHxW): weighting map
    """
    return torch.exp(-alpha * img_grad.abs().mean(dim=1, keepdim=True))


def cross_product(a, b):
    """compute cross product of vec_1 and vec_2. i.e. vec_1 X vec_2
    Args:
        a (Nx3xHxW): vec_1
        b (Nx3xHxW): vec_2
    Returns:
        cross_prod (Nx3xHxW): cross product
    """
    s1 = a[:,1:2] * b[:,2:3] - b[:,1:2] * a[:,2:3]
    s2 = a[:,2:3] * b[:,0:1] - b[:,2:3] * a[:,0:1]
    s3 = a[:,0:1] * b[:,1:2] - b[:,0:1] * a[:,1:2]
    cross_prod = torch.cat([s1, s2, s3], dim=1)
    cross_prod[:,:,-1:] = cross_prod[:,:,-2:-1]
    cross_prod[:,:,:,-1:] = cross_prod[:,:,:,-2:-1]
    return cross_prod


class Depth2Normal(nn.Module):
    """Layer to compute surface normal from depth map
    """
    def __init__(self, height, width, num_neighbour=8):
        """
        Args:
            height (int): image height
            width (int): image width
            num_neighbour (int): number of neighbours for computing surface normal
                - 2: only right and bottom neighbouring pixels are used
                - 8: 8 neighbouring pixels are used with edge-aware weighting
        """
        super(Depth2Normal, self).__init__()

        self.height = height
        self.width = width
        self.num_neighbour = num_neighbour

        self.backproj = Backprojection(height, width)

    def forward(self, depth, inv_K, img=None):
        """
        Args:
            depth (Nx1xHxW): depth map
            inv_K (Nx4x4): inverse camera intrinsics
            img (NxCxHxW): if provided, image gradients are computed for edge-aware weighting
        Returns:
            normal (Nx3xHxW): normalized surface normal
        """
        # Compute 3D point cloud
        xyz = self.backproj(depth, inv_K)
        xyz = xyz.view(depth.shape[0], 4, self.height, self.width)
        xyz = xyz[:,:3]

        # Compute surface normal
        if self.num_neighbour == 2:
            diff_vec_x, diff_vec_y = compute_difference_vector_2(xyz)
            normal = cross_product(diff_vec_y, diff_vec_x)
        elif self.num_neighbour == 8:
            # Compute surface normals
            diff_vecs = compute_difference_vectors_8(xyz)
            normal_0 = cross_product(diff_vecs[2], diff_vecs[0])
            normal_1 = cross_product(diff_vecs[3], diff_vecs[1])
            normal_2 = cross_product(diff_vecs[6], diff_vecs[4])
            normal_3 = cross_product(diff_vecs[7], diff_vecs[5])
            norm_normal_0 = normal_0 / torch.norm(normal_0, p=2, dim=1, keepdim=True)
            norm_normal_1 = normal_1 / torch.norm(normal_1, p=2, dim=1, keepdim=True)
            norm_normal_2 = normal_2 / torch.norm(normal_2, p=2, dim=1, keepdim=True)
            norm_normal_3 = normal_3 / torch.norm(normal_3, p=2, dim=1, keepdim=True)
            
            if img is not None:
                # Compute weights for each normal
                img_grads = compute_difference_vectors_8(img)
                weight_0 = img_grad_weight(img_grads[2]) * img_grad_weight(img_grads[0])
                weight_1 = img_grad_weight(img_grads[3]) * img_grad_weight(img_grads[1])
                weight_2 = img_grad_weight(img_grads[6]) * img_grad_weight(img_grads[4])
                weight_3 = img_grad_weight(img_grads[7]) * img_grad_weight(img_grads[5])
            else:
                weight_0 = norm_normal_0.clone() * 0 + 1.
                weight_1 = norm_normal_0.clone() * 0 + 1.
                weight_2 = norm_normal_0.clone() * 0 + 1.
                weight_3 = norm_normal_0.clone() * 0 + 1.
            normal = weight_0 * norm_normal_0 + weight_1 * norm_normal_1 + weight_2 * norm_normal_2 + weight_3 * norm_normal_3

        # normalize to unit vector
        norm_normal = torch.norm(normal, p=2, dim=1, keepdim=True)
        normal = normal / norm_normal
        return normal
