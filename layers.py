# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords),
            requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
                       requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1).reshape(
            self.batch_size, 4, self.height, self.width)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        points = points.view(self.batch_size, 4, -1)
        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x, sf=2):
    """Upsample input tensor by a factor
    """
    return F.interpolate(x, scale_factor=sf, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class ScaleRecovery(nn.Module):
    """Layer to estimate scale through dense geometrical constrain
    """
    def __init__(self, batch_size, height, width):
        super(ScaleRecovery, self).__init__()
        self.backproject_depth = BackprojectDepth(batch_size, height, width)
        self.batch_size = batch_size
        self.height = height
        self.width = width

    # derived from https://github.com/zhenheny/LEGO
    def get_surface_normal(self, cam_points, nei=1):
        cam_points_ctr  = cam_points[:, :-1, nei:-nei, nei:-nei]
        cam_points_x0   = cam_points[:, :-1, nei:-nei, 0:-(2*nei)]
        cam_points_y0   = cam_points[:, :-1, 0:-(2*nei), nei:-nei]
        cam_points_x1   = cam_points[:, :-1, nei:-nei, 2*nei:]
        cam_points_y1   = cam_points[:, :-1, 2*nei:, nei:-nei]
        cam_points_x0y0 = cam_points[:, :-1, 0:-(2*nei), 0:-(2*nei)]
        cam_points_x0y1 = cam_points[:, :-1, 2*nei:, 0:-(2*nei)]
        cam_points_x1y0 = cam_points[:, :-1, 0:-(2*nei), 2*nei:]
        cam_points_x1y1 = cam_points[:, :-1, 2*nei:, 2*nei:]

        vector_x0   = cam_points_x0   - cam_points_ctr
        vector_y0   = cam_points_y0   - cam_points_ctr
        vector_x1   = cam_points_x1   - cam_points_ctr
        vector_y1   = cam_points_y1   - cam_points_ctr
        vector_x0y0 = cam_points_x0y0 - cam_points_ctr
        vector_x0y1 = cam_points_x0y1 - cam_points_ctr
        vector_x1y0 = cam_points_x1y0 - cam_points_ctr
        vector_x1y1 = cam_points_x1y1 - cam_points_ctr

        normal_0 = F.normalize(torch.cross(vector_x0,   vector_y0,   dim=1), dim=1).unsqueeze(0)
        normal_1 = F.normalize(torch.cross(vector_x1,   vector_y1,   dim=1), dim=1).unsqueeze(0)
        normal_2 = F.normalize(torch.cross(vector_x0y0, vector_x0y1, dim=1), dim=1).unsqueeze(0)
        normal_3 = F.normalize(torch.cross(vector_x1y0, vector_x1y1, dim=1), dim=1).unsqueeze(0)

        normals = torch.cat((normal_0, normal_1, normal_2, normal_3), dim=0).mean(0)
        normals = F.normalize(normals, dim=1)

        refl = nn.ReflectionPad2d(nei)
        normals = refl(normals)

        return normals

    def get_ground_mask(self, cam_points, normal_map, threshold=5):
        b, _, h, w = normal_map.size()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        threshold = math.cos(math.radians(threshold))
        ones, zeros = torch.ones(b, 1, h, w).cuda(), torch.zeros(b, 1, h, w).cuda()
        vertical = torch.cat((zeros, ones, zeros), dim=1)

        cosine_sim = cos(normal_map, vertical).unsqueeze(1)
        vertical_mask = (cosine_sim > threshold) | (cosine_sim < -threshold)

        y = cam_points[:,1,:,:].unsqueeze(1)
        ground_mask = vertical_mask.masked_fill(y <= 0, False)

        return ground_mask

    def forward(self, depth, K, real_cam_height):
        inv_K = torch.inverse(K)

        cam_points = self.backproject_depth(depth, inv_K)
        surface_normal = self.get_surface_normal(cam_points)
        ground_mask = self.get_ground_mask(cam_points, surface_normal)

        cam_heights = (cam_points[:,:-1,:,:] * surface_normal).sum(1).abs().unsqueeze(1)
        cam_heights_masked = torch.masked_select(cam_heights, ground_mask)
        cam_height = torch.median(cam_heights_masked).unsqueeze(0)

        scale = torch.reciprocal(cam_height).mul_(real_cam_height)

        return scale


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_difference_vectors_8(img):
    """compute difference vectors using neighbouring 8 points.
    Outermost pixels in the image are ignored
    Args:
        img: Nx3xHxW
    Returns:
        list of difference vectors: each element is Nx3x(H-2)x(W-2)
        the order is as following:
            5 6 7
            4 * 0
            3 2 1
    """
    diff_vec_0 = img[:,:,1:-1,2:] - img[:,:,1:-1,1:-1]
    diff_vec_1 = img[:,:,2:,2:] - img[:,:,1:-1,1:-1]
    diff_vec_2 = img[:,:,2:,1:-1] - img[:,:,1:-1,1:-1]
    diff_vec_3 = img[:,:,2:,:-2] - img[:,:,1:-1,1:-1]
    diff_vec_4 = img[:,:,1:-1,:-2] - img[:,:,1:-1,1:-1]
    diff_vec_5 = img[:,:,:-2,:-2] - img[:,:,1:-1,1:-1]
    diff_vec_6 = img[:,:,:-2,1:-1] - img[:,:,1:-1,1:-1]
    diff_vec_7 = img[:,:,:-2,2:] - img[:,:,1:-1,1:-1]
    diff_vecs = [diff_vec_0, diff_vec_1, diff_vec_2, diff_vec_3, diff_vec_4, diff_vec_5, diff_vec_6, diff_vec_7]
    return diff_vecs


def compute_difference_vector_2(xyz):
    """compute difference vector using 2 neighbours (righ; bottom)
    surface normal map border (right & bottom) is set to 0
    Args:
        xyz (Nx3xHxW): 3d point cloud
    Returns:
        diff_vec_1 (Nx3xHxW) horizontal difference vector
        diff_vec_2 (Nx3xHxW) vectical difference vector
    """
    diff_vec_1 = xyz.clone().detach() * 0 # horizontal
    diff_vec_2 = xyz.clone().detach() * 0 # vectical
    diff_vec_1[:,:,:,:-1] = xyz[:,:,:,1:] - xyz[:,:,:,:-1]
    diff_vec_2[:,:,:-1,:] = xyz[:,:,1:,:] - xyz[:,:,:-1,:]
    return diff_vec_1, diff_vec_2

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

    def forward(self, depth, inv_K):
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
            weight_0 = norm_normal_0.clone() * 0 + 1.
            weight_1 = norm_normal_0.clone() * 0 + 1.
            weight_2 = norm_normal_0.clone() * 0 + 1.
            weight_3 = norm_normal_0.clone() * 0 + 1.
            normal = weight_0 * norm_normal_0 + weight_1 * norm_normal_1 + weight_2 * norm_normal_2 + weight_3 * norm_normal_3

        # normalize to unit vector
        norm_normal = torch.norm(normal, p=2, dim=1, keepdim=True)
        normal = normal / norm_normal
        return normal


class Backprojection(nn.Module):
    """Layer to backproject a depth image given the camera intrinsics
    Attributes
        xy (Nx3x(HxW)): homogeneous pixel coordinates on regular grid
    """
    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Backprojection, self).__init__()

        self.height = height
        self.width = width

        # generate regular grid
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.tensor(id_coords)

        # generate homogeneous pixel coordinates
        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False)
        self.xy = torch.unsqueeze(
                        torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
                        , 0)
        self.xy = torch.cat([self.xy, self.ones], 1)
        self.xy = nn.Parameter(self.xy, requires_grad=False)

    def forward(self, depth, inv_K, img_like_out=False):
        """
        Args:
            depth (Nx1xHxW): depth map
            inv_K (Nx4x4): inverse camera intrinsics
            img_like_out (bool): if True, the output shape is Nx4xHxW; else Nx4x(HxW)
        Returns:
            points (Nx4x(HxW)): 3D points in homogeneous coordinates
        """
        depth = depth.contiguous()

        xy = self.xy.repeat(depth.shape[0], 1, 1).cuda()
        ones = self.ones.repeat(depth.shape[0],1,1).cuda()
        points = torch.matmul(inv_K[:, :3, :3], xy)
        points = depth.view(depth.shape[0], 1, -1) * points
        points = torch.cat((points, ones), 1)

        if img_like_out:
            points = points.reshape(depth.shape[0], 4, self.height, self.width)
        return points
