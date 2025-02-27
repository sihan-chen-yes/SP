#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
from pytorch3d.renderer import (
    OrthographicCameras,
)
import cv2 as cv

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY, K = None, img_h = None, img_w = None):
    if K is None:
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
    else:
        near_fx = znear / K[0, 0]
        near_fy = znear / K[1, 1]

        left = - (img_w - K[0, 2]) * near_fx
        right = K[0, 2] * near_fx
        bottom = (K[1, 2] - img_h) * near_fy
        top = K[1, 2] * near_fy

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def index_to_uv(index, width=2048, height=1024):
    # index is a tensor of shape (N, 2)
    # Convert index to u and v in the range [0, 1]
    uv_coords = index.clone().float()
    uv_coords[:, 0] = uv_coords[:, 0] / (height - 1)  # Convert x_index to u
    uv_coords[:, 1] = uv_coords[:, 1] / (width - 1)  # Convert y_index to v
    return uv_coords

def uv_to_index(uv, width=2048, height=1024):
    # uv is a tensor of shape (N, 2) where each row is a (u, v) coordinate in [0, 1]
    # Convert u and v back to integer indices in the range [0, width-1] and [0, height-1]

    index = torch.zeros_like(uv)
    # Convert u to x_index by scaling by width and rounding
    index[:, 0] = (uv[:, 0] * (height - 1)).round()

    # Convert v to y_index by scaling by height and rounding
    index[:, 1] = (uv[:, 1] * (width - 1)).round()

    return index

def get_orthographic_camera(extr, height, width, device):
    camera = OrthographicCameras(
        focal_length=((width / 2., height / 2.),),
        principal_point=((width / 2., height / 2.),),
        R=extr[:3, :3].unsqueeze(0),
        T=extr[:3, 3].unsqueeze(0),
        in_ndc=False,
        device=device,
        image_size=((height, width),)
    )
    return camera

def position_to_depth(camera, xyz):
    assert xyz.shape[0] == 1
    xyz_cam = camera.get_world_to_view_transform().transform_points(xyz)
    # extract the depth of each point as the 3rd coord of xyz_cam
    depth = xyz_cam[:, :, 2:]

    depths = depth.squeeze(0)
    # project the points xyz to the camera
    xy = camera.transform_points(xyz)[:, :, :2]

    uv = torch.round(xy.squeeze(0)).long()
    height, width = int(camera.image_size[0, 0]), int(camera.image_size[0, 1])

    valid_mask = (uv[:, 0] >= 0) & (uv[:, 0] < width) & \
                 (uv[:, 1] >= 0) & (uv[:, 1] < height)

    uv = uv[valid_mask]
    depths = depths[valid_mask]

    linear_idx = uv[:, 1] * width + uv[:, 0]
    depths = depths.squeeze()
    flat_depth_map = torch.full((height * width,), float('inf'), device=xyz.device)

    # for idx, depth in zip(linear_idx, depths):
    #     flat_depth_map[idx] = min(flat_depth_map[idx], depth)

    flat_depth_map.scatter_reduce_(
        dim=0,
        index=linear_idx,
        src=depths,
        reduce="amin"  # min
    )

    depth_map = flat_depth_map.view(height, width)
    depth_map[depth_map == float('inf')] = 0

    return depth_map

def get_orthographic_depth_map(xyz, height=1024, width=1024):
    cano_v = xyz.cpu().detach().numpy()
    cano_center = 0.5 * (cano_v.min(0) + cano_v.max(0))
    cano_center = torch.from_numpy(cano_center).to('cuda')

    front_mv = torch.eye(4, dtype=torch.float32).to(cano_center.device)
    front_mv[:3, 3] = -cano_center + torch.tensor([0, 0, -10], dtype=torch.float32).to(cano_center.device)
    front_mv[:3, :3] = torch.linalg.inv(front_mv[:3, :3])
    front_mv[1:3, :] *= -1
    front_camera = get_orthographic_camera(front_mv, height, width, cano_center.device)

    back_mv = torch.eye(4, dtype=torch.float32).to(cano_center.device)
    rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
    rot_y = torch.from_numpy(rot_y).to(cano_center.device)
    back_mv[:3, :3] = rot_y
    back_mv[:3, 3] = -rot_y @ cano_center + torch.tensor([0, 0, -10], dtype=torch.float32).to(cano_center.device)
    back_mv[:3, :3] = torch.linalg.inv(back_mv[:3, :3])
    back_mv[1:3] *= -1
    back_camera = get_orthographic_camera(back_mv, height, width, cano_center.device)

    # position to depth
    xyz = xyz.unsqueeze(0)  # 3D points of shape (batch_size, num_points, 3)

    front_depth = position_to_depth(front_camera, xyz)
    back_depth = position_to_depth(back_camera, xyz)
    # orthographic projected depth map
    depth_map = torch.cat([front_depth, back_depth], dim=1)

    return depth_map

def depth_to_position(cameras, depth_map):
    h, w = depth_map.shape
    # homogeneous
    x, y = torch.meshgrid(torch.arange(w, device=depth_map.device), torch.arange(h, device=depth_map.device), indexing="xy")
    xy_depth = torch.stack((x, y, depth_map), dim=-1)  # (H, W, 3)
    xy_depth = xy_depth.reshape(-1, 3).to(depth_map.device, dtype=torch.float32)  # Flatten to (N, 3)
    xyz_unproj_world = cameras.unproject_points(xy_depth, world_coordinates=True)
    points_world = xyz_unproj_world.reshape(h, w, -1)
    return points_world

def depth_map_to_pos_map(depth_map, mask, return_map=False, front_camera=None, back_camera=None):
    width = depth_map.shape[1] // 2
    # split the whole depth map
    depth_front = depth_map[:, :width]
    depth_back = depth_map[:, width:]

    assert front_camera != None and back_camera != None
    points_world_front = depth_to_position(front_camera, depth_front)

    points_world_back = depth_to_position(back_camera, depth_back)

    position_map = torch.cat([points_world_front, points_world_back], dim=1)

    positions = position_map[mask]

    if return_map:
        return positions, position_map
    else:
        return positions

def gen_front_back_cameras(xyz, height=1024, width=1024):
    cano_smpl_v = xyz.cpu().detach().numpy()
    cano_center = 0.5 * (cano_smpl_v.min(0) + cano_smpl_v.max(0))
    cano_center = torch.from_numpy(cano_center).to('cuda')

    front_mv = torch.eye(4, dtype=torch.float32).to(cano_center.device)
    front_mv[:3, 3] = -cano_center + torch.tensor([0, 0, -10], dtype=torch.float32).to(cano_center.device)
    front_mv[:3, :3] = torch.linalg.inv(front_mv[:3, :3])
    front_mv[1:3, :] *= -1
    front_camera = get_orthographic_camera(front_mv, height, width, cano_center.device)

    back_mv = torch.eye(4, dtype=torch.float32).to(cano_center.device)
    rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
    rot_y = torch.from_numpy(rot_y).to(cano_center.device)
    back_mv[:3, :3] = rot_y
    back_mv[:3, 3] = -rot_y @ cano_center + torch.tensor([0, 0, -10], dtype=torch.float32).to(cano_center.device)

    back_mv[:3, :3] = torch.linalg.inv(back_mv[:3, :3])
    back_mv[1:3] *= -1
    back_camera = get_orthographic_camera(back_mv, height, width, cano_center.device)
    return front_camera, back_camera

def read_obj(obj_path):
    """
    Parameters
    ----------
    obj_path

    Returns
    -------
    vertices_v
    vertices_vt
    faces: contains both xyz vertices index and uv vertices index
    vertex_neighbors: adjacent face index

    """
    # careful these xyz positions are not canonical pose xyz positions!
    vertices_v = []
    vertices_vt = []
    faces = []
    vertex_neighbors = []

    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()
                vertices_v.append([float(x), float(y), float(z)])
            elif line.startswith('vt '):
                _, u, v = line.strip().split()
                vertices_vt.append([float(u), float(v)])
            elif line.startswith('f '):
                face_elements = line.strip().split()[1:]
                indices_v = []
                indices_vt = []
                for element in face_elements:
                    parts = element.split('/')
                    indices_v.append(int(parts[0]) - 1)
                    if len(parts) > 1 and parts[1]:
                        indices_vt.append(int(parts[1]) - 1)
                faces.append({'indices_v': indices_v, 'indices_vt': indices_vt})
    for i in range(len(vertices_v)):
        face_neighbor = []
        # iterate all faces
        for j, item in enumerate(faces):
            # contained in xyz face
            if i in item['indices_v']:
                face_neighbor.append(j)
        vertex_neighbors.append(face_neighbor)

    return {
        "vertices_v": vertices_v,
        "vertices_vt": vertices_vt,
        "faces": faces,
        "vertex_neighbors": vertex_neighbors,
    }

def get_TBN(obj_path, smplx=None):
    """
    get per face TBN and per vertex TBN
    Returns
    -------

    """
    # read data from obj file
    obj_data = read_obj(obj_path)

    # use default vertices_v from obj file
    if smplx is None:
        vertices_v = obj_data["vertices_v"]
        # normals = None
    else:
        vertices_v = smplx.vertices.astype(np.float32)
        # normals = smplx.face_normals.astype(np.float32)

    vertices_vt = obj_data["vertices_vt"]
    faces = obj_data['faces']
    vertex_neighbors = obj_data['vertex_neighbors']
    per_face_tangents, per_face_bitangents, per_face_normals = compute_per_face_TBN(vertices_v, vertices_vt, faces)
    per_vertex_tangents, per_vertex_bitangents, per_vertex_normals = compute_per_vertex_TBN(vertex_neighbors, per_face_tangents, per_face_bitangents, per_face_normals)
    obj_data.update({
        "per_face_tangents": per_face_tangents,
        "per_face_bitangents": per_face_bitangents,
        "per_face_normals": per_face_normals,
        "per_vertex_tangents": per_vertex_tangents,
        "per_vertex_bitangents": per_vertex_bitangents,
        "per_vertex_normals": per_vertex_normals,
    })
    return obj_data

def compute_per_face_TBN(vertices_v, vertices_vt, faces):
    tangent_list = []
    bitangent_list = []
    normal_list = []
    epsilon = 1e-8

    # iterate faces
    for face_index, face in enumerate(faces):
        indices_v = face['indices_v']
        indices_vt = face['indices_vt']

        # v and vt coords
        v0, v1, v2 = [vertices_v[i] for i in indices_v]
        uv0, uv1, uv2 = [vertices_vt[i] for i in indices_vt]

        delta_pos1 = np.array(v1) - np.array(v0)
        delta_pos2 = np.array(v2) - np.array(v0)
        delta_uv1 = np.array(uv1) - np.array(uv0)
        delta_uv2 = np.array(uv2) - np.array(uv0)

        normal = np.cross(delta_pos1, delta_pos2)
        # normalize
        normal /= np.linalg.norm(normal) + epsilon

        normal = np.cross(delta_pos1, delta_pos2)
        # normalize
        normal /= np.linalg.norm(normal) + epsilon

        denominator = delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0]
        if abs(denominator) < epsilon:
            if abs(denominator) < epsilon:
                # print("Warning: UV determinant is zero, using fallback tangent.")
                # pick longer edge as tangent
                if np.linalg.norm(delta_pos1) > np.linalg.norm(delta_pos2):
                    tangent = delta_pos1
                else:
                    tangent = delta_pos2
        else:
            r = 1.0 / denominator
            tangent = r * (delta_pos1 * delta_uv2[1] - delta_pos2 * delta_uv1[1])

        # Gram-Schmidt orthogonalization
        tangent = tangent - np.dot(normal, tangent) * normal
        tangent = tangent / np.linalg.norm(tangent)

        # bitangent = r * (delta_pos2 * delta_uv1[0] - delta_pos1 * delta_uv2[0])

        bitangent = np.cross(normal, tangent)

        tangent_list.append(tangent.tolist())
        bitangent_list.append(bitangent.tolist())
        normal_list.append(normal.tolist())

    # change list into numpy
    return np.array(tangent_list), np.array(bitangent_list), np.array(normal_list)

def compute_per_vertex_TBN(vertex_neighbors, per_face_tangents, per_face_bitangents, per_face_normals):
    """
    Parameters
    ----------
    vertex_neighbors: adjacent face index for each vertex

    per_face_tangents
    per_face_bitangents
    per_face_normals
    Returns
    -------
    per_vertex_tangents
    per_vertex_bitangents
    per_vertex_normals
    """
    num_vertices = len(vertex_neighbors)
    per_vertex_tangents = np.zeros((num_vertices, 3))
    per_vertex_bitangents = np.zeros((num_vertices, 3))
    per_vertex_normals = np.zeros((num_vertices, 3))

    epsilon = 1e-8

    for indices_v, neighbors in enumerate(vertex_neighbors):
        tangent_sum = np.zeros(3)
        bitangent_sum = np.zeros(3)
        normal_sum = np.zeros(3)

        for neighbor in neighbors:
            #TODO better avg methods
            tangent_sum += per_face_tangents[neighbor]
            bitangent_sum += per_face_bitangents[neighbor]
            normal_sum += per_face_normals[neighbor]

        tangent_sum /= len(neighbors)
        bitangent_sum /= len(neighbors)
        normal_sum /= len(neighbors)

        # Normalizing TBN vectors
        per_vertex_tangents[indices_v] = tangent_sum / (np.linalg.norm(tangent_sum) + epsilon)
        per_vertex_bitangents[indices_v] = bitangent_sum / (np.linalg.norm(bitangent_sum) + epsilon)
        per_vertex_normals[indices_v] = normal_sum / (np.linalg.norm(normal_sum) + epsilon)

    return per_vertex_tangents, per_vertex_bitangents, per_vertex_normals
