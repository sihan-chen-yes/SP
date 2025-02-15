import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops
import pytorch3d.transforms
import cv2 as cv

import config
from network.styleunet.dual_styleunet import DualStyleUNet
from network.volume import compute_gradient_volume
from gaussians.gaussian_model import GaussianModel
from gaussians.gaussian_renderer import render3
from utils.network_utils import VanillaCondMLP
from utils.graphics_utils import uv_to_index
import trimesh
from pytorch3d.ops import knn_points
from utils.general_utils import inverse_sigmoid
from utils.renderer.renderer_pytorch3d import Renderer
from utils.graphics_utils import get_orthographic_camera, depth_to_position, depth_map_to_pos_map
import os
from utils.obj_io import save_mesh_as_ply
import root_finding
from network.volume import CanoBlendWeightVolume

class AvatarNet(nn.Module):
    def __init__(self, opt):
        super(AvatarNet, self).__init__()
        self.opt = opt
        self.map_size = config.opt['train']['data']['map_size']

        self.random_style = opt.get('random_style', False)
        self.with_viewdirs = opt.get('with_viewdirs', True)

        # read preprocessed depth map: 1.mesh based 2.pts based
        # for 1. smplx  2. template
        cano_smpl_depth_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_depth_map_{}/cano_smpl_depth_map_pts_based.exr'.format(self.map_size), cv.IMREAD_UNCHANGED)
        self.cano_smpl_depth_map = torch.from_numpy(cano_smpl_depth_map).to(torch.float32).to(config.device)
        # cano_template_depth_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_depth_map_template_{}/cano_smpl_depth_map_pts_based.exr'.format(self.map_size), cv.IMREAD_UNCHANGED)
        # self.cano_template_depth_map = torch.from_numpy(cano_template_depth_map).to(torch.float32).to(config.device)
        self.cano_smpl_mask = self.cano_smpl_depth_map > 0.
        # self.cano_template_mask = self.cano_template_depth_map > 0.
        # change background value to 10 by default
        self.cano_smpl_depth_map[~self.cano_smpl_mask] = 10.0
        # self.cano_template_depth_map[~self.cano_template_mask] = 10.0
        self.cano_smpl_opacity_map = self.cano_smpl_mask.to(torch.float32) * 0.9
        # depth offset normalized
        self.cano_smpl_depth_offset_map = self.cano_smpl_depth_map.clone()
        self.cano_smpl_depth_offset_map = self.cano_smpl_depth_offset_map - 10
        self.cano_smpl_depth_offset_map_max = self.cano_smpl_depth_offset_map.max()
        self.cano_smpl_depth_offset_map_min = self.cano_smpl_depth_offset_map.min()

        # init canonical gausssian model
        self.max_sh_degree = 0
        self.cano_gaussian_model = GaussianModel(sh_degree = self.max_sh_degree)
        cano_smpl_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map_{}/cano_smpl_pos_map.exr'.format(self.map_size), cv.IMREAD_UNCHANGED)
        self.cano_smpl_map = torch.from_numpy(cano_smpl_map).to(torch.float32).to(config.device)
        # self.cano_smpl_mask = torch.linalg.norm(self.cano_smpl_map, dim = -1) > 0.
        self.bounding_mask = self.gen_bounding_mask()

        self.cano_init_points = self.cano_smpl_map[self.bounding_mask]
        self.lbs_init_points = self.cano_smpl_map[torch.linalg.norm(self.cano_smpl_map, dim = -1) > 0]
        self.pos_map_mask = torch.linalg.norm(self.cano_smpl_map, dim = -1) > 0.
        self.cano_gaussian_model.create_from_pcd(self.cano_init_points, torch.rand_like(self.cano_init_points), spatial_lr_scale = 2.5,
                                                 mask = self.pos_map_mask[self.bounding_mask])

        # cano_template_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map_template_{}/cano_smpl_pos_map.exr'.format(self.map_size), cv.IMREAD_UNCHANGED)
        # self.cano_template_map = torch.from_numpy(cano_template_map).to(torch.float32).to(config.device)
        # self.cano_template_mask = torch.linalg.norm(self.cano_template_map, dim = -1) > 0.
        # self.cano_template_gaussian_model = GaussianModel(sh_degree = self.max_sh_degree)
        # self.cano_template_init_points = self.cano_template_map[self.cano_template_mask]
        # self.cano_template_gaussian_model.create_from_pcd(self.cano_template_init_points, torch.rand_like(self.cano_template_init_points), spatial_lr_scale = 2.5)
        self.lbs = torch.from_numpy(np.load(config.opt['train']['data']['data_dir'] + '/smpl_pos_map_{}/init_pts_lbs.npy'.format(self.map_size))).to(torch.float32).to(config.device)

        self.cano_gaussian_model.training_setup(self.opt["gaussian"])
        self.color_net = DualStyleUNet(inp_size = self.map_size // 2, inp_ch = 3, out_ch = 3, out_size = self.map_size, style_dim = self.map_size // 2, n_mlp = 2)
        self.position_net = DualStyleUNet(inp_size = self.map_size // 2, inp_ch = 3, out_ch = 3, out_size = self.map_size, style_dim = self.map_size // 2, n_mlp = 2)
        self.other_net = DualStyleUNet(inp_size = self.map_size // 2, inp_ch = 3, out_ch = 8, out_size = self.map_size, style_dim = self.map_size // 2, n_mlp = 2)
        self.mask_net = DualStyleUNet(inp_size = self.map_size // 2, inp_ch = 3, out_ch = 1, out_size = self.map_size, style_dim = self.map_size // 2, n_mlp = 2)
        self.depth_net = DualStyleUNet(inp_size = self.map_size // 2, inp_ch = 3, out_ch = 1, out_size = self.map_size, style_dim = self.map_size // 2, n_mlp = 2)

        self.color_style = torch.ones([1, self.color_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.color_net.style_dim)
        self.position_style = torch.ones([1, self.position_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.position_net.style_dim)
        self.other_style = torch.ones([1, self.other_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.other_net.style_dim)
        self.mask_style = torch.ones([1, self.mask_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.mask_net.style_dim)
        self.depth_style = torch.ones([1, self.depth_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.depth_net.style_dim)

        # TODO separate features encoding?
        # self.feature_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = opt["feat_dim"], out_size = 1024, style_dim = 512, n_mlp = 2)
        # self.feature_style = torch.ones([1, self.feature_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.feature_net.style_dim)
        # position + rotation + scaling + opacity + color
        # self.mlp_decoder = VanillaCondMLP(opt["feat_dim"], 0, 3 + 4 + 3 + 1 + 3, opt["mlp"])

        self.template_points = trimesh.load(config.opt['train']['data']['data_dir'] + '/template_raw.ply', process = False)


        if self.with_viewdirs:
            cano_nml_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map_{}/cano_smpl_nml_map.exr'.format(self.map_size), cv.IMREAD_UNCHANGED)
            self.cano_nml_map = torch.from_numpy(cano_nml_map).to(torch.float32).to(config.device)
            self.cano_nmls = self.cano_nml_map[self.bounding_mask]
            self.viewdir_net = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(64, 128, 4, 2, 1)
            )

        # prepare front and back cameras
        cano_smpl_v = self.cano_init_points.cpu().detach().numpy()
        cano_center = 0.5 * (cano_smpl_v.min(0) + cano_smpl_v.max(0))
        cano_center = torch.from_numpy(cano_center).to('cuda')

        front_mv = torch.eye(4, dtype=torch.float32).to(cano_center.device)
        front_mv[:3, 3] = -cano_center + torch.tensor([0, 0, -10], dtype=torch.float32).to(cano_center.device)
        front_mv[:3, :3] = torch.linalg.inv(front_mv[:3, :3])
        front_mv[1:3, :] *= -1
        self.front_camera = get_orthographic_camera(front_mv, self.map_size, self.map_size, cano_center.device)

        back_mv = torch.eye(4, dtype=torch.float32).to(cano_center.device)
        rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
        rot_y = torch.from_numpy(rot_y).to(cano_center.device)
        back_mv[:3, :3] = rot_y
        back_mv[:3, 3] = -rot_y @ cano_center + torch.tensor([0, 0, -10], dtype=torch.float32).to(cano_center.device)

        back_mv[:3, :3] = torch.linalg.inv(back_mv[:3, :3])
        back_mv[1:3] *= -1
        self.back_camera = get_orthographic_camera(back_mv, self.map_size, self.map_size, cano_center.device)

        # for root finding
        self.cano_weight_volume = CanoBlendWeightVolume(config.opt['train']['data']['data_dir'] + '/cano_weight_volume.npz')
        if self.opt.get('volume_type', 'diff') == 'diff':
            self.weight_volume = self.cano_weight_volume.diff_weight_volume[0].permute(1, 2, 3, 0).contiguous()
        else:
            self.weight_volume = self.cano_weight_volume.ori_weight_volume[0].permute(1, 2, 3, 0).contiguous()
        self.grad_volume = compute_gradient_volume(self.weight_volume.permute(3, 0, 1, 2), self.cano_weight_volume.voxel_size).permute(2, 3, 4, 0, 1)\
            .reshape(self.cano_weight_volume.res_x, self.cano_weight_volume.res_y, self.cano_weight_volume.res_z, -1).contiguous()
        self.res = torch.tensor([self.cano_weight_volume.res_x, self.cano_weight_volume.res_y, self.cano_weight_volume.res_z], dtype = torch.int32, device = config.device)

    def generate_mean_hands(self):
        # print('# Generating mean hands ...')
        import glob
        # get hand mask
        lbs_argmax = self.lbs.argmax(1)
        self.hand_mask = lbs_argmax == 20
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax == 21)
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax >= 25)

        pose_map_paths = sorted(glob.glob(config.opt['train']['data']['data_dir'] + "/smpl_pos_map_{}/{:08d}.exr".format(self.map_size, config.opt['test']['fix_hand_id'])))
        smpl_pos_map = cv.imread(pose_map_paths[0], cv.IMREAD_UNCHANGED)
        pos_map_size = smpl_pos_map.shape[1] // 2
        smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
        smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
        pose_map = torch.from_numpy(smpl_pos_map).to(torch.float32).to(config.device)
        pose_map = pose_map[:3]

        cano_pts = self.get_positions(pose_map)
        opacity, scales, rotations = self.get_others(pose_map)
        colors, color_map = self.get_colors(pose_map)

        self.hand_positions = cano_pts#[self.hand_mask]
        self.hand_opacity = opacity#[self.hand_mask]
        self.hand_scales = scales#[self.hand_mask]
        self.hand_rotations = rotations#[self.hand_mask]
        self.hand_colors = colors#[self.hand_mask]

        # # debug
        # hand_pts = trimesh.PointCloud(self.hand_positions.detach().cpu().numpy())
        # hand_pts.export('./debug/hand_template.obj')
        # exit(1)

    def transform_cano2live(self, cano_gaussian_vals, items):
        # smplx transformation
        # cano 2 live space: LBS
        pts_w = self.get_lbs_pts_w(cano_gaussian_vals["positions"], self.lbs_init_points)
        # pick the corresponding transformation for each pts
        pt_mats = torch.einsum('nj,jxy->nxy', pts_w, items['cano2live_jnt_mats'])
        posed_gaussian_vals = cano_gaussian_vals.copy()
        posed_gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_gaussian_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(cano_gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        posed_gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)

        return posed_gaussian_vals

    def transform_live2cano(self, posed_gaussian_vals, items, use_root_finding=False):
        # smplx inverse transformation
        # live 2 cano space: inverse LBS
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats'])
        obs_lbs_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.lbs_init_points) + pt_mats[..., :3, 3]
        pts_w = self.get_lbs_pts_w(posed_gaussian_vals["positions"], obs_lbs_pts)
        # inverse LBS transformation matrix
        pt_mats = torch.einsum('nj,jxy->nxy', pts_w, torch.linalg.inv(items['cano2live_jnt_mats']))
        cano_gaussian_vals = posed_gaussian_vals.copy()
        cano_gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], posed_gaussian_vals['positions']) + pt_mats[..., :3, 3]

        if use_root_finding:
            argmax_lbs = torch.argmax(pts_w, -1)
            nonopt_bone_ids = [7, 8, 10, 11]
            nonopt_pts_flag = torch.zeros((cano_gaussian_vals['positions'].shape[0]), dtype=torch.bool).to(argmax_lbs.device)
            for i in nonopt_bone_ids:
                nonopt_pts_flag = torch.logical_or(nonopt_pts_flag, argmax_lbs == i)
            root_finding_flag = torch.logical_not(nonopt_pts_flag)
            if root_finding_flag.any():
                cano_pts_ = cano_gaussian_vals['positions'][root_finding_flag].unsqueeze(0)
                posed_pts_ = posed_gaussian_vals['positions'][root_finding_flag].unsqueeze(0)
                if not cano_pts_.is_contiguous():
                    cano_pts_ = cano_pts_.contiguous()
                if not posed_pts_.is_contiguous():
                    posed_pts_ = posed_pts_.contiguous()
                root_finding.root_finding(
                    self.weight_volume,
                    self.grad_volume,
                    posed_pts_,
                    cano_pts_,
                    items['cano2live_jnt_mats'],
                    self.cano_weight_volume.volume_bounds,
                    self.res,
                    cano_pts_,
                    0.1,
                    10
                )
                cano_gaussian_vals['positions'][root_finding_flag] = cano_pts_[0]

        rot_mats = pytorch3d.transforms.quaternion_to_matrix(posed_gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        cano_gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)

        return cano_gaussian_vals

    def get_others(self, pose_map, mask, return_map = False):
        other_map, _ = self.other_net([self.other_style], pose_map[None], randomize_noise = False)
        front_map, back_map = torch.split(other_map, [8, 8], 1)
        other_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0)
        others = other_map[mask]  # (N, 8)
        # others = other_map.view(-1, 8)  # (N, 8)

        opacity, scales, rotations = torch.split(others, [1, 3, 4], 1)
        # predict absolute value
        opacity = self.cano_gaussian_model.opacity_activation(opacity)
        scales = self.cano_gaussian_model.scaling_activation(scales)
        rotations = self.cano_gaussian_model.rotation_activation(rotations)
        # TODO
        # opacity = self.cano_gaussian_model.opacity_activation(opacity + self.cano_gaussian_model.get_opacity_raw)
        # scales = self.cano_gaussian_model.scaling_activation(scales + self.cano_gaussian_model.get_scaling_raw)
        # rotations = self.cano_gaussian_model.rotation_activation(rotations + self.cano_gaussian_model.get_rotation_raw)

        # opacity_map
        opacity_map = self.cano_gaussian_model.opacity_activation(other_map[:, :, 0])
        if return_map:
            return opacity, scales, rotations, opacity_map
        else:
            return opacity, scales, rotations

    def get_colors(self, pose_map, mask, front_viewdirs = None, back_viewdirs = None):
        color_style = torch.rand_like(self.color_style) if self.random_style and self.training else self.color_style
        color_map, _ = self.color_net([color_style], pose_map[None], randomize_noise = False, view_feature1 = front_viewdirs, view_feature2 = back_viewdirs)
        front_color_map, back_color_map = torch.split(color_map, [3, 3], 1)
        color_map = torch.cat([front_color_map, back_color_map], 3)[0].permute(1, 2, 0)
        colors = color_map[mask]

        return colors, color_map

    # def get_feat_map(self, pose_map):
    #     feat_map, _ = self.feature_net([self.feature_style], pose_map[None], randomize_noise = False)
    #     # TODO view direction input?
    #     # feat dim?
    #     front_feat_map, back_feat_map = torch.split(feat_map, [self.opt["feat_dim"], self.opt["feat_dim"]], 1)
    #     feat_map = torch.cat([front_feat_map, back_feat_map], 3)[0].permute(1, 2, 0)
    #     return feat_map

    def get_mask(self, pose_map):
        mask_map, _ = self.mask_net([self.mask_style], pose_map[None], randomize_noise = False)
        front_map, back_map = torch.split(mask_map, [1, 1], 1)
        mask_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0).squeeze()
        # squeeze to [0, 1]
        mask_map = torch.sigmoid(mask_map)

        return mask_map

    def get_predicted_depth_offset_map(self, pose_map):
        depth_offset_map, _ = self.depth_net([self.depth_style], pose_map[None], randomize_noise = False)
        front_map, back_map = torch.split(depth_offset_map, [1, 1], 1)
        depth_offset_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0).squeeze()
        # map to [-1, 1]
        # 0 for background
        depth_offset_map = torch.nn.functional.tanh(depth_offset_map)
        return depth_offset_map

    def get_predicted_depth_map(self, pose_map):
        depth_offset_map = self.get_predicted_depth_offset_map(pose_map)
        # recover depth offset to depth
        depth_map = depth_offset_map.clone()
        depth_map = (depth_map + 1) / 2
        depth_map = (depth_map * (self.cano_smpl_depth_offset_map_max - self.cano_smpl_depth_offset_map_min)
                                                 + self.cano_smpl_depth_offset_map_min + 10)
        return depth_map

    # def get_interpolated_feat(self, pose_map):
    #     # [1024, 2048, 64]
    #     feat_map = self.get_feat_map(pose_map)
    #     uv = self.cano_gaussian_model.get_uv
    #     width = 2048
    #     height = 1024
    #     x = uv[:, 0] * (height - 1)
    #     y = uv[:, 1] * (width - 1)
    #
    #     # Find the four surrounding points for bi-linear interpolation
    #     x0 = torch.floor(x).long()
    #     x1 = torch.clamp(x0 + 1, 0, width - 1)
    #     y0 = torch.floor(y).long()
    #     y1 = torch.clamp(y0 + 1, 0, height - 1)
    #
    #     # Compute interpolation weights based on fractional parts
    #     # gap is 1(denominator 1)
    #     wx = x - x0.float()
    #     wy = y - y0.float()
    #     w00 = (1 - wx) * (1 - wy)
    #     w10 = wx * (1 - wy)
    #     w01 = (1 - wx) * wy
    #     w11 = wx * wy
    #
    #     # Gather features at the four corners
    #     f00 = feat_map[x0, y0]  # Top-left
    #     f10 = feat_map[x1, y0]  # Bottom-left
    #     f01 = feat_map[x0, y1]  # Top-right
    #     f11 = feat_map[x1, y1]  # Bottom-right
    #
    #     # Compute the interpolated features
    #     interpolated_feat = (w00[:, None] * f00 +
    #                          w01[:, None] * f01 +
    #                          w10[:, None] * f10 +
    #                          w11[:, None] * f11)
    #     return interpolated_feat

    # def get_gaussian_feat(self, pose_map):
    #     feat = self.get_interpolated_feat(pose_map)
    #     # position + rotation + scaling + opacity + color
    #     output = self.mlp_decoder(feat)
    #
    #     delta_position, delta_rotation, delta_scales, delta_opacity, colors = torch.split(output, [3, 4, 3, 1, 3], 1)
    #
    #     delta_position = 0.05 * delta_position
    #     positions = delta_position + self.cano_gaussian_model.get_xyz
    #     rotations = self.cano_gaussian_model.rotation_activation(delta_rotation + self.cano_gaussian_model.get_rotation_raw)
    #     scales = self.cano_gaussian_model.scaling_activation(delta_scales + self.cano_gaussian_model.get_scaling_raw)
    #     opacity = self.cano_gaussian_model.opacity_activation(delta_opacity + self.cano_gaussian_model.get_opacity_raw)
    #     #TODO color activation?
    #     colors = self.cano_gaussian_model.color_activation(colors)
    #
    #     return positions, rotations, scales, opacity, colors

    def get_viewdir_feat(self, items):
        with torch.no_grad():
            pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats'])
            # find the nearest vertex
            knn_ret = pytorch3d.ops.knn_points(self.cano_init_points.unsqueeze(0),
                                               self.lbs_init_points.unsqueeze(0))
            p_idx = knn_ret.idx.squeeze()
            pt_mats = pt_mats[p_idx, :]
            live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.cano_init_points) + pt_mats[..., :3, 3]
            live_nmls = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.cano_nmls)
            cam_pos = -torch.matmul(torch.linalg.inv(items['extr'][:3, :3]), items['extr'][:3, 3])
            viewdirs = F.normalize(cam_pos[None] - live_pts, dim = -1, eps = 1e-3)
            if self.training:
                viewdirs += torch.randn(*viewdirs.shape).to(viewdirs) * 0.1
            viewdirs = F.normalize(viewdirs, dim = -1, eps = 1e-3)
            viewdirs = (live_nmls * viewdirs).sum(-1)

            viewdirs_map = torch.zeros(*self.cano_nml_map.shape[:2]).to(viewdirs)
            viewdirs_map[self.bounding_mask] = viewdirs

            viewdirs_map = viewdirs_map[None, None]
            viewdirs_map = F.interpolate(viewdirs_map, None, 0.5, 'nearest')
            front_viewdirs, back_viewdirs = torch.split(viewdirs_map, [self.map_size // 2, self.map_size // 2], -1)

        front_viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(front_viewdirs)
        back_viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(back_viewdirs)
        return front_viewdirs, back_viewdirs

    def get_pose_map(self, items):
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats_woRoot'])
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.cano_init_points) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros_like(self.cano_smpl_map)
        live_pos_map[self.cano_smpl_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = torch.cat(torch.split(live_pos_map, [self.map_size // 2, 512], 2), 0)
        items.update({
            'smpl_pos_map': live_pos_map
        })
        return live_pos_map

    # def depth_to_position(self, cameras, depth_map):
    #     h, w = depth_map.shape
    #     # recover camera view coords
    #     x, y = torch.meshgrid(torch.arange(w, device=depth_map.device), torch.arange(h, device=depth_map.device), indexing="xy")
    #     xy_depth = torch.stack((x, y, torch.ones((x.shape[0], x.shape[1]), device=depth_map.device, dtype=torch.float32)), dim=-1)  # (H, W, 3)
    #     xy_depth = xy_depth.reshape(-1, 3)
    #     fx, fy = cameras.focal_length[0]
    #     cx, cy = cameras.principal_point[0]
    #     intr = torch.tensor([
    #         [fx, 0, cx],
    #         [0, fy, cy],
    #         [0, 0, 1]
    #     ], device=depth_map.device, dtype=torch.float32)
    #     intr_inv = torch.linalg.inv(intr)
    #     xy_cam = xy_depth @ intr_inv.T
    #     # careful with orthographic: replace z with D directly!
    #     depth = depth_map.flatten()
    #     xy_cam[:, 2] = depth
    #     # recover world view coords
    #     R = cameras.R[0]
    #     T = cameras.T[0]
    #     extr = torch.eye(4, device=depth_map.device)
    #     extr[:3, :3] = R
    #     extr[:3, -1] = T
    #     extr_inv = torch.linalg.inv(extr)
    #     xy_cam_homo = torch.cat((xy_cam, torch.ones((xy_cam.shape[0], 1), device=depth_map.device)), dim=1) # (N, 4)
    #     xyz_unproj_world = xy_cam_homo @ extr_inv.T
    #     points_world = xyz_unproj_world[:, :3]  # Remove homogeneous coordinate
    #     points_world = points_world.reshape(h, w, -1)
    #     return points_world



    def render(self, items, bg_color = (0., 0., 0.), use_pca = False, use_vae = False, pretrain=False, template=False):
        """
        Note that no batch index in items.
        """
        bg_color = torch.from_numpy(np.asarray(bg_color)).to(torch.float32).to(config.device)
        pose_map = items['smpl_pos_map'][:3]
        assert not (use_pca and use_vae), "Cannot use both PCA and VAE!"
        if use_pca:
            pose_map = items['smpl_pos_map_pca'][:3]
        if use_vae:
            pose_map = items['smpl_pos_map_vae'][:3]

        # if not self.training:
        # scales = torch.clip(scales, 0., 0.03)
        if self.with_viewdirs:
            front_viewdirs, back_viewdirs = self.get_viewdir_feat(items)
        else:
            front_viewdirs, back_viewdirs = None, None

        # use depth map predicted from 2D pose map
        predicted_depth_map = self.get_predicted_depth_map(pose_map)
        if pretrain:

            opacity, scales, rotations, opacity_map = self.get_others(pose_map, self.cano_smpl_mask, return_map=True)
            cano_pts, pos_map = depth_map_to_pos_map(predicted_depth_map, self.cano_smpl_mask, return_map=True, front_camera=self.front_camera, back_camera=self.back_camera)
            colors, color_map = self.get_colors(pose_map, self.cano_smpl_mask, front_viewdirs, back_viewdirs)
            # for visualize
            visualize_mask = self.cano_smpl_mask
        else:

            opacity, scales, rotations, opacity_map = self.get_others(pose_map, self.bounding_mask, return_map=True)
            cano_pts, pos_map = depth_map_to_pos_map(predicted_depth_map, self.bounding_mask, return_map=True, front_camera=self.front_camera, back_camera=self.back_camera)
            colors, color_map = self.get_colors(pose_map, self.bounding_mask, front_viewdirs, back_viewdirs)
            # for visualize
            visualize_mask = (opacity_map >= 0.5).flatten()
        cano_pts_visualize = cano_pts[visualize_mask]

        if not self.training and config.opt['test'].get('fix_hand', False) and config.opt['mode'] == 'test':
            # print('# fuse hands ...')
            import utils.geo_util as geo_util
            #TODO meanning?
            cano_xyz = self.cano_init_points
            wl = torch.sigmoid(2.5 * (geo_util.normalize_vert_bbox(items['left_cano_mano_v'], attris = cano_xyz, dim = 0, per_axis = True)[..., 0:1] + 2.0))
            wr = torch.sigmoid(-2.5 * (geo_util.normalize_vert_bbox(items['right_cano_mano_v'], attris = cano_xyz, dim = 0, per_axis = True)[..., 0:1] - 2.0))
            wl[cano_xyz[..., 1] < items['cano_smpl_center'][1]] = 0.
            wr[cano_xyz[..., 1] < items['cano_smpl_center'][1]] = 0.

            s = torch.maximum(wl + wr, torch.ones_like(wl))
            wl, wr = wl / s, wr / s

            w = wl + wr
            cano_pts = w * self.hand_positions + (1.0 - w) * cano_pts
            opacity = w * self.hand_opacity + (1.0 - w) * opacity
            scales = w * self.hand_scales + (1.0 - w) * scales
            rotations = w * self.hand_rotations + (1.0 - w) * rotations
            # colors = w * self.hand_colors + (1.0 - w) * colors


        gaussian_vals = {
            'positions': cano_pts,
            'opacity': opacity,
            'scales': scales,
            'rotations': rotations,
            'colors': colors,
            'max_sh_degree': self.max_sh_degree,
        }

        # render_ret = render3(
        #     gaussian_vals,
        #     bg_color,
        #     items['extr'],
        #     items['intr'],
        #     items['img_w'],
        #     items['img_h']
        # )
        #
        # cano_depth_map = render_ret['depth'].permute(1, 2, 0)

        # nonrigid_offset = smplx_cano_pts - self.init_points
        nonrigid_offset = 0

        posed_gaussian_vals = self.transform_cano2live(gaussian_vals, items)

        render_ret = render3(
            posed_gaussian_vals,
            bg_color,
            items['extr'],
            items['intr'],
            items['img_w'],
            items['img_h']
        )
        rgb_map = render_ret['render'].permute(1, 2, 0)
        mask_map = render_ret['mask'].permute(1, 2, 0)
        depth_map = render_ret['depth'].permute(1, 2, 0)
        viewspace_points = render_ret['viewspace_points']
        visibility_filter = render_ret['visibility_filter']
        radii = render_ret['radii']

        # render template to supervise
        # template_positions = torch.tensor(self.template_points.vertices, dtype=torch.float, device="cuda")
        # dist2 = torch.clamp_min(knn_points(template_positions[None], template_positions[None], K = 4)[0][0, :, 1:].mean(-1), 0.0000001)
        # template_scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
        # # quaternion
        # template_rotations = torch.zeros((template_positions.shape[0], 4), dtype=torch.float, device="cuda")
        # template_rotations[:, 0] = 1
        # template_gaussian_vals = {
        #     'positions': template_positions,
        #     'opacity': torch.ones((template_positions.shape[0], 1), dtype=torch.float, device="cuda"),
        #     'scales': template_scales,
        #     'rotations': template_rotations,
        #     'colors': torch.ones((template_positions.shape[0], 3), dtype=torch.float, device="cuda"),
        #     'max_sh_degree': self.max_sh_degree
        # }

        # template_render_ret = render3(
        #     template_gaussian_vals,
        #     bg_color,
        #     items['extr'],
        #     items['intr'],
        #     items['img_w'],
        #     items['img_h']
        # )
        # cano_template_depth_map = template_render_ret['depth'].permute(1, 2, 0)

        # template_gaussian_vals = self.transform_cano2live(template_gaussian_vals, items)

        # template_render_ret = render3(
        #     template_gaussian_vals,
        #     bg_color,
        #     items['extr'],
        #     items['intr'],
        #     items['img_w'],
        #     items['img_h']
        # )
        #
        # template_mask_map = template_render_ret['mask'].permute(1, 2, 0)
        # template_depth_map = template_render_ret['depth'].permute(1, 2, 0)

        # inverse LBS
        cano_gaussian_vals = self.transform_live2cano(posed_gaussian_vals, items, use_root_finding=True)
        inverse_cano_pts_visualize = cano_gaussian_vals["positions"][visualize_mask]
        ret = {
            'rgb_map': rgb_map,
            'mask_map': mask_map,
            'offset': nonrigid_offset,

            'depth_map': depth_map,
            # 'cano_depth_map': cano_depth_map,

            'viewspace_points': viewspace_points,
            'visibility_filter': visibility_filter,
            'radii': radii,

            "colors": colors,
            'pos_map': pos_map,
            # 'template_mask_map': template_mask_map,
            # 'template_depth_map': template_depth_map,
            # 'cano_template_depth_map': cano_template_depth_map,
            "predicted_mask": opacity_map,
            "cano_pts": cano_pts_visualize,
            'predicted_depth_map': predicted_depth_map,
            "scales": scales,
            "inverse_cano_pts": inverse_cano_pts_visualize,
        }

        # if not self.training:
        #     ret.update({
        #         'cano_tex_map': color_map,
        #         'posed_gaussians': gaussian_vals
        #     })

        return ret

    def gen_bounding_mask(self, full=True):
        indices = torch.nonzero(self.cano_smpl_mask, as_tuple=False)
        y_min, x_min = indices.min(dim=0)[0]
        y_max, x_max = indices.max(dim=0)[0]
        bounding_mask = self.cano_smpl_mask.clone()

        if full:
            bounding_mask[0:self.map_size + 1, 0:self.map_size * 2 + 1] = True
        else:
            bounding_mask[y_min:y_max + 1, x_min:x_max + 1] = True

        return bounding_mask

    def get_lbs_pts_w(self, pts, lbs_pts, method="linear_interpolation"):
        """
        return the most appropriate lbs point weight each pts
        using different strategy

        method:
        NN: nearest neighbor

        LI: linear_interpolation
        """
        if method == "NN":
            # use the nearest vertex
            knn_ret = pytorch3d.ops.knn_points(pts.unsqueeze(0), lbs_pts.unsqueeze(0), K = 1)
            lbs_pts_idx = knn_ret.idx.squeeze()
            pts_w = self.lbs[lbs_pts_idx]
        elif method == "linear_interpolation":
            # use inverse distance to interpolate linearly
            knn_ret = pytorch3d.ops.knn_points(pts.unsqueeze(0), lbs_pts.unsqueeze(0), K = 4)
            lbs_pts_idx = knn_ret.idx.squeeze() #(N, 4)
            lbs_pts_dist = knn_ret.dists.squeeze() #(N, 4)
            lbs_pts_dist = torch.clamp(lbs_pts_dist, min=1e-6)
            inv_pts_dist = 1 / lbs_pts_dist # inverse distance
            weight = inv_pts_dist / inv_pts_dist.sum(dim=1, keepdim=True) # normalize
            pts_w = (weight.unsqueeze(-1) * self.lbs[lbs_pts_idx]).sum(dim=1) #(N, K, 55) -> (N, 55)

        return pts_w
