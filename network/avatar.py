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
from utils.graphics_utils import get_orthographic_camera, depth_to_position, depth_map_to_pos_map, position_to_depth, get_orthographic_depth_map
import os
from utils.obj_io import save_mesh_as_ply
import root_finding
from network.volume import CanoBlendWeightVolume
from utils.posevocab_custom_ops.nearest_face import nearest_face_pytorch3d
from utils.knn import knn_gather
from utils.geo_util import barycentric_interpolate
from utils.network_utils import hierarchical_softmax, get_skinning_mlp

class AvatarNet(nn.Module):
    def __init__(self, opt):
        super(AvatarNet, self).__init__()
        self.opt = opt
        self.map_size = config.opt['train']['data']['map_size']
        self.lbs_weights = config.opt['model']['lbs_weights']
        self.lbs_weight_interpolation = config.opt['model']['lbs_weight_interpolation']

        self.random_style = opt.get('random_style', False)
        self.with_viewdirs = opt.get('with_viewdirs', True)
        self.xy_nr_scaling = opt.get('xy_nr_scaling', 0.05)

        # read preprocessed depth map: 1.mesh based 2.pts based
        # use mesh based depth map
        cano_smpl_depth_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_depth_map_{}_rot/cano_smpl_depth_map_mesh_based.exr'.format(self.map_size), cv.IMREAD_UNCHANGED)
        self.cano_smpl_depth_map = torch.from_numpy(cano_smpl_depth_map).to(torch.float32).to(config.device)
        self.cano_smpl_mask = self.cano_smpl_depth_map > 0.
        # change background value to 10 by default
        self.cano_smpl_depth_map[~self.cano_smpl_mask] = 10.0
        self.cano_smpl_opacity_map = self.cano_smpl_mask.to(torch.float32) * 0.9
        # depth offset normalized
        self.cano_smpl_depth_offset_map = self.cano_smpl_depth_map.clone()
        self.cano_smpl_depth_offset_map = self.cano_smpl_depth_offset_map - 10
        # depth expansion
        self.cano_smpl_depth_offset_map_max = self.cano_smpl_depth_offset_map.max() * 1.1
        self.cano_smpl_depth_offset_map_min = self.cano_smpl_depth_offset_map.min() * 1.1

        # init canonical gausssian model
        self.max_sh_degree = 0
        self.cano_gaussian_model = GaussianModel(sh_degree = self.max_sh_degree)
        cano_smpl_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map_{}_rot/cano_smpl_pos_map.exr'.format(self.map_size), cv.IMREAD_UNCHANGED)
        self.cano_smpl_map = torch.from_numpy(cano_smpl_map).to(torch.float32).to(config.device)
        # changeable target region
        self.bounding_mask = self.gen_bounding_mask()

        self.cano_init_points = self.cano_smpl_map[self.bounding_mask]
        self.lbs_init_points = self.cano_smpl_map[torch.linalg.norm(self.cano_smpl_map, dim = -1) > 0]
        self.cano_gaussian_model.create_from_pcd(self.cano_init_points, torch.rand_like(self.cano_init_points), spatial_lr_scale = 2.5,
                                                 mask = self.cano_smpl_mask[self.bounding_mask])
        # for skinning weight
        cano_pos_map_size = self.cano_smpl_map.shape[1] // 2
        self.cano_pos_map = torch.concatenate([self.cano_smpl_map[:, :cano_pos_map_size], self.cano_smpl_map[:, cano_pos_map_size:]], 2)
        self.cano_pos_map = self.cano_pos_map.permute((2, 0, 1))

        self.lbs = torch.from_numpy(np.load(config.opt['train']['data']['data_dir'] + '/smpl_pos_map_{}_rot/init_pts_lbs.npy'.format(self.map_size))).to(torch.float32).to(config.device)

        self.color_net = DualStyleUNet(inp_size = self.map_size // 2, inp_ch = 3, out_ch = 3, out_size = self.map_size, style_dim = self.map_size // 2, n_mlp = 2)
        self.properties = 1 + 3 + 4
        self.other_net = DualStyleUNet(inp_size = self.map_size // 2, inp_ch = 3, out_ch = self.properties, out_size = self.map_size, style_dim = self.map_size // 2, n_mlp = 2)
        # depth and xy nr offset
        self.offset_net = DualStyleUNet(inp_size =self.map_size // 2, inp_ch = 3, out_ch =1 + 2, out_size = self.map_size, style_dim =self.map_size // 2, n_mlp = 2)

        self.color_style = torch.ones([1, self.color_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.color_net.style_dim)
        self.other_style = torch.ones([1, self.other_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.other_net.style_dim)
        self.offset_style = torch.ones([1, self.offset_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.offset_net.style_dim)
        if self.lbs_weights == "NN":
            self.skinning_net = get_skinning_mlp(3, 55+4, config.opt['model']['skinning_network'])

        if self.with_viewdirs:
            cano_nml_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map_{}_rot/cano_smpl_nml_map.exr'.format(self.map_size), cv.IMREAD_UNCHANGED)
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
        self.cano_weight_volume = CanoBlendWeightVolume(config.opt['train']['data']['data_dir'] + '/cano_weight_volume_rot.npz')
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

        pose_map_paths = sorted(glob.glob(config.opt['train']['data']['data_dir'] + "/smpl_pos_map_{}_rot/{:08d}.exr".format(self.map_size, config.opt['test']['fix_hand_id'])))
        smpl_pos_map = cv.imread(pose_map_paths[0], cv.IMREAD_UNCHANGED)
        pos_map_size = smpl_pos_map.shape[1] // 2
        smpl_pos_map = np.concatenate([smpl_pos_map[:, :pos_map_size], smpl_pos_map[:, pos_map_size:]], 2)
        smpl_pos_map = smpl_pos_map.transpose((2, 0, 1))
        pose_map = torch.from_numpy(smpl_pos_map).to(torch.float32).to(config.device)
        pose_map = pose_map[:3]

        cano_pts = self.get_positions(pose_map)

        opacity, scales, rotations, xy_nr_offset = self.get_others(pose_map)
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
        # 1. use NN skinning weight 2.interpolate lbs weight
        if self.lbs_weights == "lbs_weights":
            # pts_w = self.get_lbs_pts_w(cano_gaussian_vals["positions"], self.lbs_init_points)
            pts_w = self.get_lbs_pts_w(cano_gaussian_vals["positions"], items["cano_smpl_v"], lbs_weights=items["lbs_weights"], faces=items["smpl_faces"])
        else:
            # neural skinning weight
            pts_w = cano_gaussian_vals["skinning_weight"]
        posed_gaussian_vals = cano_gaussian_vals.copy()
        cano_gaussian_vals["skinning_weight"] = pts_w

        # pick the corresponding transformation for each pts
        pt_mats = torch.einsum('nj,jxy->nxy', pts_w, items['cano2live_jnt_mats'])
        posed_gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_gaussian_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(cano_gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        posed_gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)

        return posed_gaussian_vals

    def transform_live2cano(self, posed_gaussian_vals, items, use_root_finding=False, cano_pts_w=None):
        # smplx inverse transformation
        # live 2 cano space: inverse LBS
        # 1. use NN skinning weight 2.interpolate lbs weight
        if self.lbs_weights == "lbs_weights":
            if cano_pts_w == None:
                pts_w = self.get_lbs_pts_w(posed_gaussian_vals["positions"], items["live_smpl_v"],
                                           lbs_weights=items["lbs_weights"], faces=items["smpl_faces"])
                # inverse LBS transformation matrix
                pt_mats = torch.einsum('nj,jxy->nxy', pts_w, torch.linalg.inv(items['cano2live_jnt_mats']))
            else:
                pts_w = cano_pts_w
                pt_mats = torch.einsum('nj,jxy->nxy', pts_w, items['cano2live_jnt_mats'])
                # inverse LBS transformation matrix
                pt_mats = torch.linalg.inv(pt_mats)
        else:
            # pts_w under cano space
            pts_w = posed_gaussian_vals["skinning_weight"]
            pt_mats = torch.einsum('nj,jxy->nxy', pts_w, items['cano2live_jnt_mats'])
            # inverse LBS transformation matrix
            pt_mats = torch.linalg.inv(pt_mats)

        cano_gaussian_vals = posed_gaussian_vals.copy()
        cano_gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], posed_gaussian_vals['positions']) + pt_mats[..., :3, 3]
        posed_gaussian_vals["skinning_weight"] = pts_w

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
        front_map, back_map = torch.split(other_map, [self.properties, self.properties], 1)
        other_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0)
        height, width = other_map.shape[0], other_map.shape[1]
        others = other_map[mask]  # (N, #properties)

        opacity, scales, rotations = torch.split(others, [1, 3, 4], 1)
        # predict offset value
        opacity = self.cano_gaussian_model.opacity_activation(opacity)
        scales = self.cano_gaussian_model.scaling_activation(scales)
        rotations = self.cano_gaussian_model.rotation_activation(rotations)
        # opacity_map
        opacity_map = opacity.reshape(height, width)
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

    def get_predicted_offset_map(self, pose_map):
        offset_map, _ = self.offset_net([self.offset_style], pose_map[None], randomize_noise = False)
        front_map, back_map = torch.split(offset_map, [3, 3], 1)
        offset_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0).squeeze()
        # map to [-1, 1] for depth
        # 0 for background
        offset_map[:, :, 0] = torch.nn.functional.tanh(offset_map[:, :, 0])
        return offset_map

    def get_predicted_position_map(self, pose_map):
        offset_map = self.get_predicted_offset_map(pose_map)
        depth_offset_map, xy_nr_offset_map = torch.split(offset_map, [1, 2], 2)
        # recover depth offset to depth
        depth_map = (depth_offset_map + 1) / 2
        depth_map = (depth_map * (self.cano_smpl_depth_offset_map_max - self.cano_smpl_depth_offset_map_min)
                                                 + self.cano_smpl_depth_offset_map_min + 10)
        depth_map = depth_map.squeeze(-1)
        # TODO
        xy_nr_offset_map = xy_nr_offset_map * self.xy_nr_scaling
        return depth_map, xy_nr_offset_map

    def get_predicted_skinning_weight(self, cano_pts):
        # use cano pts as input
        assert self.lbs_weights == "NN"
        pts_w = self.skinning_net(cano_pts)
        pts_w = hierarchical_softmax(pts_w)
        return pts_w

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
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.lbs_init_points) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros_like(self.cano_smpl_map)
        live_pos_map[self.cano_smpl_mask[self.bounding_mask].view(self.map_size, self.map_size * 2)] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = torch.cat(torch.split(live_pos_map, [self.map_size // 2, self.map_size // 2], 2), 0)
        items.update({
            'smpl_pos_map': live_pos_map
        })
        return live_pos_map

    def render(self, items, bg_color = (0., 0., 0.), use_pca = False, use_vae = False, pretrain=False):
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
        predicted_depth_map, xy_nr_offset_map = self.get_predicted_position_map(pose_map)
        xy_nr_offset = xy_nr_offset_map.view(-1, 2)

        opacity, scales, rotations, opacity_map = self.get_others(pose_map, self.bounding_mask, return_map=True)
        cano_pts, pos_map = depth_map_to_pos_map(predicted_depth_map, self.bounding_mask, return_map=True,
                                                 front_camera=self.front_camera, back_camera=self.back_camera)
        # apply xy nr offset
        cano_pts[:, :2] = cano_pts[:, :2] + xy_nr_offset
        colors, color_map = self.get_colors(pose_map, self.bounding_mask, front_viewdirs, back_viewdirs)
        skinning_weight = self.get_predicted_skinning_weight(cano_pts) if self.lbs_weights == "NN" else None
        # for visualize
        filtering_mask = (opacity_map >= 0.5).flatten()

        cano_pts_filtered = cano_pts[filtering_mask]
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

        posed_gaussian_vals = self.transform_cano2live(gaussian_vals, items)
        cano_pts_w = gaussian_vals["skinning_weight"]
        cano_pts_w_filtered = cano_pts_w[filtering_mask]

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

        posed_pts = posed_gaussian_vals["positions"]
        # inverse LBS
        inverse_cano_gaussian_vals = self.transform_live2cano(posed_gaussian_vals, items, use_root_finding=True, cano_pts_w=cano_pts_w)
        inverse_cano_pts = inverse_cano_gaussian_vals["positions"]
        inverse_cano_pts_filtered = inverse_cano_gaussian_vals["positions"][filtering_mask]

        posed_pts_filtered = posed_pts[filtering_mask]
        posed_pts_w = posed_gaussian_vals["skinning_weight"]
        posed_pts_w_filtered = posed_pts_w[filtering_mask]

        # use inverse_cano_pts_filtered to generate opacity for self-supervision
        # pts -> depth map is not differentiable
        with torch.no_grad():
            inverse_depth_map = get_orthographic_depth_map(inverse_cano_pts_filtered, self.front_camera, self.back_camera)
            inverse_opacity_map = (inverse_depth_map > 0.).to(torch.float32) * 0.9

        # aiap loss
        # TODO scaling_modifier
        covariance = self.cano_gaussian_model.covariance_activation(gaussian_vals["scales"], scaling_modifier=1, rotation=gaussian_vals["rotations"])
        gaussian_vals.update({"covariance": covariance})
        posed_covariance = self.cano_gaussian_model.covariance_activation(posed_gaussian_vals["scales"], scaling_modifier=1, rotation=posed_gaussian_vals["rotations"])
        posed_gaussian_vals.update({"covariance": posed_covariance})
        # inverse_covariance = self.cano_gaussian_model.covariance_activation(inverse_cano_gaussian_vals["scales"], scaling_modifier=1, rotation=inverse_cano_gaussian_vals["rotations"])
        # inverse_cano_gaussian_vals.update({"covariance": inverse_covariance})

        ret = {
            'rgb_map': rgb_map,
            'mask_map': mask_map,
            'xy_nr_offset': xy_nr_offset,

            'depth_map': depth_map,

            'viewspace_points': viewspace_points,
            'visibility_filter': visibility_filter,
            'radii': radii,

            "colors": colors,
            'pos_map': pos_map,

            "predicted_mask": opacity_map,

            "cano_pts": cano_pts,
            "cano_pts_filtered": cano_pts_filtered,
            "cano_pts_w": cano_pts_w,
            "cano_pts_w_filtered": cano_pts_w_filtered,
            "inverse_cano_pts": inverse_cano_pts,
            "inverse_cano_pts_filtered": inverse_cano_pts_filtered,

            "inverse_depth_map": inverse_depth_map,
            "inverse_opacity_map": inverse_opacity_map,

            "posed_pts": posed_pts,
            "posed_pts_filtered": posed_pts_filtered,
            "posed_pts_w": posed_pts_w,
            "posed_pts_w_filtered": posed_pts_w_filtered,

            'predicted_depth_map': predicted_depth_map,
            "scales": scales,

            'predicted_skinning_weight': skinning_weight,

            "gaussian_vals": gaussian_vals,
            "posed_gaussian_vals": posed_gaussian_vals,
            "filtering_mask": filtering_mask,
        }
        if not self.training:
            ret.update({
                'cano_tex_map': color_map,
                'posed_gaussians': gaussian_vals
            })

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

    def get_lbs_pts_w(self, query_pts, lbs_pts, lbs_weights, faces=None):
        """
        return interpolated lbs weight for each pts
        using different interpolation methods

        input:
        query_pts: (N, 3)
        lbs_pts: (V, 3)
        lbs_weights: (V, 55)

        faces: (F, 3)
        method:
        nearest: nearest neighbor

        linear: linear_interpolation

        barycentric: barycentric_interpolation
        """
        # if no input lbs_weight
        # use smplx lbs weight as default
        assert lbs_weights != None

        # interpolation via different methods
        if self.lbs_weight_interpolation == "nearest":
            # use the nearest vertex
            knn_ret = pytorch3d.ops.knn_points(query_pts.unsqueeze(0), lbs_pts.unsqueeze(0), K = 1)
            lbs_pts_idx = knn_ret.idx.squeeze()
            pts_w = lbs_weights[lbs_pts_idx]
        elif self.lbs_weight_interpolation == "linear":
            # use inverse distance to interpolate linearly
            knn_ret = pytorch3d.ops.knn_points(query_pts.unsqueeze(0), lbs_pts.unsqueeze(0), K = 4)
            lbs_pts_idx = knn_ret.idx.squeeze() #(N, 4)
            lbs_pts_dist = knn_ret.dists.squeeze() #(N, 4)
            lbs_pts_dist = torch.clamp(lbs_pts_dist, min=1e-6)
            inv_pts_dist = 1 / lbs_pts_dist # inverse distance
            weight = inv_pts_dist / inv_pts_dist.sum(dim=1, keepdim=True) # normalize
            pts_w = (weight.unsqueeze(-1) * lbs_weights[lbs_pts_idx]).sum(dim=1) #(N, K, 55) -> (N, 55)
        elif self.lbs_weight_interpolation == "barycentric":
            assert faces != None
            dists_to_smpl, face_indices, bc_coords = nearest_face_pytorch3d(query_pts[None], lbs_pts[None], faces)
            face_vertex_ids = faces[face_indices.squeeze(0)]  # (N, 3)

            face_lbs = lbs_weights[face_vertex_ids]  # (N, 3, 55)

            pts_w = (bc_coords.squeeze(0)[..., None] * face_lbs).sum(1)  # (N, 55)

        return pts_w