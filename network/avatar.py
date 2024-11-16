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
from gaussians.gaussian_model import GaussianModel
from gaussians.gaussian_renderer import render3
from utils.network_utils import VanillaCondMLP
from utils.graphics_utils import uv_to_index
import trimesh
from pytorch3d.ops import knn_points
from utils.general_utils import inverse_sigmoid

class AvatarNet(nn.Module):
    def __init__(self, opt):
        super(AvatarNet, self).__init__()
        self.opt = opt

        self.random_style = opt.get('random_style', False)
        self.with_viewdirs = opt.get('with_viewdirs', True)

        # init canonical gausssian model
        self.max_sh_degree = 0
        self.cano_gaussian_model = GaussianModel(sh_degree = self.max_sh_degree)
        cano_smpl_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/cano_smpl_pos_map.exr', cv.IMREAD_UNCHANGED)
        self.cano_smpl_map = torch.from_numpy(cano_smpl_map).to(torch.float32).to(config.device)
        self.cano_smpl_mask = torch.linalg.norm(self.cano_smpl_map, dim = -1) > 0.

        cano_template_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map_template/cano_smpl_pos_map.exr', cv.IMREAD_UNCHANGED)
        self.cano_template_map = torch.from_numpy(cano_template_map).to(torch.float32).to(config.device)
        self.cano_template_mask = torch.linalg.norm(self.cano_template_map, dim = -1) > 0.

        self.init_points = self.cano_smpl_map[self.cano_smpl_mask]
        self.lbs = torch.from_numpy(np.load(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/init_pts_lbs.npy')).to(torch.float32).to(config.device)

        self.predict_cano_smpl_mask = torch.empty(0)
        self.cano_gaussian_model.create_from_pcd(self.init_points, torch.rand_like(self.init_points), spatial_lr_scale = 2.5)

        self.cano_gaussian_model.training_setup(self.opt["gaussian"])
        self.color_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = 3, out_size = 1024, style_dim = 512, n_mlp = 2)
        self.position_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = 3, out_size = 1024, style_dim = 512, n_mlp = 2)
        self.other_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = 8, out_size = 1024, style_dim = 512, n_mlp = 2)
        self.mask_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = 1, out_size = 1024, style_dim = 512, n_mlp = 2)

        self.color_style = torch.ones([1, self.color_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.color_net.style_dim)
        self.position_style = torch.ones([1, self.position_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.position_net.style_dim)
        self.other_style = torch.ones([1, self.other_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.other_net.style_dim)
        self.mask_style = torch.ones([1, self.mask_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.mask_net.style_dim)


        # TODO separate features encoding?
        # self.feature_net = DualStyleUNet(inp_size = 512, inp_ch = 3, out_ch = opt["feat_dim"], out_size = 1024, style_dim = 512, n_mlp = 2)
        # self.feature_style = torch.ones([1, self.feature_net.style_dim], dtype=torch.float32, device=config.device) / np.sqrt(self.feature_net.style_dim)
        # position + rotation + scaling + opacity + color
        # self.mlp_decoder = VanillaCondMLP(opt["feat_dim"], 0, 3 + 4 + 3 + 1 + 3, opt["mlp"])

        self.template_points = trimesh.load(config.opt['train']['data']['data_dir'] + '/template_raw.ply', process = False)


        if self.with_viewdirs:
            cano_nml_map = cv.imread(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/cano_smpl_nml_map.exr', cv.IMREAD_UNCHANGED)
            self.cano_nml_map = torch.from_numpy(cano_nml_map).to(torch.float32).to(config.device)
            self.cano_nmls = self.cano_nml_map[self.cano_smpl_mask]
            self.viewdir_net = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(64, 128, 4, 2, 1)
            )

    def generate_mean_hands(self):
        # print('# Generating mean hands ...')
        import glob
        # get hand mask
        lbs_argmax = self.lbs.argmax(1)
        self.hand_mask = lbs_argmax == 20
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax == 21)
        self.hand_mask = torch.logical_or(self.hand_mask, lbs_argmax >= 25)

        pose_map_paths = sorted(glob.glob(config.opt['train']['data']['data_dir'] + '/smpl_pos_map/%08d.exr' % config.opt['test']['fix_hand_id']))
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

    def transform_cano2live(self, gaussian_vals, items):
        # smplx transformation
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats'])
        # find the nearest vertex
        knn_ret = pytorch3d.ops.knn_points(gaussian_vals['positions'].unsqueeze(0), self.init_points.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pt_mats = pt_mats[p_idx, :]
        gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)

        return gaussian_vals

    def get_positions(self, pose_map, mask, return_map = False):
        position_map, _ = self.position_net([self.position_style], pose_map[None], randomize_noise = False)
        front_position_map, back_position_map = torch.split(position_map, [3, 3], 1)
        position_map = torch.cat([front_position_map, back_position_map], 3)[0].permute(1, 2, 0)
        delta_position = 0.05 * position_map[mask]

        # delta_position = position_map[self.cano_smpl_mask]

        positions = delta_position + self.cano_gaussian_model.get_xyz
        if return_map:
            return positions, position_map
        else:
            return positions

    def get_others(self, pose_map, mask):
        other_map, _ = self.other_net([self.other_style], pose_map[None], randomize_noise = False)
        front_map, back_map = torch.split(other_map, [8, 8], 1)
        other_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0)
        others = other_map[mask]  # (N, 8)

        opacity, scales, rotations = torch.split(others, [1, 3, 4], 1)
        opacity = self.cano_gaussian_model.opacity_activation(opacity + self.cano_gaussian_model.get_opacity_raw)
        scales = self.cano_gaussian_model.scaling_activation(scales + self.cano_gaussian_model.get_scaling_raw)
        rotations = self.cano_gaussian_model.rotation_activation(rotations + self.cano_gaussian_model.get_rotation_raw)

        return opacity, scales, rotations

    def get_colors(self, pose_map, mask, front_viewdirs = None, back_viewdirs = None):
        color_style = torch.rand_like(self.color_style) if self.random_style and self.training else self.color_style
        color_map, _ = self.color_net([color_style], pose_map[None], randomize_noise = False, view_feature1 = front_viewdirs, view_feature2 = back_viewdirs)
        front_color_map, back_color_map = torch.split(color_map, [3, 3], 1)
        color_map = torch.cat([front_color_map, back_color_map], 3)[0].permute(1, 2, 0)
        colors = color_map[mask]

        return colors, color_map

    def get_feat_map(self, pose_map):
        feat_map, _ = self.feature_net([self.feature_style], pose_map[None], randomize_noise = False)
        # TODO view direction input?
        # feat dim?
        front_feat_map, back_feat_map = torch.split(feat_map, [self.opt["feat_dim"], self.opt["feat_dim"]], 1)
        feat_map = torch.cat([front_feat_map, back_feat_map], 3)[0].permute(1, 2, 0)
        return feat_map

    def get_mask(self, pose_map):
        mask_map, _ = self.mask_net([self.mask_style], pose_map[None], randomize_noise = False)
        front_map, back_map = torch.split(mask_map, [1, 1], 1)
        mask_map = torch.cat([front_map, back_map], 3)[0].permute(1, 2, 0).squeeze()
        # squeeze to [0, 1]
        mask_map = torch.sigmoid(mask_map)

        return mask_map

    def get_interpolated_feat(self, pose_map):
        # [1024, 2048, 64]
        feat_map = self.get_feat_map(pose_map)
        uv = self.cano_gaussian_model.get_uv
        width = 2048
        height = 1024
        x = uv[:, 0] * (height - 1)
        y = uv[:, 1] * (width - 1)

        # Find the four surrounding points for bi-linear interpolation
        x0 = torch.floor(x).long()
        x1 = torch.clamp(x0 + 1, 0, width - 1)
        y0 = torch.floor(y).long()
        y1 = torch.clamp(y0 + 1, 0, height - 1)

        # Compute interpolation weights based on fractional parts
        # gap is 1(denominator 1)
        wx = x - x0.float()
        wy = y - y0.float()
        w00 = (1 - wx) * (1 - wy)
        w10 = wx * (1 - wy)
        w01 = (1 - wx) * wy
        w11 = wx * wy

        # Gather features at the four corners
        f00 = feat_map[x0, y0]  # Top-left
        f10 = feat_map[x1, y0]  # Bottom-left
        f01 = feat_map[x0, y1]  # Top-right
        f11 = feat_map[x1, y1]  # Bottom-right

        # Compute the interpolated features
        interpolated_feat = (w00[:, None] * f00 +
                             w01[:, None] * f01 +
                             w10[:, None] * f10 +
                             w11[:, None] * f11)
        return interpolated_feat

    def get_gaussian_feat(self, pose_map):
        feat = self.get_interpolated_feat(pose_map)
        # position + rotation + scaling + opacity + color
        output = self.mlp_decoder(feat)

        delta_position, delta_rotation, delta_scales, delta_opacity, colors = torch.split(output, [3, 4, 3, 1, 3], 1)

        delta_position = 0.05 * delta_position
        positions = delta_position + self.cano_gaussian_model.get_xyz
        rotations = self.cano_gaussian_model.rotation_activation(delta_rotation + self.cano_gaussian_model.get_rotation_raw)
        scales = self.cano_gaussian_model.scaling_activation(delta_scales + self.cano_gaussian_model.get_scaling_raw)
        opacity = self.cano_gaussian_model.opacity_activation(delta_opacity + self.cano_gaussian_model.get_opacity_raw)
        #TODO color activation?
        colors = self.cano_gaussian_model.color_activation(colors)

        return positions, rotations, scales, opacity, colors

    def get_viewdir_feat(self, items):
        # TODO
        with torch.no_grad():
            pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats'])
            live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.init_points) + pt_mats[..., :3, 3]
            live_nmls = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.cano_nmls)
            cam_pos = -torch.matmul(torch.linalg.inv(items['extr'][:3, :3]), items['extr'][:3, 3])
            viewdirs = F.normalize(cam_pos[None] - live_pts, dim = -1, eps = 1e-3)
            if self.training:
                viewdirs += torch.randn(*viewdirs.shape).to(viewdirs) * 0.1
            viewdirs = F.normalize(viewdirs, dim = -1, eps = 1e-3)
            viewdirs = (live_nmls * viewdirs).sum(-1)

            viewdirs_map = torch.zeros(*self.cano_nml_map.shape[:2]).to(viewdirs)
            viewdirs_map[self.cano_smpl_mask] = viewdirs

            viewdirs_map = viewdirs_map[None, None]
            viewdirs_map = F.interpolate(viewdirs_map, None, 0.5, 'nearest')
            front_viewdirs, back_viewdirs = torch.split(viewdirs_map, [512, 512], -1)

        front_viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(front_viewdirs)
        back_viewdirs = self.opt.get('weight_viewdirs', 1.) * self.viewdir_net(back_viewdirs)
        return front_viewdirs, back_viewdirs

    def get_pose_map(self, items):
        pt_mats = torch.einsum('nj,jxy->nxy', self.lbs, items['cano2live_jnt_mats_woRoot'])
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.cano_gaussian_model.get_init_pts()) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros_like(self.cano_smpl_map)
        live_pos_map[self.cano_smpl_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = torch.cat(torch.split(live_pos_map, [512, 512], 2), 0)
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

        mask = self.get_mask(pose_map)
        if pretrain:
            cano_pts, pos_map = self.get_positions(pose_map, self.cano_smpl_mask, return_map = True)
            opacity, scales, rotations = self.get_others(pose_map, self.cano_smpl_mask)
            colors, color_map = self.get_colors(pose_map, self.cano_smpl_mask, front_viewdirs, back_viewdirs)

        else:
            # update cano gs
            self.cano_gaussian_model.create_from_pcd(self.cano_smpl_map[mask >= 0.5], torch.rand_like(self.cano_smpl_map[mask >= 0.5]), spatial_lr_scale=2.5, cano_smpl_mask=mask)
            cano_pts, pos_map = self.get_positions(pose_map, mask >= 0.5, return_map = True)
            opacity, scales, rotations = self.get_others(pose_map, mask >= 0.5)
            colors, color_map = self.get_colors(pose_map, mask >= 0.5, front_viewdirs, back_viewdirs)


        if not self.training and config.opt['test'].get('fix_hand', False) and config.opt['mode'] == 'test':
            # print('# fuse hands ...')
            import utils.geo_util as geo_util
            #TODO meanning?
            cano_xyz = self.init_points
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
            'max_sh_degree': self.max_sh_degree
        }

        # nonrigid_offset = gaussian_vals['positions'] - self.init_points[self.cano_gaussian_model.get_gs_index()]

        gaussian_vals = self.transform_cano2live(gaussian_vals, items)

        render_ret = render3(
            gaussian_vals,
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
        template_positions = torch.tensor(self.template_points.vertices, dtype=torch.float, device="cuda")
        dist2 = torch.clamp_min(knn_points(template_positions[None], template_positions[None], K = 4)[0][0, :, 1:].mean(-1), 0.0000001)
        template_scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
        # quaternion
        template_rotations = torch.zeros((template_positions.shape[0], 4), dtype=torch.float, device="cuda")
        template_rotations[:, 0] = 1
        template_gaussian_vals = {
            'positions': template_positions,
            'opacity': torch.ones((template_positions.shape[0], 1), dtype=torch.float, device="cuda"),
            'scales': template_scales,
            'rotations': template_rotations,
            'colors': torch.ones((template_positions.shape[0], 3), dtype=torch.float, device="cuda"),
            'max_sh_degree': self.max_sh_degree
        }

        template_gaussian_vals = self.transform_cano2live(template_gaussian_vals, items)

        template_render_ret = render3(
            template_gaussian_vals,
            bg_color,
            items['extr'],
            items['intr'],
            items['img_w'],
            items['img_h']
        )

        template_mask_map = template_render_ret['mask'].permute(1, 2, 0)
        template_depth_map = template_render_ret['depth'].permute(1, 2, 0)

        ret = {
            'rgb_map': rgb_map,
            'mask_map': mask_map,
            # 'offset': nonrigid_offset,

            'depth_map': depth_map,
            'viewspace_points': viewspace_points,
            'visibility_filter': visibility_filter,
            'radii': radii,

            "colors": colors,
            'pos_map': pos_map,
            'template_mask_map': template_mask_map,
            'template_depth_map': template_depth_map,
            "mask": mask
        }

        # if not self.training:
        #     ret.update({
        #         'cano_tex_map': color_map,
        #         'posed_gaussians': gaussian_vals
        #     })

        return ret
