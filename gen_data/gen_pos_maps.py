import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
import trimesh
import yaml
import tqdm

import smplx
from network.volume import CanoBlendWeightVolume
import config
from utils.renderer.renderer_pytorch3d import Renderer
from utils.visualize_util import colormap
from utils.graphics_utils import get_orthographic_depth_map, depth_map_to_pos_map, gen_front_back_cameras
from utils.obj_io import save_mesh_as_ply

def save_pos_map(pos_map, path):
    mask = np.linalg.norm(pos_map, axis = -1) > 0.
    positions = pos_map[mask]
    print('Point nums %d' % positions.shape[0])
    pc = trimesh.PointCloud(positions)
    pc.export(path)


def interpolate_lbs(pts, vertices, faces, vertex_lbs):
    from utils.posevocab_custom_ops.nearest_face import nearest_face_pytorch3d
    from utils.geo_util import barycentric_interpolate
    dists, indices, bc_coords = nearest_face_pytorch3d(
        torch.from_numpy(pts).to(torch.float32).cuda()[None],
        torch.from_numpy(vertices).to(torch.float32).cuda()[None],
        torch.from_numpy(faces).to(torch.int64).cuda()
    )
    # print(dists.mean())
    lbs = barycentric_interpolate(
        vert_attris = vertex_lbs[None].to(torch.float32).cuda(),
        faces = torch.from_numpy(faces).to(torch.int64).cuda()[None],
        face_ids = indices,
        bc_coords = bc_coords
    )
    return lbs[0].cpu().numpy()

def gen_depth_map(pos_map_dir, depth_map_dir, pc_dir, height=1024, width=1024):
    # create pts from pose map
    # init canonical gausssian model
    cano_smpl_map = cv.imread(pos_map_dir + '/cano_smpl_pos_map.exr', cv.IMREAD_UNCHANGED)
    cano_smpl_map = torch.from_numpy(cano_smpl_map).to(torch.float32).to(config.device)
    cano_smpl_mask = torch.linalg.norm(cano_smpl_map, dim = -1) > 0.
    cano_init_points = cano_smpl_map[cano_smpl_mask]
    front_camera, back_camera = gen_front_back_cameras(cano_init_points, height, width)

    # get orthographic projected depth map using pts in canonical space
    with torch.no_grad():
        cano_smpl_depth_map = get_orthographic_depth_map(cano_init_points, front_camera, back_camera)

        # unproject depth back to position
        mask = cano_smpl_depth_map > 0

        position = depth_map_to_pos_map(cano_smpl_depth_map, mask, front_camera=front_camera, back_camera=back_camera)
        # save depth map from canonical template
        cv.imwrite(depth_map_dir + '/cano_smpl_depth_map_pts_based.exr', cano_smpl_depth_map.cpu().numpy())

        os.makedirs(pc_dir, exist_ok=True)
        # save ply canonical template
        save_mesh_as_ply(pc_dir + '/cano_smpl.ply', position.cpu().numpy())

# 1. 1024 2. 512
map_size = 512
template = True

if __name__ == '__main__':
    from argparse import ArgumentParser
    import importlib

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    args = arg_parser.parse_args()

    opt = yaml.load(open(args.config_path, encoding = 'UTF-8'), Loader = yaml.FullLoader)
    dataset_module = opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
    MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
    dataset = MvRgbDataset(**opt['train']['data'])
    data_dir, frame_list = dataset.data_dir, dataset.pose_list

    pos_map_dir = data_dir + ('/smpl_pos_map_{}_smpl'.format(map_size) if not template else '/smpl_pos_map_{}_template'.format(map_size))
    os.makedirs(pos_map_dir, exist_ok = True)
    depth_map_dir = data_dir + ('/smpl_depth_map_{}_smpl'.format(map_size) if not template else '/smpl_depth_map_{}_template'.format(map_size))
    os.makedirs(depth_map_dir, exist_ok = True)
    pc_dir = data_dir + ('/pc_smpl' if not template else '/pc_template')

    cano_renderer = Renderer(map_size, map_size, shader_name = 'vertex_attribute')

    smpl_model = smplx.SMPLX(config.PROJ_DIR + '/smpl_files/smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, batch_size = 1)
    smpl_data = np.load(data_dir + '/smpl_params.npz')
    smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}

    with torch.no_grad():
        cano_smpl = smpl_model.forward(
            betas = smpl_data['betas'],
            global_orient = config.cano_smpl_global_orient[None],
            transl = config.cano_smpl_transl[None],
            body_pose = config.cano_smpl_body_pose[None]
        )
        cano_smpl_v = cano_smpl.vertices[0].cpu().numpy()
        cano_center = 0.5 * (cano_smpl_v.min(0) + cano_smpl_v.max(0))
        cano_smpl_v_min = cano_smpl_v.min()
        smpl_faces = smpl_model.faces.astype(np.int64)

    if os.path.exists(data_dir + '/template.ply') and template:
        print('# Loading template from %s' % (data_dir + '/template.ply'))
        template = trimesh.load(data_dir + '/template.ply', process = False)
        using_template = True
    else:
        print(f'# Cannot find template.ply from {data_dir}, using SMPL-X as template')
        template = trimesh.Trimesh(cano_smpl_v, smpl_faces, process = False)
        using_template = False

    cano_smpl_v = template.vertices.astype(np.float32)
    smpl_faces = template.faces.astype(np.int64)
    cano_smpl_v_dup = cano_smpl_v[smpl_faces.reshape(-1)]
    cano_smpl_n_dup = template.vertex_normals.astype(np.float32)[smpl_faces.reshape(-1)]

    # define front & back view matrices
    front_mv = np.identity(4, np.float32)
    front_mv[:3, 3] = -cano_center + np.array([0, 0, -10], np.float32)
    front_mv[1:3] *= -1

    back_mv = np.identity(4, np.float32)
    rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
    back_mv[:3, :3] = rot_y
    back_mv[:3, 3] = -rot_y @ cano_center + np.array([0, 0, -10], np.float32)
    back_mv[1:3] *= -1

    # render canonical smpl position maps
    cano_renderer.set_camera(front_mv)
    cano_renderer.set_model(cano_smpl_v_dup, cano_smpl_v_dup)
    front_cano_pos_map, front_cano_depth_map = cano_renderer.render()
    front_cano_pos_map = front_cano_pos_map[:, :, :3]

    cano_renderer.set_camera(back_mv)
    cano_renderer.set_model(cano_smpl_v_dup, cano_smpl_v_dup)
    back_cano_pos_map, back_cano_depth_map = cano_renderer.render()
    back_cano_pos_map = back_cano_pos_map[:, :, :3]
    cano_pos_map = np.concatenate([front_cano_pos_map, back_cano_pos_map], 1)
    cano_depth_map = np.concatenate([front_cano_depth_map, back_cano_depth_map], 1)
    # cano_depth_map_normalized = (cano_depth_map - cano_depth_map.min()) / (cano_depth_map.max() - cano_depth_map.min())
    cano_depth_map_visual = colormap(cano_depth_map).numpy()

    cv.imwrite(pos_map_dir + '/cano_smpl_pos_map.exr', cano_pos_map)
    cv.imwrite(depth_map_dir + '/cano_smpl_depth_map_mesh_based.exr', cano_depth_map)
    print("generated mesh based depth map")

    # gen point based depth map
    gen_depth_map(pos_map_dir, depth_map_dir, pc_dir, width=map_size, height=map_size)
    print("generated pts based depth map")

    # render canonical smpl normal maps
    cano_renderer.set_model(cano_smpl_v_dup, cano_smpl_n_dup)
    cano_renderer.set_camera(front_mv)
    front_cano_nml_map = cano_renderer.render()[0][:, :, :3]

    cano_renderer.set_camera(back_mv)
    back_cano_nml_map = cano_renderer.render()[0][:, :, :3]
    cano_nml_map = np.concatenate([front_cano_nml_map, back_cano_nml_map], 1)
    cv.imwrite(pos_map_dir + '/cano_smpl_nml_map.exr', cano_nml_map)

    body_mask = np.linalg.norm(cano_pos_map, axis = -1) > 0.
    cano_pts = cano_pos_map[body_mask]
    if using_template:
        weight_volume = CanoBlendWeightVolume(data_dir + '/cano_weight_volume.npz')
        pts_lbs = weight_volume.forward_weight(torch.from_numpy(cano_pts)[None].cuda())[0]
    else:
        pts_lbs = interpolate_lbs(cano_pts, cano_smpl_v, smpl_faces, smpl_model.lbs_weights)
        pts_lbs = torch.from_numpy(pts_lbs).cuda()
    np.save(pos_map_dir + '/init_pts_lbs.npy', pts_lbs.cpu().numpy())

    inv_cano_smpl_A = torch.linalg.inv(cano_smpl.A).cuda()
    body_mask = torch.from_numpy(body_mask).cuda()
    cano_pts = torch.from_numpy(cano_pts).cuda()
    pts_lbs = pts_lbs.cuda()

    # gen pos map
    for pose_idx in tqdm.tqdm(frame_list, desc = 'Generating positional maps...'):
        with torch.no_grad():
            live_smpl_woRoot = smpl_model.forward(
                betas = smpl_data['betas'],
                # global_orient = smpl_data['global_orient'][pose_idx][None],
                # transl = smpl_data['transl'][pose_idx][None],
                body_pose = smpl_data['body_pose'][pose_idx][None],
                jaw_pose = smpl_data['jaw_pose'][pose_idx][None],
                expression = smpl_data['expression'][pose_idx][None],
                # left_hand_pose = smpl_data['left_hand_pose'][pose_idx][None],
                # right_hand_pose = smpl_data['right_hand_pose'][pose_idx][None]
            )

        cano2live_jnt_mats_woRoot = torch.matmul(live_smpl_woRoot.A.cuda(), inv_cano_smpl_A)[0]
        pt_mats = torch.einsum('nj,jxy->nxy', pts_lbs, cano2live_jnt_mats_woRoot)
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros((map_size, 2 * map_size, 3)).to(live_pts)
        live_pos_map[body_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = live_pos_map.permute(1, 2, 0).cpu().numpy()
        cv.imwrite(pos_map_dir + '/{:08d}.exr'.format(pose_idx), live_pos_map)