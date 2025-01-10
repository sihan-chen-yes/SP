
import importlib
import torch
from utils.net_util import to_cuda
import cv2 as cv
import torch.nn as nn
from network.styleunet.dual_styleunet import SimpleNet
import torch.nn.functional as F
import os
import utils.net_util as net_util
import numpy as np
from utils.obj_io import save_mesh_as_ply
from pytorch3d.renderer import (
    OrthographicCameras,
)

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
front_mask_net = SimpleNet().to(device)
back_mask_net = SimpleNet().to(device)
front_depth_net = SimpleNet().to(device)
back_depth_net = SimpleNet().to(device)
all_params = (list(front_mask_net.parameters()) +
              list(back_mask_net.parameters()) +
              list(front_depth_net.parameters()) +
              list(back_depth_net.parameters()))
optm = torch.optim.Adam(
    all_params, lr=5e-3
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optm, 'min', patience=50, threshold=1e-3,
                                                            cooldown=50, factor=0.9)


cano_smpl_depth_map = cv.imread('../avatarrex/lbn1/smpl_depth_map/cano_smpl_depth_map_pts_based.exr',
                                cv.IMREAD_UNCHANGED)
gt_cano_smpl_depth_map = torch.from_numpy(cano_smpl_depth_map).to(torch.float32).to(device)
gt_cano_smpl_mask = gt_cano_smpl_depth_map > 0.
cano_smpl_map = cv.imread('../avatarrex/lbn1/smpl_pos_map/cano_smpl_pos_map.exr',
                          cv.IMREAD_UNCHANGED)
cano_smpl_map = torch.from_numpy(cano_smpl_map).to(torch.float32).to(device)
cano_init_points = cano_smpl_map[gt_cano_smpl_mask]

cano_smpl_v = cano_init_points.cpu().detach().numpy()
cano_center = 0.5 * (cano_smpl_v.min(0) + cano_smpl_v.max(0))
cano_center = torch.from_numpy(cano_center).to('cuda')

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

front_mv = torch.eye(4, dtype=torch.float32).to(cano_center.device)
front_mv[:3, 3] = -cano_center + torch.tensor([0, 0, -10], dtype=torch.float32).to(cano_center.device)
front_mv[:3, :3] = torch.linalg.inv(front_mv[:3, :3])
front_mv[1:3, :] *= -1
front_camera = get_orthographic_camera(front_mv, 1024, 1024, cano_center.device)

back_mv = torch.eye(4, dtype=torch.float32).to(cano_center.device)
rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
rot_y = torch.from_numpy(rot_y).to(cano_center.device)
back_mv[:3, :3] = rot_y
back_mv[:3, 3] = -rot_y @ cano_center + torch.tensor([0, 0, -10], dtype=torch.float32).to(cano_center.device)

back_mv[:3, :3] = torch.linalg.inv(back_mv[:3, :3])
back_mv[1:3] *= -1
back_camera = get_orthographic_camera(back_mv, 1024, 1024, cano_center.device)


def get_predicted_depth_map(pose_map):
    front_depth_map = front_depth_net(pose_map)
    back_depth_map = back_depth_net(pose_map)
    depth_map = torch.cat([front_depth_map, back_depth_map], 2).permute(1, 2, 0).squeeze()
    # clamp negative depth
    # depth_map = torch.clamp(depth_map, min=0)
    depth_map = torch.nn.functional.softplus(depth_map)
    return depth_map

def get_mask(pose_map):
    front_mask_map = front_mask_net(pose_map)
    back_mask_map = back_mask_net(pose_map)
    mask_map = torch.cat([front_mask_map, back_mask_map], 2).permute(1, 2, 0).squeeze()
    # squeeze to [0, 1]
    mask_map = torch.sigmoid(mask_map)

    return mask_map

def depth_map_to_pos_map(depth_map, mask, return_map=False):
    depth_front = depth_map[:, :1024]
    depth_back = depth_map[:, 1024:]

    # mask_front = mask[:, :1024]
    # mask_back = mask[:, 1024:]

    points_world_front = depth_to_position(front_camera, depth_front)

    points_world_back = depth_to_position(back_camera, depth_back)

    position_map = torch.cat([points_world_front, points_world_back], dim=1)

    # mask = torch.cat([mask_front, mask_back], dim=1)
    positions = position_map[mask]

    if return_map:
        return positions, position_map
    else:
        return positions

def depth_to_position(cameras, depth_map):
    h, w = depth_map.shape
    # homogeneous
    x, y = torch.meshgrid(torch.arange(w, device=depth_map.device), torch.arange(h, device=depth_map.device), indexing="xy")
    xy_depth = torch.stack((x, y, depth_map), dim=-1)  # (H, W, 3)
    xy_depth = xy_depth.reshape(-1, 3).to(depth_map.device, dtype=torch.float32)  # Flatten to (N, 3)
    xyz_unproj_world = cameras.unproject_points(xy_depth, world_coordinates=True)
    points_world = xyz_unproj_world.reshape(h, w, -1)
    return points_world

if __name__ == '__main__':
    dataset_module = 'MvRgbDatasetAvatarReX'
    MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)

    dataset = MvRgbDataset(**{'data_dir': '../avatarrex/lbn1', 'frame_range': [0, 1901, 1], 'load_smpl_pos_map': True, 'subject_name': 'avatarrex_lbn1', 'used_cam_ids': [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15]})
    batch_size = 1
    num_workers = 8
    batch_num = len(dataset) // batch_size
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             drop_last=True)


    l1_loss = torch.nn.L1Loss()
    # tb writer
    smooth_interval = 10
    smooth_count = 0
    smooth_losses = {}
    iter_idx = 0
    output_dir = "./results/avatarrex_lbn1/avatar/debug/"
    os.makedirs(output_dir + '/cano_pts', exist_ok=True)
    os.makedirs(output_dir + '/predicted', exist_ok=True)

    for epoch_idx in range(0, 9999999):
        epoch_idx = epoch_idx
        for batch_idx, items in enumerate(dataloader):
            total_loss = 0
            batch_losses = {}
            iter_idx = batch_idx + epoch_idx * batch_num
            items = to_cuda(items)
            items = net_util.delete_batch_idx(items)
            pose_map = items['smpl_pos_map'][:3]

            predicted_mask = get_mask(pose_map)
            predicted_depth = get_predicted_depth_map(pose_map)

            mask_loss = F.binary_cross_entropy(predicted_mask, gt_cano_smpl_mask.float())
            total_loss += mask_loss

            cano_depth_loss = l1_loss(predicted_depth, gt_cano_smpl_depth_map)
            total_loss += cano_depth_loss

            batch_losses.update({
                'smpl_predicted_mask_loss': mask_loss.item()
            })

            batch_losses.update({
                'cano_smpl_predicted_depth_loss': cano_depth_loss.item()
            })

            total_loss.backward()

            # update params
            optm.step()
            optm.zero_grad()
            scheduler.step(total_loss)

            # record batch loss
            for key, loss in batch_losses.items():
                if key in smooth_losses:
                    smooth_losses[key] += loss
                else:
                    smooth_losses[key] = loss
            smooth_count += 1

            if iter_idx % smooth_interval == 0:
                log_info = 'epoch %d, batch %d, iter %d ' % (epoch_idx, batch_idx, iter_idx)
                for key, loss in batch_losses.items():
                    log_info = log_info + ('%s: %f, ' % (key, batch_losses[key]))
                print(log_info)
            # if iter_idx % smooth_interval == 0:
            #     log_info = 'epoch %d, batch %d, iter %d ' % (epoch_idx, batch_idx, iter_idx)
                # for key in smooth_losses.keys():
                #     smooth_losses[key] /= smooth_count
                #     log_info = log_info + ('%s: %f, ' % (key, smooth_losses[key]))
                #     smooth_losses[key] = 0.
                # smooth_count = 0
                # print(log_info)

            if iter_idx % 200 == 0 and iter_idx != 0:
                if iter_idx % (1000) == 0:
                    eval_cano_pts = True
                else:
                    eval_cano_pts = False
                with torch.no_grad():
                    if eval_cano_pts:
                        mask = predicted_mask > 0.5
                        position = depth_map_to_pos_map(predicted_depth, mask)

                        save_mesh_as_ply(output_dir + '/cano_pts/iter_%d.ply' % iter_idx,
                                         position.cpu().numpy())
                    predicted_mask_image = (predicted_mask.cpu().numpy() * 255).astype(np.uint8)
                    predicted_depth_map = predicted_depth.cpu().numpy()

                    cv.imwrite(output_dir + '/predicted/predicted_mask_iter_%d.jpg' % iter_idx,
                               predicted_mask_image)
                    cv.imwrite(output_dir + '/predicted/predicted_depth_map_iter_%d.jpg' % iter_idx,
                               predicted_depth_map)

