# To compute FID, first install pytorch_fid
# pip install pytorch-fid

import os
import cv2 as cv
from tqdm import tqdm
import shutil
import config
import yaml
import subprocess
from eval.score import *

def find_full_dataset_dir(data_dir, cam_id):
    assert os.path.exists(data_dir)
    if dataset == "avatarrex":
        cam_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
        cam_folder = os.path.join(data_dir, cam_folders[cam_id])
        gt_dir = cam_folder
        mask_dir = os.path.join(cam_folder, "mask", "pha")
        return gt_dir, mask_dir
    else:
        raise NotImplementedError

if __name__ == '__main__':

    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    arg_parser.add_argument('-e', '--exp', type=str, help='result directory path')
    args = arg_parser.parse_args()

    # load config
    config.load_global_opt(args.config_path)
    if args.exp is not None:
        config.opt['exp'] = args.exp

    cam_id = config.opt["cam_id"]
    iters = config.opt["iters"]

    dataset = config.opt["dataset"]
    subject = config.opt["subject"]
    data_dir = f'../{dataset}/{subject}'
    exp = config.opt["exp"]
    dir = f"./test_results/{dataset}_{subject}/{exp}/training__cam_{cam_id:03d}/batch_{iters:06d}/vanilla"
    eval_dir = os.path.join(dir, "eval")
    os.makedirs(eval_dir, exist_ok = True)
    rgb_dir = os.path.join(dir, "rgb_map")
    frame_range = config.opt["frame_range"]
    frame_list = list(range(frame_range[0], frame_range[1], frame_range[2]))
    results_file = os.path.join(eval_dir, "results.txt")
    print(yaml.safe_dump(config.opt, sort_keys=False))
    with open(results_file, 'w') as fp:
        fp.write(yaml.safe_dump(config.opt, sort_keys=False) + '\n')

    # construct gt and mask directory for different datasets and subjects
    gt_dir, mask_dir = find_full_dataset_dir(data_dir, cam_id)

    ours_metrics = Metrics()

    eval_ours_dir = os.path.join(eval_dir, "ours")
    eval_gt_dir = os.path.join(eval_dir, "gt")
    os.makedirs(eval_ours_dir, exist_ok = True)
    os.makedirs(eval_gt_dir, exist_ok = True)

    for frame_id in tqdm(frame_list):
        ours_img = (cv.imread(rgb_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        gt_img = (cv.imread(gt_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) / 255.).astype(np.float32)
        mask_img = cv.imread(mask_dir + '/%08d.jpg' % frame_id, cv.IMREAD_UNCHANGED) > 128
        gt_img[~mask_img] = 1.

        ours_img_cropped, gt_img_cropped = \
            crop_image(
                mask_img,
                512,
                ours_img,
                # posevocab_img,
                # slrf_img,
                # tava_img,
                # arah_img,
                gt_img
            )

        cv.imwrite(f'{eval_ours_dir}/{frame_id:08d}.png', (ours_img_cropped * 255).astype(np.uint8))
        cv.imwrite(f'{eval_gt_dir}/{frame_id:08d}.png', (gt_img_cropped * 255).astype(np.uint8))

        if ours_img is not None:
            ours_metrics.psnr += compute_psnr(ours_img, gt_img)
            ours_metrics.ssim += compute_ssim(ours_img, gt_img)
            ours_metrics.lpips += compute_lpips(ours_img_cropped, gt_img_cropped)
            ours_metrics.count += 1

    print('Ours metrics: ', ours_metrics)
    with open(results_file, 'a') as fp:
        fp.write(f"Ours metrics:\n{ours_metrics}\n")

    print('--- Ours ---')
    command = f'python -m pytorch_fid --device cuda {eval_ours_dir} {eval_gt_dir}'
    process = subprocess.run(command, shell=True, capture_output=True, text=True)

    output = process.stdout + process.stderr

    with open(results_file, 'a') as fp:
        fp.write("--- Ours ---\n")
        fp.write(output)



