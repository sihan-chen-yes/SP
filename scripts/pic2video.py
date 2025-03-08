import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import config
from moviepy.editor import ImageSequenceClip


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
    rgb_dir = os.path.join(dir, "rgb_map")

    fps = 24

    extensions = ['*.jpg', '*.png', '*.jpeg']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(rgb_dir, ext)))

    image_files = sorted(image_files)

    print(f"found {len(image_files)} rgb images")

    print(f"generating video...")
    clip = ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(os.path.join(rgb_dir, 'output.mp4'), codec='libx264')
