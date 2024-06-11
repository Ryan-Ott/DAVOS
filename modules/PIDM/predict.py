# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config as DiffConfig
import numpy as np
from config.diffconfig import DiffusionConfig, get_model_conf
import torch.distributed as dist
import os, glob, cv2, time, shutil
from models.unet_autoenc import BeatGANsAutoencConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
import torchvision.transforms as transforms
import torchvision
import data as deepfashion_data
from config.dataconfig import Config as DataConfig
import argparse


def generate_ex(diffusion, ema, img, target_pose, args, betas, distributed, val):
    sample_type = "validation" if val else "training"
    print(f'Generating {sample_type} samples')
    with torch.no_grad():
        if args.sample_algorithm == 'ddpm':
            samples = diffusion.p_sample_loop(ema, x_cond=[img, target_pose], progress=True, cond_scale=args.cond_scale)
        elif args.sample_algorithm == 'ddim':
            nsteps = 50
            noise = torch.randn(img.shape).cuda()
            seq = range(0, 1000, 1000 // nsteps)
            xs, x0_preds = ddim_steps(noise, seq, ema, betas.cuda(), [img, target_pose])
            samples = xs[-1].cuda()
    return samples

def init_distributed():
    """
    Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    """
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_main_process():
    #? Always true for single GPU
    try:
        if dist.get_rank()==0:
            return True
        else:
            return False
    except:
        return True


def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0
    
    while True:
        try:
            yield epoch, next(loader_iter)
        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)
            yield epoch, next(loader_iter)


class Predictor():
    def __init__(self, args, DataConf, DiffConf):
        """Load the model into memory to make running multiple predictions efficient"""

        conf = DiffConf

        self.model = get_model_conf().make_model()
        ckpt = torch.load("checkpoints/lisa/model_epoch_149.pt")  # change the checkpoint path here
        self.model.load_state_dict(ckpt["ema"])
        self.model = self.model.cuda()
        self.model.eval()

        self.betas = conf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart = False)
        DiffConf.distributed = False

        local_rank = 0 if DiffConf.distributed == False else int(os.environ['LOCAL_RANK']) 
        DataConf.data.train.batch_size = args.batch_size
        self.DiffConf = DiffConf
        self.val_dataset, _ = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required=True, distributed=DiffConf.distributed)
    def save_tensor(self, args):
        imgs, targets, samples = None, None, None
        for batch in tqdm(self.val_dataset, total=len(self.val_dataset)):
            device = args.device
            img = batch['image']
            target_img = batch['target']
            target_pose = batch['bcc']

            img = img.to(device)
            target_img = target_img.to(device)
            target_pose = target_pose.to(device)
            with torch.no_grad():
                if args.sample_algorithm == 'ddpm':
                    sample = generate_ex(self.diffusion, self.model, img, target_pose, args, self.betas, self.DiffConf.distributed, val=True)
            if imgs is None:
                imgs = img
                targets = target_pose
                samples = sample
            else:
                imgs = torch.cat((imgs, img), dim=0)
                targets = torch.cat((targets, target_pose), dim=0)
                samples = torch.cat((samples, sample), dim=0)
        dic = {'imgs':imgs, 'targets':targets, 'samples':samples}
        torch.save(dic, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='pidm_deepfashion')
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf')
    parser.add_argument('--DataConfigPath', type=str, default='./config/data.yaml')
    parser.add_argument('--dataset_path', type=str, default='./dataset/deepfashion')
    parser.add_argument('--save_path', type=str, default='outputs/output_data.pt')
    parser.add_argument('--cond_scale', type=int, default=2)
    parser.add_argument('--guidance_prob', type=int, default=0.1)
    parser.add_argument('--sample_algorithm', type=str, default='ddim')  # ddpm, ddim
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_wandb_logs_every_iters', type=int, default=50)
    parser.add_argument('--save_checkpoints_every_epochs', type=int, default=20)
    parser.add_argument('--save_wandb_images_every_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_gpu', type=int, default=8)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--checkpoint', type=str, default=None)  # 'checkpoints/last.pt'
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    
    if args.n_gpu > 1:
        init_distributed()  #* Necessary for multi-GPU

    if is_main_process():
        print(f'Using {args.n_gpu} GPUs')
        print(f'Experiment: {args.exp_name}')
        print(f'Diffusion Config Path: {args.DiffConfigPath}')
        print(f'Data Config Path: {args.DataConfigPath}')
        print(f'Dataset Path: {args.dataset_path}')
        print(f'Current path: {os.getcwd()}')
        print(f'Sampling algorithm used: {args.sample_algorithm.upper()}')

    DiffConf = DiffConfig(DiffusionConfig,  args.DiffConfigPath, args.opts, False)
    DataConf = DataConfig(args.DataConfigPath)  # Gets set up using a yaml file

    DiffConf.training.ckpt_path = os.path.join(args.save_path, args.exp_name)
    DataConf.data.path = args.dataset_path
    
    obj = Predictor(args, DataConf, DiffConf)

    obj.save_tensor(args)
    
    # ref_img = "data/deepfashion_256x256/target_edits/reference_img_0.png"
    # ref_mask = "data/deepfashion_256x256/target_mask/lower/reference_mask_0.png"
    # ref_pose = "data/deepfashion_256x256/target_pose/reference_pose_0.npy"

    # #obj.predict_appearance(image='test.jpg', ref_img = ref_img, ref_mask = ref_mask, ref_pose = ref_pose, sample_algorithm = 'ddim',  nsteps = 50)
