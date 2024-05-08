import os
import warnings

warnings.filterwarnings("ignore")

import time, cv2, torch, wandb
import argparse
import torch.distributed as dist
from config.diffconfig import DiffusionConfig, get_model_conf
from config.dataconfig import Config as DataConfig
from tensorfn import load_config as DiffConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
from tensorfn.optim import lr_scheduler
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import data as deepfashion_data
from model import UNet

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
    """
    Disables printing when not in master process
    """
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


def accumulate(model1, model2, decay=0.9999):
    """
    Accumulates the parameters of two models using exponential moving average.

    This function is used in the context of a diffusion model to accumulate the parameters
    of two models, `model1` and `model2`, using exponential moving average. The accumulated
    parameters are stored in `model1`.
    - `model1` is the EMA model and will be used during inference
    - `model2` is the training model and is updated with each training batch

    Parameters:
        model1 (torch.nn.Module): The first model whose parameters will be accumulated.
        model2 (torch.nn.Module): The second model whose parameters will be used for accumulation.
        decay (float, optional): The decay factor for the exponential moving average. Default is 0.9999.

    Returns:
        None
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def generate_samples(diffusion, ema, img, target_pose, args, betas, distributed, val):
    sample_type = "validation" if val else "training"
    print(f'Generating {sample_type} samples')
    sample_start_time = time.time()
    with torch.no_grad():
        if args.sample_algorithm == 'ddpm':
            samples = diffusion.p_sample_loop(ema, x_cond=[img, target_pose], progress=True, cond_scale=args.cond_scale)
        elif args.sample_algorithm == 'ddim':
            nsteps = 50
            noise = torch.randn(img.shape).cuda()
            seq = range(0, 1000, 1000 // nsteps)
            xs, x0_preds = ddim_steps(noise, seq, ema, betas.cuda(), [img, target_pose])
            samples = xs[-1].cuda()

    grid = torch.cat([img, target_pose[:, :3], samples], -1)
    if distributed:
        gathered_samples = [torch.zeros_like(grid) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, grid)
        if is_main_process():
            wandb.log({f'{sample_type}_samples': wandb.Image(torch.cat(gathered_samples, -2))})
    else:
        wandb.log({f'{sample_type}_samples': wandb.Image(grid)})

    print(f'Sampling took {int(time.time() - sample_start_time)} secs')


def train(conf, loader, val_loader, model, ema, diffusion, betas, optimizer, scheduler, args, wandb):
    device = args.device
    i = -1
    loss_list = []
    loss_mean_list = []
    loss_vb_list = []

    for epoch in range(args.epochs):
        if is_main_process: print(f'#Epoch - {epoch}')

        start_time = time.time()
        for batch in tqdm(loader):
            i = i + 1
            #? Forming a combined batch that includes both the original and target images, 
            #? Possibly for processing both directions in a single pass (source to target and target to source), enhancing the training efficiency.
            img = torch.cat([batch['source_image'], batch['target_image']], 0)  #? Not sure yet why these get concatenated
            target_img = torch.cat([batch['target_image'], batch['source_image']], 0)  #? Same here
            target_pose = torch.cat([batch['target_skeleton'], batch['source_skeleton']], 0)  #? Providing the model with both perspectives (source pose and target pose) in each batch

            img = img.to(device)
            target_img = target_img.to(device)
            target_pose = target_pose.to(device)
            time_t = torch.randint(  #? Randomly sample a time step from the diffusion schedule for each image in the batch. Shape: [batch_size]
                0,
                conf.diffusion.beta_schedule["n_timestep"],
                (img.shape[0],),
                device=device,
            )

            #? Calculate loss
            loss_dict = diffusion.training_losses(model, x_start = target_img, t = time_t, cond_input = [img, target_pose], prob = 1 - args.guidance_prob)
            loss = loss_dict['loss'].mean()
            loss_mse = loss_dict['mse'].mean()
            loss_vb = loss_dict['vb'].mean()
        
            #? Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            scheduler.step()
            optimizer.step()
            
            #? Logging
            loss = loss_dict['loss'].mean()
            loss_list.append(loss.detach().item())
            loss_mean_list.append(loss_mse.detach().item())
            loss_vb_list.append(loss_vb.detach().item())
            
            if conf.distributed:  #* Added this if else for single-GPU setup, there is no model.module if we're not using torch.distributed
                model_module = model.module
            else:
                model_module = model
            
            #? Accumulate model parameters using EMA
            accumulate(ema, model_module, 0 if i < conf.training.scheduler.warmup else 0.9999)

            if i % args.save_wandb_logs_every_iters == 0 and is_main_process():
                wandb.log({'loss': (sum(loss_list)/len(loss_list)),
                           'loss_vb': (sum(loss_vb_list)/len(loss_vb_list)),
                           'loss_mean': (sum(loss_mean_list)/len(loss_mean_list)),
                           'epoch': epoch,
                           'steps': i})
                loss_list = []
                loss_mean_list = []
                loss_vb_list = []


        if is_main_process():            
            if ((epoch % args.save_checkpoints_every_epochs == 0) or (epoch == args.epochs-1)):
                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "ema": ema.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "conf": conf,
                    },
                    conf.training.ckpt_path + f"/model_epoch_{str(epoch).zfill(3)}.pt")
                print(f'Saved model checkpoint after epoch {epoch} (step {i}) under {conf.training.ckpt_path}')

            if (epoch % args.save_wandb_images_every_epochs == 0) or (epoch == args.epochs-1):
                print(f'Generating samples after epoch {epoch} (step {i})')
                generate_samples(diffusion, ema, img, target_pose, args, betas, conf.distributed, val=False)  # ToDo: check whether just passing the img and pose like this is sufficient for training samples
                
                val_batch = next(val_loader)
                val_img = val_batch['source_image'].cuda()
                val_pose = val_batch['target_skeleton'].cuda()
                generate_samples(diffusion, ema, val_img, val_pose, args, betas, conf.distributed, val=True)
                
        print(f'Epoch Time: {int(time.time()-start_time)} secs')


def main(settings, EXP_NAME):
    [args, DiffConf, DataConf] = settings

    if is_main_process():
        wandb.init(project="DAVOS", entity="DAVOS-CV", name=EXP_NAME, settings=wandb.Settings(code_dir="."), config=vars(args))

    if DiffConf.ckpt is not None:  #? If training from checkpoint skip warmup
        DiffConf.training.scheduler.warmup = 0

    DiffConf.distributed = args.n_gpu > 1  #* This needs to be False for single GPU, was originally True
    local_rank = 0 if DiffConf.distributed == False else int(os.environ['LOCAL_RANK']) #* Changed from int(os.environ['LOCAL_RANK']) to 0 for single GPU

    #? args.batch_size probably denotes how many actual imgs are in a batch. During training we do both noising & denoising, so both directions are needed -> //2
    DataConf.data.train.batch_size = args.batch_size//2  #src -> tgt , tgt -> src

    #* distributed changed from True to False for single GPU running
    #* Might also need to create a new modules/PIDM/config/data.yaml and change the type, path & batch_size there
    #? deepfashion_data class name is confusing because its just data, it's supposedly fine with handling non-fashion data so maybe rename this
    val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required=True, distributed=DiffConf.distributed)

    def cycle(iterable):  #? Cycle through the dataset indefinitely
        while True:
            for x in iterable:
                yield x

    val_dataset = iter(cycle(val_dataset))  #? Iterator for the validation set

    #? Creates BeatGANsAutoencModels
    model = get_model_conf().make_model()
    model = model.to(args.device)
    ema = get_model_conf().make_model()
    ema = ema.to(args.device)

    if DiffConf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True
        )

    optimizer = DiffConf.training.optimizer.make(model.parameters())
    scheduler = DiffConf.training.scheduler.make(optimizer)

    if DiffConf.ckpt is not None:
        ckpt = torch.load(DiffConf.ckpt, map_location=lambda storage, loc: storage)

        if DiffConf.distributed:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])

        ema.load_state_dict(ckpt["ema"])
        scheduler.load_state_dict(ckpt["scheduler"])

        if is_main_process():
            print('Model loaded successfully!')

    betas = DiffConf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, predict_xstart = False)

    train(
        DiffConf, train_dataset, val_dataset, model, ema, diffusion, betas, optimizer, scheduler, args, wandb
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='pidm_deepfashion')
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf')
    parser.add_argument('--DataConfigPath', type=str, default='./config/data.yaml')
    parser.add_argument('--dataset_path', type=str, default='./dataset/deepfashion')
    parser.add_argument('--save_path', type=str, default='checkpoints')
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

    print(f'Experiment: {args.exp_name}')
    print(f'Diffusion Config Path: {args.DiffConfigPath}')
    print(f'Data Config Path: {args.DataConfigPath}')
    print(f'Dataset Path: {args.dataset_path}')
    print(f'Current path: {os.getcwd()}')
    print(f'Sampling algorithm used: {args.sample_algorithm.upper()}')

    #? Configuration objects storing things like paths, schedules, flags, hyperparameters etc...
    DiffConf = DiffConfig(DiffusionConfig,  args.DiffConfigPath, args.opts, False)
    DataConf = DataConfig(args.DataConfigPath)  # Gets set up using a yaml file

    DiffConf.training.ckpt_path = os.path.join(args.save_path, args.exp_name)
    DataConf.data.path = args.dataset_path

    if is_main_process():  #? Always returns True on single GPU
        if not os.path.isdir(args.save_path): os.mkdir(args.save_path)
        if not os.path.isdir(DiffConf.training.ckpt_path): os.mkdir(DiffConf.training.ckpt_path)

    #? Training from a checkpoint
    DiffConf.ckpt = args.checkpoint

    main(settings = [args, DiffConf, DataConf], EXP_NAME = args.exp_name)
