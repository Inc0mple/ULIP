'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * Changed from SLIP
 * https://github.com/facebookresearch/SLIP
 * By Le Xue
'''
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from skimage.transform import resize
from nltk.tokenize import word_tokenize
from torchvision.transforms.functional import pad
from torch.utils.data import Dataset
from PIL import Image
import nibabel as nib
import numpy as np
import pandas as pd
import os
import argparse
from collections import OrderedDict
import math
import time
import wandb

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import collections

import matplotlib.pyplot as plt
from numpy.random import randint

from data.dataset_3d import *

from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from data.dataset_3d import customized_collate_fn


def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP training and evaluation', add_help=False)
    # Data
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='shapenet', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    # parser.add_argument('--model', default='ULIP_PN_SSG', type=str)
    parser.add_argument('--model', default='ULIP_CUSTOMIZED', type=str)
    # Training
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')

    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')

    parser.add_argument('--test_ckpt_addr', default='',
                        help='the ckpt to test 3d zero shot')
    parser.add_argument('--augment', action='store_true',
                        help='perform random crop instead of simple resizing')
    return parser


best_acc1 = 0

# Computes the start and end indices for slicing a list (or a dimension of an array)
# in a way that grows outwards from the middle.
def compute_slice_indices(slice_num, max_idx):
    mid_idx = max_idx // 2
    slice_half_length = slice_num // 2
    start_idx = mid_idx - slice_half_length
    end_idx = mid_idx + slice_half_length
    # Ensure indices are within valid range
    start_idx = max(0, start_idx)
    end_idx = min(max_idx, end_idx)
    return start_idx, end_idx


class CustomTrainDataset_3Views(Dataset):
    def __init__(self, nifti_dir, txt_dir, png_dir_1, png_dir_2, png_dir_3, tokenizer, augment=False):
        self.tokenizer = tokenizer
        self.nifti_files = sorted(os.listdir(nifti_dir))
        self.png_files_1 = sorted(os.listdir(png_dir_1))
        self.png_files_2 = sorted(os.listdir(png_dir_2))
        self.png_files_3 = sorted(os.listdir(png_dir_3))
        self.txt_files = sorted(os.listdir(txt_dir))
        self.nifti_dir = nifti_dir
        self.png_dir_1 = png_dir_1
        self.png_dir_2 = png_dir_2
        self.png_dir_3 = png_dir_3
        self.txt_dir = txt_dir
        self.augment = augment

        # Compute the min_slice_number across all nifti files
        # self.min_slice_number = min(
        #     [nib.load(os.path.join(nifti_dir, f)).shape[2] for f in self.nifti_files])
        self.min_slice_number = 64
        
    def __len__(self):
        return len(self.nifti_files)

    def __getitem__(self, idx):
        # Load and process the NIfTI file
        nifti_path = os.path.join(self.nifti_dir, self.nifti_files[idx])
        # print(f"acessing {nifti_path}")
        nifti_img = nib.load(nifti_path).get_fdata()
        
        max_idx = nifti_img.shape[2]  # Assuming the slice is along the third dimension
        
        
        if self.augment: # Resize to 256 256 n, then random crop to 96 96 96
            # First resize to 256, 256, slice_num
            # nifti_img = resize(nifti_img, (96, 96, max_idx))
            nifti_img = resize(nifti_img, (128, 128, max_idx))

            # Random crop to (96, 96, 96)
            # start_x = randint(0, nifti_img.shape[0] - 96)
            # start_y = randint(0, nifti_img.shape[1] - 96)
            # start_z = randint(0, nifti_img.shape[2] - 96)
            
            # start_x = randint(0, nifti_img.shape[0] - 96)
            # start_y = randint(0, nifti_img.shape[1] - 96)
            start_z = randint(0, nifti_img.shape[2] - self.min_slice_number)
            
            # nifti_img = nifti_img[start_x:start_x+96,
            #                       start_y:start_y+96, start_z:start_z+96]

            nifti_img = nifti_img[:, :, start_z:start_z+self.min_slice_number]
        else: # simple resizing to 96 96 with the middle 96 slices
            start_idx, end_idx = compute_slice_indices(
                self.min_slice_number, max_idx)
            nifti_img = nifti_img[:, :, start_idx:end_idx]
            nifti_img = resize(nifti_img, (96, 96, 96))
        
        # Load and process the PNG file
        png_path_1 = os.path.join(self.png_dir_1, self.png_files_1[idx])
        png_path_2 = os.path.join(self.png_dir_2, self.png_files_2[idx])
        png_path_3 = os.path.join(self.png_dir_3, self.png_files_3[idx])
        
        png_img_1 = np.array(Image.open(png_path_1))
        # Had to resize from 512 to 224 for the ULIP image model
        png_img_1 = resize(png_img_1, (224, 224))
        # We will use the np.newaxis to add a dimension at the beginning
        # and then use np.repeat to duplicate the image across three channels
        png_img_1 = np.repeat(png_img_1[np.newaxis, :, :], 3, axis=0)
        
        png_img_2 = np.array(Image.open(png_path_2))
        png_img_2 = resize(png_img_2, (224, 224))
        png_img_2 = np.repeat(png_img_2[np.newaxis, :, :], 3, axis=0)
        
        png_img_3 = np.array(Image.open(png_path_3))
        png_img_3 = resize(png_img_3, (224, 224))
        png_img_3 = np.repeat(png_img_3[np.newaxis, :, :], 3, axis=0)

        # Load and process the text file
        txt_path = os.path.join(self.txt_dir, self.txt_files[idx])
        with open(txt_path, 'r') as file:
            txt = file.read()
        tokens = self.tokenizer(txt)

        return nifti_img, tokens, png_img_1, png_img_2, png_img_3
    
    
class CustomTrainDataset(Dataset):
    def __init__(self, nifti_dir, txt_dir, png_dir, tokenizer, augment=False):
        self.tokenizer = tokenizer
        self.nifti_files = sorted(os.listdir(nifti_dir))
        self.png_files = sorted(os.listdir(png_dir))
        self.txt_files = sorted(os.listdir(txt_dir))
        self.nifti_dir = nifti_dir
        self.png_dir = png_dir
        self.txt_dir = txt_dir
        self.augment = augment

        # Compute the min_slice_number across all nifti files
        # self.min_slice_number = min(
        #     [nib.load(os.path.join(nifti_dir, f)).shape[2] for f in self.nifti_files])
        self.min_slice_number = 96

    def __len__(self):
        return len(self.nifti_files)

    def __getitem__(self, idx):
        # Load and process the NIfTI file
        nifti_path = os.path.join(self.nifti_dir, self.nifti_files[idx])
        # print(f"acessing {nifti_path}")
        nifti_img = nib.load(nifti_path).get_fdata()

        # Assuming the slice is along the third dimension
        max_idx = nifti_img.shape[2]

        if self.augment:  # Resize to 256 256 n, then random crop to 96 96 96
            # First resize to 256, 256, slice_num
            # nifti_img = resize(nifti_img, (96, 96, max_idx))
            nifti_img = resize(nifti_img, (96, 96, max_idx))

            # Random crop to (96, 96, 96)
            # start_x = randint(0, nifti_img.shape[0] - 96)
            # start_y = randint(0, nifti_img.shape[1] - 96)
            # start_z = randint(0, nifti_img.shape[2] - 96)

            # start_x = randint(0, nifti_img.shape[0] - 96)
            # start_y = randint(0, nifti_img.shape[1] - 96)
            start_z = randint(0, nifti_img.shape[2] - 96)

            # nifti_img = nifti_img[start_x:start_x+96,
            #                       start_y:start_y+96, start_z:start_z+96]

            nifti_img = nifti_img[:, :, start_z:start_z+96]
        else:  # simple resizing to 96 96 with the middle 96 slices
            start_idx, end_idx = compute_slice_indices(
                self.min_slice_number, max_idx)
            nifti_img = nifti_img[:, :, start_idx:end_idx]
            nifti_img = resize(nifti_img, (96, 96, 96))

        # Load and process the PNG file
        png_path = os.path.join(self.png_dir, self.png_files[idx])
        png_img = np.array(Image.open(png_path))
        # Had to resize from 512 to 224 for the ULIP image model
        png_img = resize(png_img, (224, 224))

        # We will use the np.newaxis to add a dimension at the beginning
        # and then use np.repeat to duplicate the image across three channels
        png_img = np.repeat(png_img[np.newaxis, :, :], 3, axis=0)

        # Load and process the text file
        txt_path = os.path.join(self.txt_dir, self.txt_files[idx])
        with open(txt_path, 'r') as file:
            txt = file.read()
        tokens = self.tokenizer(txt)

        return nifti_img, png_img, tokens


class CustomValDataset(Dataset):
    def __init__(self, nifti_val_dir, labels_csv):
        self.nifti_files = sorted(os.listdir(nifti_val_dir))
        self.nifti_val_dir = nifti_val_dir
        self.labels_df = pd.read_csv(labels_csv)
        # self.min_slice_number = min([nib.load(os.path.join(nifti_dir, f)).shape[2] for f in self.nifti_files])
        self.min_slice_number = 64

    def __len__(self):
        return len(self.nifti_files)

    def __getitem__(self, idx):
        # Load and process the NIfTI file
        nifti_path = os.path.join(self.nifti_val_dir, self.nifti_files[idx])
        # print(f"acessing {nifti_path}")
        nifti_img = nib.load(nifti_path).get_fdata()
        
        # Assuming the slice is along the third dimension
        max_idx = nifti_img.shape[2]
        start_idx, end_idx = compute_slice_indices(self.min_slice_number, max_idx)
        nifti_img = nifti_img[:, :, ]

        nifti_img = nifti_img[:, :, start_idx:end_idx]
        # nifti_img = resize(nifti_img, (96, 96, 96))
        nifti_img = resize(nifti_img, (128, 128, 64))
        
        filename = self.nifti_files[idx].split(".")[0]
        
        # Extract labels and label_name from the DataFrame
        label = int(self.labels_df.loc[self.labels_df['filename'] == filename, 'labels'].values[0])
        label_name = self.labels_df.loc[self.labels_df['filename'] == filename, 'label_name'].values[0]

        return nifti_img, label, label_name

def main(args):
    start_time = time.time()
    utils.init_distributed_mode(args)

    global best_acc1

    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='ULIP', id=wandb_id, config=args, reinit=True, entity='lxue')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.evaluate_3d:
        zero_stats = test_zeroshot_3d(args)
        print(zero_stats)
        return

    # create model
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args=args)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=False)

    # define loss function (criterion) and optimizer
    # criterion = models.get_loss(args).cuda(args.gpu)
    criterion = models.get_loss_3_views(args).cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            print('in optimizer freeze {}'.format(n))
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from the latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])

    # train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    # val_dataset = get_dataset(None, tokenizer, args, 'val')
    nifti_path = "NNI_Data/3D_stack_nii"
    png_path = "NNI_Data/2D_projection_AP_png"
    text_path = "NNI_Data/Report_txt"
    
    png_path_AP = "NNI_Data/2D_projection_AP_png"
    png_path_LR = "NNI_Data/2D_projection_LR_png"
    png_path_SI = "NNI_Data/2D_projection_SI_png"
    
    nifti_path_valid = "NNI_Data_valid/3D_stack_nii"
    labels_path_valid = "NNI_Data_valid/labels.csv"
    
    # train_dataset = CustomTrainDataset(nifti_path, text_path, png_path, SimpleTokenizer(), augment=args.augment)
    train_dataset = CustomTrainDataset_3Views(
        nifti_path, text_path, png_path_AP, png_path_LR, png_path_SI, SimpleTokenizer(), augment=args.augment)
    val_dataset = CustomValDataset(nifti_path_valid, labels_path_valid)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    print("1")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        collate_fn=customized_collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(
    #         train_sampler is None), drop_last=False)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None), drop_last=False)

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    print(args)

    print("=> beginning training")
    print(f"Batch_size: {args.batch_size}")

    best_epoch = -1
    
    logs_list = []

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)
        val_stats = {"acc1": -1}

        if epoch % 1 == 0:

            val_stats = test_zeroshot_3d_core(val_loader, model, tokenizer, args)
            acc1 = val_stats["acc1"]
            print(val_stats)

            is_best = acc1 > best_acc1
            if is_best:
                best_epoch = epoch

            best_acc1 = max(acc1, best_acc1)

            if is_best or epoch % 50 == 0:
                print("=> saving checkpoint")
                utils.save_on_master({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc1': best_acc1,
                        'args': args,
                    }, is_best, args.output_dir)

            if epoch + 1 == args.epochs:
                print("=> saving last checkpoint")
                utils.save_on_master({
                    'epoch': 'last',
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'best_acc1': best_acc1,
                     'best_epoch': best_epoch}
        
        logs_list.append(log_stats)
        
        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
                # wandb.watch(model)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
                
    

    print("Saving graphs...")
    graphs_folder = os.path.join(args.output_dir, "graphs")
    os.makedirs(graphs_folder, exist_ok=True)
    keys = set().union(*logs_list)
    keys.remove('epoch')

    # For each key, create a plot
    for key in keys:
        epochs = [d['epoch'] for d in logs_list]
        values = [d[key] for d in logs_list]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, values, marker='o')
        plt.title(f'{key} across epochs')
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.grid(True)

        # Save the figure in the output folder
        plt.savefig(os.path.join(graphs_folder, f'{key}.png'))
        plt.close()
    end_time = time.time()
    
    # Calculate the total execution time
    total_time = end_time - start_time

    # Open the text file in write mode
    with open(os.path.join(args.output_dir, "execution_time.txt"), "w") as file:
        # Write the execution time to the file
        file.write(f"The script executed in {total_time} seconds.")


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    # metric_names = models.get_metric_names(args.model)
    metric_names = models.get_metric_names_3_views(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]

        # pc = inputs[3]
        # texts = inputs[2]
        # image = inputs[4]
        
        # Changed
        # pc = inputs[0]
        # # Needed to prevent einops.EinopsError: Expected 5 dimensions, got 4
        # pc = torch.unsqueeze(pc, 1)
        # image = inputs[1]
        # texts = inputs[2]
        
        pc = inputs[0]
        # Needed to prevent einops.EinopsError: Expected 5 dimensions, got 4
        pc = torch.unsqueeze(pc, 1)
        texts = inputs[1]
        image_1 = inputs[2]
        image_2 = inputs[3]
        image_3 = inputs[4]
        inputs = [[inputs]]
        
        # print(
        #     f"inputs length: {len(inputs)} | pc.shape: {pc.shape} | image.shape: {image.shape} | texts.shape: {texts.shape}")
        
        inputs = [pc, texts, image_1, image_2, image_3]

        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
        # print(
        #     f"inputs length after line 398: {len(inputs)} | pc.shape: {pc.shape} | image.shape: {image.shape} | texts.shape: {texts.shape}")
        
        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]

        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale(),
                        'logit': logit_scale})
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def test_zeroshot_3d_core(test_loader, model, tokenizer, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top2 = AverageMeter('Acc@2', ':6.2f')
    f1_meter = AverageMeter('F1_Macro@1', ':6.2f')
    precision_meter = AverageMeter('Precision@1', ':6.2f')
    recall_meter = AverageMeter('Recall@1', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, f1_meter, precision_meter, recall_meter],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    print('=> encoding captions')
    with open(os.path.join("./NNI_Data_valid", 'templates_v2.json')) as f:
        # templates = json.load(f)[args.validate_dataset_prompt]
        templates = json.load(f)["MRA"]

    with open(os.path.join("./NNI_Data_valid", 'labels.json')) as f:
        # labels = json.load(f)[args.validate_dataset_name]
        labels = json.load(f)["MRA"]


    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = utils.get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        # per_class_correct_top2 = collections.defaultdict(int)

        for i, (pc, target, target_name) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            # Needed to prevent einops.EinopsError: Expected 5 dimensions, got 4
            pc = torch.unsqueeze(pc, 1)

            # encode pc
            pc_features = utils.get_model(model).encode_pc(pc)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_pc = pc_features @ text_features.t()

            # measure accuracy and record loss
            (acc1), (f1), (precision), (recall), correct = accuracy(logits_per_pc, target, topk=(1,))
            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, f1, precision, recall = utils.scaled_all_reduce([acc1, f1, precision, recall])
            print(acc1, f1, precision, recall)
            top1.update(acc1[0].item(), pc.size(0))
            # top2.update(acc2[0].item(), pc.size(0))
            
            # Our own metrics for imbalanced classes
            f1_meter.update(f1[0].item(), pc.size(0))
            precision_meter.update(precision[0].item(), pc.size(0))
            recall_meter.update(recall[0].item(), pc.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            top1_accurate = correct[:1].squeeze()
            # top2_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            # print(f"top1_accurate: {top1_accurate}")
            for idx, name in enumerate(target_name):
                
                if len(top1_accurate.shape) > 0:  # Check if tensor is not 0-dimensional
                    if top1_accurate[idx].item():
                        per_class_correct_top1[name] += 1
                else:
                    if top1_accurate.item():
                        per_class_correct_top1[name] += 1
                # if top1_accurate[idx].item():
                #     per_class_correct_top1[name] += 1
                # if len(top2_accurate.shape) > 0:  # Check if tensor is not 0-dimensional
                #     if top2_accurate[idx].item():
                #         per_class_correct_top2[name] += 1
                # else:
                #     if top2_accurate.item():
                #         per_class_correct_top2[name] += 1
                # if top2_accurate[idx].item():
                #     per_class_correct_top2[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        # top2_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            # top2_accuracy_per_class[name] = per_class_correct_top2[name] / per_class_stats[name]

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        # top2_accuracy_per_class = collections.OrderedDict(top2_accuracy_per_class)
        print(','.join(top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in top1_accuracy_per_class.values()]))
        # print(','.join([str(value) for value in top2_accuracy_per_class.values()]))

    progress.synchronize()
    print(
        '0-shot * Acc@1 {top1.avg:.3f} f1 {f1_meter.avg:.3f} precision {precision_meter.avg:.3f} recall {recall_meter.avg:.3f}')
    return {'acc1': top1.avg, 'f1': f1_meter.avg, 'precision': precision_meter.avg, 'recall': recall_meter.avg}

def test_zeroshot_3d(args):
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    try:
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))

    tokenizer = SimpleTokenizer()

    test_dataset = get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )
    results = test_zeroshot_3d_core(test_loader, model, tokenizer, args)

    return results


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print("output: ", output)
        # print("target: ", target)
        # print("pred: ", pred)
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res_acc = []
        res_f1 = []
        res_precision = []
        res_recall = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_acc.append(correct_k.mul_(100.0 / batch_size))
            
            pred_k = pred[:k].reshape(-1)
            f1_k = f1_score(target.cpu().numpy(),
                            pred_k.cpu().numpy(), average='macro', zero_division=0)
            precision_k = precision_score(
                target.cpu().numpy(), pred_k.cpu().numpy(), zero_division=0)
            recall_k = recall_score(
                target.cpu().numpy(), pred_k.cpu().numpy(), zero_division=0)
            res_f1.append(torch.tensor(f1_k).to(target.device))
            res_precision.append(torch.tensor(precision_k).to(target.device))
            res_recall.append(torch.tensor(recall_k).to(target.device))
        return res_acc, res_f1, res_precision, res_recall, correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
