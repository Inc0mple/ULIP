# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 09:10:08 2022

drafted script for 3D CNN on BraTS

https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

@author: yuya0
"""

## Setup environment
# conda install -c conda-forge monai
# also install numpy, matplotlib, and other optional packages etc...


#----- suggested from the website ------
#Compulsory dependencies:
#MONAI version: 0.4.0+618.g69b44596
#Numpy version: 1.20.3
#Pytorch version: 1.9.0a0+c3d40fd
#MONAI flags: HAS_EXT = False, USE_COMPILED = False
#MONAI rev id: 69b4459650fb6943b9e729e724254d2db2b2a1f2

#Optional dependencies:
#Pytorch Ignite version: 0.4.5 done 
#Nibabel version: 3.2.1
#scikit-image version: 0.15.0
#Pillow version: 8.3.1
#Tensorboard version: 2.5.0
#gdown version: 3.13.0
#TorchVision version: 0.10.0a0
#tqdm version: 4.53.0
#lmdb version: 1.2.1
#psutil version: 5.8.0
#pandas version: 1.1.4
#einops version: 0.3.0

#For details about installing the optional dependencies, please visit:
#    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies

#----- installed on the workstation (Yang) ------
#Compulsory dependencies:
#MONAI version: 1.1.0
#Numpy version: 1.24.3
#Pytorch version: 1.12.0
#MONAI flags: HAS_EXT = False, USE_COMPILED = False, USE_META_DICT = False
#MONAI rev id: 356d2d2f41b473f588899d705bbc682308cee52c
#MONAI __file__: /home/yang/anaconda3/envs/3D_CNN/lib/python3.9/site-packages/monai/__init__.py
#
#Optional dependencies:
#Pytorch Ignite version: 0.4.12
#Nibabel version: 5.1.0
#scikit-image version: 0.19.2
#Pillow version: 8.2.0
#Tensorboard version: NOT INSTALLED or UNKNOWN VERSION.
#gdown version: NOT INSTALLED or UNKNOWN VERSION.
#TorchVision version: 0.13.0
#tqdm version: 4.65.0
#lmdb version: NOT INSTALLED or UNKNOWN VERSION.
#psutil version: NOT INSTALLED or UNKNOWN VERSION.
#pandas version: 1.5.2
#einops version: NOT INSTALLED or UNKNOWN VERSION.
#transformers version: NOT INSTALLED or UNKNOWN VERSION.
#mlflow version: NOT INSTALLED or UNKNOWN VERSION.
#pynrrd version: 0.4.2
#matplotlib version: 3.7.1




## Setup import
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

import torch.nn as nn
from functools import partial
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai import transforms
from monai import data
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism

import torch

print_config()

# change working directory
os.chdir('/home/user/Documents/MONAI_3D_UNet_Yang_BraTS21_MICCAI 2023/')

# define GPU 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

## Setup data directory
directory = '/home/user/Documents/MONAI_3D_UNet_Yang_BraTS21_MICCAI 2023'
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
model_dir = '/home/user/Documents/MONAI_3D_UNet_Yang_BraTS21_MICCAI 2023/results_160_160_72_DP_uni_s1'

## Set deterministic training for reproducibility
set_determinism(seed=0)

## Define a new transform to convert brain tumor labels
# Here we convert the multi-classes labels into multi-labels segmentation task in One-Hot format.
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    This one is used for BraTS 20162017
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    
    This one is used for BraTS 2021
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
#            # merge label 4 and label 1 to construct TC
#            result.append(torch.logical_or(d[key] == 4, d[key] == 1))
#            # merge labels 4, 2 and 1 to construct WT
#            result.append(
#                torch.logical_or(
#                    torch.logical_or(d[key] == 4, d[key] == 1), d[key] == 2
#                )
#            )
            # label 1 is TC (original label 4 and label 1 added and converted)
            result.append(d[key] == 1)
            d[key] = torch.stack(result, axis=0).float()
        return d

## Setup folder reader
def datafold_read(datalist, basedir, fold_val=3, fold_test = 4, key="training"):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    test = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold_val:
            val.append(d)
        elif "fold" in d and d["fold"] == fold_test:
            test.append(d)
        else:
            tr.append(d)

    return tr, val, test

## Setup dataloader
def get_loader(batch_size, data_dir, json_list, fold_val, fold_test, roi):
    data_dir = data_dir
    datalist_json = json_list
    train_files, validation_files, test_files = datafold_read(
        datalist=datalist_json, basedir=data_dir, fold_val=fold_val, fold_test=fold_test
    )
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
           transforms.CropForegroundd(
               keys=["image", "label"],
               source_key="image",
               k_divisible=[roi[0], roi[1], roi[2]],
           ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
               random_size=False,
               random_center=False
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=False, channel_wise=True
            ),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=False, channel_wise=True
            ),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_ds = data.Dataset(data=test_files, transform=val_transform)
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds



## Set dataset root directory and hyper-parameters
data_dir = "/home/user/Documents/MONAI_3D_UNet_Yang_BraTS21_MICCAI 2023/data/brats2021challenge"
json_list = "/home/user/Documents/MONAI_3D_UNet_Yang_BraTS21_MICCAI 2023/data/brats2021challenge/brats21_folds_uni.json"
roi = (160, 160, 72)
batch_size = 16
#sw_batch_size = 4
fold_val = 3
fold_test = 4
#infer_overlap = 0.5
#max_epochs = 100
#val_every = 10
train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_loader(batch_size, data_dir, json_list, fold_val, fold_test, roi)


#Verified 'Task01_BrainTumour.tar', md5: 240a19d752f0d9e9101544901065d872.
#File exists: /workspace/data/medical/Task01_BrainTumour.tar, skipped downloading.
#Non-empty folder exists in /workspace/data/medical/Task01_BrainTumour, skipped extracting.


## Check data shape and visualize on validation
# pick one image from DecathlonDataset to visualize and check the 4 channels
train_data_example = train_ds[2]
print(f"image shape: {train_data_example['image'].shape}")
fig = plt.figure("image", (24, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(train_data_example["image"][i, :, :, 50].detach().cpu(), cmap="gray")
plt.show()
fig.savefig(os.path.join(model_dir, "Train_image_4channel.pdf"))
# also visualize the 3 channels label corresponding to this image
print(f"label shape: {train_data_example['label'].shape}")
fig = plt.figure("label", (18, 6))
for i in range(1):
    plt.subplot(1, 1, i + 1)
    plt.title(f"label channel {i}")
    plt.imshow(train_data_example["label"][i, :, :, 50].detach().cpu())
plt.show()
fig.savefig(os.path.join(model_dir, "Train_mask_1channel.pdf"))


## Create Model, Loss, Optimizer
max_epochs = 120
val_interval = 3
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")

print(device)
model = nn.DataParallel(SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=1,
    dropout_prob=0.2,
).to(device))
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)


# define inference method
def inference(input):

    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            # although the evaluation data is loaded in whole, the analysis (prediction) will be done using patching methods
            # that's why we have sliding window inference here to save the memory
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True


## Execute a typical PyTorch training process
best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
#metric_values_wt = []
#metric_values_et = []

#------------------------------------
total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    torch.save(
        model.state_dict(),
        os.path.join(model_dir, str(epoch+1)+"_best_metric_model_temp.pth"),
    )

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
#            metric_wt = metric_batch[1].item()
#            metric_values_wt.append(metric_wt)
#            metric_et = metric_batch[2].item()
#            metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(model_dir, "best_metric_model.pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} "
#                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

## Plot the loss and metric
fig = plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y, color="green")
plt.show()
fig.savefig(os.path.join(model_dir, "Train_Loss_and_Val_DICE_all.pdf"))

fig = plt.figure("train", (18, 6))
plt.subplot(1, 1, 1)
plt.title("Val Mean Dice TC")
x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
y = metric_values_tc
plt.xlabel("epoch")
plt.plot(x, y, color="blue")
#plt.subplot(1, 3, 2)
#plt.title("Val Mean Dice WT")
#x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
#y = metric_values_wt
#plt.xlabel("epoch")
#plt.plot(x, y, color="brown")
#plt.subplot(1, 3, 3)
#plt.title("Val Mean Dice ET")
#x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
#y = metric_values_et
#plt.xlabel("epoch")
#plt.plot(x, y, color="purple")
plt.show()
fig.savefig(os.path.join(model_dir, "Val_DICE_various.pdf"))

#------------------------------------

## Check best model output with the input image and label on testing data
model.load_state_dict(
    torch.load(os.path.join(model_dir, "best_metric_model.pth"))
)
model.eval()
with torch.no_grad():
    # select one image to evaluate and visualize the model output
    test_input = test_ds[6]["image"].unsqueeze(0).to(device)
    roi_size = (128, 128, 72)
    sw_batch_size = 4
    test_output = inference(test_input)
    test_output = post_trans(test_output[0])
    fig = plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(test_ds[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
    plt.show()
    fig.savefig(os.path.join(model_dir, "Test_image_4channel.pdf"))
    # visualize the 3 channels label corresponding to this image
    fig = plt.figure("label", (18, 6))
    for i in range(1):
        plt.subplot(1, 1, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(test_ds[6]["label"][i, :, :, 70].detach().cpu())
    plt.show()
    fig.savefig(os.path.join(model_dir, "Test_mask_1channel_truth.pdf"))
    # visualize the 3 channels model output corresponding to this image
    fig = plt.figure("output", (18, 6))
    for i in range(1):
        plt.subplot(1, 1, i + 1)
        plt.title(f"output channel {i}")
        plt.imshow(test_output[i, :, :, 70].detach().cpu())
    plt.show()
    fig.savefig(os.path.join(model_dir, "Test_mask_1channel_predict.pdf"))


## Evaluation on the testing set
# dataloader
test_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(
            keys="image", nonzero=True, channel_wise=True
        ),
    ]
)

post_transforms = Compose([
    Invertd(
        keys="pred",
        transform=test_transform,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
    Activationsd(keys="pred", sigmoid=True),
    AsDiscreted(keys="pred", threshold=0.5),
])

# inference
model.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model.pth")))
model.eval()

with torch.no_grad():
    for test_data in test_loader:
        test_inputs = test_data["image"].to(device)
#        test_labels = test_data["label"].to(device)
        test_data["pred"] = inference(test_inputs)
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]
        test_outputs, test_labels = from_engine(["pred", "label"])(test_data)

        test_labels[0] = test_labels[0].cuda()
        # here test_outputs is torch.cuda.Tensor (on GPU memory), and test_labels are torch.Tensor (on CPU memory)
        # so we have to add .cuda() to convert test_labels to torch.cuda.Tensor
        
#        print(test_outputs[0].type())
#        print(test_labels[0].type())
        dice_metric(y_pred=test_outputs, y=test_labels)
        dice_metric_batch(y_pred=test_outputs, y=test_labels)

    metric_org = dice_metric.aggregate().item()
    metric_batch_org = dice_metric_batch.aggregate()

    dice_metric.reset()
    dice_metric_batch.reset()

metric_tc = metric_batch_org[0].item()

#print("Metric on original image spacing: ", metric)
print("Metric on testing dataset: ", metric_org)
print(f"metric_tc: {metric_tc:.4f}")
#print(f"metric_wt: {metric_wt:.4f}")
#print(f"metric_et: {metric_et:.4f}")

















