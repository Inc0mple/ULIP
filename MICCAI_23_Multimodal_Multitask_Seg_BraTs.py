#!/usr/bin/env python
# coding: utf-8
# with help from Jiahao
# created by Yu Yang
# Date Apr 2021
# update Date Oct 2021
# update Date Jan 2023

# updated with mixed supervision
# updated with 2023 MICCAI sub
# image abd mask for FA data
# image and label for WA data

# In[1]: import module


import torch.utils.data as Data
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses
import segmentation_models_pytorch.utils as smp_utils
#from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import datetime
import random
import sys
#from torchsummary import summary
#from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
from copy import deepcopy
import torchvision.utils as vutils


# In[2]: define GPU


#get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
#get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')
torch.set_num_threads(6)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[2]: define seed (global, just for guidance, have to re-call them everytime when it's needed

seed_cus = 4

random.seed(seed_cus)
np.random.seed(seed_cus)
torch.manual_seed(seed_cus)
torch.cuda.manual_seed(seed_cus)
torch.cuda.manual_seed_all(seed_cus)

#torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#os.environ['PYTHONHASHSEED'] = str(2)

# may want to check/print the train/val data first

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed_cus)


# In[3]: define and load pretrained model

## define axu parameters for model
#aux_params=dict(
#    pooling='avg',             # one of 'avg', 'max'
#    dropout=0.5,               # dropout ratio, default is None
#    activation='sigmoid',      # activation function, default is None
#    classes=1,                 # define number of output labels
#)

# load model with classification head
model_inf = smp.Unet('resnext50_32x4d',encoder_weights='imagenet', in_channels=3, classes=2)
model_rev = smp.Unet('resnext50_32x4d',encoder_weights='imagenet', in_channels=4, classes=2)

class Dual_NN(nn.Module):
    """Network to learn text representations"""
    def __init__(self):
        super(Dual_NN, self).__init__()
        self.inf_net = model_inf
        self.rev_net = model_rev
#        self.linearLayer3 = nn.Linear(minus_one_dim, output_dim)

    def forward(self, img, img_noi):
        inf_predict = self.inf_net(img)
        rev_predict = self.rev_net(img_noi)
        return inf_predict, rev_predict

model_phx = Dual_NN()

# this is for running data on all GPU for linux workstation
model_phx= nn.DataParallel(model_phx)
model_phx.to(device)

# load pre-trained model from classifictaion  if available
#model_phx.load_state_dict(torch.load('/home/yang/Documents/SIIM-ACR pneumothorax/Mixed-supervision-TD filing/Classification results/unet-resnext50-mixed-superv-base_4FC.pt'))

# In[4]: define dataloader


import torch.utils.data as Data
class My_Datasets(Data.Dataset):
    def __init__(self, img_dir, maskN_dir, maskG_dir, transform1=None,transform2=None):
        super().__init__()
        self.img_dir = img_dir
        self.maskN_dir = maskN_dir
        self.maskG_dir = maskG_dir


        self.img_list = os.listdir(self.img_dir)
        self.maskN_list = os.listdir(self.maskN_dir)
        self.maskG_list = os.listdir(self.maskG_dir)

        
        self.transform1 = transform1
        self.transform2 = transform2
      
    def __getitem__(self, index):
        img_name = self.img_list[index]
        maskN_name = self.maskN_list[index]
        maskG_name = self.maskG_list[index]
        
        if img_name.endswith('.png'):
            img = cv2.imread(os.path.join(self.img_dir,self.img_list[index])) # here we load as RGB
            maskN= cv2.imread(os.path.join(self.maskN_dir,self.maskN_list[index])) # here we load as RGB
            maskG= cv2.imread(os.path.join(self.maskG_dir,self.maskG_list[index])) # here we load as RGB
            

            LABImg = img/[255.0]
#            LABmaskN = maskN/[255.0]
#            LABmaskG = maskG/[255.0]


            LABImg = cv2.resize(LABImg,(224,224))
            LABmaskN = cv2.resize(maskN,(224,224))
            LABmaskG = cv2.resize(maskG,(224,224))


            LABImg = LABImg.astype(np.float32)
            LABmaskN = LABmaskN.astype(np.float32)
            LABmaskN = LABmaskN[:,:,0]
            LABmaskG = LABmaskG.astype(np.float32)
            LABmaskG = LABmaskG[:,:,0]


               
        if self.transform1:
            LABImg = self.transform1(LABImg)
            LABmaskN = self.transform2(LABmaskN)
            LABmaskG = self.transform2(LABmaskG)
        
#        print(LABImg.shape)
#        print(LABmaskN.shape)
        LABImgN = torch.cat((LABImg, LABmaskN), 0) # combine all channels together
#        print(LABImgN.shape)
        
        return LABImg, LABImgN, LABmaskG 
    
    def __len__(self):
        return len(self.img_list)


# In[5]: define transform


transforms_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transforms_mask = transforms.Compose([
    transforms.ToTensor(),
])


# In[6]: define data path and load data


img_path = 'D:/PhD and Work/I2R/MICCAI 2023_main_BraTs/data_processed/train_val/img/'
maskN_path = 'D:/PhD and Work/I2R/MICCAI 2023_main_BraTs/data_processed/train_val/mask_noi_new/'
maskG_path = 'D:/PhD and Work/I2R/MICCAI 2023_main_BraTs/data_processed/train_val/mask_gro/'


batch_size_train = 8
batch_size_val = 8

train_set = My_Datasets(img_path,maskN_path, maskG_path, transform1=transforms_img,transform2=transforms_mask)
n_train = len(train_set)
split = n_train // 5
a = list(range(n_train))
# define fixed seed
random.seed(seed_cus)
indices = random.sample(a, len(a))
# define fixed seed
torch.manual_seed(seed_cus)
torch.cuda.manual_seed(seed_cus)
torch.cuda.manual_seed_all(seed_cus)
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
# define fixed seed
torch.manual_seed(seed_cus)
torch.cuda.manual_seed(seed_cus)
torch.cuda.manual_seed_all(seed_cus)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
train_loader = Data.DataLoader(
    dataset=train_set,
    sampler=train_sampler,
    num_workers=0,
    batch_size=batch_size_train,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g)
valid_loader = Data.DataLoader(
    dataset=train_set,
    sampler=valid_sampler,
    num_workers=0,
    batch_size=batch_size_val,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g)


# In[9]: define training para


num_epoch = 40
learning_rate = 1e-2
#writer = SummaryWriter('D:/PhD and Work/I2R\SIIM-ACR pneumothorax/Pseudo-labeling Yang V2-with lung seg prior/log/test')
BATCH_SIZE = 2
# criterion = nn.BCEWithLogitsLoss()
#criterion_FA = smp.utils.losses.DiceLoss() # default loss in util, not working with multi class problem
criterion_sup = smp_losses.dice.DiceLoss(smp_losses.MULTICLASS_MODE, from_logits=True) # new loss (multi) in losses
criterion_dis = nn.MSELoss()
#criterion_WA = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(),
#                           lr=learning_rate)
# optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8)

# remember to add model_phx() to make it an instance instead of simple class if use self defined net
optimizer = optim.SGD(model_phx.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=4e-5)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=2, total_epoch=5, after_scheduler=scheduler_cosine)
device = torch.device('cuda')


# In[10]: define training


def train(epoch, alpha, beta, gamma):
    loss_sum = 0
    prev_time = time.time()
    for i, (images, imagesN, masksG) in enumerate(train_loader):
        imgs = images.to(device=device,dtype=torch.float32)
        imgsN = imagesN.to(device=device,dtype=torch.float32)
        masksG = masksG.to(device=device,dtype=torch.float32)
        
        masks_inf, masks_rev = model_phx(imgs, imgsN) # seg output

#        masks_pred = model_phx(imgs)
#        masksA_pred = model_phx(imgsA)
        loss_seg_inf = alpha * criterion_sup(masks_inf, masksG) 
        loss_seg_rev = beta * criterion_sup(masks_rev, masksG) 
        loss_seg_dis = gamma * criterion_dis(masks_inf, masks_rev) 
        
        loss_all = loss_seg_inf + loss_seg_rev + loss_seg_dis
#       
        iter_loss = loss_all.data.item() 
        loss_sum += loss_all.item()
#        writer.add_scalar('loss/train',iter_loss,epoch*len(train_loader) + i)
        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        
        batches_done = epoch * len(train_loader) + i
        batches_left = num_epoch * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss_seg_inf: %f] [loss_seg_rev: %f] [loss_seg_dis: %f] [loss: %f]  ETA: %s"
            % (
                epoch,
                num_epoch,
                i,
                len(train_loader),
                loss_seg_inf.item(),
                loss_seg_rev.item(),
                loss_seg_dis.item(),
                loss_all.item(),
                time_left,
            )
        )


# In[10]: define validation


def val(epoch, alpha, beta, gamma):
    loss_sum = 0
    global best_seg_loss
    running_loss_all = 0.0
    running_loss_seg_inf = 0.0
    running_loss_seg_rev = 0.0
    running_loss_seg_dis = 0.0
    prev_time = time.time()
    for i, (images, imagesN, masksG) in enumerate(valid_loader):
        imgs = images.to(device=device,dtype=torch.float32)
        imgsN = imagesN.to(device=device,dtype=torch.float32)
        masksG = masksG.to(device=device,dtype=torch.float32)
        batch_size = imgs.shape[0]
        
        with torch.no_grad():
            model_phx.eval()
        
            masks_inf, masks_rev = model_phx(imgs, imgsN) # seg output
    #        masks_pred = model_phx(imgs)
    #        masksA_pred = model_phx(imgsA)
            loss_seg_inf = alpha * criterion_sup(masks_inf, masksG) 
            loss_seg_rev = beta * criterion_sup(masks_rev, masksG) 
#            print(masks_inf.shape)
#            print(masks_inf[0].shape)
#            print(masks_inf[:,1,:,:].shape)
            loss_seg_dis = gamma * criterion_dis(masks_inf, masks_rev)  
            
            loss_all = loss_seg_inf + loss_seg_rev + loss_seg_dis
    #       
            iter_loss = loss_all.data.item() 
            loss_sum += loss_all.item()
            running_loss_seg_inf += loss_seg_inf.item() * batch_size
            running_loss_seg_rev += loss_seg_rev.item() * batch_size
            running_loss_seg_dis += loss_seg_dis.item() * batch_size
            running_loss_all += loss_all.item() * batch_size
        
            # this is used to self-examine the results for val
    #        writer.add_scalar('loss/train',iter_loss,epoch*len(train_loader) + i)
            batches_done = epoch * len(valid_loader) + i
            batches_left = num_epoch * len(valid_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss_seg_inf: %f] [loss_seg_rev: %f] [loss_seg_dis: %f] [loss: %f]  ETA: %s"
            % (
                epoch,
                num_epoch,
                i,
                len(valid_loader),
                loss_seg_inf.item(),
                loss_seg_rev.item(),
                loss_seg_dis.item(),
                loss_all.item(),
                time_left,
            )
        )
    
    # len of valid_loader here is the total number of image, thus divided by 10
    epoch_loss_seg_inf = running_loss_seg_inf / (len(valid_loader.dataset)/5)
    epoch_loss_seg_rev = running_loss_seg_rev / (len(valid_loader.dataset)/5)
    epoch_loss_seg_dis = running_loss_seg_dis / (len(valid_loader.dataset)/5)
    epoch_loss_all = running_loss_all / (len(valid_loader.dataset)/5)
    
    # skip one line    
    keyword = os.linesep 
    print(keyword)  
    
    sys.stdout.write(
        "[Epoch %d/%d] [loss_seg_inf: %f] [loss_seg_rev: %f] [loss_seg_dis: %f] [loss: %f]"
        % (
            epoch,
            num_epoch,
            epoch_loss_seg_inf,
            epoch_loss_seg_rev,
            epoch_loss_seg_dis,
            epoch_loss_all,
#                    time_left,
        )
    )

    epoch_loss_seg_all = epoch_loss_all
    if epoch_loss_seg_all < best_seg_loss:
        best_seg_loss = epoch_loss_seg_all
#                best_epoch = epoch
        torch.save(model_phx.state_dict(), 'D:/PhD and Work/I2R/MICCAI 2023_main_BraTs/exp_3_dual_label_seg/results/unet-resnext50-fully-superv-multiclass-smploss-s2.pt')

# In[11]: to train and valid the model


criterion_sup = criterion_sup.cuda()
criterion_dis = criterion_dis.cuda()

alpha = 1 # weight for seg inf loss
beta = 1 # weight for seg rev loss
gamma = 0.01 # weight for seg dis loss
# remember to add model_phx() to make it an instance instead of simple class if use self defined net
model_phx = model_phx.cuda()

    
best_seg_loss = 999999
for epoch in range(num_epoch):
    print('Starting training epoch {}/{}.'.format(epoch + 1, num_epoch))
    # remember to add model_phx() to make it an instance
    model_phx.train()
    train(epoch, alpha, beta, gamma)
    scheduler_warmup.step()
    print(scheduler_warmup.get_lr())
    print('Starting validation epoch {}/{}.'.format(epoch + 1, num_epoch))
    model_phx.eval()
    val(epoch, alpha, beta, gamma)
#    print('\n') # newline
    print(scheduler_warmup.get_lr())
    print('Best all seg loss for validation dataset: {}.'.format(best_seg_loss))
#    print('\n') # newline


# In[11]: to save the model
#torch.save(model_phx.state_dict(), 'D:/PhD and Work/I2R/SIIM-ACR pneumothorax/Mixed-supervision/unet-resnext50-mixed-superv-V1.pt')

# In[12]: iterate the data


img, imgN, mask = next(iter(valid_loader))


# In[12.1]: load trained model

#model_test = smp.Unet('resnext50_32x4d',encoder_weights='imagenet',classes=1,activation='sigmoid')
model_phx.load_state_dict(torch.load('D:/PhD and Work/I2R/MICCAI 2023_main_BraTs/exp_3_dual_label_seg/results/unet-resnext50-fully-superv-multiclass-smploss-s2.pt'))
# remember to add model_phx() to make it an instance instead of simple class if use self defined net
model_test = model_phx.cuda()


# In[13]: define prediction


with torch.no_grad():
    model_test.eval()
    mask_inf, mask_rev = model_test(img.cuda(), imgN.cuda()) # for FA images input
    
#    pre_2, pre_2A = torch.sigmoid(pre, preA)


# In[17]: show groud truth labels and predicted labels
#print(labelC)
#print(label_preC)


# In[18]: show original image


plt.figure(figsize=(64,64))
plt.axis("off")
plt.title("Testing Images")
plt.imshow(np.transpose(vutils.make_grid(img[0:], padding=2, normalize=True).cpu(),(1,2,0)))
fig = plt.imshow(np.transpose(vutils.make_grid(img[0:], padding=2, normalize=True).cpu(),(1,2,0)))
fig.figure.savefig("D:/PhD and Work/I2R/MICCAI 2023_main_BraTs/exp_3_dual_label_seg/results/Fully S-multiclass-smploss-s2/Testing Images V1.png", dpi=224)

# In[15]: show mask


plt.figure(figsize=(64,64))
plt.axis("off")
plt.title("Annotation masks")
plt.imshow(np.transpose(vutils.make_grid(mask[0:], padding=2, normalize=True).cpu(),(1,2,0)))
fig = plt.imshow(np.transpose(vutils.make_grid(mask[0:], padding=2, normalize=True).cpu(),(1,2,0)))
fig.figure.savefig("D:/PhD and Work/I2R/MICCAI 2023_main_BraTs/exp_3_dual_label_seg/results/Fully S-multiclass-smploss-s2/Annotation masks V1.png", dpi=224)


# In[19]: show predicted results
# here we have to use mask_pre[0:,1].unsqueeze(1) - select 2nd channel as mask output and add one more dimension for 16,1,224,224 for image display

plt.figure(figsize=(64,64))
plt.axis("off")
plt.title("Predicted results")
plt.imshow(np.transpose(vutils.make_grid(mask_inf[0:,1].unsqueeze(1), padding=2, normalize=True).cpu(),(1,2,0)))
fig = plt.imshow(np.transpose(vutils.make_grid(mask_inf[0:,1].unsqueeze(1), padding=2, normalize=True).cpu(),(1,2,0)))
fig.figure.savefig("D:/PhD and Work/I2R/MICCAI 2023_main_BraTs/exp_3_dual_label_seg/results/Fully S-multiclass-smploss-s2/Predicted results inf V1.png", dpi=224)

plt.figure(figsize=(64,64))
plt.axis("off")
plt.title("Predicted results")
plt.imshow(np.transpose(vutils.make_grid(mask_rev[0:,1].unsqueeze(1), padding=2, normalize=True).cpu(),(1,2,0)))
fig = plt.imshow(np.transpose(vutils.make_grid(mask_rev[0:,1].unsqueeze(1), padding=2, normalize=True).cpu(),(1,2,0)))
fig.figure.savefig("D:/PhD and Work/I2R/MICCAI 2023_main_BraTs/exp_3_dual_label_seg/results/Fully S-multiclass-smploss-s2/Predicted results rev V1.png", dpi=224)

