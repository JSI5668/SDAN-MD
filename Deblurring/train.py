import os
from config import Config

opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import pandas as pd
from ignite.metrics.ssim import SSIM
import matplotlib.pyplot as plt

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np
from skimage import img_as_ubyte

import utils
from data_RGB import get_training_data, get_validation_data
from model import SDAN_MD
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm


# Set  Seeds
RANDOM_SEED = 2021
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR
train_val_dir = opt.TRAINING.TRAIN_VAL_DIR
######### Model ###########
model_restoration = SDAN_MD()
utils.load_checkpoint(model_restoration, './base/model_deblurring.pth')
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                        eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_perceptual = losses.PerceptualLoss()
criterion_perceptual.initialize()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=0,
                          drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                        pin_memory=True)

train_val_dataset = get_validation_data(train_val_dir, {'patch_size': opt.TRAINING.VAL_PS})
train_val_loader = DataLoader(dataset=train_val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                        pin_memory=True)


print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')


for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()
        restored = model_restoration(input_)

        filenames = data[2]

        loss_char = 0
        loss_edge = 0
        loss_perceptual = 0
        # Compute loss at each stage
        for j in range(3):
            loss_char += criterion_char(restored[j], target)
        for j in range(3):
            loss_edge += criterion_edge(restored[j], target)
        for j in range(3):
            loss_perceptual += criterion_perceptual(restored[j], target)

        loss = (loss_char) + (0.05 * loss_edge) + loss_perceptual

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #
        if i % 50 == 0:
            for batch in range(len(restored)):
                restored_img = torch.clamp(restored[batch], 0, 1)
                restored_img = restored_img.permute(0, 2, 3, 1).cpu().detach().numpy()
                restored_img = img_as_ubyte(restored_img)

                result_sub_dir = os.path.join(result_dir + '/train', str(epoch))
                utils.mkdir(result_sub_dir)
                utils.save_img((os.path.join(result_sub_dir, filenames[0] + '_' + str(batch) + '.png')),
                               restored_img[0])

            tar_sav = target.permute(0, 2, 3, 1).cpu().detach().numpy()
            tar_sav = img_as_ubyte(tar_sav)

            utils.save_img((os.path.join(result_sub_dir, filenames[0] + '_' + str(batch) + 'tar.png')), tar_sav[0])


    #### Evaluation ####
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()

        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            filenames = data_val[2]

            with torch.no_grad():
                restored = model_restoration(input_)
            # restored = restored[0]

            for res, tar in zip(restored[0], target):
                res = torch.clamp(res, 0, 1)
                tar = torch.clamp(tar, 0, 1)
                res = torch.unsqueeze(res, dim=0)
                tar = torch.unsqueeze(tar, dim=0)

            if ii % 50 == 0:
                for batch in range(len(restored)):
                    restored_img = torch.clamp(restored[batch], 0, 1)
                    restored_img = restored_img.permute(0, 2, 3, 1).cpu().detach().numpy()
                    restored_img = img_as_ubyte(restored_img)

                    result_sub_dir = os.path.join(result_dir + '/test', str(epoch))
                    utils.mkdir(result_sub_dir)

                    utils.save_img((os.path.join(result_sub_dir, filenames[0] + '_' + str(batch) + '.png')),
                                   restored_img[0])

                tar_sav = tar.permute(0, 2, 3, 1).cpu().detach().numpy()
                tar_sav = img_as_ubyte(tar_sav)

                utils.save_img((os.path.join(result_sub_dir, filenames[0] + '_' + str(batch) + 'tar.png')), tar_sav[0])

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")




