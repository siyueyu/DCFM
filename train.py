import torch
import torch.nn as nn
import torch.optim as optim
from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from dataset import get_loader
from loss import *
import torchvision.utils as vutils


from models.main import *

import torch.nn.functional as F
import pytorch_toolbelt.losses as PTL

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Parameter from command line
parser = argparse.ArgumentParser(description='')

parser.add_argument('--loss',
                    default='Scale_IoU',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--bs', '--batch_size', default=1, type=int)
parser.add_argument('--lr',
                    '--learning_rate',
                    default=1e-4,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='CoCo',
                    type=str,
                    help="Options: 'CoCo'")
parser.add_argument('--size',
                    default=224,
                    type=int,
                    help='input size')
parser.add_argument('--tmp', default='./temp', help='Temporary folder')

args = parser.parse_args()


# Prepare dataset
if args.trainset == 'CoCo':
    train_img_path = './data/CoCo/img/'
    train_gt_path = './data/CoCo/gt/'
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              args.size,
                              args.bs,
                              max_num=16, #20,
                              istrain=True,
                              shuffle=False,
                              num_workers=8, #4,
                              pin=True)

else:
    print('Unkonwn train dataset')
    print(args.dataset)


# make dir for tmp
os.makedirs(args.tmp, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.tmp, "log.txt"))
set_seed(123)

# Init model
device = torch.device("cuda")

model = DCFM()
model = model.to(device)
model.apply(weights_init)

model.dcfmnet.backbone._initialize_weights(torch.load('./models/vgg16-397923af.pth'))

backbone_params = list(map(id, model.dcfmnet.backbone.parameters()))
base_params = filter(lambda p: id(p) not in backbone_params,
                     model.dcfmnet.parameters())

all_params = [{'params': base_params}, {'params': model.dcfmnet.backbone.parameters(), 'lr': args.lr*0.1}]

# Setting optimizer
optimizer = optim.Adam(params=all_params,lr=args.lr, weight_decay=1e-4, betas=[0.9, 0.99])

for key, value in model.named_parameters():
    if 'dcfmnet.backbone' in key and 'dcfmnet.backbone.conv5.conv5_3' not in key:
        value.requires_grad = False

for key, value in model.named_parameters():
    print(key,  value.requires_grad)

# log model and optimizer pars
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
# logger.info(scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
exec('from loss import ' + args.loss)
IOUloss = eval(args.loss+'()')


def main():

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.dcfmnet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    print(args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(epoch)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save(model.dcfmnet.state_dict(), args.tmp + '/model-' + str(epoch + 1) + '.pt')

        if epoch > 188:
            torch.save(model.dcfmnet.state_dict(), args.tmp+'/model-' + str(epoch + 1) + '.pt')
    dcfmnet_dict = model.dcfmnet.state_dict()
    torch.save(dcfmnet_dict, os.path.join(args.tmp, 'final.pth'))


def SCloss(x, p, n):
    dist_p = (1 + compute_cos_dis(x, p)) * 0.5
    dist_n = (1 + compute_cos_dis(x, n)) * 0.5
    loss = - torch.log(dist_p + 1e-5) - torch.log(1. + 1e-5 - dist_n)
    return loss.sum()


def train(epoch):
    # Switch to train mode
    model.train()
    model.set_mode('train')
    loss_sum = 0.0
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        preds, proto, proto_pos, proto_neg = model(inputs, gts)
        loss_iou = IOUloss(preds, gts)
        loss_sc = SCloss(proto, proto_pos, proto_neg)
        loss = loss_iou + loss_sc*0.1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss.detach().item()

        if batch_idx % 20 == 0:
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]  '
                        'Train Loss: loss_iou: {4:.3f}, loss_sc: {5:.3f}, '.format(
                            epoch,
                            args.epochs,
                            batch_idx,
                            len(train_loader),
                            loss_iou,
                            loss_sc,
                        ))
    loss_mean = loss_sum / len(train_loader)
    logger.info('CkptIndex={}:    TrainLoss={} \n'.format(epoch, loss_mean))
    return loss_sum


if __name__ == '__main__':
    main()
