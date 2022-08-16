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
from config import Config
from evaluation.dataloader import EvalDataset
from evaluation.evaluator import Eval_thread


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
parser.add_argument('--testsets',
                    default='CoCA',
                    type=str,
                    help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")
parser.add_argument('--size',
                    default=224,
                    type=int,
                    help='input size')
parser.add_argument('--tmp', default='/data1/dcfm/temp', help='Temporary folder')
parser.add_argument('--save_root', default='./CoSODmaps/pred', type=str, help='Output folder')

args = parser.parse_args()
config = Config()

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

for testset in ['CoCA']:
    if testset == 'CoCA':
        test_img_path = './data/images/CoCA/'
        test_gt_path = './data/gts/CoCA/'

        saved_root = os.path.join(args.save_root, 'CoCA')
    elif testset == 'CoSOD3k':
        test_img_path = './data/images/CoSOD3k/'
        test_gt_path = './data/gts/CoSOD3k/'
        saved_root = os.path.join(args.save_root, 'CoSOD3k')
    elif testset == 'CoSal2015':
        test_img_path = './data/images/CoSal2015/'
        test_gt_path = './data/gts/CoSal2015/'
        saved_root = os.path.join(args.save_root, 'CoSal2015')
    elif testset == 'CoCo':
        test_img_path = './data/images/CoCo/'
        test_gt_path = './data/gts/CoCo/'
        saved_root = os.path.join(args.save_root, 'CoCo')
    else:
        print('Unkonwn test dataset')
        print(args.dataset)

    test_loader = get_loader(
        test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

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
    val_measures = []
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
        if config.validation:
            measures = validate(model, test_loader, args.testsets)
            val_measures.append(measures)
            print(
                'Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with S_measure {:.4f}'.format(
                    epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
                    np.max(np.array(val_measures)[:, 0]))
            )
            # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.dcfmnet.state_dict(),
                #'scheduler': scheduler.state_dict(),
            },
            path=args.tmp)
        if config.validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                best_weights_before = [os.path.join(args.tmp, weight_file) for weight_file in
                                       os.listdir(args.tmp) if 'best_' in weight_file]
                for best_weight_before in best_weights_before:
                    os.remove(best_weight_before)
                torch.save(model.dcfmnet.state_dict(),
                           os.path.join(args.tmp, 'best_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save(model.dcfmnet.state_dict(), args.tmp + '/model-' + str(epoch + 1) + '.pt')
       
        if epoch > 188:
            torch.save(model.dcfmnet.state_dict(), args.tmp+'/model-' + str(epoch + 1) + '.pt')
    #dcfmnet_dict = model.dcfmnet.state_dict()
    #torch.save(dcfmnet_dict, os.path.join(args.tmp, 'final.pth'))

def sclloss(x, xt, xb):
    cosc = (1+compute_cos_dis(x, xt))*0.5
    cosb = (1+compute_cos_dis(x, xb))*0.5
    loss = -torch.log(cosc+1e-5)-torch.log(1-cosb+1e-5)
    return loss.sum()

def train(epoch):
    # Switch to train mode
    model.train()
    model.set_mode('train')
    loss_sum = 0.0
    loss_sumkl = 0.0
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        pred, proto, protogt, protobg = model(inputs, gts)
        loss_iou = IOUloss(pred, gts)
        loss_scl = sclloss(proto, protogt, protobg)
        loss = loss_iou+0.1*loss_scl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss_iou.detach().item()

        if batch_idx % 20 == 0:
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]  '
                        'Train Loss: loss_iou: {4:.3f}, loss_scl: {5:.3f} '.format(
                            epoch,
                            args.epochs,
                            batch_idx,
                            len(train_loader),
                            loss_iou,
                            loss_scl,
                        ))
    loss_mean = loss_sum / len(train_loader)
    return loss_sum


def validate(model, test_loaders, testsets):
    model.eval()

    testsets = testsets.split('+')
    measures = []
    for testset in testsets[:1]:
        print('Validating {}...'.format(testset))
        #test_loader = test_loaders[testset]

        saved_root = os.path.join(args.save_root, testset)

        for batch in test_loader:
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                scaled_preds = model(inputs, gts)[-1].sigmoid()

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True)
                save_tensor_img(res, os.path.join(saved_root, subpath))

        eval_loader = EvalDataset(
            saved_root,  # preds
            os.path.join('./data/gts', testset)  # GT
        )
        evaler = Eval_thread(eval_loader, cuda=True)
        # Use S_measure for validation
        s_measure = evaler.Eval_Smeasure()
        if s_measure > config.val_measures['Smeasure']['CoCA'] and 0:
            # TODO: evluate others measures if s_measure is very high.
            e_max = evaler.Eval_Emeasure().max().item()
            f_max = evaler.Eval_fmeasure().max().item()
            print('Emax: {:4.f}, Fmax: {:4.f}'.format(e_max, f_max))
        measures.append(s_measure)

    model.train()
    return measures

if __name__ == '__main__':
    main()
