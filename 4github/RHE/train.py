import numpy as np
import os

import evaluate
import cv2
import torchgeometry as tgm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from network import IHN
from utils import *
import datasets_4cor_img as datasets


from torch.cuda.amp import GradScaler
import numpy as np
import time
import torch.optim as optim
import matplotlib


matplotlib.use('Agg')

setup_seed(2022)

MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def sequence_loss(flow_preds, flowiter1, flowiter2,flow_gt, valid, gamma):
    """ Loss function defined over sequence of flow predictions """

    n_predictions1 = 6
    n_predictions2 = 3
    flow_loss = 0.0

    for i in range(n_predictions1):

        i_weight = gamma**(n_predictions1 - i - 1)
        i_loss = (flowiter1[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss.mean()

    for i in range(n_predictions2):

        i_weight = gamma**(n_predictions2 - i - 1)
        i_loss = (flowiter2[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss.mean()

    mace = torch.sum((flow_preds - flow_gt) ** 2, dim=0).sqrt().mean()

    # 算分类的百分比
    metrics = {
        'mace': mace.mean().item(),
        '1px': (mace < 1).float().mean().item(),
        '3px': (mace < 3).float().mean().item(),
        '5px': (mace < 5).float().mean().item(),
    }

    return flow_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model,  batch_size=4, args=None):
    assert batch_size > 0, "batchsize > 0"

    print("Parameter Count: %d" % count_parameters(model))

    model.cuda()
    model.train()


    total_steps = 0

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args)


    timeall = []


    VAL_FREQ = 5000

    should_keep_training = True

    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            img1, img2, flow_gt, H ,img1_g ,img2_g= [x.cuda() for x in data_blob]        # img1是warp的，img2是正的


            img1 = img1.cuda()

            img2 = img2.cuda()
            img1_g = img1_g.cuda()

            img2_g= img2_g.cuda()

            img1_e = img1[:,:1,:,:]
            img1_e = img1_e / 255.0
            img2_e = img2[:,:1, :, :]
            img2_e = img2_e / 255.0

            img1 = img1[:, 1:,:,:]
            img2 = img2[:, 1:,:,:]

            time_start = time.time()

            four_pred, flowiter1, flowiter2, e1 ,e2, img2w ,e2ww= model(img1, img2,img1_g,img2_g,iters_lev0=args.iters_lev0, iters_lev1=args.iters_lev1, test_mode=True)

            flow_4cor = torch.zeros((four_pred.shape[0], 2, 2, 2))
            flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
            flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
            flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
            flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
            flow_4cor = flow_4cor.cuda()
            flowiter1 = flowiter1.cuda()
            flowiter2 = flowiter2.cuda()
            e1 = e1.cuda()
            e2 = e2.cuda()


            LOSS_E = torch.nn.L1Loss()
            eloss1 = LOSS_E(img1,img2w)
            eloss2 = LOSS_E(e1,e2ww)



            loss1, metrics = sequence_loss(four_pred, flowiter1, flowiter2, flow_4cor, H, args.gamma)
            loss = loss1+eloss1+eloss2

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            time_end = time.time()
            timeall.append(time_end - time_start)
            metrics['time'] = time_end - time_start
            logger.push(metrics)

            if logger.total_steps % args.val_freq == args.val_freq - 1:

                results = {}
                for val_dataset in args.validation:
                    results.update(evaluate.validate_process(model,args))
                for key in results.keys():
                    if key not in logger.val_results_dict.keys():
                        logger.val_results_dict[key] = []
                    logger.val_results_dict[key].append(results[key])
                PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)
                model.train()
            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)
    return PATH



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'./pidinet-master/trained_models/table5_pidinet.pth',
                        help="restore checkpoint")
    parser.add_argument('--modeledge', default=r'./pidinet-master/trained_models/table5_pidinet.pth',
                        help="restore checkpoint")
    parser.add_argument('--modelpre', default=r'./79999_IHN.pth',
                        help="restore checkpoint")
    parser.add_argument('--name', default='IHN', help="name your experiment")
    parser.add_argument('--lev0', default=False, action='store_true',
                        help='warp no')
    parser.add_argument('--lev1', default=False, action='store_true',
                        help='warp once')
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=3)
    parser.add_argument('--validation', type=str, nargs='+', default='None')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--output', type=str, default='checkpoints',
                        help='output directory to save checkpoints and plots')
    parser.add_argument('--dataset', type=str, default='mscoco', help='dataset')

    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--num_steps', type=int, default=180000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])

    parser.add_argument('--weight', default=False, action='store_true',
                        help='weight')

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')

    parser.add_argument('--val_freq', type=int, default=5000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='printing frequency')

    parser.add_argument('--mixed_precision', default=False, help='use mixed precision')

    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.gpuid[0]))
    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')


    model = IHN(args)

    model.to(device)
    pretrain_model = torch.load(args.modeledge, map_location=device)
    pretrain_model1 = torch.load(args.model, map_location=device)
    model_dict = model.state_dict()
    #
    # # 新建权重字典，并更新
    state_dict = pretrain_model['state_dict']
    state_dict2 = {k.replace('module', 'EN1'): v for k, v in state_dict.items()}
    state_dict3 = {k.replace('module', 'EN2'): v for k, v in state_dict.items()}
    state_dict1 = {k: v for k, v in state_dict2.items()  if k in model_dict.keys()}
    state_dict5 = {k: v for k, v in state_dict3.items() if k in model_dict.keys()}

    # 更新现有模型的权重字典
    model_med = torch.load(args.modelpre, map_location='cuda:0')
    model_dict.update(model_med)
    model_dict.update(state_dict1)
    model_dict.update(state_dict5)
    # 载入更新后的权重字典
    model.load_state_dict(model_dict)

    torch.manual_seed(1234)
    np.random.seed(1234)

    train(model,  args=args)





