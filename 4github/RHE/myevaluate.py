import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import argparse
from network import IHN
from utils import *
import datasets_4cor_img as datasets
import scipy.io as io
import torchvision
import numpy as np
import time
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchgeometry as tgm
setup_seed(2024)
def evaluate_SNet(model, val_dataset, batch_size=0, args = None):

    assert batch_size > 0, "batchsize > 0"

    total_mace = torch.empty(0)
    timeall=[]
    flag = 0

    for i_batch, data_blob in enumerate(val_dataset):
        img1, img2, flow_gt,  H ,img1_g,img2_g= [x.to(model.device) for x in data_blob]

        img1 = img1.to(model.device)
        img2 = img2.to(model.device)
        img1 = img1[:, 1:, :, :]
        img2 = img2[:, 1:, :, :]

        time_start = time.time()
        four_pred, flow1, flow2,_,_= model(img1, img2, img1_g, img2_g,iters_lev0=args.iters_lev0, iters_lev1=args.iters_lev1, test_mode=True)
        time_end = time.time()
        timeall.append(time_end-time_start)
        flag = flag + 1
        print(flag)

        flow_4cor = torch.zeros((four_pred.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

        mace_ = (flow_4cor - four_pred.cpu().detach())**2
        mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
        mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)
      
        total_mace = torch.cat([total_mace,mace_vec], dim=0)
        final_mace = torch.mean(total_mace).item()

        print(mace_.mean())
        print("MACE Metric: ", final_mace)
        with open("./test2017mace.txt",'a') as f:
            f.write('%.6f'%mace_.mean())
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 忘了是不是这个checkpoint了，可能会和结果差一点点
    parser.add_argument('--model', default=r'./checkpoints/IHN.pth',help="restore checkpoint")
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=3)
    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--savemat', type=str,  default='resmat')
    parser.add_argument('--savedict', type=str, default='resnpy')
    parser.add_argument('--dataset', type=str, default='mscoco', help='dataset')    
    parser.add_argument('--lev0', default=False, action='store_true',
                        help='warp no')
    parser.add_argument('--lev1', default=False, action='store_true',
                        help='warp once')
    parser.add_argument('--weight', default=False, action='store_true',
                        help='weight')
    parser.add_argument('--model_name_lev0', default='', help='specify model0 name')
    parser.add_argument('--model_name_lev1', default='', help='specify model0 name')

    args = parser.parse_args()
    device = torch.device('cuda:'+ str(args.gpuid[0]))

    model = IHN(args)
    model_med = torch.load(args.model, map_location='cuda:0')

    model.load_state_dict(model_med)

    model.to(device) 
    model.eval()

    batchsz = 1

    if args.dataset=='ggearth' or args.dataset=='ggmap':
        import dataset as datasets

    args.batch_size = batchsz
    val_dataset = datasets.fetch_dataloader(args, split='val')
    evaluate_SNet(model, val_dataset, batch_size=batchsz, args=args)