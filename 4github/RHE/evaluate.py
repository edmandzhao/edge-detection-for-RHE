import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import torchvision

import datasets_4cor_img as datasets
from utils import *

@torch.no_grad()
def validate_process(model, args):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    mace_list = []
    val_dataset = datasets.fetch_dataloader(args, split='validation')
    for i_batch, data_blob in enumerate(val_dataset):
        image1, image2, flow_gt,  H,img1_g,img2_g  = [x.to(model.device) for x in data_blob]


        image1 = image1.to(model.device)
        image2 = image2.to(model.device)
        img1_e = image1[:, :1, :, :]
        img2_e = image2[:, :1, :, :]
        img2_e = 2 * (img2_e / 255.0) - 1.0
        img1_g = img1_g.cuda()

        img2_g = img2_g.cuda()
        image1 = image1[:, 1:, :, :]
        image2 = image2[:, 1:, :, :]
        four_pr, flowiter1, flowiter2,_= model(image1, image2, img2_e, img1_g,iters_lev0 = args.iters_lev0, iters_lev1 = args.iters_lev1, test_mode=True)
        flow_4cor = torch.zeros((four_pr.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
        mace = torch.sum((four_pr[0, :, :, :].cpu() - flow_4cor) ** 2, dim=0).sqrt()
        mace_list.append(mace.view(-1).numpy())
        torch.cuda.empty_cache()
        if i_batch>300:
            break

    model.train()
    mace = np.mean(np.concatenate(mace_list))
    print("Validation MACE: %f" % mace)
    return {'chairs_mace': mace}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', default='mscoco', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name')
    parser.add_argument('--batch_size', type=int, default=4)
    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    args = parser.parse_args()
