import sys
import os
import argparse
import logging
import json
import time
import cv2

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import pandas as pd
from Classification.data.wsi_producer import GridWSIPatchDataset  # noqa
from Classification.model import MODELS  # noqa
from PIL import Image
parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                             ' patch predictions given a WSI')
parser.add_argument('--wsi_path', default='./data/svs/', metavar='WSI_PATH',
                    type=str, help='Path to the input WSI file')
parser.add_argument('--ckpt_path', default='model/MobileNetV2_0/best.ckpt', metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('--cfg_path', default='./configs/resnet18_tcga.json',
                    metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                         ' the ckpt file')
parser.add_argument('--mask_path', default='./data/tissue_mask/',
                    metavar='MASK_PATH', type=str, help='Path to the tissue mask of the input WSI file')
parser.add_argument('--probs_map_path', default='./data/npy/', metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--visualize_path', default='./data/vis/', metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                                                         ', default 0')
parser.add_argument('--num_workers', default=0, type=int, help='number of '
                                                               'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                                                             ' of the 8 direction predictions for each patch,'
                                                             ' default 0, which means disabled')


def get_probs_map(model, dataloader):
    probs_map = np.zeros((dataloader.dataset._mask.shape))
    num_batch = len(dataloader)
    # only use the prediction of the center patch within the grid
    idx_center = dataloader.dataset._grid_size // 2

    count = 0
    time_now = time.time()
    with torch.no_grad():
        for (data, x_mask, y_mask) in dataloader:
            data = torch.autograd.Variable(data.cuda())

            output,_ = model(data)

            batch_size, grid_size, _ = output.size()
            output = output.sigmoid()
            predict = torch.zeros_like(output)
            predict[output > 0.5] = 1
            predict = predict + 1
            probs_map[x_mask, y_mask] = predict[:, idx_center].cpu().data.numpy().flatten()

            count += 1

            time_spent = time.time() - time_now
            time_now = time.time()
            logging.info(
                '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
                    .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                    dataloader.dataset._rotate, count, num_batch, time_spent))

    return probs_map


def make_dataloader(args, cfg, tif_pth, mask_pth, flip='NONE', rotate='NONE'):
    batch_size = cfg['wsi_test_batch_size'] * 8
    num_workers = args.num_workers

    dataloader = DataLoader(
        GridWSIPatchDataset(tif_pth, mask_pth,
                            image_size=cfg['image_size'],
                            patch_size=cfg['patch_size'],
                            crop_size=cfg['crop_size'], normalize=True,
                            flip=flip,
                            rotate=rotate,level=0),

        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args, tif_pth, mask_pth, tif):
    start_t = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if cfg['image_size'] % cfg['patch_size'] != 0:
        raise Exception('Image size / patch size != 0 : {} / {}'.
                        format(cfg['image_size'], cfg['patch_size']))

    patch_per_side = cfg['image_size'] // cfg['patch_size']
    grid_size = patch_per_side * patch_per_side
    mask = cv2.imread(mask_pth)
    ckpt = torch.load(args.ckpt_path)
    model = MODELS[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model.eval()
    if not args.eight_avg:
        dataloader = make_dataloader(
            args, cfg, tif_pth, mask_pth, flip='NONE', rotate='NONE')
        probs_map = get_probs_map(model, dataloader)
    else:
        probs_map = np.zeros(mask.shape)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        probs_map /= 8
    nsp = args.probs_map_path + '/' + tif
    np.save(nsp, probs_map)
    all_time = time.time() - start_t
    print("Classification need time:", all_time)


def main():
    args = parser.parse_args()
    if not os.path.isdir(args.visualize_path):
        os.mkdir(args.visualize_path)
    if not os.path.isdir(args.probs_map_path):
        os.mkdir(args.probs_map_path)
    data_path = args.wsi_path
    mask_path = args.mask_path
    tiffs = os.listdir(data_path)

    for tif in tiffs:

            if tif.strip('.svs') + '.npy' in os.listdir(args.probs_map_path):
                continue
            tif_pth = os.path.join(data_path, tif)
            print(tif_pth)
            mask = tif.strip('.svs') + '_tissue_mask_64.png'
            mask_pth = os.path.join(mask_path, mask)
            run(args, tif_pth, mask_pth, tif.strip('.svs'))


if __name__ == '__main__':
    main()

