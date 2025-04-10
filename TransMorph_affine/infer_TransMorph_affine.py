import os, utils, glob, losses, random, math
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph_affine import CONFIGS as CONFIGS_TM
import models.TransMorph_affine as TransMorph
import torch.nn as nn
import argparse
from skimage.metrics import structural_similarity as ssim
import pickle

import pandas as pd

def affine_aug(im, im_label=None, seed=10):
    # mode = 'bilinear' or 'nearest'
    with torch.no_grad():
        random.seed(seed)
        angle_range = 10
        trans_range = 0.1
        scale_range = 0.1
        # scale_range = 0.15

        angle_xyz = (random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180)
        scale_xyz = (random.uniform(-scale_range, scale_range), random.uniform(-scale_range, scale_range),
                     random.uniform(-scale_range, scale_range))
        trans_xyz = (random.uniform(-trans_range, trans_range), random.uniform(-trans_range, trans_range),
                     random.uniform(-trans_range, trans_range))

        rotation_x = torch.tensor([
            [1., 0, 0, 0],
            [0, math.cos(angle_xyz[0]), -math.sin(angle_xyz[0]), 0],
            [0, math.sin(angle_xyz[0]), math.cos(angle_xyz[0]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_y = torch.tensor([
            [math.cos(angle_xyz[1]), 0, math.sin(angle_xyz[1]), 0],
            [0, 1., 0, 0],
            [-math.sin(angle_xyz[1]), 0, math.cos(angle_xyz[1]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_z = torch.tensor([
            [math.cos(angle_xyz[2]), -math.sin(angle_xyz[2]), 0, 0],
            [math.sin(angle_xyz[2]), math.cos(angle_xyz[2]), 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        trans_shear_xyz = torch.tensor([
            [1. + scale_xyz[0], 0, 0, trans_xyz[0]],
            [0, 1. + scale_xyz[1], 0, trans_xyz[1]],
            [0, 0, 1. + scale_xyz[2], trans_xyz[2]],
            [0, 0, 0, 1]
        ], requires_grad=False).unsqueeze(0).cuda()

        theta_final = torch.matmul(rotation_x, rotation_y)
        theta_final = torch.matmul(theta_final, rotation_z)
        theta_final = torch.matmul(theta_final, trans_shear_xyz)

        output_disp_e0_v = F.affine_grid(theta_final[:, 0:3, :], im.shape, align_corners=False)

        im = F.grid_sample(im, output_disp_e0_v, mode='bilinear', padding_mode="border", align_corners=False)

        if im_label is not None:
            im_label = F.grid_sample(im_label, output_disp_e0_v, mode='nearest', padding_mode="border",
                                     align_corners=False)
            return im, im_label
        else:
            return im


def calc_ssim(x_torch, y_torch):
    x_np = x_torch.squeeze().detach().cpu().numpy()
    y_np = y_torch.squeeze().detach().cpu().numpy()
    
    data_range = max(x_np.max(), y_np.max()) - min(x_np.min(), y_np.min())
    
    if data_range == 0 or np.isnan(data_range) or np.isinf(data_range):
        raise Exception("Data range invalido")
    else:
       return ssim(x_np, y_np, data_range= data_range)
   
def calc_mae(x_torch, y_torch):
    return torch.mean(torch.abs(x_torch - y_torch)).item()

def calc_rmse(x_torch, y_torch):
    return torch.sqrt(torch.mean((y_torch - x_torch) ** 2)).item()

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)



def args_input():
    parser = argparse.ArgumentParser(description='Affine TransMorph Affine')
    parser.add_argument('--train_dir', type=str, default='train', help='path to train data')
    parser.add_argument('--val_dir', type=str,  default='val', help='path to val data')
    parser.add_argument('--save', type=str,  default='transmorph_affine', help='Save folder')
    parser.add_argument('--epoch', type=int, default=0, help='Start epoch for training')
    parser.add_argument('--no_save_infer', action='store_true', help="flag for not saving infered results")
    parser.add_argument('--train_infer', action='store_true', help="eval train images. infered train images will not be saved")
    return parser.parse_args()

def main():
    args = args_input()
    
    train_infer = args.train_infer
    no_save = args.no_save_infer

    if train_infer:
        train_dir = args.train_dir
    val_dir = args.val_dir
    save_dir = args.save
    epoch = args.epoch
    
    model_dir = os.path.join('experiments', save_dir)
    if not os.path.exists(model_dir):
        raise Exception("Pasta do modelo deve existir")
    
    infer_dir = os.path.join(model_dir,'infer')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize model
    # TODO: Pensar em como setar h, w, d
    H, W, D = 64, 512, 512
    config = CONFIGS_TM['TransMorph_Affine']
    config.img_size = (H, W, D)
    config.window_size = (H // 16, W // 128, D // 128)
    
    model = TransMorph.TransMorphAffine(config)
    affine_trans = TransMorph.AffineTransform()#AffineTransformer((H, W, D)).cuda()
    
    
    best_model = torch.load(os.path.join(model_dir, f' epc_{epoch}.pth.tar'))['model_state']
    print(f'Model: epc_{epoch}.pth.tar loaded!')
    model.load_state_dict(best_model)
    model.cuda()
        

    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(),
                                       trans.NumpyType((np.float32, np.float32))])
    
    val_set = datasets.OrCaScoreDataSet(glob.glob(val_dir + '/*.pkl'), transforms=val_composed)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    infer_data = {
        "image": [],
        "mae_pre": [],
        "mae_post": [],
        "rmse_pre": [],
        "rmse_post": [],
        "ssim_pre": [],
        "ssim_post": [],
        "phase": []
    }
    
    with torch.no_grad():
        print('========================== Infer Set ==========================')
        for data in val_loader:
            model.eval()
            data = [t.cuda() for t in data[:2]]
            id_name = data[2]
            infer_data['image'].append(id_name)
            print('- infer ' + id_name)
            x = data[0]
            y = data[1]
            infer_data['phase'].append('infer')
            
            infer_data['mae_pre'].append(calc_mae(x, y))
            infer_data['rmse_pre'].append(calc_rmse(x, y))
            try:
                infer_data['ssim_pre'].append(calc_ssim(x, y))
            except:
                infer_data['ssim_pre'].append(np.nan)
                print("\tErro ao calcular ssim pre")
            
            aff, scl, transl, shr = model((x, y))
            x_trans, mat, inv_mat = affine_trans(x, aff, scl, transl, shr)
            
            if not no_save:
                x_trans_np = x_trans.cpu().numpy()
                params = {
                    'aff': aff.cpu().numpy(),
                    'scl': scl.cpu().numpy(),
                    'transl': transl.cpu().numpy(),
                    'shr': shr.cpu().numpy()
                }
                save_pickle(os.path.join(infer_dir, f'{id_name}_transformed.pkl'), x_trans_np)
                save_pickle(os.path.join(infer_dir, f'{id_name}_params.pkl'), params)
            
            infer_data['mae_post'].append(calc_mae(x, y))
            infer_data['rmse_post'].append(calc_rmse(x, y))
            try:
                infer_data['ssim_post'].append(calc_ssim(x_trans, y))
            except:
                infer_data['ssim_post'].append(np.nan)
                print("\tErro ao calcular ssim post")
        
    df_infer = pd.DataFrame(infer_data)
    print(f'Average SSMI pre {np.nanmean(infer_data["ssim_pre"])} | Average SSMI post {np.nanmean(infer_data["ssim_post"])}')
    print(f'Average MAE pre {np.nanmean(infer_data["mae_pre"])} | Average MAE post {np.nanmean(infer_data["mae_post"])}')
    print(f'Average RMSE pre {np.nanmean(infer_data["rmse_pre"])} | Average RMSE post {np.nanmean(infer_data["rmse_post"])}')
    
    if train_infer:
        train_set = datasets.OrCaScoreDataSet(glob.glob(train_dir + '/*.pkl'), transforms=train_composed)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        
        train_data = {
            "image": [],
            "mae_pre": [],
            "mae_post": [],
            "rmse_pre": [],
            "rmse_post": [],
            "ssim_pre": [],
            "ssim_post": [],
            "phase": []
        }
        
        with torch.no_grad():
            print('========================== Training Set ==========================')
            for data in train_loader:
                model.eval()
                data = [t.cuda() for t in data[:2]]
                id_name = data[2]
                train_data['image'].append(id_name)
                print('- infer ' + id_name)
                x = data[0]
                y = data[1]
                train_data['phase'].append('train')
                
                train_data['mae_pre'].append(calc_mae(x, y))
                train_data['rmse_pre'].append(calc_rmse(x, y))
                try:
                    train_data['ssim_pre'].append(calc_ssim(x, y))
                except:
                    train_data['ssim_pre'].append(np.nan)
                    print("\tErro ao calcular ssim pre")
                
                aff, scl, transl, shr = model((x, y))
                x_trans, mat, inv_mat = affine_trans(x, aff, scl, transl, shr)
                

                
                train_data['mae_post'].append(calc_mae(x, y))
                train_data['rmse_post'].append(calc_rmse(x, y))
                try:
                    train_data['ssim_post'].append(calc_ssim(x_trans, y))
                except:
                    train_data['ssim_post'].append(np.nan)
                    print("\tErro ao calcular ssim post")
                
        print(f'Average SSMI pre {np.nanmean(train_data["ssim_pre"])} | Average SSMI post {np.nanmean(train_data["ssim_post"])}')
        print(f'Average MAE pre {np.nanmean(train_data["mae_pre"])} | Average MAE post {np.nanmean(train_data["mae_post"])}')
        print(f'Average RMSE pre {np.nanmean(train_data["rmse_pre"])} | Average RMSE post {np.nanmean(train_data["rmse_post"])}')
        df_infer = pd.concat([df_infer, pd.DataFrame(train_data)])
    df_infer.to_csv(os.path.join(infer_dir, 'results.csv'))
    

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
