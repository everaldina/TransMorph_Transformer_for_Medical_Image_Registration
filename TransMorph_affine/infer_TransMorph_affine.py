import os, glob, metrics
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from models.TransMorph_affine import CONFIGS as CONFIGS_TM
import models.TransMorph_affine as TransMorph
import torch.nn as nn
import argparse
from utils.file_utils import save_pickle
from utils.model_utils import load_model

import pandas as pd

def args_input():
    parser = argparse.ArgumentParser(description='Affine TransMorph Affine')
    parser.add_argument('--train_dir', type=str, default='train', help='path to train data')
    parser.add_argument('--val_dir', type=str,  default='val', help='path to val data')
    parser.add_argument('--save', type=str,  default='transmorph_affine', help='Save folder')
    parser.add_argument('--epoch', type=int, default=0, help='Start epoch for training')
    parser.add_argument('--no_save_infer', action='store_true', help="flag for not saving infered results")
    parser.add_argument('--train_infer', action='store_true', help="eval train images. infered train images will not be saved")
    parser.add_argument('--calc_padding', action='store_true', help="flag for calculating results without padding slices")
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
    calc_padding = args.calc_padding
    
    model_dir = os.path.join('experiments', save_dir)
    if not os.path.exists(model_dir):
        raise Exception("Pasta do modelo deve existir")
    
    infer_dir = os.path.join(model_dir,'infer')
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)

    # Initialize model
    H, W, D = 64, 512, 512
    config = CONFIGS_TM['TransMorph_Affine']
    config.img_size = (H, W, D)
    config.window_size = (H // 16, W // 128, D // 128)
    
    model = TransMorph.TransMorphAffine(config)
    affine_trans = TransMorph.AffineTransform()#AffineTransformer((H, W, D)).cuda()
    
    best_model, _ = load_model(os.path.join(model_dir, f'epc_{epoch}.pth.tar'))
    print(f'Model: epc_{epoch}.pth.tar loaded!')
    model.load_state_dict(best_model)
    model.cuda()
        

    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    
    val_set = datasets.OrCaScoreDataSet(glob.glob(val_dir + '/*.pkl'), transforms=val_composed)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    if calc_padding:
        clean_data = []
    
    infer_data = []
    with torch.no_grad():
        print('========================== Infer Set ==========================')
        for data in val_loader:
            model.eval()
            id_name = data['id_image'][0]
            print('- infer ' + id_name)
            x = data['x'].cuda()
            y = data['y'].cuda()
            
            aff, scl, transl, shr = model((x, y))
            x_trans, mat, inv_mat = affine_trans(x, aff, scl, transl, shr)
            
            infer_data.append(metrics.calc_metrics(id_name, x, y, x_trans, y))
            
            if not no_save:
                save_file = {
                    'x': x.detach().cpu().squeeze().numpy(),
                    'x_trans': x_trans.detach().cpu().squeeze().numpy(),
                    'transformation': {
                        'aff': aff.detach().cpu().squeeze().numpy(),
                        'scl': scl.detach().cpu().squeeze().numpy(),
                        'transl': transl.detach().cpu().squeeze().numpy(),
                        'shr': shr.detach().cpu().squeeze().numpy()
                    },
                    'y': y.detach().cpu().squeeze().numpy()
                }
                save_pickle(os.path.join(infer_dir, f'{id_name}.pkl'), save_file)
                
            if calc_padding:
                padding_start = data['padding_start']
                padding_end = data['padding_end']
                
                print(padding_start, padding_end)
                x_og =  x[:, :, padding_start:63-padding_end+1, :, :]
                y_og = y[:, :, padding_start:63-padding_end+1, :, :]
                print(x_og.shape, y_og.shape)
                
                # Calculating metrics without padding
                x_trans_clean = x_trans[:, :, padding_start:63-padding_end+1, :, :]
                clean_data.append(metrics.calc_metrics(id_name, x_og, y_og, x_trans_clean, y_og))

    if calc_padding:
        df_clean = pd.DataFrame(clean_data)
        df_clean.to_csv(os.path.join(infer_dir, 'results_clean.csv'))

    metrics.print_metrics(infer_data)
    df_infer = pd.DataFrame(infer_data)
    
    if train_infer:
        train_set = datasets.OrCaScoreDataSet(glob.glob(train_dir + '/*.pkl'), transforms=train_composed)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        
        train_data = []
        
        with torch.no_grad():
            print('========================== Training Set ==========================')
            for data in train_loader:
                model.eval()
                id_name = data['id_image'][0]
                print('- infer ' + id_name)
                x = data['x'].cuda()
                y = data['y'].cuda()
                
                aff, scl, transl, shr = model((x, y))
                x_trans, mat, inv_mat = affine_trans(x, aff, scl, transl, shr)
                
                train_data.append(metrics.calc_metrics(id_name, x, y, x_trans, y, 'train'))
                
        metrics.print_metrics(train_data)
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
