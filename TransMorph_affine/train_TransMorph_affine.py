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
from datetime import datetime

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

def args_input():
    parser = argparse.ArgumentParser(description='Affine TransMorph Affine')
    parser.add_argument('--train_dir', type=str, default='train', help='path to train data')
    parser.add_argument('--val_dir', type=str,  default='val', help='path to val data')
    parser.add_argument('--save', type=str,  default='transmorph_affine', help='Save folder')
    parser.add_argument('--lr', type=float,  default=0.0001, help='Learning Rate')
    parser.add_argument('--batch_size', type=int,  default=1, help='Batch size')
    parser.add_argument('--continue_train', action='store_true', help='Flag for continue training a model')
    parser.add_argument('--num_epochs', type=int, default=500, help='Total number of epochs')
    parser.add_argument('--epoch_start', type=int, default=0, help='Start epoch for training')
    return parser.parse_args()

def main():
    args = args_input()

    # atlas_dir = 'D:/DATA/IXI/atlas.pkl'
    train_dir = args.train_dir
    val_dir = args.val_dir
    save_dir = args.save
    
    model_dir = os.path.join('experiments', save_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    img_dir = os.path.join(model_dir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    batch_size = args.batch_size
    lr = args.lr
    epoch_start = args.epoch_start
    max_epoch = args.num_epochs
    cont_training = args.continue_train

    log_dir = os.path.join('logs', save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = utils.Logger(log_dir, f"logfile_epoch_str{epoch_start}.log")
    sys.stderr = sys.stdout
    
    

    # Initialize model
    # TODO: Pensar em como setar h, w, d
    H, W, D = 64, 512, 512
    config = CONFIGS_TM['TransMorph_Affine']
    config.img_size = (H, W, D)
    config.window_size = (H // 16, W // 128, D // 128)
    
    model = TransMorph.TransMorphAffine(config)
    affine_trans = TransMorph.AffineTransform()#AffineTransformer((H, W, D)).cuda()


    # If continue from previous training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if cont_training:
        if epoch_start == 0:
            raise Exception('Set a model to load')
        # updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9),8)
        # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        best_model, epoch_optimizer = utils.load_model(os.path.join(model_dir, f' epc_{epoch_start}.pth.tar'))
        print(f'Model: epc_{epoch_start}.pth.tar loaded!')
        model.load_state_dict(best_model)
        optimizer.load_state_dict(epoch_optimizer)
    model.cuda()
        
    # else:
    #     updated_lr = lr

    # Initialize training
    # TODO: verificar modficcao vit
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    train_set = datasets.OrCaScoreDataSet(glob.glob(train_dir + '/*.pkl'), transforms=train_composed)
    val_set = datasets.OrCaScoreDataSet(glob.glob(val_dir + '/*.pkl'), transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    criterion = nn.MSELoss()
    for epoch in range(epoch_start, max_epoch):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now}] ==== Epoch start {epoch + 1}/{max_epoch} ====")
        
        # Training
        idx = 0
        print('Training Starts')
        for data in train_loader:
            idx += 1
            model.train()
            x = data['x'].cuda()
            y = data['y'].cuda()
            x_ = affine_aug(x, seed=idx)
            y_ = y
            aff, scl, transl, shr = model((x_, y_))
            x_trans, mat, inv_mat = affine_trans(x_, aff, scl, transl, shr)
            loss = criterion(x_trans, y_)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}'.format(idx, len(train_loader), loss.item()))
       
        # validation
        idd = 0
        print('Validation start')
        with torch.no_grad():
            avg_ssim = 0
            ssim_count = 0
            for data in val_loader:
                model.eval()
                x = data['x'].cuda()
                y = data['y'].cuda()
                x_ = affine_aug(x, seed=idd)
                y_ = y  # affine_aug(y_half)
                aff, scl, transl, shr = model((x_, y_))
                x_trans, mat, inv_mat = affine_trans(x_, aff, scl, transl, shr)
                y_trans = affine_trans.apply_affine(y_, inv_mat)

                x_np = x_trans.squeeze().detach().cpu().numpy()
                y_np = y_.squeeze().detach().cpu().numpy()
                
                data_range = max(x_np.max(), y_np.max()) - min(x_np.min(), y_np.min())
                
                if data_range == 0 or np.isnan(data_range) or np.isinf(data_range):
                    continue
                else:
                    avg_ssim += ssim(x_np, y_np, data_range= data_range)
                    ssim_count += 1
                
                if (epoch % 10 == 0) or (epoch == max_epoch - 1):
                    # print(mat)
                    plt.figure()
                    plt.subplot(2, 2, 1)
                    plt.imshow(x_.cpu().detach().numpy()[0, 0, :, 32, ])
                    plt.title('Imagem original - X')
                    
                    plt.subplot(2, 2, 2)
                    plt.imshow(y_.cpu().detach().numpy()[0, 0, :, 32, ])
                    plt.title('Imagem alvo - Y')
                    
                    plt.subplot(2, 2, 4)
                    plt.imshow(x_trans.cpu().detach().numpy()[0, 0, :, 32, ])
                    plt.title('Transformação de Y')
                    
                    plt.subplot(2, 2, 3)
                    plt.imshow(y_trans.cpu().detach().numpy()[0, 0, :, 32, ])
                    plt.title('Transformação de X')
                    
                    
                    plt.suptitle('Comparação de transformação afim - epoca ' + str(epoch+1), fontsize=16)  # Título geral
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(os.path.join(img_dir, f'reg_results{idd}-epc{epoch+1}'))
                    plt.close()
                idd += 1
            avg_ssim = avg_ssim/ssim_count
            print(f'Average SSMI = {avg_ssim}')
        if (epoch % 10 == 0) or (epoch == max_epoch - 1):
            print('Save epoch', epoch+1)
            save_state = {
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(save_state, os.path.join(model_dir, f'epc_{epoch + 1}.pth.tar'))
        # TODO:  colocar avg ssmi ??


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, os.path.join(save_dir, filename))
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

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
