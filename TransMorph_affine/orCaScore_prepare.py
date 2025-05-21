import os
import pickle
import SimpleITK as sitk
import pandas as pd
import shutil
import json
from math import ceil, floor
import numpy as np
from utils.file_utils import json_load, pkload, save_pickle

def get_image_array(image_path):
    '''This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.'''
    
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(image_path)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    return ct_scan

def save(x_image, y_image, save_path, normalize=True, padding_mode = None, result_slices = None, artery_image = None):
    data = {}
    if normalize:
        x_image = normalize_image(x_image)
        y_image = normalize_image(y_image)
        
    if padding_mode is not None:
        if result_slices is None:
            raise ValueError("result_slices must be provided when padding_mode is not None.")
        x_image, y_image, add_str, add_end = pad_img(x_image, y_image, padding_mode, result_slices)
        data['padding_start'] = add_str
        data['padding_end'] = add_end
        
        
    data['moved'] = x_image
    data['fixed'] = y_image
    
    if artery_image is not None:
        data['artery'] = pad_img(artery_image, artery_image, padding_mode, result_slices)[0]
    else:
        data['artery'] = None   
    save_pickle(save_path, data)
        


def get_ids(split_path):
    id_list = pd.read_csv(split_path)
    id_list = list(id_list.iloc[:, 0])
    return id_list

def normalized2normalized(ids, config, ct_type, result_folder):
    orca_folder = config['orca_folder']
    padding_mode = config.get('padding_mode')
    result_slices = config.get('result_slices')
    for i in ids:
        x_path = os.path.join(orca_folder, f"{i}{ct_type[config['moved']]}.mhd")
        x_image = get_image_array(x_path)
        x_image = normalize_cytran(x_image)
        y_path = os.path.join(orca_folder, f"{i}{ct_type[config['fixed']]}.mhd")
        y_image = get_image_array(y_path)
        y_image = normalize_cytran(y_image)
        if config['has_arteries_labels']:
            artery_path = os.path.join(orca_folder, f"{i}.label.nii.gz")
            artery_image = sitk.GetArrayFromImage(sitk.ReadImage(artery_path))
            
        save(x_image, y_image, os.path.join(result_folder, f"{i}.pkl"), padding_mode=padding_mode, 
                 result_slices=result_slices, artery_image=artery_image)
        
def transformed2normalized(ids, config, ct_type, result_folder):
    orca_folder = config['orca_folder']
    padding_mode = config.get('padding_mode')
    result_slices = config.get('result_slices')
    netA_folder = config['cytran']['net_A_folder']
    for i in ids:
        x_image = pkload(os.path.join(netA_folder, f"{i}.pkl"))
        y_path = os.path.join(orca_folder, f"{i}{ct_type[config['fixed']]}.mhd")
        y_image = get_image_array(y_path)
        y_image = normalize_cytran(y_image)
        if config['has_arteries_labels']:
            artery_path = os.path.join(orca_folder, f"{i}.label.nii.gz")
            artery_image = sitk.GetArrayFromImage(sitk.ReadImage(artery_path))
        save(x_image, y_image, os.path.join(result_folder, f"{i}.pkl"), 
             padding_mode=padding_mode, result_slices=result_slices,
             artery_image=artery_image)
    
def fuse_splits(folderA, folderB, result_folder, suffixA = "A", suffixB = "B"):
    # TODO: Verificar como fica como os labels de arterias
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for filename in os.listdir(folderA):
        if filename.endswith('.pkl'):
            base_name = os.path.splitext(filename)[0]
            novo_nome = f"{base_name}_{suffixA}.pkl"
            origem = os.path.join(folderA, filename)
            destino = os.path.join(result_folder, novo_nome)
            # Move o arquivo
            shutil.copy(origem, destino)

    # Renomeie e mova os arquivos da pasta B para a pasta C
    for filename in os.listdir(folderB):
        if filename.endswith('.pkl'):
            # Extrai o nome do arquivo sem extensão
            base_name = os.path.splitext(filename)[0]
            # Cria o novo nome com o sufixo 'B.pkl'
            novo_nome = f"{base_name}_{suffixB}.pkl"
            # Caminho completo do arquivo de origem e de destino
            origem = os.path.join(folderB, filename)
            destino = os.path.join(result_folder, novo_nome)
            # Move o arquivo
            shutil.copy(origem, destino)


def normalize_cytran(image):
    result = image.copy()
    result = result + 1024
    result[result < 0] = 0
    result = result / 1e3
    result = result - 1
    return result

def normalize_image(image):
    array_min = image.min()
    array_max = image.max()
    normalized_array = (image - array_min) / (array_max - array_min)
    return normalized_array


def pad_img(x_image, y_image, padding_mode, result_size):
    diff = result_size - x_image.shape[0]
    start_add = ceil(diff / 2)
    end_add = floor(diff / 2)
    
    if padding_mode == 'min_value':
        min_x = np.full(x_image.shape[1:], np.min(x_image))
        start_pad_x = np.repeat(min_x, start_add, axis=0)
        end_pad_x = np.repeat(min_x, end_add, axis=0)
        min_y = np.full(y_image.shape[1:], np.min(y_image))
        start_pad_y = np.repeat(min_y, start_add, axis=0)
        end_pad_y = np.repeat(min_y, end_add, axis=0)
    elif padding_mode == 'slices':
        start_pad_x = np.repeat(x_image[0:1, :, :], start_add, axis=0)
        end_pad_x = np.repeat(x_image[-1:, :, :], end_add, axis=0)
        start_pad_y = np.repeat(y_image[0:1, :, :], start_add, axis=0)
        end_pad_y = np.repeat(y_image[-1:, :, :], end_add, axis=0)
    else:
        raise ValueError("Invalid padding mode. Use 'min_value' or 'slices'.")

        
    x_image_cropped = np.concatenate([start_pad_x, x_image, end_pad_x], axis=0)
    y_image_cropped = np.concatenate([start_pad_y, y_image, end_pad_y], axis=0)
    
    return x_image_cropped, y_image_cropped, start_add, end_add

def main(config):
    mode = config['train']
    phases = ['train', 'test']
    for phase in phases:
        result_folder = f'{config["result_folder"]}/{phase}'
        
        # deve conter um train_data.csv e test.csv
        split_path = f'{config["split_path"]}/{phase}_data.csv'
        
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        
        ct_type = {
            'ARTERIAL': 'CTAI',
            'NATIVE': 'CTI',
        }
        
        ids = get_ids(split_path)
        
        # T64 normal
        match (mode):
            case 'T64_normalized':
                '''
                    x = 64 CTAI, normalizados pelo normilized_cytran
                    y = 64 CTI, normalizados pelo normilized_cytran
                '''
                normalized2normalized(ids, config, ct_type, result_folder)
            case 'T64_transformed':
                '''
                    x = 64 CTAI, com mudança de estilo cytran
                    y = 64 CTI, normalizados pelo normilized_cytran
                '''
                transformed2normalized(ids, config, ct_type, result_folder)

            case 'T128_normalized':
                '''64 pares de imagem T64_normalized + 64 pares de imagem T64_transformed'''
                fA = f'{config["result_folder"]}/T64_normalized/{phase}'
                fB = f'{config["result_folder"]}/T64_transformed/{phase}'
                fuse_splits(fA, fB, result_folder, suffixA = "N", suffixB = "T")
            
            case _:
                print("Invalid mode")
    
    

if __name__ == '__main__':
    json_input = json_load("input_prepare.json")
    main(json_input)
