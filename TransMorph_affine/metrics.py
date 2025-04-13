from torch import sqrt, mean, abs
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calc_ssim(x_torch, y_torch):
    x_np = x_torch.squeeze().detach().cpu().numpy()
    y_np = y_torch.squeeze().detach().cpu().numpy()
    
    data_range = max(x_np.max(), y_np.max()) - min(x_np.min(), y_np.min())
    
    if data_range == 0 or np.isnan(data_range) or np.isinf(data_range):
        raise Exception("Data range invalido")
    else:
       return ssim(x_np, y_np, data_range= data_range)
   
def calc_mae(x_torch, y_torch):
    return mean(abs(x_torch - y_torch)).item()

def calc_rmse(x_torch, y_torch):
    return sqrt(mean((y_torch - x_torch) ** 2)).item()


def calc_metrics(image_id, x_torch_pre, y_torch_pre, x_torch_post=None, y_torch_post=None, phase='infer'):
    result = {}
    result['image_id'] = image_id
    result['phase'] = phase
    result['mae_pre'] = calc_mae(x_torch_pre, y_torch_pre)
    result['rmse_pre'] = calc_rmse(x_torch_pre, y_torch_pre)
    
    try:
        result['ssim_pre'] = calc_ssim(x_torch_pre, y_torch_pre)
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        result['ssim_pre'] = None
    
    
    if x_torch_post and x_torch_post:
        result['mae_post'] = calc_mae(x_torch_post, y_torch_post)
        result['rmse_post'] = calc_rmse(x_torch_post, y_torch_post)
        try:
            result['ssim_post'] = calc_ssim(x_torch_post, y_torch_post)
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            result['ssim_post'] = None
    else:
        result['mae_post'] = None
        result['rmse_post'] = None
        result['ssim_post'] = None
    return result

def calc_list_statistic(data_list, key, stat_function):
    """
    Calcula uma estatística específica para uma chave em uma lista de dicionários.
    
    Parameters:
        data_list (list): Lista de dicionários contendo os dados.
        key (str): A chave para a qual calcular a estatística.
        stat_function (callable): Função estatística (ex: np.mean, np.median, np.min, np.max, np.std).
    
    Returns:
        float: O resultado da estatística calculada ou None se não houver valores válidos.
    """
    # Filtra os valores válidos (não None) para a chave especificada
    values = [item[key] for item in data_list if item[key] is not None]
    
    # Retorna o cálculo da estatística se houver valores válidos
    return stat_function(values) if values else None

def print_metrics(data_list, function='average'):
    map_function = {
        'average': np.mean,
        'min': np.min,
        'max': np.max,
        'std': np.std
    }
    
    if function not in map_function.keys():
        raise ValueError(f"Function {function} not in {map_function.keys()}")
    
    result = {
        'ssim_pre': calc_list_statistic(data_list, 'ssim_pre', map_function[function]),
        'ssim_post': calc_list_statistic(data_list, 'ssim_post', map_function[function]),
        'mae_pre': calc_list_statistic(data_list, 'mae_pre', map_function[function]),
        'mae_post': calc_list_statistic(data_list, 'mae_post', map_function[function]),
        'rmse_pre': calc_list_statistic(data_list, 'rmse_pre', map_function[function]),
        'rmse_post': calc_list_statistic(data_list, 'rmse_post', map_function[function]),
    }
    
    print(f'{function.capitalize()} SSMI pre {result["ssim_pre"]} | {function.capitalize()} SSMI post {result["ssim_post"]}')
    print(f'{function.capitalize()} MAE pre {result["mae_pre"]} | {function.capitalize()} MAE post {result["mae_post"]}')
    print(f'{function.capitalize()} RMSE pre {result["rmse_pre"]} | {function.capitalize()} RMSE post {result["rmse_post"]}')
    