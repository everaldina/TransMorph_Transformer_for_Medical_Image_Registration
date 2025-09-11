# TransMorph Affine

Este repositório contém a implementação do **TransMorph Affine**, um modelo baseado em Transformer para registro de imagens médicas com transformações afins. O projeto inclui scripts para treinamento, inferência, preparação de dados e avaliação de métricas.

## Modelos
Para execução de inferencias ou continuação de treinos os modelos devem estar na pasta `experiments/` com a seguinte estrutura:

```
experiments/
    <nome_modelo>/
        epc_<num_epoch>.pth.tar
        ...
```

Segue link para download do modelo da epoca 500 com o treino de T64_transformed:
- [Google Drive](https://drive.google.com/file/d/1M0oaqUBTaAfZ-GUitD6xaBgv9x6m8Xu7/view?usp=sharing)

## Preparação dos Dados

Os dados devem estar no formato `.pkl` (pickle), contendo os seguintes campos obrigatórios por amostra:

- `id_image`: identificador da imagem
- `x`: tensor da imagem de entrada (moving)
- `y`: tensor da imagem alvo (fixed)
- `padding_start`, `padding_end`: informações de padding para pós-processamento
- `artery` (opcional): máscara de artérias, se disponível

Exemplo de estrutura de um arquivo `.pkl`:

```python
{
    'id_image': '001',
    'x': np.ndarray shape (1, D, H, W),
    'y': np.ndarray shape (1, D, H, W),
    'padding_start': [d, h, w],
    'padding_end': [d, h, w],
    'artery': np.ndarray shape (1, D, H, W)  # opcional
}
```

O script `orCaScore_prepare.py` foi o utilizado para fabricação dos arquivos `.pkl` a partir de imagens médicas.

Os arquivos `.pkl` devem ser organizados em pastas para treino e validação, por exemplo:

```
data/train/
    001.pkl
    002.pkl
    ...
data/val/
    101.pkl
    102.pkl
    ...
```

## Treinamento

O script principal para treinamento é o `train_TransMorph_affine.py`.

### Parâmetros principais

- `--train_dir`: Caminho para a pasta com arquivos `.pkl` de treino
- `--val_dir`: Caminho para a pasta com arquivos `.pkl` de validação
- `--save`: Nome da pasta para salvar os resultados em `experiments/`
- `--epochs`: Número de épocas de treinamento
- `--batch_size`: Tamanho do batch
- `--lr`: Taxa de aprendizado
- `--has_artery_label`: Indica se há máscara de artéria nos dados (`True`/`False`)
- `--no_save_train`: Não salva arquivos intermediários de treino
- `--no_save_infer`: Não salva arquivos intermediários de inferência

### Exemplo de execução

```bash
python train_TransMorph_affine.py \
    --train_dir data/train \
    --val_dir data/val \
    --save T64_transformed_min \
    --epochs 500 \
    --batch_size 1 \
    --lr 1e-4 \
    --has_artery_label True
```

## Inferência

O script de inferência é o `infer_TransMorph_affine.py`.

### Parâmetros principais

- `--val_dir`: Caminho para a pasta com arquivos `.pkl` de validação
- `--save`: Nome da pasta de experimento para salvar resultados
- `--epoch`: Qual checkpoint carregar (ex: `500` para `epc_500.pth.tar`)
- `--has_artery_label`: Indica se há máscara de artéria nos dados
- `--no_save_infer`: Não salva arquivos intermediários de inferência
- `--calc_padding`: Calcula métricas removendo padding

### Exemplo de execução

```bash
python infer_TransMorph_affine.py \
    --val_dir data/val \
    --save T64_transformed_min \
    --epoch 500 \
    --has_artery_label True \
    --calc_padding True
```

Os resultados serão salvos em `experiments/<save>/infer/`, incluindo arquivos `.pkl` com as imagens transformadas e arquivos `.csv` com as métricas.

## Docker

Para facilitar a execução em ambientes controlados, utilize o Dockerfile disponível em `Docker/`.

