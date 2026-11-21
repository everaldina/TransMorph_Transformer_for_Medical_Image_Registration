# Docker do TransMorph no OrCaScore

Ambiente único para as duas redes: o **TransMorph-Affine** (`TransMorph_affine/`) e o
**TransMorph deformável** (`TransMorph/`). A pasta `Docker/` da raiz do repositório é a dos autores
(registro de cérebro em CPU) e não é usada aqui.

A imagem só tem o ambiente (torch 2.0.1 com CUDA 11.8 e as bibliotecas). O código não é copiado
para dentro dela: o repositório é montado quando o container sobe. Por isso, editar o código não
exige rebuild; só mudar o `requirements.txt` ou o `Dockerfile`.

## 1. Build (uma vez)

Na pasta `docker_orcascore/`, na máquina de treino:

```bash
cd docker_orcascore
docker build -t transmorph_orcascore .
```

Os pesos da AlexNet usados pelo LPIPS são baixados no build. Assim, a inferência não precisa de
internet.

## 2. Configurar os caminhos

O `run.sh` monta duas pastas:

| no host | no container | variável |
|---|---|---|
| o repositório (a pasta acima de `docker_orcascore/`) | `/workspace/repo` | `REPO_DIR` |
| a pasta com os datasets gerados pelo `pipeline_dataset.py` | `/workspace/data` | `DATA_DIR` |

Edite os valores padrão no topo do `run.sh` ou exporte as variáveis antes de chamar:

```bash
export DATA_DIR=/home/tcc-repository/orcascore/datasets   # contém dataset_original_comum/, ...
```

Os experimentos (checkpoints, logs, figuras, inferência) ficam em
`<repo>/experiments/<nome_do_experimento>/`, que está no host. Eles continuam lá depois que o container termina.
O container roda com o seu usuário (`--user`), então os arquivos não ficam como root.

## 3. Rodar em background (para desligar o notebook)

**Antes**, é preciso:
1. ter feito o build da imagem (seção 1), uma vez por máquina;
2. ter os caminhos certos no `run.sh` (seção 2);
3. ter o dataset gerado na pasta `DATA_DIR`.

Não é preciso subir nenhum container antes: cada chamada do `run.sh` cria o seu.

```bash
./run.sh <nome_do_container> <gpu> <pasta> <comando...>
```

- **nome_do_container**: um nome que você escolhe, livre, só para achar o container depois
  (acompanhar o log, parar). Não pode repetir o de um container que ainda existe (ver
  `docker rm` abaixo). Para não se perder, use o mesmo nome do experimento (`--name`), como nos
  exemplos da seção 4. Mas são coisas diferentes:
  - o nome do container é do Docker;
  - o `--name` do script é a pasta do experimento em `experiments/`.
- **gpu**: índice da GPU no host (0 ou 1 na curupira). Dentro do container ela aparece como
  `cuda:0`, por isso os scripts usam o padrão `--gpu 0`.
- **pasta**: `TransMorph_affine` ou `TransMorph`, a pasta onde está o script.
- **comando**: o comando do script, como rodaria fora do Docker (ver seção 4).

Exemplo:

```bash
./run.sh afim_v1 0 TransMorph_affine \
    python train_TransMorph_affine.py --dataset_dir /workspace/data/dataset_original_comum --name afim_v1
```

O container sobe com `docker run -d`. Ele continua rodando depois de fechar o SSH e desligar o
notebook.

| o quê | comando |
|---|---|
| acompanhar o log | `docker logs -f <nome_do_container>` (Ctrl+C só para de acompanhar, não para o treino) |
| ver o que está rodando | `docker ps` (`docker ps -a` mostra também os que já terminaram) |
| parar | `docker stop <nome_do_container>` |
| apagar o container depois que terminou | `docker rm <nome_do_container>` (necessário para reusar o nome) |
| log em arquivo | `experiments/<nome_do_experimento>/logs/train_*.log` |
| histórico por época | `experiments/<nome_do_experimento>/history.csv` |

Apagar o container não apaga nada do experimento: checkpoints, logs e resultados ficam em
`experiments/`, no host.

Para retomar um treino interrompido, rode o mesmo comando com `--resume`. Ele lê o
`checkpoints/last.pth.tar`, que é salvo a cada época. Antes, apague o container antigo
(`docker rm <nome_do_container>`) ou use outro nome de container.

### Modo interativo

Para testar ou depurar, com `-i` o container fica preso ao terminal:

```bash
./run.sh -i teste 0 TransMorph bash
```

## 4. Receita completa

Os exemplos usam o dataset `dataset_original_comum` (64×800×800). O split padrão é
`splits/orcascore/train.csv` (64 pacientes) e `test.csv` (8 pacientes). Para outro split, passe
`--train_split` e `--val_split`.

```bash
DS=/workspace/data/dataset_original_comum

# 1. treinar a afim (GPU 0)
./run.sh afim_v1 0 TransMorph_affine \
    python train_TransMorph_affine.py --dataset_dir $DS --name afim_v1 --save_every 10

# 2. inferir a afim (usa o best.pth.tar; volumes 64x800x800 + metrics.csv)
./run.sh afim_v1_infer 0 TransMorph_affine \
    python infer_TransMorph_affine.py --dataset_dir $DS --name afim_v1 --sets val train

# 3. treinar o deformável depois da afim (pode rodar na GPU 1 em paralelo com outro experimento)
./run.sh def_v1 1 TransMorph \
    python train_TransMorph.py --dataset_dir $DS --name def_v1 \
    --affine_checkpoint /workspace/repo/experiments/afim_v1/checkpoints/best.pth.tar

# 4. inferir o deformável (usa a mesma afim do treino; volumes 64x800x800 + metrics.csv)
./run.sh def_v1_infer 1 TransMorph \
    python infer_TransMorph.py --dataset_dir $DS --name def_v1 --sets val train
```

Antes do primeiro treino do deformável, meça a memória da GPU (seção 7 de
`TransMorph/GUIA_TREINO_INFERENCIA.md`). Se não couber em 16 GB, use `--use_checkpointing` ou
`--config TransMorph-Small`.

As flags de cada script aparecem com `--help`, por exemplo
`./run.sh -i ajuda 0 TransMorph python train_TransMorph.py --help`.

## 5. Sem Docker

Os scripts também rodam direto num ambiente Python com as mesmas bibliotecas (ver
`requirements.txt`, mais torch 2.0.1). Para não depender da sessão SSH:

```bash
# tmux: a sessão continua depois de desconectar
tmux new -s afim_v1
cd TransMorph_affine && python train_TransMorph_affine.py --dataset_dir ... --name afim_v1
# Ctrl+B depois D para sair sem parar; tmux attach -t afim_v1 para voltar

# ou nohup
cd TransMorph_affine
nohup python -u train_TransMorph_affine.py --dataset_dir ... --name afim_v1 > afim_v1.out 2>&1 &
```

## 6. O que cada experimento salva

```
experiments/<nome_do_experimento>/
  experiment.json            argumentos, config da rede, commit do git, GPU, config.json do dataset
  experiment_resume_*.json   o mesmo, a cada --resume
  history.csv                por época: loss de treino, SSIM e MSE da validação (pré e pós), lr, tempo
  checkpoints/
    last.pth.tar             toda época (para --resume)
    best.pth.tar             maior SSIM na validação (padrão da inferência)
    epc_<n>.pth.tar          a cada --save_every épocas
  figures/epoch_*.png        fatia axial e coronal de um par de validação
  logs/train_*.log
  infer_<checkpoint>/
    infer.json               o que foi usado na inferência
    metrics.csv              uma linha por par (ver o guia de cada rede)
    <val|train>/<pair_id>/   volumes 64x800x800 com o header da cti.nii.gz
```
