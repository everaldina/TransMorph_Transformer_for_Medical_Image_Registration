# Guia: treino, inferência e uso da transformação (TransMorph_affine)

Como treinar, inferir e **aplicar a transformação prevista em outras imagens** (artéria,
calcificação, exportar para SimpleITK). Para rodar na curupira com Docker e em background, veja
[../docker_orcascore/README.md](../docker_orcascore/README.md).

---

## 1. Configuração

Em [models/configs_TransMorph.py](models/configs_TransMorph.py), `get_3DTransMorph_Affine_config`:

| campo | valor | significado |
|---|---|---|
| `full_size` | (64, 800, 800) | tamanho dos volumes do dataset. A transformação é aplicada e a loss/métricas são calculadas aqui. |
| `img_size` | (64, 96, 96) | tamanho da **entrada da rede**. O modelo reduz sozinho de `full_size` para `img_size`. |
| `window_size` | (2, 3, 3) | precisa dividir a resolução de todos os estágios: `img_size`/4, /8, /16, /32 = 16×24×24, 8×12×12, 4×6×6, 2×3×3. |

Regras:
- `img_size` divisível por 32 em todos os eixos;
- `window_size` dividindo `img_size/32·2^k` para k = 0..3.

O modelo valida essas regras ao ser criado (`check_config`). Ao mudar `full_size`, não precisa
mudar mais nada; ao mudar `img_size`, reveja `window_size`. A config usada fica salva dentro de
cada checkpoint e no `experiment.json`.

---

## 2. Treino

```bash
cd TransMorph_affine
python train_TransMorph_affine.py --dataset_dir <dataset> --name afim_v1
# retomar de onde parou (lê checkpoints/last.pth.tar):
python train_TransMorph_affine.py --dataset_dir <dataset> --name afim_v1 --resume
```

| flag | padrão | o que faz |
|---|---|---|
| `--dataset_dir` | obrigatória | pasta do dataset (`dataset.csv`, `config.json`), ex.: `dataset_original_comum` |
| `--name` | obrigatória | nome do experimento (pasta em `--output_dir`) |
| `--output_dir` | `<repo>/experiments` | onde ficam os experimentos |
| `--train_split` / `--val_split` | `<repo>/splits/orcascore/train.csv` / `test.csv` | CSVs com a coluna `ID` (case_id); 64 e 8 pacientes |
| `--lr`, `--batch_size`, `--num_epochs` | 1e-4, 1, 500 | padrões dos autores |
| `--save_every` | 10 | guarda `epc_N.pth.tar` e uma figura a cada N épocas (`0` desliga as cópias) |
| `--resume [ckpt]` | – | retoma de `last` (ou `best`, `epc_N`, caminho) |
| `--no_flip` | – | desliga o flip aleatório |
| `--seed` | – | semente opcional |
| `--gpu` / `--device` | 0 / cuda | GPU a usar; `--device cpu` para testar |

O que acontece em cada iteração (receita dos autores):
1. `x` = CTAI (móvel), `y` = CTI (fixa), ambos em `full_size`, lidos do NIfTI sem renormalizar.
   O flip aleatório espelha x e y juntos.
2. `x` recebe uma afim aleatória em resolução cheia (`affine_aug`, que usa
   `random_affine_matrix`). Cada valor é sorteado de forma uniforme, por eixo, dentro de:
   - rotação: ±`ANGLE_RANGE_DEGREES` (10°);
   - escala: 1 ± `SCALE_RANGE` (0,1, ou seja, de 0,9 a 1,1);
   - translação: ±`TRANSLATION_RANGE` (0,1 em coordenadas normalizadas, ver seção 4; dá
     ~12,6 mm em XY e 8 mm em z).

   As constantes ficam no topo de `train_TransMorph_affine.py`. A semente é o índice da
   iteração, como no código dos autores. Com batch > 1, cada amostra recebe a sua matriz.
3. O modelo reduz `(x, y)` para `img_size` com anti-aliasing e prevê rotação, escala,
   translação e cisalhamento.
4. `AffineTransform` monta a matriz e reamostra `x` **em resolução cheia**. A loss é o MSE com `y`
   (AdamW).

**Validação (toda época):**
- é feita nos pares reais, sem augmentation, que é o que a inferência faz;
- mede SSIM por fatia e MSE **sem padding**, antes e depois da afim;
- o maior SSIM vira o `best.pth.tar`.

**Saídas** (`experiments/<name>/`):
- `experiment.json`: argumentos, config, commit, GPU e `config.json` do dataset;
- `history.csv`: uma linha por época;
- `checkpoints/`: `last`, `best` e `epc_N`;
- `figures/` e `logs/`.

---

## 3. Inferência

```bash
python infer_TransMorph_affine.py --dataset_dir <dataset> --name afim_v1 [--checkpoint best] \
    [--sets val train] [--no_lpips] [--no_save_volumes]
```

Grava em `experiments/<name>/infer_<checkpoint>/`.

Para cada par, em `<val|train>/<pair_id>/`, **todos em 64×800×800 e com o header (spacing,
origem, direção) da `cti.nii.gz` do par**:
- `ctai_afim.nii.gz`: CTAI depois da afim (bilinear);
- `arteria_afim.nii.gz`: artéria depois da afim (nearest), se o par tiver artéria;
- `transformacao.json`: `mat`, `inv_mat` (3×4) e os parâmetros (`rotation`, `scale`,
  `translation`, `shear`).

Na pasta da inferência ficam ainda:
- `metrics.csv`: uma linha por par. As colunas são `<métrica>_<estágio>_<região>`:
  - métrica: `ssim`, `lpips`, `mse`, `psnr`, `mae`, `rmse`;
  - estágio: `pre`, `affine`;
  - região: `padded` (volume inteiro) ou `cropped` (sem padding).

  Há também `pp_calc_in_artery_<estágio>`, a % da calcificação dentro da artéria. SSIM e
  LPIPS são calculados por fatia; MSE, PSNR, MAE e RMSE, no volume. Todas usam
  `data_range = 1`, porque os volumes já estão em [0, 1].
- `infer.json`: o checkpoint e os argumentos usados.

**Para os resultados, use a região `cropped`.** No padding as duas imagens valem 0 e coincidem,
o que infla as métricas.

---

## 4. O que a matriz significa

A rede não devolve a imagem alinhada: devolve uma **transformação afim**, guardada como a
matriz `mat` no `transformacao.json`. Para usá-la em outras imagens, três conceitos importam.

### 4.1 Transformação afim e a matriz `[M | t]`

Uma transformação afim leva um ponto `p = (x, y, z)` para um ponto `p'`:

```
p' = M · p + t
```

- `M` (3×3) junta rotação, escala e cisalhamento;
- `t` (3 números) é a translação, o deslocamento;
- `mat` guarda os dois juntos numa matriz 3×4: as 3 primeiras colunas são `M` e a última é `t`.

A afim mantém retas como retas e paralelas como paralelas. Ela move, gira, estica e inclina o
volume inteiro de uma vez, mas não deforma regiões separadamente (isso é o trabalho do
TransMorph deformável).

### 4.2 Reamostrar é "puxar", não "empurrar"

Para gerar a imagem de saída, o PyTorch (`affine_grid` + `grid_sample`) percorre os voxels da
**saída** e pergunta, para cada um: "de onde da imagem de entrada eu tiro este valor?". A
resposta é `M · p + t`.

> para cada voxel `p` da imagem de saída (grid da CTI), o valor vem da imagem de entrada (CTAI)
> no ponto `M · p + t`.

Consequências:
- `mat` leva coordenadas da **CTI para a CTAI**, mesmo sendo usada para trazer a CTAI até a CTI;
- o conteúdo da imagem anda no sentido **contrário** de `t`. Com `t = (0,1; 0; 0)`, o voxel `x`
  da saída pega o valor de `x + 0,1` da entrada, então a imagem se desloca 0,1 para o lado
  negativo de x;
- `inv_mat` é a inversa de `mat` e faz o caminho de volta.

Qual matriz usar:

| quero levar... | para... | uso | interpolação |
|---|---|---|---|
| algo no espaço da CTAI (artéria, a própria CTAI) | espaço da CTI | **`mat`** | imagem: bilinear; rótulo: nearest |
| algo no espaço da CTI (calcificação, a CTI) | espaço da CTAI | **`inv_mat`** | idem |

### 4.3 Coordenadas normalizadas

Os pontos `p` não estão em voxels nem em milímetros. Em cada eixo, o volume inteiro vai de **−1
a +1**:
- −1 é a borda externa do primeiro voxel, +1 é a borda externa do último, e 0 é o centro do
  volume (convenção `align_corners=False`);
- o voxel de índice `i`, num eixo com `n` voxels, fica em `(2i + 1) / n − 1`;
- a ordem do vetor é **(x, y, z)**, o contrário da ordem do array, que é `(z, y, x)`.

O que isso significa neste dataset (caixa de 160 mm em z e ~251,6 mm em XY):
- 1 unidade normalizada vale metade da extensão do eixo: ~125,8 mm em x e y, 80 mm em z;
- a translação `t = 0,1` vale ~12,6 mm em XY e 8 mm em z.

Como as coordenadas só dependem da caixa, e não da quantidade de voxels, **a mesma matriz vale
em qualquer resolução da mesma caixa**. É por isso que a rede pode estimar a matriz olhando
64×96×96 e ela ser aplicada sem ajuste em 64×800×800.

### 4.4 Quando a matriz deixa de valer

A matriz está presa à caixa física da entrada da rede: o volume do par **com padding**. Ela
deixa de valer se a caixa mudar, por exemplo depois de cortar o padding, recortar ou reamostrar
com outra extensão. **Aplique sempre no volume com padding, e só depois corte o padding.**

---

## 5. Artéria já pareada → espaço da CTI

A `arteria.nii.gz` do dataset está no grid do par (o mesmo de `ctai.nii.gz` e `cti.nii.gz`, com
padding). Ela acompanha a CTAI, então usa `mat`:

```python
import json
import numpy as np, torch, torch.nn.functional as F, SimpleITK as sitk

res = json.load(open('experiments/<name>/infer_best/val/<pair_id>/transformacao.json'))
mat = torch.tensor(res['mat'], dtype=torch.float32)[None] # shape (1, 3, 4)

art_img = sitk.ReadImage('<pasta_do_par>/arteria.nii.gz')
art = torch.from_numpy(sitk.GetArrayFromImage(art_img).astype(np.float32))[None, None]  # shape (1,1,z,y,x)

grid = F.affine_grid(mat, art.shape, align_corners=False)   # grid do tamanho do volume
art_w = F.grid_sample(art, grid, mode='nearest', align_corners=False)[0, 0].numpy().astype(np.uint8)

# salvar no grid da CTI (mesma geometria do par)
out = sitk.GetImageFromArray(art_w)
out.CopyInformation(art_img)
sitk.WriteImage(out, 'arteria_affine.nii.gz')

# só depois, se quiser medir sem padding:
# from orcascore.metrics import crop_padding
# art_sem_pad = crop_padding(art_w, json.load(open('<pasta_do_par>/metadados.json'))['padding'])
```

Alternativa a `nearest` com bordas menos serrilhadas: `mode='bilinear'` e depois `>= 0.5`.

**Calcificação (espaço da CTI) → espaço da CTAI:** mesmo código com `inv_mat`.

**Rótulo em outro grid**: primeiro leve-o para o grid do par (seção "Reaplicar o alinhamento
numa segmentação nova" do `README_pipeline_dataset.md`, no repositório do alinhamento), e **só
então** aplique `mat`.

---

## 6. Exportar para mm / SimpleITK

Para usar a transformação fora do PyTorch (3D Slicer, ITK, `sitk.Resample`). Testado contra o
`grid_sample`:

```python
import numpy as np, SimpleITK as sitk

def mat_to_sitk(mat, ref_img):
    """mat: 3x4 normalizado (transformacao.json). ref_img: sitk.Image do par (cti.nii.gz, COM padding)."""
    mat = np.asarray(mat, dtype=np.float64)
    M, t = mat[:, :3], mat[:, 3]
    size = np.array(ref_img.GetSize(), dtype=np.float64)       # (x, y, z)
    sp = np.array(ref_img.GetSpacing(), dtype=np.float64)
    S = np.diag(size * sp / 2)                                  # normalizado -> mm (meia extensão)
    D = np.array(ref_img.GetDirection()).reshape(3, 3)
    c = np.array(ref_img.TransformContinuousIndexToPhysicalPoint(((size - 1) / 2).tolist()))
    tr = sitk.AffineTransform(3)
    tr.SetMatrix((D @ S @ M @ np.linalg.inv(S) @ D.T).flatten().tolist())
    tr.SetCenter(c.tolist())
    tr.SetTranslation((D @ S @ t).tolist())
    return tr   # fixa(CTI) -> móvel(CTAI) em mm, mesma convenção de "puxar" do SimpleITK

cti_img = sitk.ReadImage('<pasta_do_par>/cti.nii.gz')
art_img = sitk.ReadImage('<pasta_do_par>/arteria.nii.gz')
tr = mat_to_sitk(res['mat'], cti_img)   # res: o transformacao.json da inferência (seção 5)
ctai_w = sitk.Resample(sitk.ReadImage('<pasta_do_par>/ctai.nii.gz'), cti_img, tr, sitk.sitkBSpline, 0.0)
art_w = sitk.Resample(art_img, cti_img, tr, sitk.sitkNearestNeighbor, 0)
```

Observação: as caixas têm extensões físicas diferentes por eixo (160 mm em z, ~251,6 mm em XY).
Uma "rotação" em coordenadas normalizadas vira, em mm, rotação com uma leve escala/cisalhamento.
Isso não é erro (o modelo já é afim completo), mas os parâmetros `rotation`/`scale`/`shear` **não** são
ângulos e escalas físicas. Para reportar ângulos em graus, decomponha a matriz em mm.

