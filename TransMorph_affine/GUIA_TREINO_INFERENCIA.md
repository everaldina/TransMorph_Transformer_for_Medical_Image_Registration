# Guia: treino, inferência e uso da transformação (TransMorph_affine)

Como treinar, inferir e, principalmente, **aplicar a transformação prevista em outras imagens
sem errar** (artéria, calcificação, exportar para SimpleITK).

---

## 1. Estado atual

| item | status |
|---|---|
| Config 96×800×800 (dados) e 96×96×96 (rede), janela (3,3,3) | ✅ |
| Redução da entrada com anti-aliasing (`resize_antialias`), dentro do modelo | ✅ testado |
| Inferência salva `mat` e `inv_mat` | ✅ |
| **Loader do dataset NIfTI** (`pipeline_dataset.py`) | ❌ o `OrCaScoreDataSet` ainda lê `.pkl` (`ctai_aligned`/`cti`) |
| Tirar `MinMax_norm` e `PadToSize` do treino | ❌ ver `Alinhamento_Axial_LPIPS/NOTAS_TRANSMORPH.md` item 5 e 7 |
| `get_clean_data` com `padding_end == 0` devolve vazio | ❌ ver NOTAS item 3 |
| Early stopping / salvar melhor modelo | ❌ hoje salva a cada 10 épocas |

Enquanto o loader não for trocado, os comandos abaixo valem para os `.pkl` antigos.

---

## 2. Configuração

Em [models/configs_TransMorph.py](models/configs_TransMorph.py), `get_3DTransMorph_Affine_config`:

| campo | valor | significado |
|---|---|---|
| `full_size` | (96, 800, 800) | tamanho dos volumes do dataset. A transformação é aplicada e a loss/métricas são calculadas aqui. |
| `img_size` | (96, 96, 96) | tamanho da **entrada da rede**. O modelo reduz sozinho de `full_size` para `img_size`. |
| `window_size` | (3, 3, 3) | precisa dividir a resolução de todos os estágios: `img_size`/4, /8, /16, /32 = 24, 12, 6, 3. |

Regras: `img_size` divisível por 32; `window_size` dividindo `img_size/32·2^k` para k = 0..3. O
modelo valida isso ao ser criado (`check_config`). Se mudar `full_size`, não precisa mudar mais
nada; se mudar `img_size`, reveja `window_size`.

Checkpoints treinados com outro `img_size` não carregam (os heads mudam de tamanho).

---

## 3. Treino

```bash
cd TransMorph_affine
python train_TransMorph_affine.py --train_dir <pasta_treino> --val_dir <pasta_val> \
    --save <nome_experimento> --num_epochs 500 --lr 1e-4 --batch_size 1
# continuar: --continue_train --epoch_start <época do checkpoint>
```

O que acontece em cada iteração:
1. `x` = CTAI (móvel), `y` = CTI (fixa), ambos em `full_size`.
2. `x` recebe uma afim aleatória (`affine_aug`: ±10° por eixo, ±10% escala, ±0,1 translação
   normalizada), em resolução cheia.
3. O modelo reduz `(x, y)` para `img_size` com anti-aliasing e prevê rotação, escala,
   translação e cisalhamento.
4. `AffineTransform` monta a matriz e reamostra `x` **em resolução cheia**; loss = MSE com `y`.

Saídas: `experiments/<save>/epc_<n>.pth.tar`, imagens em `experiments/<save>/images/` e log em
`logs/<save>/`. GPU: `GPU_iden = 0` no fim do script (troque para 1 para rodar outro experimento
na segunda placa).

---

## 4. Inferência

```bash
python infer_TransMorph_affine.py --val_dir <pasta> --save <nome_experimento> --epoch <n> \
    [--has_artery_label] [--calc_padding] [--train_infer --train_dir <pasta> --save_train]
```

Para cada caso grava `experiments/<save>/infer/<id>.pkl` com:
- `x`, `x_trans`, `y`, e `artery` (se `--has_artery_label`), **já sem o padding em z**;
- `transformation`: `aff`, `scl`, `transl`, `shr` (parâmetros) e **`mat`, `inv_mat`** (3×4).

E `results.csv` (e `results_clean.csv` com `--calc_padding`) com as métricas.

Resultados de inferências antigas não têm `mat`; rode a inferência de novo.

---

## 5. O que a matriz significa (leia antes de usar)

`mat` é uma matriz 3×4 `[M | t]` em **coordenadas normalizadas do PyTorch**, no sentido
**"puxar"** (é o que `affine_grid` + `grid_sample` fazem):

> para cada voxel `p` da imagem de saída (grid da CTI), o valor é buscado na imagem de entrada
> (CTAI) no ponto `M·p + t`.

Consequências:

| quero levar... | para... | uso | interpolação |
|---|---|---|---|
| algo no espaço da CTAI (artéria, a própria CTAI) | espaço da CTI | **`mat`** | imagem: bilinear; rótulo: nearest |
| algo no espaço da CTI (calcificação, a CTI) | espaço da CTAI | **`inv_mat`** | idem |

**Coordenadas normalizadas** (`align_corners=False`):
- a ordem do vetor é **(x, y, z) = (último, penúltimo, primeiro eixo)** do array `(z, y, x)`;
- −1 e +1 são as **bordas externas** do primeiro e do último voxel; 0 é o centro do volume;
- por isso a mesma matriz vale em **qualquer resolução da mesma caixa física**: foi estimada em
  96×96×96 e é aplicada em 96×800×800 sem ajuste (testado: reduzir-e-transformar ≈
  transformar-e-reduzir, mesmo centroide).

A matriz **deixa de valer** se a caixa mudar: depois de cortar o padding, recortar, ou
reamostrar com outra extensão. **Aplique sempre no volume com o mesmo tamanho físico da entrada da
rede (com padding), e só depois corte o padding.**

---

## 6. Receita: artéria já pareada → espaço da CTI

A `arteria.nii.gz` do dataset está no grid do par (o mesmo de `ctai.nii.gz` e `cti.nii.gz`, com
padding). Ela acompanha a CTAI, então usa `mat`:

```python
import pickle, json
import numpy as np, torch, torch.nn.functional as F, SimpleITK as sitk

res = pickle.load(open('experiments/<save>/infer/<id>.pkl', 'rb'))
mat = torch.from_numpy(res['transformation']['mat']).float()[None]          # (1, 3, 4)

art_img = sitk.ReadImage('<pasta_do_par>/arteria.nii.gz')
art = torch.from_numpy(sitk.GetArrayFromImage(art_img).astype(np.float32))[None, None]  # (1,1,z,y,x)

grid = F.affine_grid(mat, art.shape, align_corners=False)                     # grid do tamanho do volume
art_w = F.grid_sample(art, grid, mode='nearest', align_corners=False)[0, 0].numpy().astype(np.uint8)

# salvar no grid da CTI (mesma geometria do par)
out = sitk.GetImageFromArray(art_w); out.CopyInformation(art_img)
sitk.WriteImage(out, 'arteria_affine.nii.gz')

# só depois, se quiser medir sem padding:
# from pipeline_dataset import crop_padding
# art_sem_pad = crop_padding(art_w, json.load(open('<pasta_do_par>/metadados.json'))['padding'])
```

Alternativa a `nearest` com bordas menos serrilhadas: `mode='bilinear'` e depois `>= 0.5`.

**Calcificação (espaço da CTI) → espaço da CTAI:** mesmo código com `inv_mat`.

**Rótulo em outro grid** (ex.: o `pre_label.nii.gz` original, no grid da CTAI original): primeiro
leve-o para o grid do par (seção "Reaplicar o alinhamento numa segmentação nova" do
`README_pipeline_dataset.md`: `apply_alignment` → mesmo resample → mesmo padding), e **só então**
aplique `mat`.

---

## 7. Receita: exportar para mm / SimpleITK

Para usar a transformação fora do PyTorch (3D Slicer, ITK, `sitk.Resample`). Testado contra o
`grid_sample` (erro ~5e-7, inclusive com direção invertida):

```python
import numpy as np, SimpleITK as sitk

def mat_to_sitk(mat, ref_img):
    """mat: 3x4 normalizado (do pkl). ref_img: sitk.Image do par (cti.nii.gz, COM padding)."""
    M, t = mat[:, :3].astype(np.float64), mat[:, 3].astype(np.float64)
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
tr = mat_to_sitk(res['transformation']['mat'], cti_img)   # res: o pkl da inferência (seção 6)
ctai_w = sitk.Resample(sitk.ReadImage('<pasta_do_par>/ctai.nii.gz'), cti_img, tr, sitk.sitkBSpline, 0.0)
art_w = sitk.Resample(art_img, cti_img, tr, sitk.sitkNearestNeighbor, 0)
```

Observação: as caixas têm extensões físicas diferentes por eixo (~240 mm em z, ~251 mm em XY).
Uma "rotação" em coordenadas normalizadas vira, em mm, rotação com uma leve escala/cisalhamento.
Isso não é erro (o modelo já é afim completo), mas os parâmetros `aff`/`scl`/`shr` **não** são
ângulos e escalas físicas. Para reportar ângulos em graus, decomponha a matriz em mm.

---

## 8. Checklist para não errar

- [ ] Artéria/CTAI → CTI com `mat`; calcificação/CTI → CTAI com `inv_mat`.
- [ ] Aplicar no volume **com padding** (mesma caixa física da entrada da rede); cortar depois.
- [ ] `align_corners=False` em `affine_grid` **e** `grid_sample`.
- [ ] Array em `(z, y, x)` (SimpleITK `GetArrayFromImage`). Com MONAI `LoadImaged`, que costuma
      dar `(x, y, z)`, transponha antes.
- [ ] `nearest` (ou bilinear + limiar) para rótulos; nunca bilinear sem limiar.
- [ ] Rótulo fora do grid do par: primeiro o alinhamento do pipeline, depois `mat`.
- [ ] Ao encadear com o TransMorph deformável: use `warp_affine_then_flow` (afim + campo numa
      interpolação só). Ver [../TransMorph/GUIA_TREINO_INFERENCIA.md](../TransMorph/GUIA_TREINO_INFERENCIA.md).
