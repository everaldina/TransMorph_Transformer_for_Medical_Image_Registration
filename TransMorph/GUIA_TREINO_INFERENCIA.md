# Guia: TransMorph deformável no OrCaScore

Como treinar, inferir e aplicar o campo de deformação **em resolução cheia** sem errar. A parte
afim está em [../TransMorph_affine/GUIA_TREINO_INFERENCIA.md](../TransMorph_affine/GUIA_TREINO_INFERENCIA.md);
leia a seção 5 de lá (convenções da matriz) antes desta.

---

## 1. Ideia

A rede deformável devolve um deslocamento **por voxel**, então o decoder trabalha no tamanho
inteiro da entrada. Em 96×800×800 isso não cabe em 16 GB. Por isso:

1. a rede trabalha em **96×256×256** (~0,98 mm em XY), com a entrada reduzida com anti-aliasing;
2. o campo previsto é **ampliado para 96×800×800** e aplicado na imagem cheia.

## 2. Estado atual

| item | status |
|---|---|
| Config `TransMorph-OrCaScore`: `full_size` 96×800×800, `img_size` 96×256×256, janela (3,8,8) | ✅ 46,76 M parâmetros, igual à Tabela 2 do artigo |
| Redução da entrada dentro do modelo (`resize_antialias`) | ✅ |
| Ampliar e aplicar o campo (`upsample_flow`, `warp`, `model.warp_full`) | ✅ testado |
| Afim + deformável numa interpolação só (`warp_affine_then_flow`) | ✅ testado |
| **`train_TransMorph.py` / `infer_TransMorph.py` para o OrCaScore** | ❌ ainda são os do IXI (atlas `.pkl`, `JHUBrainDataset`); dependem do loader NIfTI |
| Memória real na Quadro RTX 5000 | ❌ medir (ver seção 6) |

## 3. Configuração

Em [models/configs_TransMorph.py](models/configs_TransMorph.py), `get_3DTransMorph_OrCaScore_config`:

| campo | valor | por quê |
|---|---|---|
| `full_size` | (96, 800, 800) | tamanho dos volumes do dataset |
| `img_size` | (96, 256, 256) | entrada da rede; ~6,3 M voxels, perto dos ~6,9 M do artigo |
| `window_size` | (3, 8, 8) | estágios 24×64×64, 12×32×32, 6×16×16, 3×8×8; divide todos e, como no artigo, é igual ao último estágio |

Use `CONFIGS['TransMorph-OrCaScore']`. Se faltar memória: `config.use_checkpoint = True` (no teste em
CPU economizou ~16%) ou o TransMorph-Small (`embed_dim` 48).

## 4. Treino (receita para o loop)

A loss é calculada **na resolução da rede** (o campo e a imagem deformada saem em `img_size`):

```python
from models.TransMorph import CONFIGS, TransMorph
import losses

config = CONFIGS['TransMorph-OrCaScore']
model = TransMorph(config).cuda()
sim, reg = nn.MSELoss(), losses.Grad3d(penalty='l2')
weights = [1, 0.02]   # os do train_TransMorph.py atual (MSE + difusão)

# x: CTAI JÁ PASSADA PELA AFIM, y: CTI, ambos (1, 1, 96, 800, 800)
out, flow = model(torch.cat((x, y), dim=1))      # out e flow em 96x256x256
y_low = model.downsample(y)                      # mesma redução que a rede aplicou
loss = weights[0] * sim(out, y_low) + weights[1] * reg(flow, y_low)
```

- A entrada `x` é a CTAI depois da afim (o `x_trans` do TransMorph_affine), na resolução cheia.
- Não reduza `y` com `F.interpolate` direto: use `model.downsample`, senão a fixa e a móvel ficam
  em resoluções "diferentes" (uma com anti-aliasing e a outra sem).

## 5. Inferência e aplicação em resolução cheia

```python
from models.TransMorph import warp_affine_then_flow

with torch.no_grad():
    out_low, flow = model(torch.cat((x_aff, y), dim=1))   # flow em 96x256x256

    # (a) imagem já afim -> deformável, em 96x800x800
    x_def = model.warp_full(x_aff, flow)                      # bilinear

    # (b) RECOMENDADO para rótulos: CTAI original -> afim -> deformável numa interpolação só
    #     mat = res['transformation']['mat'] da inferência do TransMorph_affine
    art_def = warp_affine_then_flow(art, mat, flow, mode='nearest')
```

- `art` é a `arteria.nii.gz` do par (grid do par, **com padding**), como tensor `(1, 1, z, y, x)`.
- Use `warp_affine_then_flow` em vez de aplicar a afim e depois o campo: duas reamostragens
  `nearest` seguidas acumulam erro. No teste sintético, a versão composta errou ~35% menos voxels
  do rótulo.
- Corte o padding **só depois** de aplicar tudo.
- Para salvar: `out = sitk.GetImageFromArray(art_def[0, 0].numpy()); out.CopyInformation(cti_img)`.

## 6. O que significa o campo (para não errar)

- `flow` tem forma `(B, 3, 96, 256, 256)`: deslocamento **em voxels da grade reduzida**, canais na
  ordem **(z, y, x)**, sentido "puxar": o voxel `p` da CTI busca a móvel em `p + flow[p]`.
- Ao ampliar (`upsample_flow`), mudam duas coisas: a **posição** de cada vetor (interpolação
  trilinear com `align_corners=False`, a mesma convenção da redução) e o **tamanho** do
  deslocamento (×800/256 = 3,125 em y e x; ×1 em z). Ampliar sem multiplicar dá deformação 3× menor.
- O `SpatialTransformer`/`warp` usa `align_corners=True` sobre índices de voxel; o `affine_grid` da
  afim usa `align_corners=False` em coordenadas normalizadas. Não troque um pelo outro. A função
  `warp_affine_then_flow` já faz a conversão.

Testes feitos (volume sintético): translação de 3 voxels reduzidos move o centroide exatamente o
esperado; para um campo suave não uniforme, deformar no cheio e reduzir ≈ reduzir e deformar
(erro máx. 0,16% do pico, mesmo centroide).

## 7. Medir a memória na GPU antes de treinar

No teste em CPU com 1/4 dos voxels (96×128×128), um passo de treino usou ~2,5 GB. Extrapolando, a
estimativa para 96×256×256 é de ~10 GB ou mais, perto do limite de 16 GB. Confirme na curupira:

```python
torch.cuda.reset_peak_memory_stats()
out, flow = model(torch.rand(1, 2, 96, 800, 800, device='cuda'))
(out.mean() + flow.mean()).backward()
print(torch.cuda.max_memory_allocated() / 2**30, 'GiB')
```
