# Registro de mudanças metodológicas

Este repositório é parte de um mestrado. Toda mudança de código que **altera o método** (o que um
artigo descreveria na seção de Metodologia) deve ser registrada no log de mudanças do repositório:

`G:\Meu Drive\Mestrado\mudancas\<nome_repo>.md`

onde `<nome_repo>` é o nome da pasta raiz do repositório git (aqui:
`TransMorph_Transformer_for_Medical_Image_Registration`).

## Quando registrar

Registre quando a mudança altera o que foi feito ou como foi feito, por exemplo:
- arquitetura do modelo (camadas, dimensões, hiperparâmetros, como um tamanho é calculado);
- pré-processamento e dados (tamanho de imagem, espaçamento, reamostragem, normalização, padding, crop, split);
- treino (loss, otimizador, learning rate, épocas, augmentation, batch size);
- avaliação (definição de uma métrica, protocolo de validação);
- correção de bug que muda resultados ou desvia do artigo original.

**Não** registre o que não muda o método: caminhos e pastas de saída, logging, flags de conveniência
(ex.: flag a mais para calcular uma métrica que já existia), refatoração sem efeito no resultado,
plots, nomes de arquivos, ambiente/Docker.

## Como registrar

1. Ao final de toda alteração metodológica, leia o arquivo de log (crie se não existir).
2. Verifique se a mudança já foi anotada. Se já estiver, só atualize a entrada quando houver novidade.
3. Se não estiver, acrescente uma entrada no fim com:
   - data (AAAA-MM-DD) e título curto;
   - **O que mudou**: antes → depois, com os valores;
   - **Por quê**: a justificativa técnica;
   - **Onde**: arquivos/funções afetados;
   - **Impacto / diferença em relação ao artigo original**, quando houver.
4. Escreva em português, em tom de texto de metodologia (pronto para adaptar no artigo).
   Não invente números: o que não estiver confirmado, marque como `[A CONFIRMAR]`.
5. Avise na resposta ao usuário que o log foi atualizado (ou que a mudança não entra em metodologia).

# Contexto do trabalho (resumo da sessão de 2026-11-18)

O que já foi decidido e implementado está no log de mudanças (acima) e nos guias. Esta seção guarda
só o contexto que não está lá.

## Dados e ambiente
- Pares do OrCaScore: CTAI = móvel (`x`), CTI = fixa (`y`). Os dados são gerados no repositório
  irmão `../Alinhamento_Axial_LPIPS` (`pipeline_dataset.py`). Leia `README_pipeline_dataset.md`,
  `RESUMO.md` e, principalmente, `NOTAS_TRANSMORPH.md`, que tem o checklist dos ajustes de leitura.
- Os pares já chegam pré-alinhados em z e em translação XY pelo PDR proposto. O affine corrige só
  um resíduo; no treino, o desalinhamento vem do `affine_aug`.
- Máquina de treino: `curupira`, com 2× Quadro RTX 5000 de **16 GB**. O Docker usa torch 2.0.1
  (cu118). Os scripts usam uma GPU só (`GPU_iden`); a segunda serve para rodar outro experimento
  em paralelo.
- O espaçamento final (0,314 ou 0,383 mm em XY) será conferido pelo usuário. Se mudar, basta ajustar
  `full_size` nas configs: a rede reduz para `img_size` sozinha.

## Decisões do usuário (não reabrir sem motivo)
- Evitar acrescentar etapas de metodologia além do necessário: a preparação dos dados já tem uma
  metodologia grande.
- A interpolação tem que ser cuidadosa, sem perder informação: anti-aliasing antes de reduzir e
  geometria preservada. A referência de estilo é `upsample_xy` em
  `../Alinhamento_Axial_LPIPS/preprocessing.py` (B-spline, centro preservado, clamp).
- Affine: entrada da rede reduzida (opção B). Pooling antes dos heads e manter 800×800 foram
  descartados.
- Deformável: rede em resolução menor com o campo ampliado (opção 1). Recorte em volta do coração e
  treino por patches foram descartados.
- Early stopping: ainda não decidido. Se implementar, entra no log, porque muda o protocolo de
  treino.

## Guias de uso (convenções da matriz e do campo, receitas para a artéria)
- `TransMorph_affine/GUIA_TREINO_INFERENCIA.md`
- `TransMorph/GUIA_TREINO_INFERENCIA.md`

## Pendências, em ordem
1. **Loader NIfTI** para o dataset novo. Bloqueia o treino: o `OrCaScoreDataSet` ainda lê `.pkl`.
   Junto: tirar `MinMax_norm` e `PadToSize` e corrigir `get_clean_data` (vazio com
   `padding_end == 0`). Ver itens do `NOTAS_TRANSMORPH.md`.
2. Adaptar `TransMorph/train_TransMorph.py` e `infer_TransMorph.py`, que ainda são os do IXI. A
   receita do loop está no guia do deformável.
3. Medir a memória do deformável na GPU (snippet na seção 7 do guia). Se não couber:
   `use_checkpoint=True` ou TransMorph-Small.
4. Encadear affine → deformável no pipeline, com `warp_affine_then_flow` para os rótulos.

## Estado dos testes
- Validado só em CPU, com volumes sintéticos: forward, anti-aliasing, transferência da matriz e do
  campo entre resoluções, conversão para SimpleITK e composição afim + campo.
- Ainda não rodou com dados reais nem em GPU.
