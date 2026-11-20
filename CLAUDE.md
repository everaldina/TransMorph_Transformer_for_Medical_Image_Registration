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
