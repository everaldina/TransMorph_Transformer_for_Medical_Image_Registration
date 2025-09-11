# Build image

>> docker build -f Dockerfile -t transmorph_affine .
- Comando tem que ser executado na pasta atual


# Run

1. Rodando iterativo
```bash
docker run --rm -it --gpus all --runtime=nvidia --ipc=host\
  -v /home/everaldina/TransMorph_Transformer_for_Medical_Image_Registration/TransMorph_affine:/workspace/repositorio \
  -v /home/tcc-repository/orcascore/aligned:/workspace/orcascore/data/aligned \
  -v /home/Diagnostico_Segmentacao-Source/cycle-transformer/test_results/aligned_orca_cytran/images:/workspace/orcascore/data/aligned_orca_cytran \
  -v /home/Diagnostico_Segmentacao-Source/cycle-transformer/test_results/aligned/images:/workspace/orcascore/data/aligned_cytran \
  -v /home/tcc-repository/orcascore/datasplits:/workspace/orcascore/datasplits \
  -w /workspace \
  transmorph_affine \
  bash
```
