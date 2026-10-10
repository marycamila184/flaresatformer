# FlareSatFormer 2.0 — Design do experimento

**Data:** 2026-10-09
**Status:** aprovado em conversa, aguardando revisão da spec escrita

## 1. Objetivo

Refazer o experimento do FlareSatFormer como uma comparação rigorosa e reprodutível entre CNN, transformers e foundation models para a segmentação de gas flares em Landsat 8. Os alvos são subpixel: cerca de 14 pixels positivos por patch de 256×256, ou 0,02%.

**Hipótese de pesquisa:** "transformers e foundation models vs. CNN para alvos subpixel". O resultado é publicável mesmo que o UNet vença, desde que a comparação seja justa: mesmo protocolo, mesmo orçamento de ajuste e saída em alta resolução para todos.

**Critério de sucesso:**
- As prioridades 1 e 2 da matriz (Seção 5) concluídas.
- Tabelas com média ± desvio padrão das seeds, por fold.
- Intervalos de confiança por bootstrap de ΔF1 em relação ao UNet.
- Métricas por objeto e de falso positivo nos negativos.
- Tudo reprodutível a partir dos splits e configs versionados.

## 2. Restrições

- **Hardware:** apenas um notebook com RTX 4070 Laptop (8 GB de VRAM), 24 threads e 62 GB de RAM.
- **Dados:** 37 GB de TIFF float32 no pendrive KINGSTON (USB). O SSD interno tem cerca de 238 GB livres.
- **Ambiente:** o pendrive não suporta symlinks, e o `venv/` atual está quebrado. O novo venv precisa ficar fora do pendrive.
- **Volume de treino:** cerca de 116 treinos ao todo. O notebook vai ser desligado durante o processo, então a retomada de treino é obrigatória.

## 3. Dados

### 3.1 Inventário

| Fonte | Patches | Máscara | Agrupamento disponível |
|---|---|---|---|
| Flare (`dataset/flare_dataset.csv`) | 11.505 | 0/255, cerca de 14 px positivos por patch | `continent`, `cntry_name` (83 países); cena e path/row extraídos do nome do arquivo (`fire_LC8PPPRRRYYYYDDD...`) |
| Urbano (`dataset/urban_dataset.csv`) | 683 | Toda zero (negativo difícil) | `city` (239 cidades) |
| Fogo (`dataset/fire_dataset.csv`) | 265 | Toda zero (negativo difícil) | Path/row extraído do nome do arquivo (`LC08_L1TP_PPPRRR_...`) |

Vulcões ficam fora do escopo.

**Formato dos TIFFs:** 256×256×10, float32, reflectância TOA de 0 a 1. A ordem dos canais, confirmada no FlareSat (`models/transfer_learning/unet_sentinel_landcover.py`), é:

| Índice | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Banda | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B9 | B10 | B11 |

### 3.2 Configurações de bandas

- **Principal:** B2, B6, B7, ou seja, índices `[1, 5, 6]`. Usada por todos os modelos.
- **Variante HLS:** B2 a B7, ou seja, índices `[1, 2, 3, 4, 5, 6]`. Usada só pelo Prithvi (prioridade 3).

### 3.3 Splits (`scripts/make_splits.py` → `splits/fold{1..4}.csv`)

Gerados uma única vez, de forma determinística (`random_state=42`), e versionados no git. **As seeds de treino não alteram os splits.**

**Fold de teste:**

| Fonte | Agrupamento |
|---|---|
| Flare | `GroupKFold(4)` por `continent`. Gera Ásia (4.570) \| África (2.581) \| Europa+Oceania (2.283) \| Américas (2.071). O desbalanceamento é reportado no paper. |
| Urbano | `GroupKFold(4)` por `city` |
| Fogo | `GroupKFold(4)` por path/row WRS-2. Hoje esse split é aleatório; passa a ser espacial. |

**Validação:** cerca de 10% dos patches de treino de cada fold, via `GroupShuffleSplit` agrupado por **path/row WRS-2**, separadamente para cada fonte. Isso impede que a mesma instalação, em outra data, apareça em treino e validação ao mesmo tempo.

**Colunas de cada CSV:** `tiff_file, mask_file, source{flare,urban,fire}, group, path_row, scene, country, continent, role{train,val,test}`. Para urbano e fogo, `country` e `continent` ficam vazios.

### 3.4 Cache (`scripts/build_cache.py`)

- Lê os TIFFs uma vez e grava, por configuração de bandas, um array `float16` com shape `(N, C, 256, 256)` e as máscaras em `uint8` `{0, 1}`, mais um índice `tiff_file → linha`.
- **Destino configurável** (`cache_dir` em `configs/base.yaml`), com padrão `~/.cache/flaresatformer`.
- Tamanho: cerca de 5 GB com 3 bandas e cerca de 10 GB com 6 bandas.
- O dataset carrega o cache inteiro em RAM (memmap seguido de cópia), e o treino deixa de depender do disco.
- O cache é validado pelo número de patches e por um hash da lista de arquivos e bandas. Se não bater, é reconstruído.

### 3.5 Dataset (`flaresatformer/data/dataset.py`)

- **Saída:** `image (C, 256, 256) float32` normalizada e `mask (1, 256, 256) float32` com valores {0, 1}. Não há nenhum resize.
- **Normalização:** média e desvio padrão por banda, calculados **só sobre o `role=train` de cada fold**, salvos em `outputs/.../norm.json`.
  - **Exceção do Prithvi:** usa a média e o desvio padrão HLS do checkpoint, convertendo a reflectância 0–1 para a escala 0–10000 esperada pelo modelo. A diferença TOA vs. reflectância de superfície é discutida no paper.
- **Augmentation (só no treino):** flip horizontal, flip vertical e rot90 k∈{0, 1, 2, 3}, sempre aplicados juntos à imagem e à máscara. Não há augmentation de brilho, contraste ou cor, para preservar o sinal físico do SWIR.

## 4. Modelos (`flaresatformer/models/`)

Interface única: `build_model(cfg) -> nn.Module`, com `forward(x: (B, C, 256, 256)) -> logits (B, 1, 256, 256)`. Cada modelo expõe ainda `param_groups(lr)`, que separa encoder pré-treinado de decoder/head para aplicar o lr diferencial.

### 4.1 Cabeça de alta resolução (FRH)

Módulo compartilhado `FullResHead(in_ch, dec_ch)`:
1. Duas camadas conv3×3, BN e ReLU sobre a entrada original (256×256), com 32 canais.
2. O mesmo é feito sobre a saída do decoder, após upsample bilinear para 256×256.
3. As duas saídas são concatenadas e passam por conv3×3, BN, ReLU e conv1×1, gerando o logit.

Aplicada a todos os modelos exceto o UNet, que já tem skip em resolução cheia. Pode ser desligada com `frh: false` (ablação).

### 4.2 Modelos

| ID | Arquitetura | Origem | Pré-treino | Encoder no fine-tuning |
|---|---|---|---|---|
| `unet` | UNet atual (`base_filters=32`, dropout 0.1), mantido por continuidade com o FlareSat | Código próprio | Nenhum | — |
| `segformer` | MiT-B2 + head MLP + FRH | HF `transformers` (`SegformerForSemanticSegmentation`, `nvidia/mit-b2`) | ImageNet-1k | Completo |
| `segformer_scratch` | Idêntico ao `segformer` | HF `SegformerConfig` com inicialização aleatória | Nenhum | Completo |
| `swin_unet` | SwinV2-T (`swinv2_tiny_window8_256`, `features_only`) + decoder UNet (4 estágios) + FRH | `timm` | ImageNet-1k | Completo |
| `transunet` | R50-ViT-B/16 híbrido (`vit_base_r50_s16_224`, pos-embed interpolado de 14² para 16²) + decoder CUP com skips do ResNet + FRH | `timm` + decoder próprio | ImageNet-21k | Completo |
| `prithvi` | Prithvi-EO-2.0-300M + decoder UperNet + FRH | `terratorch` | HLS | LoRA (r=16, α=32) em qkv/proj; decoder e FRH treináveis |

**Primeira camada com 3 bandas:** os modelos ImageNet usam os pesos RGB como estão (3→3 canais). O Prithvi seleciona as bandas pelo nome (`BLUE, SWIR_1, SWIR_2` ou as 6 HLS), descartando os pesos de patch embedding das bandas ausentes.

**Memória (8 GB):**
- bf16 e gradient checkpointing quando necessário (`grad_checkpointing: true` no YAML do modelo).
- Batch físico ajustado por modelo e batch efetivo de 16 via accumulation.
- **Fallback do Prithvi:** se o LoRA não couber com batch físico 2, usar o encoder congelado. A decisão é registrada no YAML e no paper.

### 4.3 Removidos

`models/swinunet_model.py` (implementação quebrada), `models/segformer_model.py` e `models/transunet_model.py` (substituídos pelas versões pré-treinadas) e o pipeline antigo de `train/`. Tudo continua disponível no histórico do git.

## 5. Matriz de experimentos (`configs/experiments.yaml`)

| Prioridade | Conjunto | Treinos |
|---|---|---|
| 0 | Ajuste de lr: 6 modelos × lr {1e-4, 3e-4, 1e-3}, fold 1, seed 1, 30 épocas, avaliado só na validação | 18 (curtos) |
| 1 | 6 modelos × 4 folds × seed 1, bandas principais | 24 |
| 2 | 6 modelos × 4 folds × seeds 2 e 3 | 48 |
| 3 | `prithvi` com 6 bandas HLS e `segformer` com `frh: false`, ambos × 4 folds × seed 1 | 8 |
| 4 | Curva de aprendizado: `unet`, `segformer` e `prithvi` × frações {10, 25, 50}% do treino (amostradas por path/row) × 4 folds × seed 1 | 36 |

A fração de 100% da curva de aprendizado vem da prioridade 1. A subamostragem por fração é determinística e aninhada: os 10% estão contidos nos 25%, que estão contidos nos 50%.

## 6. Protocolo de treino (`configs/base.yaml`, `flaresatformer/engine.py`)

Igual para todos os modelos. O único hiperparâmetro ajustado por modelo é o lr (prioridade 0), escolhido pelo melhor F1 de validação.

| Item | Valor |
|---|---|
| Loss | `FocalLoss(α=0.25 para positivos e 0.75 para negativos, γ=2)` + `DiceLoss` em nível de batch (soma sobre o batch inteiro, smooth=1), com pesos 1:1 |
| Otimizador | AdamW, weight decay 0.01. Encoder pré-treinado com 0.1× o lr; demais parâmetros com o lr do modelo |
| Scheduler | Warmup linear de 5% dos steps, depois cosine até 0 |
| Épocas | Máximo de 80, early stopping com paciência 15 sobre o F1 de validação com threshold 0.5 |
| Batch | Efetivo de 16, com batch físico por modelo e accumulation |
| Precisão | bf16 autocast, gradient clipping na norma 1.0 |
| Determinismo | Seed global (python, numpy, torch, workers do DataLoader), `cudnn.deterministic=True`, `cudnn.benchmark=False` |
| Seleção | `best.pt` é o maior F1 de validação com 0.5 |
| Threshold | Após o treino, grade de 0.05 a 0.95 (passo 0.05) na validação; o melhor vai para `threshold.json`. O teste é reportado com 0.5 e com o threshold ajustado |

### 6.1 Organização de saída e retomada

```
outputs/{model}/{bands}/{frac}/fold{k}/seed{s}/
  config.yaml      # config resolvida + hash do commit git + versões das libs
  norm.json        # estatísticas de normalização do fold
  history.csv      # época, train_loss, val_loss, val_P, val_R, val_F1, lr, tempo
  last.pt          # modelo, otimizador, scheduler, scaler, RNG, época, contador do early stopping
  best.pt
  threshold.json
  test_patches.csv # estatísticas por patch (Seção 7.1)
  DONE             # marcador de conclusão
```

- O `scripts/run_queue.py` expande o `experiments.yaml` em ordem de prioridade, **pula os treinos com `DONE`** e retoma os que têm `last.pt`.
- `--only-priority N` e `--dry-run` listam o que vai rodar.
- `outputs/` vai para o `.gitignore`. Só `results/` é versionado.

## 7. Avaliação (`scripts/evaluate.py`, `scripts/aggregate.py`)

### 7.1 Estatísticas por patch (`test_patches.csv`)

Para cada patch de teste e cada threshold (0.5 e o ajustado):
- `tp, fp, fn` em pixels.
- `gt_objects, gt_detected, pred_objects, pred_fp_objects`.
- `source, country, continent, path_row`.

Todas as métricas agregadas são recalculadas a partir desse arquivo.

### 7.2 Métricas

- **Pixel:** P, R, F1 e IoU da classe flare (micro, somadas no fold). Também separadas por continente.
- **Objeto:** componentes conexas com conectividade 8 (`scipy.ndimage.label`).
  - Um objeto real conta como detectado se houver algum pixel previsto na sua máscara dilatada de 1 px.
  - Um objeto previsto é falso positivo se não tocar nenhum objeto real dilatado de 1 px.
  - Daí saem P, R e F1 por objeto.
- **Negativos (urbano e fogo, separadamente):** taxa de alarme falso (% de patches com pelo menos um objeto previsto) e média de pixels FP por patch.
- **Custo:** parâmetros totais e treináveis, tempo de treino por época, pico de VRAM e velocidade de inferência (patches/s, batch 16, bf16).

### 7.3 Estatística

- **Tabelas:** média ± desvio padrão das 3 seeds, por fold e no agregado dos 4 folds.
- **ΔF1 (pixel e objeto) de cada modelo vs. `unet`:** bootstrap por grupos, com 2.000 reamostragens dos path/rows de teste, estatísticas pareadas pelos mesmos patches e seeds em média. Reporto o IC de 95% por fold e no agregado.
- **Saídas em `results/`:** tabelas em CSV e LaTeX; figuras de curva de aprendizado, F1 por continente, custo vs. F1 e um painel qualitativo (RGB, B7, máscara real e predição de cada modelo) para um conjunto fixo de patches, incluindo urbano e fogo.

## 8. Estrutura do repositório

```
configs/
  base.yaml                  # protocolo comum, caminhos, cache_dir
  models/{unet,segformer,segformer_scratch,swin_unet,transunet,prithvi}.yaml
  experiments.yaml           # matriz e prioridades
flaresatformer/
  data/splits.py  data/cache.py  data/dataset.py
  models/factory.py  models/frh.py  models/unet.py  models/segformer.py
  models/swin_unet.py  models/transunet.py  models/prithvi.py
  losses.py  metrics.py  engine.py  utils.py
scripts/
  make_splits.py  build_cache.py  train.py  run_queue.py  evaluate.py  aggregate.py  infer.py
splits/fold{1..4}.csv
results/
tests/
docs/superpowers/specs/
```

- `train.py` aceita `--model --bands --fold --seed --frac --lr --epochs --smoke`.
- `--smoke` roda 2 épocas em 64 patches e imprime o tempo e o pico de VRAM.
- O `pyproject.toml` passa a incluir o pacote `flaresatformer*`.
- O `requirements.txt` fica com versões fixadas (torch, transformers, timm, terratorch, peft, scipy, pandas, scikit-learn, tifffile, pyyaml, pytest).

## 9. Testes (pytest)

| Área | O que é verificado |
|---|---|
| Splits | Nenhum `group` ou path/row compartilhado entre papéis no mesmo fold; cada patch é teste em exatamente um fold; mesma saída em duas execuções; proporção de validação perto de 10% |
| Cache e dataset | Shape e dtype; bandas certas (comparando com a leitura direta do TIFF); máscara em {0, 1}; a mesma transformação aplicada à imagem e à máscara |
| Losses | α_t da Focal em casos manuais; Dice de batch finito e correto com máscaras todas vazias |
| Métricas | Pixel e objeto em máscaras sintéticas com resposta conhecida, incluindo tolerância de 1 px, objeto perdido e FP isolado |
| Modelos | Forward com saída `(2, 1, 256, 256)` para 3 e 6 bandas na CPU; `param_groups` separa encoder e head. O Prithvi, por exigir download de pesos, fica marcado como `@pytest.mark.slow` |
| Engine | Treinar N steps, salvar, retomar e conferir que pesos, otimizador e RNG são idênticos a uma execução contínua |

O smoke test na GPU (`--smoke` para os 6 modelos) é executado antes da prioridade 0 e preenche a tabela de custo e memória.

## 10. Ambiente

- venv em `~/venvs/flaresatformer`, fora do pendrive. Python 3.12.
- Cache em `~/.cache/flaresatformer`.
- O `venv/` antigo do repositório deixa de ser usado. Já está no `.gitignore`.

## 11. Documentação

- O README é reescrito ao final, com os números reais (11.505 patches de flare, 83 países, 683 urbanos, 265 de fogo), o pipeline novo (splits → cache → fila → avaliação → agregação) e as instruções de reprodução.
- O `infer.py` é atualizado para carregar um treino de `outputs/` e gerar RGB, B7 e predição.

## 12. Fora de escopo

- Vulcões.
- Pré-treino self-supervised (MAE) nos próprios dados, possível trabalho futuro.
- Fine-tuning completo do Prithvi (não cabe em 8 GB).
- Bandas térmicas B10 e B11 nos modelos pré-treinados.
- Variantes maiores (MiT-B3+, Swin-B, Prithvi-600M).
