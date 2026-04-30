# Resultados Fieis ao Paper

Esta pasta/documento registra a implementacao separada da versao fiel do baseline descrito no paper e no Kaggle release, sem uso de RGB e sem reutilizar o pipeline CHM do projeto principal.

Arquivos principais:

- Implementacao: [src/lidaragrosat/paper_faithful/method.py](/home/murilo/Documentos/LiDARAgroSat/src/lidaragrosat/paper_faithful/method.py)
- Runner: [scripts/run_faithful_paper_baseline.py](/home/murilo/Documentos/LiDARAgroSat/scripts/run_faithful_paper_baseline.py)
- Config usada na execucao: [outputs_faithful_paper/config.json](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper/config.json)
- Resumo canonico: [outputs_faithful_paper/canonical_summary.json](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper/canonical_summary.json)
- Sweep global: [outputs_faithful_paper/window_sweep_global.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper/window_sweep_global.csv)
- Sweep macro: [outputs_faithful_paper/window_sweep_macro.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper/window_sweep_macro.csv)
- Resultados por parcela no caso canonico: [outputs_faithful_paper/canonical_per_plot.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper/canonical_per_plot.csv)
- Resumo de pre-processamento: [outputs_faithful_paper/cache/preprocessing_summary.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper/cache/preprocessing_summary.csv)

## Implementacao fiel

### Pre-processamento

- Remocao de duplicatas com tolerancia de `1 mm` sobre `(x, y, z, classification)`.
- Filtro manual de ruido implementado, mas desligado na execucao final porque o release publico ja parece vendor-cleaned.
- Normalizacao de altura a partir dos pontos de solo `classification == 2`, com interpolacao linear e fallback por vizinho mais proximo.

### Baseline

- Filtro de maximos locais em `3D` com `cKDTree`.
- Vizinhança esferica com raio `window_size`.
- Descarte de pontos com `z_norm < 3 m`.
- Parametro canonico do paper: `window_size = 2.0 m`, `height_threshold = 3.0 m`.

### Matching

- Matching guloso deterministico ordenado por distancia 2D crescente.
- Restricoes: `max_dist = 5.0 m`, `max_height_diff = 3.0 m`.
- A restricao de altura so entra quando a altura de campo esta disponivel.
- Nao foi usado Hungarian matching.

## Resumo de pre-processamento

Na execucao sobre o release publico:

- o filtro manual de ruido removeu `0` pontos em todas as parcelas;
- a remocao de duplicatas retirou entre `3513` e `11611` pontos por parcela;
- as alturas normalizadas ficaram aproximadamente no intervalo `[-0.32, 34.59] m`.

## Resultado canonico do paper

Parametros:

- `window_size = 2.0 m`
- `height_threshold = 3.0 m`
- `max_dist = 5.0 m`
- `max_height_diff = 3.0 m`

### Agregado global

- `TP = 2100`
- `FP = 4977`
- `FN = 1502`
- `Precision = 0.2967`
- `Recall = 0.5830`
- `F1 = 0.3933`

### Media macro entre parcelas

- `Precision = 0.3073`
- `Recall = 0.5983`
- `F1 = 0.3923`
- `Distance = 0.9080 m`
- `Height error = 0.9624 m`

## Melhor janela observada no sweep

No sweep de `window_size` de `0.5` a `5.0 m` com passo `0.5 m`, a melhor configuracao global foi:

- `window_size = 2.5 m`
- `TP = 1611`
- `FP = 2328`
- `FN = 1991`
- `Precision = 0.4090`
- `Recall = 0.4473`
- `F1 = 0.4273`

Isso mostra um trade-off claro:

- janelas menores (`0.5-1.5 m`) maximizam recall, mas explodem FP;
- janelas maiores (`3.0-5.0 m`) aumentam precision, mas derrubam recall;
- o melhor equilibrio medido por `F1` nesta implementacao ficou em `2.5 m`, ligeiramente acima do `2.0 m` reportado como configuracao canonica.

## Resultados por parcela no caso canonico

| Plot | TP | FP | FN | Precision | Recall | F1 | Distance (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| plot_01 | 293 | 616 | 127 | 0.3223 | 0.6976 | 0.4409 | 0.7421 |
| plot_02 | 186 | 355 | 179 | 0.3438 | 0.5096 | 0.4106 | 0.8535 |
| plot_03 | 216 | 568 | 116 | 0.2755 | 0.6506 | 0.3871 | 1.1471 |
| plot_04 | 154 | 459 | 107 | 0.2512 | 0.5900 | 0.3524 | 1.0369 |
| plot_05 | 142 | 402 | 66 | 0.2610 | 0.6827 | 0.3777 | 0.9973 |
| plot_06 | 190 | 666 | 100 | 0.2220 | 0.6552 | 0.3316 | 0.8005 |
| plot_07 | 256 | 624 | 152 | 0.2909 | 0.6275 | 0.3975 | 0.9311 |
| plot_08 | 233 | 639 | 108 | 0.2672 | 0.6833 | 0.3842 | 0.8097 |
| plot_09 | 227 | 446 | 232 | 0.3373 | 0.4946 | 0.4011 | 0.9542 |
| plot_10 | 203 | 202 | 315 | 0.5012 | 0.3919 | 0.4399 | 0.8071 |

## Observacoes

- Esta implementacao fiel e propositalmente independente do pipeline principal.
- Ela opera diretamente sobre a nuvem `LAS` normalizada por pontos de solo e sobre o inventario de campo.
- A ortofoto RGB nao participa do baseline em nenhum momento.

## Optuna

Tambem foi executado um estudo `Optuna` separado sobre esta versao fiel, sem tocar no pipeline principal:

- Runner: [scripts/run_optuna_faithful_baseline.py](/home/murilo/Documentos/LiDARAgroSat/scripts/run_optuna_faithful_baseline.py)
- Saida: [outputs_faithful_paper_optuna](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_optuna)
- Banco do estudo: [study.sqlite3](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_optuna/study.sqlite3)
- Trials: [trials.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_optuna/trials.csv)
- Melhor resumo: [best_summary.json](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_optuna/best_summary.json)
- Melhor resultado por parcela: [best_per_plot.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_optuna/best_per_plot.csv)

Configuracao do estudo:

- `80` trials
- sampler `TPE`
- objetivo: maximizar `F1_global`
- matching mantido fixo em `max_dist = 5 m` e `max_height_diff = 3 m`
- espaco de busca:
  - `window_size_m` em `[0.5, 5.0]` com passo `0.1`
  - `height_threshold_m` em `[1.5, 5.0]` com passo `0.1`

Melhor trial:

- `window_size_m = 2.4`
- `height_threshold_m = 5.0`
- `Precision_global = 0.4253`
- `Recall_global = 0.4542`
- `F1_global = 0.4393`

Comparacao direta:

- caso canonico fiel: `F1 = 0.3933`
- melhor sweep manual anterior: `F1 = 0.4273`
- melhor Optuna: `F1 = 0.4393`

Ganhos:

- `+0.0460` sobre o caso canonico
- `+0.0120` sobre o melhor sweep manual

Interpretacao:

- o estudo reforcou o mesmo padrao do sweep manual: a regiao otima fica perto de `window_size` entre `2.3` e `2.5 m`;
- o `height_threshold` empurrou para o limite superior da busca (`5.0 m`), sugerindo que ainda existe espaco para um estudo adicional com limite superior mais alto, se isso for desejavel.
