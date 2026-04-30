# Resultados Faithful Point Cloud Refiner

## Escopo

Foi implementada uma variante `point-cloud deep learning` baseada em `PointNet` local sobre recortes reais da nuvem de pontos ao redor de propostas LiDAR.

Arquivos principais:

- [src/lidaragrosat/paper_faithful/point_cloud_refiner.py](/home/murilo/Documentos/LiDARAgroSat/src/lidaragrosat/paper_faithful/point_cloud_refiner.py)
- [scripts/run_faithful_point_cloud_refiner.py](/home/murilo/Documentos/LiDARAgroSat/scripts/run_faithful_point_cloud_refiner.py)

Observação metodológica importante:

- isto não é um detector denso 3D no estilo `PointNet++` em cena completa;
- é um refinador local baseado diretamente na nuvem, treinado sobre propostas LiDAR já existentes.

Ou seja:

- a entrada principal da rede são vizinhanças reais da nuvem (`point sets`);
- mas ainda existe um estágio prévio de geração de candidatos.

## Protocolo

Como esse método exige treino supervisionado, ele não foi avaliado no protocolo “treina em tudo e mede em tudo”, porque isso vazaria informação.

Foi usado um protocolo simples e defensável com `3` splits fixos:

- `split_plot10`: treino em `8` plots, validação em `plot_09`, teste em `plot_10`
- `split_plot07`: treino em `8` plots, validação em `plot_08`, teste em `plot_07`
- `split_plot06`: treino em `8` plots, validação em `plot_05`, teste em `plot_06`

Foram testadas duas fontes de proposta:

- `adaptive`
- `union` (`adaptive + watershed`)

## Resultados

Saídas:

- [outputs_faithful_point_cloud_refiner/metrics_by_split.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_point_cloud_refiner/metrics_by_split.csv)
- [outputs_faithful_point_cloud_refiner/summary_by_source_mode.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_point_cloud_refiner/summary_by_source_mode.csv)
- [outputs_faithful_point_cloud_refiner/comparison_summary.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_point_cloud_refiner/comparison_summary.csv)
- [outputs_faithful_point_cloud_refiner/baseline_summary_on_test_plots.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_point_cloud_refiner/baseline_summary_on_test_plots.csv)

Resumo por modo de proposta:

| Modo | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `union` | 0.4454 | 0.5847 | 0.5025 |
| `adaptive` | 0.4487 | 0.5386 | 0.4865 |

Melhor modo:

- `union`
- `F1_mean = 0.5025`

## Resultados por split

| Modo | Test plot | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| `adaptive` | `plot_06` | 0.3659 | 0.5552 | 0.4411 |
| `adaptive` | `plot_07` | 0.4924 | 0.5588 | 0.5235 |
| `adaptive` | `plot_10` | 0.4878 | 0.5019 | 0.4948 |
| `union` | `plot_06` | 0.4055 | 0.6586 | 0.5020 |
| `union` | `plot_07` | 0.4497 | 0.5588 | 0.4984 |
| `union` | `plot_10` | 0.4810 | 0.5367 | 0.5073 |

## Comparação com os métodos atuais nesses mesmos 3 plots de teste

Médias dos baselines nesses `3` plots:

| Método | Precision mean | Recall mean | F1 mean |
|---|---:|---:|---:|
| `fixed` | 0.4574 | 0.4451 | 0.4209 |
| `adaptive` | 0.4680 | 0.5667 | 0.5092 |
| `watershed` | 0.5580 | 0.5330 | 0.5412 |
| `object_fusion` | 0.5155 | 0.5260 | 0.5149 |

Delta do melhor `PointNet` (`union`) contra esses baselines:

- contra `fixed`: `+0.0817` de `F1`
- contra `adaptive`: `-0.0067`
- contra `watershed`: `-0.0387`
- contra `object_fusion`: `-0.0123`

## Leitura técnica

O que funcionou:

- o branch direto na nuvem aumentou bastante o `recall`;
- a melhor variante foi usar propostas `union`, o que sugere que a rede filtra melhor do que a heurística crua;
- houve ganho claro sobre o baseline fixo antigo.

O que não aconteceu:

- não houve ganho sobre o `watershed`;
- não houve ganho sobre a melhor fusão RGB+LiDAR orientada a objetos;
- o ganho esperado de `+0.15 a +0.20` não apareceu neste desenho.

Interpretação:

- o gargalo provável não está mais na representação local da nuvem, e sim no estágio de propostas e no volume de dados;
- o modelo ainda depende de candidatos geométricos e não explora contexto de cena completa;
- para competir com resultados na faixa `> 0.7`, provavelmente seria preciso um detector 3D mais forte, com mais tuning e protocolo mais longo.

## Validação

Verificações executadas:

- `python3 -m compileall src scripts tests`
- `PYTHONPATH=src:. ./.venv/bin/pytest -q tests`

Resultado: `14 passed`.
