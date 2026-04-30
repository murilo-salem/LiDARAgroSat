# Resultados Faithful Local Maxima com Janela Adaptativa

## Implementação

Variante separada em:

- [src/lidaragrosat/paper_faithful/adaptive_local_maxima.py](/home/murilo/Documentos/LiDARAgroSat/src/lidaragrosat/paper_faithful/adaptive_local_maxima.py)
- [scripts/run_optuna_faithful_adaptive_local_maxima.py](/home/murilo/Documentos/LiDARAgroSat/scripts/run_optuna_faithful_adaptive_local_maxima.py)

Desenho da ablação:

- mesmo cache fiel do baseline atual;
- mesmo matching guloso determinístico;
- mesmo `height_threshold = 5.0 m` do melhor baseline fixo atual;
- única mudança: substituir a janela fixa por `window_size(z) = a * z + b`, com `clip` em `[0.5, 5.0] m`.

Detalhe geométrico importante:

- a janela adaptativa foi aplicada no plano `xy`, não em esfera 3D, porque o objetivo é modelar extensão horizontal de copa.

## Estudo Optuna

Saídas:

- [outputs_faithful_paper_adaptive_optuna/best_summary.json](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_adaptive_optuna/best_summary.json)
- [outputs_faithful_paper_adaptive_optuna/best_per_plot.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_adaptive_optuna/best_per_plot.csv)
- [outputs_faithful_paper_adaptive_optuna/comparison_vs_current_local_maxima.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_adaptive_optuna/comparison_vs_current_local_maxima.csv)
- [outputs_faithful_paper_adaptive_optuna/trials.csv](/home/murilo/Documentos/LiDARAgroSat/outputs_faithful_paper_adaptive_optuna/trials.csv)

Melhor configuração:

- `a = 0.035`
- `b = 2.5`
- `height_threshold = 5.0`
- `min_window = 0.5`
- `max_window = 5.0`

Isso produziu uma janela média efetiva de aproximadamente `3.24 m`.

## Resultado Global

Variante adaptativa:

- `TP = 2073`
- `FP = 2480`
- `FN = 1529`
- `Precision = 0.4553`
- `Recall = 0.5755`
- `F1 = 0.5084`
- `Distance = 1.0403 m`

Baseline fixo atual otimizado:

- `TP = 1636`
- `FP = 2211`
- `FN = 1966`
- `Precision = 0.4253`
- `Recall = 0.4542`
- `F1 = 0.4393`
- `Distance = 0.9886 m`

Delta global da janela adaptativa contra o baseline fixo atual:

- `delta Precision = +0.0300`
- `delta Recall = +0.1213`
- `delta F1 = +0.0691`
- `delta Distance = +0.0516 m`

## Resultados por Parcela

| Parcela | F1 adaptativo | F1 atual | delta F1 |
|---|---:|---:|---:|
| `plot_01` | 0.5507 | 0.5153 | +0.0354 |
| `plot_02` | 0.4566 | 0.4214 | +0.0352 |
| `plot_03` | 0.5106 | 0.4349 | +0.0757 |
| `plot_04` | 0.4223 | 0.4298 | -0.0076 |
| `plot_05` | 0.5423 | 0.4412 | +0.1010 |
| `plot_06` | 0.5014 | 0.4052 | +0.0962 |
| `plot_07` | 0.5258 | 0.4643 | +0.0615 |
| `plot_08` | 0.5371 | 0.4646 | +0.0725 |
| `plot_09` | 0.5355 | 0.3936 | +0.1418 |
| `plot_10` | 0.5005 | 0.3931 | +0.1073 |

## Leitura Técnica

O resultado ficou consistente com a hipótese:

- a principal melhora veio de `recall`, não de `precision`;
- o método adaptativo recuperou muito mais árvores reais em parcelas difíceis;
- houve um leve custo em erro médio de localização e em `FP` em algumas parcelas.

Resumo por contagem de parcelas:

- melhorou `F1` em `9/10` parcelas;
- piorou `F1` apenas em `plot_04`.

As maiores melhoras ocorreram em:

- `plot_09`: `+0.1418`
- `plot_10`: `+0.1073`
- `plot_05`: `+0.1010`
- `plot_06`: `+0.0962`

## Comparação com o Watershed

Para contexto:

- `local maxima` fixo atual: `F1 = 0.4393`
- `local maxima` adaptativo: `F1 = 0.5084`
- `CHM + watershed`: `F1 = 0.5354`

Ou seja:

- a janela adaptativa entregou ganho real e relevante sobre o baseline fixo;
- mas o `CHM + watershed` ainda permaneceu melhor no benchmark atual.

## Validação

Verificações executadas:

- `python3 -m compileall src scripts tests`
- `PYTHONPATH=src:. ./.venv/bin/pytest -q tests`

Resultado: `13 passed`.
