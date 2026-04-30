# Benchmark Protocol V2

## Objetivo
Congelar a régua oficial de comparação da próxima iteração para evitar comparações entre regimes incompatíveis.

## Protocolos Oficiais

### Protocolo A
- `protocol_id`: `protocol_a_v2_3split`
- uso: métodos aprendidos, refinadores locais, ablações de labels e contribuição de modalidades
- partição: suíte comparável de 3 splits em nível de parcela
- métricas obrigatórias:
  - `precision`
  - `recall`
  - `f1`
  - `location_mae_m`
  - `height_mae_m`
  - `tp`
  - `fp`
  - `fn`
- baselines obrigatórias:
  - `baseline_local_maxima`
  - `chm_watershed`
  - `hybrid_proposal_refinement_multimodal_v2`

### Protocolo B
- `protocol_id`: `protocol_b_v2_faithful10`
- uso: métodos sem treino, Stage-1 geométrico e treino local leve
- escopo: benchmark fiel nas 10 parcelas
- métricas obrigatórias:
  - `precision`
  - `recall`
  - `f1`
  - `location_mae_m`
  - `height_mae_m`
  - `tp`
  - `fp`
  - `fn`
- baselines obrigatórias:
  - `baseline_local_maxima`
  - `chm_watershed`
  - `hybrid_proposal_refinement_multimodal_v2`

## Regras Operacionais
- toda run agregável precisa carregar `protocol_id`
- toda run agregável precisa ter `experiment_manifest.json`
- o naming canônico é `family.stage1.stage2.labels.protocol.seed`
- scripts de avaliação devem rejeitar runs sem `protocol_id`
- relatórios consolidados devem preservar `protocol_id` em todas as tabelas

## Manifesto Obrigatório
Toda run nova deve persistir `experiment_manifest.json` com:
- `dataset`
- `split`
- `stage1`
- `refiner`
- `labels`
- `decode_params`
- `commit_hash`

## Observação
Runs antigas sem manifesto permanecem úteis como artefato histórico, mas não entram em tabelas oficiais do V2 sem migração explícita.
