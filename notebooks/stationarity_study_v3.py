# %% [markdown]
# # Estudo de Estacionaridade v3 — Análise em Massa
#
# Processa **todas** as séries de `series_spec.csv` de uma vez.
#
# **Outputs principais:**
# 1. `df_full` — Todas as combinações variável × pipeline, com testes de estacionaridade + métricas de estabilidade de variância
# 2. `df_best` — Melhor transformação **mensal** e **anual** para cada variável (com base em estacionaridade + estabilidade de variância)
#
# Transformações classificadas como:
# - **Mensal**: operações com lag=1 (diff, pct_change MoM, power transforms)
# - **Anual**: operações com lag=12/4 (sdiff12, sdiff4, pct_change YoY)

# %% ── Seção 1: Setup ──────────────────────────────────────────────────────

import warnings
warnings.filterwarnings('ignore')

import datetime as dt
import numpy as np
import pandas as pd
from IPython.display import display

from pib_nowcast.config import SERIES_SPEC, LAST_DATA, DATA_DIR
from pib_nowcast.utils.transformations import (
    stationarity_tests,
    deflate,
    seas_adj_stl_parallel,
)
from pib_nowcast.utils.transformations.transform_pipeline import (
    PIPELINE_REGISTRY,
    MONTHLY_PIPELINE_IDS,
    QUARTERLY_PIPELINE_IDS,
    apply_transform_pipeline,
)

# %% ── Seção 2: Carregamento e Pré-processamento ───────────────────────────

start = dt.datetime.now()

specs_df = pd.read_csv(SERIES_SPEC, sep=';')

try:
    full_data = pd.read_excel(LAST_DATA, sheet_name='full_dataset', index_col='Date')
except Exception:
    from pib_nowcast.utils.get_data import get_data
    full_data = get_data(specs_df, '1996-01-01')

# Deflação + ajuste sazonal
full_data = deflate(full_data, specs_df)
full_data = seas_adj_stl_parallel(full_data, specs_df)

# Mapa variável → frequência
freq_map = dict(zip(specs_df['variable'], specs_df['frequency']))

print(f"Séries: {full_data.shape[1]} | Obs: {full_data.shape[0]}")
print(f"Período: {full_data.index.min().date()} a {full_data.index.max().date()}")


# %% ── Seção 3: Classificação Mensal / Anual ────────────────────────────────

# Transformações MENSAIS (lag dominante = 1)
MONTHLY_TRANSFORM_IDS: set[int] = {0, 1, 2, 3, 4, 5, 6, 7, 13}

# Transformações ANUAIS (lag dominante = 12 para mensal, 4 para trimestral)
ANNUAL_TRANSFORM_IDS_MONTHLY: set[int] = {8, 9, 10, 11, 12, 14}
ANNUAL_TRANSFORM_IDS_QUARTERLY: set[int] = {15, 16, 17, 18, 19, 20, 21}

# Classificação para lookup
def _classify_transform(pid: int, freq: str) -> str:
    """Retorna 'Mensal' ou 'Anual' para uma transformação."""
    if pid in MONTHLY_TRANSFORM_IDS:
        return 'Mensal'
    if freq == 'Monthly' and pid in ANNUAL_TRANSFORM_IDS_MONTHLY:
        return 'Anual'
    if freq == 'Quarterly' and pid in ANNUAL_TRANSFORM_IDS_QUARTERLY:
        return 'Anual'
    # Fallback: transformações compartilhadas entre freq (0-7 cobrem ambos)
    if pid in MONTHLY_TRANSFORM_IDS:
        return 'Mensal'
    return 'Anual'


# %% ── Seção 4: Loop em Massa — Todas as Séries × Todos os Pipelines ───────

def compute_variance_stability(s: pd.Series, window: int = 24) -> tuple[float, float]:
    """Calcula CV e IQR ratio da variância rolling.
    
    Returns (cv_var, iqr_ratio) — ambos adimensionais, menor = mais estável.
    """
    rolling_var = s.rolling(window=window, min_periods=window // 2).var().dropna()
    
    if len(rolling_var) < 5:
        return np.nan, np.nan
    
    mean_rv = rolling_var.mean()
    if mean_rv == 0 or np.isnan(mean_rv):
        return np.nan, np.nan
    
    cv = rolling_var.std() / mean_rv
    
    median_rv = rolling_var.median()
    if median_rv == 0 or np.isnan(median_rv):
        return cv, np.nan
    
    q25, q75 = rolling_var.quantile(0.25), rolling_var.quantile(0.75)
    iqr_ratio = (q75 - q25) / median_rv
    
    return cv, iqr_ratio


all_results = []

for col in full_data.columns:
    raw = full_data[col].dropna()
    freq = freq_map.get(col, 'Monthly')
    
    # Seleciona pipelines aplicáveis pela frequência
    pipe_ids = QUARTERLY_PIPELINE_IDS if freq == 'Quarterly' else MONTHLY_PIPELINE_IDS
    
    # Janela para variância rolling (2 anos)
    rv_window = 8 if freq == 'Quarterly' else 24
    
    for pid in pipe_ids:
        pipe_name, steps, _ = PIPELINE_REGISTRY[pid]
        categoria = _classify_transform(pid, freq)
        
        transformed = apply_transform_pipeline(raw, pid)
        
        if transformed is None:
            all_results.append({
                'variable': col,
                'frequency': freq,
                'categoria': categoria,
                'pipeline_id': pid,
                'pipeline': pipe_name,
                'n_steps': len(steps),
                'n_obs': 0,
                'ADF': None,
                'KPSS': None,
                'Phillips-Perron': None,
                'DFGLS': None,
                'n_stationary': 0,
                'is_stationary': False,
                'applicable': False,
                'cv_var': np.nan,
                'iqr_ratio': np.nan,
            })
            continue
        
        # ── Testes de estacionaridade ──
        try:
            tests_df = stationarity_tests(transformed)
            test_results = tests_df.set_index('test')['is_stationary'].to_dict()
            valid_results = {k: v for k, v in test_results.items() if v is not None}
            n_stat = sum(valid_results.values())
            n_valid = len(valid_results)
            is_stat = n_stat > n_valid / 2 if n_valid > 0 else False
        except Exception:
            test_results = {}
            n_stat = 0
            is_stat = False
        
        # ── Estabilidade de variância ──
        cv, iqr_r = compute_variance_stability(transformed, window=rv_window)
        
        all_results.append({
            'variable': col,
            'frequency': freq,
            'categoria': categoria,
            'pipeline_id': pid,
            'pipeline': pipe_name,
            'n_steps': len(steps),
            'n_obs': len(transformed),
            'ADF': test_results.get('ADF'),
            'KPSS': test_results.get('KPSS'),
            'Phillips-Perron': test_results.get('Phillips-Perron'),
            'DFGLS': test_results.get('DFGLS'),
            'n_stationary': n_stat,
            'is_stationary': is_stat,
            'applicable': True,
            'cv_var': cv,
            'iqr_ratio': iqr_r,
        })
    
    print(f'✓ {col}')

df_full = pd.DataFrame(all_results)

elapsed = dt.datetime.now() - start
print(f'\nTotal de combinações: {len(df_full)}')
print(f'Inaplicáveis: {(~df_full["applicable"]).sum()}')
print(f'Tempo: {elapsed.seconds // 60}m {elapsed.seconds % 60}s')


# %% ── Seção 5: Tabela Resumo Completa ──────────────────────────────────────

def fmt_bool(v):
    if v is None:
        return '—'
    return '✓' if v else '✗'

display_df = df_full[df_full['applicable']].copy()

for test_col in ['ADF', 'KPSS', 'Phillips-Perron', 'DFGLS', 'is_stationary']:
    display_df[test_col] = display_df[test_col].map(fmt_bool)

display_df = display_df.rename(columns={
    'variable': 'Variável',
    'pipeline_id': 'ID',
    'pipeline': 'Pipeline',
    'categoria': 'Categoria',
    'n_steps': 'Etapas',
    'n_obs': 'N obs',
    'n_stationary': 'Votos',
    'is_stationary': 'Estacionária',
    'cv_var': 'CV Var',
    'iqr_ratio': 'IQR Ratio',
})

print(f"Combinações aplicáveis: {len(display_df)}")
display(display_df.head(30))


# %% ── Seção 6: Seleção da Melhor Transformação por Variável ────────────────

# ── Critérios de seleção (separados para Mensal e Anual) ──
#
# 1. Apenas pipelines que atingiram estacionaridade
# 2. Desempate ordenado por:
#    a. Maior nº de votos de estacionaridade (mais testes concordam)
#    b. Menor CV de variância rolling (mais estável)
#    c. Menor nº de etapas (mais parcimonioso)
#    d. Mais observações preservadas

# Filtrar apenas estacionárias e aplicáveis
stationary = df_full[df_full['is_stationary'] & df_full['applicable']].copy()

# Ordenar pelos critérios de desempate
stationary = stationary.sort_values(
    ['n_stationary', 'cv_var', 'n_steps', 'n_obs'],
    ascending=[False, True, True, False],
)

# ── Melhor MENSAL por variável ──
best_monthly = (
    stationary[stationary['categoria'] == 'Mensal']
    .groupby('variable')
    .first()
    .reset_index()
)
best_monthly = best_monthly[[
    'variable', 'frequency', 'pipeline_id', 'pipeline',
    'n_stationary', 'cv_var', 'iqr_ratio', 'n_steps', 'n_obs',
]].rename(columns={
    'pipeline_id': 'tid_mensal',
    'pipeline': 'pipeline_mensal',
    'n_stationary': 'votos_mensal',
    'cv_var': 'cv_var_mensal',
    'iqr_ratio': 'iqr_mensal',
    'n_steps': 'steps_mensal',
    'n_obs': 'nobs_mensal',
})

# ── Melhor ANUAL por variável ──
best_annual = (
    stationary[stationary['categoria'] == 'Anual']
    .groupby('variable')
    .first()
    .reset_index()
)
best_annual = best_annual[[
    'variable', 'pipeline_id', 'pipeline',
    'n_stationary', 'cv_var', 'iqr_ratio', 'n_steps', 'n_obs',
]].rename(columns={
    'pipeline_id': 'tid_anual',
    'pipeline': 'pipeline_anual',
    'n_stationary': 'votos_anual',
    'cv_var': 'cv_var_anual',
    'iqr_ratio': 'iqr_anual',
    'n_steps': 'steps_anual',
    'n_obs': 'nobs_anual',
})

# ── Combinar: uma linha por variável ──
df_best = best_monthly.merge(best_annual, on='variable', how='outer')

# Preencher variáveis que estão em apenas um dos dois
all_vars = set(full_data.columns)
existing_vars = set(df_best['variable'].values)
missing_vars = all_vars - existing_vars

if missing_vars:
    missing_rows = pd.DataFrame({'variable': sorted(missing_vars)})
    df_best = pd.concat([df_best, missing_rows], ignore_index=True)

# Garantir frequência para todos
for var in df_best['variable']:
    if pd.isna(df_best.loc[df_best['variable'] == var, 'frequency']).all():
        df_best.loc[df_best['variable'] == var, 'frequency'] = freq_map.get(var, 'Monthly')

# df_best = df_best.sort_values('variable').reset_index(drop=True)

print('═' * 100)
print('  MELHOR TRANSFORMAÇÃO POR VARIÁVEL (Mensal + Anual)')
print('═' * 100)
display(df_best)


# %% ── Seção 7: Diagnósticos ────────────────────────────────────────────────

# ── 7a. Variáveis sem NENHUMA transformação estacionária ──
no_monthly = set(full_data.columns) - set(best_monthly['variable'].values)
no_annual = set(full_data.columns) - set(best_annual['variable'].values)
no_any = no_monthly & no_annual

if no_any:
    print('⚠️  Variáveis sem NENHUMA transformação estacionária (mensal ou anual):')
    for v in sorted(no_any):
        print(f'   - {v}')
else:
    print('✅ Todas as variáveis possuem pelo menos uma transformação estacionária.')

if no_monthly - no_any:
    print(f'\n⚠️  Sem transformação MENSAL estacionária ({len(no_monthly - no_any)}):')
    for v in sorted(no_monthly - no_any):
        print(f'   - {v}')

if no_annual - no_any:
    print(f'\n⚠️  Sem transformação ANUAL estacionária ({len(no_annual - no_any)}):')
    for v in sorted(no_annual - no_any):
        print(f'   - {v}')

# ── 7b. Comparação com transformation_id atual do series_spec ──
print('\n' + '═' * 100)
print('  COMPARAÇÃO COM TRANSFORMATION_ID ATUAL (series_spec.csv)')
print('═' * 100)

comparison_rows = []
for _, row in specs_df.iterrows():
    var = row['variable']
    current_tid = row.get('transformation_id', -1)
    
    if pd.isna(current_tid) or current_tid < 0:
        continue
    
    current_tid = int(current_tid)
    best_row = df_best[df_best['variable'] == var]
    
    if best_row.empty:
        comparison_rows.append({
            'variable': var,
            'tid_atual': current_tid,
            'cat_atual': _classify_transform(current_tid, freq_map.get(var, 'Monthly')),
            'tid_melhor_mensal': None,
            'tid_melhor_anual': None,
            'mudou_mensal': None,
            'mudou_anual': None,
        })
        continue
    
    br = best_row.iloc[0]
    tid_m = br.get('tid_mensal')
    tid_a = br.get('tid_anual')
    
    cat_atual = _classify_transform(current_tid, freq_map.get(var, 'Monthly'))
    
    # Verificar se o atual coincide com o melhor da sua categoria
    if cat_atual == 'Mensal':
        coincide = (not pd.isna(tid_m)) and int(tid_m) == current_tid
    else:
        coincide = (not pd.isna(tid_a)) and int(tid_a) == current_tid
    
    comparison_rows.append({
        'variable': var,
        'tid_atual': current_tid,
        'cat_atual': cat_atual,
        'pipeline_atual': PIPELINE_REGISTRY[current_tid][0],
        'tid_melhor_mensal': int(tid_m) if pd.notna(tid_m) else None,
        'pipeline_melhor_mensal': br.get('pipeline_mensal', '—'),
        'tid_melhor_anual': int(tid_a) if pd.notna(tid_a) else None,
        'pipeline_melhor_anual': br.get('pipeline_anual', '—'),
        'atual_coincide': '✓' if coincide else '✗',
    })

df_comparison = pd.DataFrame(comparison_rows)

n_match = (df_comparison['atual_coincide'] == '✓').sum()
n_total = len(df_comparison)
print(f'\nCoincidência com melhor: {n_match}/{n_total} ({100*n_match/n_total:.0f}%)')

display(df_comparison)


# %% ── Seção 8: Exportação ──────────────────────────────────────────────────

csv_dir = DATA_DIR / 'stationarity'
csv_dir.mkdir(parents=True, exist_ok=True)

# Resultados completos (todas as combinações)
df_full.to_csv(csv_dir / 'stationarity_full_v3.csv', index=False, sep=';')
print(f'✓ Resultados completos: {csv_dir / "stationarity_full_v3.csv"}')

# Recomendação por variável (mensal + anual)
df_best.to_csv(csv_dir / 'stationarity_best_v3.csv', index=False, sep=';')
print(f'✓ Recomendações: {csv_dir / "stationarity_best_v3.csv"}')

# Comparação com spec atual
df_comparison.to_csv(csv_dir / 'stationarity_comparison_v3.csv', index=False, sep=';')
print(f'✓ Comparação: {csv_dir / "stationarity_comparison_v3.csv"}')

elapsed = dt.datetime.now() - start
print(f'\nTempo total: {elapsed.seconds // 60}m {elapsed.seconds % 60}s')

# %%
