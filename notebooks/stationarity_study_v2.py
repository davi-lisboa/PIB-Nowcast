# %% [markdown]
# # Estudo de Estacionaridade v2
#
# Notebook para análise visual e quantitativa de transformações de estacionaridade.
#
# **Funcionalidades:**
# 1. Classificação de transformações como **mensais** (lag=1) vs. **anuais** (lag=12/4)
# 2. Visualização lado-a-lado mensal/anual para cada série
# 3. Testes de estacionaridade (ADF, KPSS, Phillips-Perron, DFGLS)
# 4. Ranking de estabilidade de variância (CV da variância rolling + IQR ratio)
#
# **Uso:** Altere `selected_series` na Seção 2 e re-execute as células subsequentes.

# %% ── Seção 1: Setup e Carregamento ────────────────────────────────────────

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from IPython.display import display

from pib_nowcast.config import SERIES_SPEC, LAST_DATA
from pib_nowcast.utils.transformations import (
    seas_adj_stl_parallel,
    deflate,
    stationarity_tests,
)
from pib_nowcast.utils.transformations.transform_pipeline import (
    PIPELINE_REGISTRY,
    MONTHLY_PIPELINE_IDS,
    QUARTERLY_PIPELINE_IDS,
    apply_transform_pipeline,
)

# Estilo dos gráficos
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
})

# %% Carregamento dos dados

specs_df = pd.read_csv(SERIES_SPEC, sep=';')
full_data = pd.read_excel(LAST_DATA, sheet_name='full_dataset', index_col='Date')

print(f"Séries: {full_data.shape[1]} | Observações: {full_data.shape[0]}")
print(f"Período: {full_data.index.min().date()} a {full_data.index.max().date()}")

# %% Pré-processamento (deflação + ajuste sazonal)

# Deflacionar
full_data_defl = deflate(full_data, specs_df)

# Ajuste sazonal
full_data_sa = seas_adj_stl_parallel(full_data_defl, specs_df)

print("✓ Dados pré-processados (deflação + ajuste sazonal)")

# %% ── Seção 2: Classificação Mensal / Anual ────────────────────────────────

# Transformações MENSAIS: operações com lag=1 (diff, pct_change(1), power transforms)
MONTHLY_TRANSFORMS: dict[int, str] = {
    0:  "nível",
    1:  "diff",
    2:  "log",
    3:  "log→diff",
    4:  "boxcox",
    5:  "boxcox→diff",
    6:  "yeojohnson",
    7:  "yeojohnson→diff",
    13: "mom_pct",
}

# Transformações ANUAIS: operações com lag=12 (sdiff12, pct_change(12))
ANNUAL_TRANSFORMS: dict[int, str] = {
    8:  "sdiff12",
    9:  "sdiff12→diff",
    10: "log→sdiff12",
    11: "log→sdiff12→diff",
    12: "boxcox→sdiff12→diff",
    14: "yoy_pct_monthly",
}

# ──────────────────────────────────────────────────────────────────────────────
# ⬇️  ALTERE AQUI a série de interesse e re-execute as células abaixo ⬇️
# ──────────────────────────────────────────────────────────────────────────────
selected_series = 'abras'
# ──────────────────────────────────────────────────────────────────────────────

# Verifica se a série existe
assert selected_series in full_data_sa.columns, \
    f"'{selected_series}' não encontrada. Opções:\n{list(full_data_sa.columns)}"

series_raw = full_data_sa[selected_series].dropna()
freq_str = specs_df.loc[specs_df['variable'] == selected_series, 'frequency'].iloc[0]

print(f"\nSérie selecionada: {selected_series}")
print(f"  Frequência: {freq_str}")
print(f"  Obs: {len(series_raw)} ({series_raw.index.min().date()} — {series_raw.index.max().date()})")

# Define transformações aplicáveis por frequência
if freq_str == 'Monthly':
    applicable_monthly = MONTHLY_TRANSFORMS
    applicable_annual = ANNUAL_TRANSFORMS
else:
    # Para séries trimestrais, ajustar a classificação
    applicable_monthly = {
        k: v for k, v in MONTHLY_TRANSFORMS.items()
        if k in QUARTERLY_PIPELINE_IDS
    }
    applicable_annual = {
        k: v for k, v in {
            15: "qoq_pct",
            16: "yoy_pct",
            17: "sdiff4",
            18: "sdiff4→diff",
            19: "log→sdiff4",
            20: "log→sdiff4→diff",
            21: "qoq_annualized",
        }.items()
    }

print(f"\n  Transformações mensais aplicáveis: {list(applicable_monthly.keys())}")
print(f"  Transformações anuais aplicáveis:  {list(applicable_annual.keys())}")


# %% ── Seção 3: Visualização Comparativa (Mensal vs. Anual) ─────────────────

def _apply_and_collect(
    series: pd.Series,
    transforms: dict[int, str],
) -> dict[int, tuple[str, pd.Series]]:
    """Aplica cada pipeline e retorna {id: (nome, série_transformada)} para os que funcionam."""
    results = {}
    for pid, label in transforms.items():
        s_transf = apply_transform_pipeline(series, pid)
        if s_transf is not None:
            results[pid] = (label, s_transf)
    return results


def _plot_panel(
    ax: plt.Axes,
    pid: int,
    label: str,
    series: pd.Series,
    color: str,
    is_current: bool = False,
):
    """Plota uma série transformada em um eixo."""
    ax.plot(series.index, series.values, color=color, linewidth=0.8, alpha=0.85)
    
    title = f"[{pid}] {label}"
    if is_current:
        title += "  ◀ atual"
        ax.set_title(title, fontsize=9, fontweight='bold', color='#c0392b')
    else:
        ax.set_title(title, fontsize=9)
    
    ax.tick_params(axis='both', labelsize=7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))


# Aplica todas as transformações
monthly_results = _apply_and_collect(series_raw, applicable_monthly)
annual_results = _apply_and_collect(series_raw, applicable_annual)

# Pega o transformation_id atual do spec
current_tid = specs_df.loc[
    specs_df['variable'] == selected_series, 'transformation_id'
].iloc[0]
current_tid = int(current_tid) if pd.notna(current_tid) else -1

# Cores diferenciadas para cada painel
MONTHLY_COLOR = '#2980b9'
ANNUAL_COLOR  = '#27ae60'

# ── Plot ──
n_monthly = len(monthly_results)
n_annual  = len(annual_results)
n_rows = max(n_monthly, n_annual)

if n_rows == 0:
    print("Nenhuma transformação aplicável para esta série.")
else:
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(16, 2.5 * n_rows),
        squeeze=False,
    )
    
    fig.suptitle(
        f'Transformações de Estacionaridade — {selected_series}  (tid atual = {current_tid})',
        fontsize=13, fontweight='bold', y=1.01,
    )
    
    # Painel esquerdo: Mensais
    for i, (pid, (label, s)) in enumerate(monthly_results.items()):
        _plot_panel(axes[i, 0], pid, label, s, MONTHLY_COLOR, is_current=(pid == current_tid))
    
    # Limpa subplots vazios do painel mensal
    for i in range(n_monthly, n_rows):
        axes[i, 0].set_visible(False)
    
    # Header dos painéis
    axes[0, 0].annotate(
        'MENSAL (lag=1)', xy=(0.5, 1.25), xycoords='axes fraction',
        fontsize=11, fontweight='bold', color=MONTHLY_COLOR,
        ha='center', va='bottom',
    )
    
    # Painel direito: Anuais
    for i, (pid, (label, s)) in enumerate(annual_results.items()):
        _plot_panel(axes[i, 1], pid, label, s, ANNUAL_COLOR, is_current=(pid == current_tid))
    
    # Limpa subplots vazios do painel anual
    for i in range(n_annual, n_rows):
        axes[i, 1].set_visible(False)
    
    axes[0, 1].annotate(
        'ANUAL (lag=12)', xy=(0.5, 1.25), xycoords='axes fraction',
        fontsize=11, fontweight='bold', color=ANNUAL_COLOR,
        ha='center', va='bottom',
    )
    
    fig.tight_layout()
    plt.show()


# %% ── Seção 4: Testes de Estacionaridade ───────────────────────────────────

def run_stationarity_battery(
    series: pd.Series,
    transforms: dict[int, str],
    category_label: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Executa bateria de testes em todas as transformações e retorna tabela resumo."""
    rows = []
    
    for pid, label in transforms.items():
        s_transf = apply_transform_pipeline(series, pid)
        if s_transf is None:
            rows.append({
                'categoria': category_label,
                'pipeline_id': pid,
                'nome': label,
                'ADF_p': np.nan,
                'KPSS_p': np.nan,
                'PP_p': np.nan,
                'DFGLS_p': np.nan,
                'n_estac': np.nan,
                'n_testes': np.nan,
                'veredicto': 'N/A (não aplicável)',
            })
            continue
        
        df_tests = stationarity_tests(s_transf, alpha=alpha)
        
        # Extrai p-values por teste
        p_vals = df_tests.set_index('test')['p_value'].to_dict()
        is_stat = df_tests.set_index('test')['is_stationary'].to_dict()
        
        # Contagem de testes que indicam estacionaridade
        valid_results = df_tests['is_stationary'].dropna()
        n_stationary = valid_results.sum()
        n_total = len(valid_results)
        
        # Veredicto por maioria
        if n_total == 0:
            verdict = '❓ Inconclusivo'
        elif n_stationary > n_total / 2:
            verdict = '✅ Estacionária'
        else:
            verdict = '❌ Não estacionária'
        
        rows.append({
            'categoria': category_label,
            'pipeline_id': pid,
            'nome': label,
            'ADF_p': p_vals.get('ADF', np.nan),
            'KPSS_p': p_vals.get('KPSS', np.nan),
            'PP_p': p_vals.get('Phillips-Perron', np.nan),
            'DFGLS_p': p_vals.get('DFGLS', np.nan),
            'n_estac': n_stationary,
            'n_testes': n_total,
            'veredicto': verdict,
        })
    
    return pd.DataFrame(rows)


# Executa testes para ambas categorias
df_tests_monthly = run_stationarity_battery(
    series_raw, applicable_monthly, 'Mensal'
)
df_tests_annual = run_stationarity_battery(
    series_raw, applicable_annual, 'Anual'
)

df_tests_all = pd.concat([df_tests_monthly, df_tests_annual], ignore_index=True)

# Formatação
styled = (
    df_tests_all
    .style
    .format({
        'ADF_p':   '{:.4f}',
        'KPSS_p':  '{:.4f}',
        'PP_p':    '{:.4f}',
        'DFGLS_p': '{:.4f}',
    }, na_rep='—')
    .set_caption(f'Testes de Estacionaridade — {selected_series}')
    .set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '14px'), ('font-weight', 'bold')]},
    ])
)

display(styled)


# %% ── Seção 5: Ranking de Estabilidade de Variância ────────────────────────

def compute_variance_stability(
    series: pd.Series,
    transforms: dict[int, str],
    category_label: str,
    rolling_window: int = 24,
) -> pd.DataFrame:
    """Calcula métricas de estabilidade de variância para cada transformação.
    
    Métricas:
    - cv_var: Coeficiente de variação da variância rolling (menor = mais estável)
    - iqr_ratio: IQR / mediana da variância rolling (menor = mais estável)
    
    Ambas são invariantes à escala — não requerem normalização prévia.
    """
    rows = []
    
    for pid, label in transforms.items():
        s_transf = apply_transform_pipeline(series, pid)
        if s_transf is None:
            rows.append({
                'categoria': category_label,
                'pipeline_id': pid,
                'nome': label,
                'cv_var': np.nan,
                'iqr_ratio': np.nan,
                'mean_var': np.nan,
                'obs': 'Transformação não aplicável',
            })
            continue
        
        # Variância rolling
        rolling_var = s_transf.rolling(window=rolling_window, min_periods=rolling_window // 2).var().dropna()
        
        if len(rolling_var) < 5 or rolling_var.mean() == 0:
            rows.append({
                'categoria': category_label,
                'pipeline_id': pid,
                'nome': label,
                'cv_var': np.nan,
                'iqr_ratio': np.nan,
                'mean_var': np.nan,
                'obs': 'Dados insuficientes para rolling',
            })
            continue
        
        # CV da variância rolling: std / mean
        cv = rolling_var.std() / rolling_var.mean()
        
        # IQR ratio: IQR / median
        q25, q75 = rolling_var.quantile(0.25), rolling_var.quantile(0.75)
        median_var = rolling_var.median()
        iqr_ratio = (q75 - q25) / median_var if median_var > 0 else np.nan
        
        rows.append({
            'categoria': category_label,
            'pipeline_id': pid,
            'nome': label,
            'cv_var': cv,
            'iqr_ratio': iqr_ratio,
            'mean_var': rolling_var.mean(),
            'obs': '',
        })
    
    return pd.DataFrame(rows)


# Janela rolling: 24 meses para mensal, 8 trimestres para trimestral
window = 24 if freq_str == 'Monthly' else 8

df_var_monthly = compute_variance_stability(
    series_raw, applicable_monthly, 'Mensal', rolling_window=window
)
df_var_annual = compute_variance_stability(
    series_raw, applicable_annual, 'Anual', rolling_window=window
)

df_var_all = pd.concat([df_var_monthly, df_var_annual], ignore_index=True)

# Ranking separado por categoria
df_var_all['rank_cv'] = (
    df_var_all
    .groupby('categoria')['cv_var']
    .rank(method='min', ascending=True)
)
df_var_all['rank_iqr'] = (
    df_var_all
    .groupby('categoria')['iqr_ratio']
    .rank(method='min', ascending=True)
)

# Ranking combinado (média dos ranks)
df_var_all['rank_combinado'] = (df_var_all['rank_cv'] + df_var_all['rank_iqr']) / 2

# Ordena por categoria e ranking combinado
df_var_ranked = df_var_all.sort_values(['categoria', 'rank_combinado'])

# Display
styled_var = (
    df_var_ranked[['categoria', 'pipeline_id', 'nome', 'cv_var', 'iqr_ratio', 
                   'rank_cv', 'rank_iqr', 'rank_combinado', 'obs']]
    .style
    .format({
        'cv_var':         '{:.4f}',
        'iqr_ratio':      '{:.4f}',
        'rank_cv':        '{:.0f}',
        'rank_iqr':       '{:.0f}',
        'rank_combinado': '{:.1f}',
    }, na_rep='—')
    .background_gradient(
        subset=['cv_var'], cmap='RdYlGn_r', axis=0,
    )
    .background_gradient(
        subset=['iqr_ratio'], cmap='RdYlGn_r', axis=0,
    )
    .set_caption(
        f'Ranking de Estabilidade de Variância — {selected_series} '
        f'(janela rolling = {window} períodos)'
    )
    .set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '14px'), ('font-weight', 'bold')]},
    ])
)

display(styled_var)

# Destaca melhor de cada categoria
for cat in ['Mensal', 'Anual']:
    subset = df_var_ranked[df_var_ranked['categoria'] == cat].head(1)
    if not subset.empty:
        row = subset.iloc[0]
        print(f"\n🏆 Melhor {cat}: [{int(row['pipeline_id'])}] {row['nome']}  "
              f"(CV={row['cv_var']:.4f}, IQR ratio={row['iqr_ratio']:.4f})")


# %% ── Seção 6: Visualização da Variância Rolling ───────────────────────────

def plot_rolling_variance(
    series: pd.Series,
    transforms: dict[int, str],
    category_label: str,
    rolling_window: int,
    current_tid: int,
    color: str,
    top_n: int = 3,
    rank_df: pd.DataFrame | None = None,
):
    """Plota variância rolling para as top N transformações + a transformação atual."""
    
    # Identifica as top N do ranking
    if rank_df is not None:
        cat_rank = rank_df[rank_df['categoria'] == category_label].head(top_n)
        top_pids = cat_rank['pipeline_id'].astype(int).tolist()
    else:
        top_pids = list(transforms.keys())[:top_n]
    
    # Garante que o tid atual esteja incluído (se pertencer a esta categoria)
    pids_to_plot = list(dict.fromkeys(top_pids + ([current_tid] if current_tid in transforms else [])))
    
    n_plots = len(pids_to_plot)
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), squeeze=False)
    fig.suptitle(
        f'Variância Rolling ({rolling_window} períodos) — {category_label} — {selected_series}',
        fontsize=12, fontweight='bold',
    )
    
    for i, pid in enumerate(pids_to_plot):
        ax = axes[i, 0]
        label = transforms.get(pid, PIPELINE_REGISTRY[pid][0])
        s_transf = apply_transform_pipeline(series, pid)
        
        if s_transf is None:
            ax.text(0.5, 0.5, f'[{pid}] {label} — N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            continue
        
        rolling_var = s_transf.rolling(window=rolling_window, min_periods=rolling_window // 2).var()
        
        ax.fill_between(rolling_var.index, 0, rolling_var.values, alpha=0.3, color=color)
        ax.plot(rolling_var.index, rolling_var.values, color=color, linewidth=1)
        
        title = f'[{pid}] {label}'
        if pid == current_tid:
            title += '  ◀ atual'
            ax.set_title(title, fontsize=9, fontweight='bold', color='#c0392b')
        elif pid in top_pids[:1]:
            title += '  🏆 melhor'
            ax.set_title(title, fontsize=9, fontweight='bold', color='#27ae60')
        else:
            ax.set_title(title, fontsize=9)
        
        ax.tick_params(axis='both', labelsize=7)
    
    fig.tight_layout()
    plt.show()


# Plot para transformações mensais
plot_rolling_variance(
    series_raw, applicable_monthly, 'Mensal',
    rolling_window=window, current_tid=current_tid,
    color=MONTHLY_COLOR, top_n=3, rank_df=df_var_ranked,
)

# Plot para transformações anuais
plot_rolling_variance(
    series_raw, applicable_annual, 'Anual',
    rolling_window=window, current_tid=current_tid,
    color=ANNUAL_COLOR, top_n=3, rank_df=df_var_ranked,
)

# %% [markdown]
# ## Como Usar
#
# 1. **Altere `selected_series`** na Seção 2 para a série desejada
# 2. **Re-execute** todas as células a partir da Seção 2
# 3. Compare os painéis **mensal** (esquerda/azul) vs. **anual** (direita/verde)
# 4. Consulte a **tabela de testes** para verificar se a transformação atual é estacionária
# 5. Consulte o **ranking de variância** para encontrar a transformação que melhor estabiliza
# 6. Visualize a **variância rolling** para confirmar visualmente
#
# **Transformação marcada com ◀ atual** corresponde ao `transformation_id` do `series_spec.csv`.

# %%


