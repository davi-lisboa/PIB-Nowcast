import datetime as dt
import matplotlib.pyplot as plt
from pib_nowcast.config import RECESSIONS, FIG_DIR

def _add_recessions(recessions, ax):
    for recession in recessions:
        if len(recession) > 0:
            ax.axvspan(recession[0], recession[-1], color='black', alpha=0.3, lw=0)

def plot_factors(model, factor_type='both', show_recessions=True, save_fig=False):
    """
    Plota os fatores do modelo.
    
    Parâmetros:
    - model: O objeto do modelo contendo os fatores.
    - factor_type: 'filtered', 'smooth' ou 'both' (padrão).
    - show_recessions: Booleano indicando se deve mostrar as recessões (padrão True).
    - save_fig: Booleano indicando se deve salvar a figura no FIG_DIR.
    """
    if factor_type not in ['filtered', 'smooth', 'smoothed', 'both']:
        raise ValueError("factor_type deve ser 'filtered', 'smooth', ou 'both'")

    filtered_factors = model.factors['filtered']
    smoothed_factors = model.factors['smoothed']
    
    n_fatores = len(filtered_factors.columns)
    
    cols = 3
    rows = max(1, (n_fatores + cols - 1) // cols)
    
    fig, ax = plt.subplots(rows, cols, figsize=(14, max(8, rows * 3)), dpi=300)
    
    # Garantir que ax seja iterável e achatado
    if rows == 1 and cols == 1:
        ax = [ax]
    else:
        ax = ax.ravel()
        
    for i, factor in enumerate(filtered_factors.columns):
        ax[i].set_title(factor)
        
        if factor_type in ['filtered', 'both']:
            ax[i].plot(filtered_factors.index, filtered_factors[factor], label='Filtered', color='blue')
        if factor_type in ['smooth', 'smoothed', 'both']:
            ax[i].plot(smoothed_factors.index, smoothed_factors[factor], label='Smoothed', color='orange')
            
        ax[i].legend()
        
        if show_recessions:
            _add_recessions(RECESSIONS, ax[i])

    # Esconder eixos vazios
    for i in range(n_fatores, len(ax)):
        ax[i].axis('off')
        
    fig.tight_layout()
    
    if save_fig:
        now = dt.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        fig.savefig(FIG_DIR / f'fatores_{now}.png', dpi=300)
        
    return fig, ax


def plot_factors_vs_pib(model, pib_series, factor_names=None, show_recessions=True, save_fig=False):
    """Plota fatores (smoothed) vs PIB observado (transformado) em eixos duplos.
    
    Parâmetros:
    - model: Objeto do modelo estimado contendo os fatores.
    - pib_series: pd.Series com o PIB já transformado (e.g. YoY%, QoQ%).
    - factor_names: Lista de nomes dos fatores a plotar. Default: ['Global', 'Output'].
    - show_recessions: Booleano indicando se deve mostrar as recessões.
    - save_fig: Booleano indicando se deve salvar a figura no FIG_DIR.
    """
    import numpy as np

    if factor_names is None:
        factor_names = ['Global', 'Output']
    
    smoothed_factors = model.factors['smoothed']
    
    # Identificar colunas que correspondem a cada fator solicitado
    factor_cols = {}
    for name in factor_names:
        matching = [c for c in smoothed_factors.columns if c == name or c.startswith(f'{name}.')]
        if matching:
            # Usa o primeiro (e.g. 'Global' ou 'Global.1')
            factor_cols[name] = matching[0]
    
    if not factor_cols:
        print(f"[WARN] Nenhum dos fatores {factor_names} encontrado nas colunas: {list(smoothed_factors.columns)}")
        return None, None
    
    n_plots = len(factor_cols)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5), dpi=300)
    
    if n_plots == 1:
        axes = [axes]
    
    # PIB como série trimestral (pode ter NaNs nas datas mensais)
    pib = pib_series.dropna()
    
    for i, (label, col_name) in enumerate(factor_cols.items()):
        ax1 = axes[i]
        factor_data = smoothed_factors[col_name]
        
        # Plot do fator (eixo esquerdo)
        color_factor = '#2563EB'
        ax1.plot(factor_data.index, factor_data, color=color_factor, linewidth=1.2, alpha=0.85, label=f'Fator {label} (smoothed)')
        ax1.set_ylabel(f'Fator {label}', color=color_factor)
        ax1.tick_params(axis='y', labelcolor=color_factor)
        
        # Plot do PIB (eixo direito)
        ax2 = ax1.twinx()
        color_pib = '#DC2626'
        ax2.plot(pib.index, pib.values, color=color_pib, linewidth=1.5, alpha=0.9, label='PIB observado', marker='o', markersize=2)
        ax2.set_ylabel('PIB (transformado)', color=color_pib)
        ax2.tick_params(axis='y', labelcolor=color_pib)
        
        # Calcular correlação no período comum
        common_idx = factor_data.dropna().index.intersection(pib.index)
        if len(common_idx) > 5:
            corr = np.corrcoef(factor_data.loc[common_idx].values, pib.loc[common_idx].values)[0, 1]
            ax1.set_title(f'{label} vs PIB  |  ρ = {corr:.3f}')
        else:
            ax1.set_title(f'{label} vs PIB')
        
        # Legendas combinadas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        if show_recessions:
            _add_recessions(RECESSIONS, ax1)
    
    fig.tight_layout()
    
    if save_fig:
        now = dt.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        fig.savefig(FIG_DIR / f'fatores_vs_pib_{now}.png', dpi=300)
    
    return fig, axes
