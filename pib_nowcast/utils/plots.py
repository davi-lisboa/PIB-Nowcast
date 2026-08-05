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
