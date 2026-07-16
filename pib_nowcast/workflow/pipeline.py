# %% Bibliotecas
import sys
import gc
import ast

import pandas as pd
import numpy as np
import datetime as dt

from statsmodels.tsa.api import DynamicFactorMQ

from pib_nowcast.config import SERIES_SPEC, LAST_DATA, DATA_DIR
from pib_nowcast.utils.get_data import get_data, get_data_parallel
from pib_nowcast.utils.transformations import seas_adj_stl_parallel, make_stationary, deflate
from pib_nowcast.utils.news import get_news_impacts, get_new_forecasts

# %%

### Especifica caminho e primeira data
specs_df = pd.read_csv(SERIES_SPEC, sep=';')
start_date = '1996-01-01'

### Especificação de datas
today = dt.date.today()

# %% Coletas

## Dataset completo, última run
old_full_data = pd.read_excel(LAST_DATA, sheet_name='full_dataset', index_col='Date')

## Coleta dados mais recentes
new_full_data = get_data_parallel(specs_df, start_date)
# Reordenar as colunas para evitar erros no apply do statsmodels
new_full_data = new_full_data[old_full_data.columns]

new_full_data
# %% Comparação

if old_full_data.equals(new_full_data):
    print("Sem dados novos ou revisões, encerrando processso.")
    sys.exit(0)

else:
    print('Houve atualização/revisão nos dados, prosseguindo com o processo.')

# %% Tratamentos 

## -> Deflacionar valores nominais
old_full_data_defl = deflate(old_full_data, specs_df)
new_full_data_defl = deflate(new_full_data, specs_df)

## -> Ajuste sazonal
old_full_data_sa = seas_adj_stl_parallel(old_full_data_defl, specs_df)
new_full_data_sa = seas_adj_stl_parallel(new_full_data_defl, specs_df)

## -> Estacionarização
old_full_data_stat = make_stationary(old_full_data_sa, specs_df)
new_full_data_stat = make_stationary(new_full_data_sa, specs_df)

## Filtro de data
old_full_data_stat = old_full_data_stat.loc['2007-01-01':, :]
new_full_data_stat = new_full_data_stat.loc['2007-01-01':, :]


# %% Separar datas e dfs relevantes do PIB (Antes da Limpeza de Memória)

## Caso onde houve atualização do PIB
if old_full_data['pib'].last_valid_index() < new_full_data['pib'].last_valid_index():
    pib_series = new_full_data[['pib']].dropna()
else:
    pib_series = old_full_data[['pib']].dropna()

last_pib_date_timestamp = pib_series.last_valid_index()

# Definir próximo trimestre do PIB
next_pib_quarter_timestamp = last_pib_date_timestamp + pd.DateOffset(months=3)

# %% Limpeza de Memória
del old_full_data, new_full_data, old_full_data_sa, new_full_data_sa
gc.collect()

# %% Estimação do modelo com dados antigos

# Extrai os fatores especificados e corrige o tipo dos dados
factors = specs_df.set_index('variable')['factors'].to_dict()
factors = {
    k: ast.literal_eval(v) if isinstance(v, str) else v
    for k, v in factors.items()
}

old_model = DynamicFactorMQ(
    endog = old_full_data_stat,
    k_endog_monthly = specs_df.query("frequency == 'Monthly' ").shape[0],
    factors = factors,
    # factor_multiplicities={ 'Global': 2 },
    factor_orders = {
        'Global': 1,
        ('Output', 'Employment', 'Prices', 'Sentiment', 'Credit'): 1
    }
).fit(
    disp=True,
    maxiter=120,
    tolerance=1e-5,
)

print(old_model.summary())

# %% Estimação do modelo com novos dados

new_model = old_model.apply(
    endog = new_full_data_stat,
    k_endog_monthly = specs_df.query("frequency == 'Monthly' ").shape[0],

)


# %% Estimar news
news = new_model.news(
    comparison=old_model, 
    impacted_variable='pib', 
    impact_date=next_pib_quarter_timestamp.strftime('%Y-%m-%d'),
    # tolerance=1e-5,
    comparison_type='previous',
    revisions_details_start=-12  # Limita as matrizes de revisões apenas para os últimos 12 meses
)
print(news.summary())

# %%

filtered_factors = new_model.factors['filtered']
smoothed_factors = new_model.factors['smoothed']

# recessions1 =    pd.date_range(start='1998-01-01', end='1999-03-01', freq='MS').to_list() 
# recessions2 =    pd.date_range(start='2001-04-01', end='2001-12-01', freq='MS').to_list()
# recessions3 =    pd.date_range(start='2003-01-01', end='2003-06-01', freq='MS').to_list()
recessions4 =    pd.date_range(start='2008-10-01', end='2009-03-01', freq='MS').to_list()
recessions5 =    pd.date_range(start='2014-04-01', end='2016-12-01', freq='MS').to_list()
recessions6 =    pd.date_range(start='2020-01-01', end='2020-06-01', freq='MS').to_list()

recessions = [
    # recessions1, recessions2, recessions3, 
    recessions4, recessions5, recessions6
    ]

def _add_recessions(recessions, ax, ymin, ymax):
    for recession in recessions:
        ax.fill_between(x=recession, y1=ymin, y2=ymax, color='black', alpha=0.3)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 3, figsize=(14, 6))

ax = ax.ravel()

for i, factor in enumerate(filtered_factors.columns):

    ax[i].set_title(factor)
    ax[i].plot(filtered_factors.index, filtered_factors[factor], label='Filtered', color='blue')
    ax[i].plot(smoothed_factors.index, smoothed_factors[factor], label='Smoothed', color='orange')
    ax[i].legend()
    _add_recessions(
                    recessions, 
                    ax[i], 
                    ymin=min(filtered_factors[factor].min(), smoothed_factors[factor].min()), 
                    ymax=max(filtered_factors[factor].max(), smoothed_factors[factor].max())
                )



# %% Impactos e forecasts

## -> Salvar impactos no histórico
get_news_impacts(news, save_to=DATA_DIR / 'news_impacts.xlsx')


## -> Salvar novos forecasts no histórico
forecasts_df = get_new_forecasts(
    news=news, 
    new_model_res=new_model, 
    last_pib_date_timestamp=last_pib_date_timestamp, 
    next_pib_quarter_timestamp=next_pib_quarter_timestamp, 
    historical_pib_index=pib_series,
    save_to=DATA_DIR / 'forecasts.xlsx'
)

# %%
