# %% Bibliotecas
import sys
import gc
import ast

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from statsmodels.tsa.api import DynamicFactorMQ

from pib_nowcast.config import SERIES_SPEC, LAST_DATA, DATA_DIR, START_DATE, OUTLIER_THRESHOLD, RECESSIONS, MODEL_PARAMS_FILE, FIG_DIR
from pib_nowcast.utils.get_data import get_data, get_data_parallel
from pib_nowcast.utils.transformations import seas_adj_stl_parallel, make_stationary, deflate, remove_outliers
from pib_nowcast.utils.news import get_news_impacts, get_new_forecasts
from pib_nowcast.utils.plots import plot_factors

# %%

### Especifica caminho e primeira data
specs_df = pd.read_csv(SERIES_SPEC, sep=';')
start_date = START_DATE
fit_start_date = '1996-01-01'

### Especificação de datas
today = dt.date.today()

# %% Coletas

## Dataset completo, última run
old_full_data = pd.read_excel(LAST_DATA, sheet_name='full_dataset', index_col='Date')

## Coleta dados mais recentes
new_full_data = get_data_parallel(specs_df, start_date)
# Reordenar as colunas para evitar erros no apply do statsmodels
# old_full_data = old_full_data[new_full_data.columns]
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
old_full_data_stat = old_full_data_stat.loc[fit_start_date:, :]
new_full_data_stat = new_full_data_stat.loc[fit_start_date:, :]

## -> Remoção de Outliers
old_full_data_stat = remove_outliers(old_full_data_stat, threshold=OUTLIER_THRESHOLD)
new_full_data_stat = remove_outliers(new_full_data_stat, threshold=OUTLIER_THRESHOLD)


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
# del old_full_data, new_full_data, old_full_data_sa, new_full_data_sa
gc.collect()

# %% Estimação do modelo com dados antigos

# Extrai os fatores especificados e corrige o tipo dos dados
factors = specs_df.set_index('variable')['factors'].to_dict()
factors = {
    k: ast.literal_eval(v) if isinstance(v, str) else v
    for k, v in factors.items()
}

old_model_base = DynamicFactorMQ(
    endog = old_full_data_stat,
    k_endog_monthly = specs_df.query("frequency == 'Monthly' ").shape[0],
    factors = factors,
    factor_multiplicities={ 'Global': 2 },
    factor_orders = {
        'Global': 4,
        'Output': 4,
        'Employment': 4,
        'Prices': 4,
        'Sentiment': 4,
        'Credit': 4

    }
    # factor_orders = 1
        # ('Global', 'Output', 'Employment', 'Prices', 'Sentiment', 'Credit'): 4
)
# 'Global': 1,
# ('Output', 'Employment', 'Prices', 'Sentiment', 'Credit'): 1,

refit = True
save_params = True

# Se o arquivo de parâmetros existir, carrega e faz smooth
if MODEL_PARAMS_FILE.exists() and not refit:
    print(f"[{dt.datetime.now().time()}] Carregando parâmetros do modelo de cache: {MODEL_PARAMS_FILE.name}")
    # Injeta parâmetros estáticos sem rodar EM, ~0.1s
    params = pd.read_csv(MODEL_PARAMS_FILE, index_col=0).squeeze("columns")
    old_model = old_model_base.smooth(params)
else:
    print(f"[{dt.datetime.now().time()}] Treinando modelo completo (isso pode demorar)...")
    old_model = old_model_base.fit(
        disp=True,
        maxiter=120,
        tolerance=1e-5,
    )
    if save_params:
        old_model.params.to_csv(MODEL_PARAMS_FILE)
        print(f"[{dt.datetime.now().time()}] Parâmetros salvos em {MODEL_PARAMS_FILE.name}")

print(old_model.summary())

plot_factors(old_model, factor_type='both', show_recessions=True, save_fig=True)

# %% Estimação do modelo com novos dados

new_model = old_model.apply(
    endog = new_full_data_stat,
    k_endog_monthly = specs_df.query("frequency == 'Monthly' ").shape[0],

)

# %% Plot dos fatores
plot_factors(new_model, factor_type='both', show_recessions=True, save_fig=True)

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


# %% Impactos e forecasts

export = False

if export:
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

