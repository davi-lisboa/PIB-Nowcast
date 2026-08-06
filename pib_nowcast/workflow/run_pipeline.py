# %% Configurações e Bibliotecas
import ast
import pandas as pd
import datetime as dt

from statsmodels.tsa.api import DynamicFactorMQ

from pib_nowcast.config import (
    SERIES_SPEC, LAST_DATA, DATA_DIR, START_DATE, 
    OUTLIER_THRESHOLD, MODEL_PARAMS_FILE, SKIP_SEAS_ADJ
)
from pib_nowcast.utils.get_data import get_data_parallel
from pib_nowcast.utils.transformations import preprocess_data
from pib_nowcast.utils.news import get_news_impacts, get_new_forecasts, get_factors_history
from pib_nowcast.utils.plots import plot_factors, plot_factors_vs_pib

# --- CONFIGURAÇÕES DO USUÁRIO ---
REFIT_MODEL = True
SAVE_PARAMS = True
EXPORT_RESULTS = True
UPDATE_BASE_DATA = False
FIT_START_DATE = '2005-01-01'
# --------------------------------

# %% Execução Principal
def main():
    print("Iniciando pipeline de Nowcast...")

    specs = pd.read_csv(SERIES_SPEC, sep=';')
    old_data = pd.read_excel(LAST_DATA, sheet_name='full_dataset', index_col='Date')
    new_data = get_data_parallel(specs, START_DATE)

    # Alinhar colunas para garantir comparação justa e apply do modelo
    old_data = old_data.reindex(columns=new_data.columns)
    new_data = new_data.reindex(columns=old_data.columns)

    if old_data.equals(new_data):
        print("Nenhum dado novo encontrado. Encerrando processo com sucesso.")
        return

    print('Atualização detectada. Processando dados...')
    
    # Pré-processamento sem repetição de código (função encapsulada)
    old_clean, new_clean = [
        preprocess_data(d, specs, SKIP_SEAS_ADJ, FIT_START_DATE, OUTLIER_THRESHOLD) 
        for d in (old_data, new_data)
    ]
    
    # Identificar pib_series da versão com dados mais recentes
    pib_series = (new_data if old_data['pib'].last_valid_index() < new_data['pib'].last_valid_index() else old_data)[['pib']].dropna()
    next_pib_quarter = pib_series.last_valid_index() + pd.DateOffset(months=3)

    # Parametrização dos fatores
    factors = {
        k: ast.literal_eval(v) if isinstance(v, str) else v
        for k, v in specs.set_index('variable')['factors'].to_dict().items()
    }

    base_model = DynamicFactorMQ(
        endog=old_clean,
        k_endog_monthly=specs.query("frequency == 'Monthly'").shape[0],
        factors=factors,
        factor_multiplicities={'Global': 1},
        factor_orders={'Global': 2, 'Output': 1, 'Employment': 1, 'Prices': 1, 'Sentiment': 1, 'Credit': 1}
    )

    if MODEL_PARAMS_FILE.exists() and not REFIT_MODEL:
        print(f"[{dt.datetime.now().time()}] Carregando cache...")
        old_model = base_model.smooth(pd.read_csv(MODEL_PARAMS_FILE, index_col=0).squeeze("columns"))
    else:
        print(f"[{dt.datetime.now().time()}] Treinando modelo completo...")
        old_model = base_model.fit(disp=True, maxiter=120, tolerance=1e-5)
        if SAVE_PARAMS:
            old_model.params.to_csv(MODEL_PARAMS_FILE)

    # Modelo com dados novos (apenas apply)
    new_model = old_model.apply(endog=new_clean, k_endog_monthly=base_model.k_endog_monthly)

    # Gráficos
    plot_factors(new_model, factor_type='both', show_recessions=True, save_fig=True)
    plot_factors_vs_pib(new_model, pib_series=new_clean['pib'], save_fig=True)

    # Estimar impactos (news)
    news = new_model.news(
        comparison=old_model, 
        impacted_variable='pib', 
        impact_date=next_pib_quarter.strftime('%Y-%m-%d'),
        comparison_type='previous',
        revisions_details_start=-12
    )
    print(news.summary())

    if EXPORT_RESULTS:
        get_news_impacts(news, save_to=DATA_DIR / 'news_impacts.xlsx')
        get_new_forecasts(
            news=news, new_model_res=new_model, 
            last_pib_date_timestamp=pib_series.last_valid_index(), 
            next_pib_quarter_timestamp=next_pib_quarter, 
            historical_pib_index=pib_series, save_to=DATA_DIR / 'forecasts.xlsx'
        )
        get_factors_history(new_model, save_to=DATA_DIR / 'factors.xlsx')
        print("Resultados exportados.")

    if UPDATE_BASE_DATA:
        new_data.to_excel(LAST_DATA, sheet_name='full_dataset')
        print(f"Base antiga sobrescrita com sucesso em {LAST_DATA.name}.")

if __name__ == '__main__':
    main()
