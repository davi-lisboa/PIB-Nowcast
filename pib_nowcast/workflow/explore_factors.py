# %% Imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pib_nowcast.config import DATA_DIR, LAST_DATA, MODEL_PARAMS_FILE

# %% 1. Base de Dados
factors_long = pd.read_excel(DATA_DIR / 'factors.xlsx')
data = pd.read_excel(LAST_DATA, sheet_name='full_dataset', index_col='Date')
params = pd.read_csv(MODEL_PARAMS_FILE, index_col=0).squeeze("columns")

# %% 2. Cargas Fatoriais (O que move cada fator?)
# Isola as cargas fatoriais (loadings) dos parâmetros salvos
loadings = params[params.index.str.contains('loading')].reset_index()
loadings.columns = ['param', 'value']
loadings = loadings.assign(
                            factor = lambda df: [factor.replace('loading.', "") for factor, _ in df['param'].str.split('->').to_list()],
                            variable = lambda df: [var for _, var in df['param'].str.split('->').to_list()],
                        )
loadings  = loadings[['factor', 'variable', 'value']]

# O formato típico do statsmodels é 'loading.L1.Fator.Variavel' ou 'loading.Fator.Variavel'
# Aqui tratamos genericamente extraindo a variável e o fator
loadings['abs_value'] = loadings['value'].abs()

loadings = loadings.sort_values(by=['factor', 'variable', 'abs_value'], ascending=[True, True, False])

print("--- Top Variáveis com Maior Peso por Fator ---")
# Como os nomes podem variar, ordenamos e exibimos os maiores pesos absolutos.
# Uma lógica customizada de string split pode ser feita se o nome do fator for fixo.
# top_loadings = loadings.nlargest(20, 'abs_value')
# for _, row in top_loadings.iterrows():
#     print(f"{row['param']}: {row['value']:.4f}")
for f in loadings['factor'].unique():
    display(loadings.query("factor == @f").nlargest(5, 'abs_value'))

# %% 3. Explicabilidade (Fatores x PIB)
# Pegamos apenas fatores 'filtered' mais recentes e pivotamos para formato largo
recent_factors = factors_long[factors_long['type'] == 'filtered'].pivot_table(
    index='reference date', columns='factor', values='value'
)

# Junta com o PIB
df_model = data[['pib']].assign(
                                pib_yoy = data['pib'].pct_change(12).multiply(100),
                        ).join(recent_factors).dropna()

y = df_model['pib_yoy']
X = sm.add_constant(df_model.drop(columns=['pib', 'pib_yoy']))

# Roda regressão para avaliar a capacidade explicativa
ols_full = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 4})
# print("\n--- Regressão: Poder Explicativo dos Fatores sobre o PIB ---")
# print(ols.summary().tables[0])
# print(ols.summary().tables[1])
# print(ols_full.summary())

for factor in recent_factors.columns:
    X_factor = sm.add_constant(df_model[[factor]])
    ols_factor = sm.OLS(y, X_factor, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    print(f"\n--- Regressão: Poder Explicativo do Fator {factor} sobre o PIB ---")
    # print(ols_factor.summary())
    print("R²:", ols_factor.rsquared.round(4), "R² Adj:", ols_factor.rsquared_adj.round(4))
    display(pd.DataFrame({'Params': ols_factor.params.round(4), 'P-Values': ols_factor.pvalues.round(4)}))
