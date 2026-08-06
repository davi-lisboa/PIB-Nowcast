# %% Imports
import pandas as pd
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
# O formato típico do statsmodels é 'loading.L1.Fator.Variavel' ou 'loading.Fator.Variavel'
# Aqui tratamos genericamente extraindo a variável e o fator
loadings['abs_value'] = loadings['value'].abs()

print("--- Top Variáveis com Maior Peso por Fator ---")
# Como os nomes podem variar, ordenamos e exibimos os maiores pesos absolutos.
# Uma lógica customizada de string split pode ser feita se o nome do fator for fixo.
top_loadings = loadings.nlargest(15, 'abs_value')
for _, row in top_loadings.iterrows():
    print(f"{row['param']}: {row['value']:.4f}")

# %% 3. Explicabilidade (Fatores x PIB)
# Pegamos apenas fatores 'filtered' mais recentes e pivotamos para formato largo
recent_factors = factors_long[factors_long['type'] == 'filtered'].pivot_table(
    index='reference date', columns='factor', values='value'
)

# Junta com o PIB
df_model = data[['pib']].join(recent_factors).dropna()

y = df_model['pib']
X = sm.add_constant(df_model.drop(columns='pib'))

# Roda regressão para avaliar a capacidade explicativa
ols = sm.OLS(y, X).fit()
print("\n--- Regressão: Poder Explicativo dos Fatores sobre o PIB ---")
print(ols.summary().tables[0])
print(ols.summary().tables[1])
