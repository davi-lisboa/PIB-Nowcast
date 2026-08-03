# %%
# Bibliotecas
import pandas as pd
from pib_nowcast.config import SERIES_SPEC
from tenacity import wait_fixed, retry, stop_after_attempt

# %%
# get_pib()

@retry(stop=stop_after_attempt(5))
def get_pib():
    import pandas as pd
    import sidrapy
    # Sem ajuste sazonal
    # https://apisidra.ibge.gov.br/values/t/1620/n1/all/v/all/p/all/c11255/90707/d/v583%202

    # Com ajuste sazonal
    # https://apisidra.ibge.gov.br/values/t/1621/n1/all/v/all/p/all/c11255/90707/d/v584%204

    pib = sidrapy.get_table(
                                table_code='1621', # 1620 
                                territorial_level='1', 
                                ibge_territorial_code='all', 
                                period='all',
                                variable='all', 
                                classification='11255/90707', 
                            )
    
    pib = pib[['D2C', 'V']].iloc[1:].reset_index(drop=True)
    pib.columns = ['Date', 'pib']
    pib = pib.assign(
                        pib = lambda df: pd.to_numeric(df['pib'], errors='coerce'),
                        year = pib['Date'].str[:4],
                        month = pib['Date'].str[-2:].astype(int).multiply(3).astype(str).str.zfill(2),
                        Date = lambda df: df['year'] + df['month'],
                    ) \
            .assign(
                        Date = lambda df: pd.to_datetime(df['Date'], format='%Y%m')
                    ) \
            .drop(columns=['year', 'month']) \
            .set_index('Date')

    return pib
# %%
# get_pib_v2
@retry(stop=stop_after_attempt(5))
def get_pib_v2(
                series: str | dict| pd.DataFrame | None = None, 
                start_date: str|None = None
                ):
    import pandas as pd
    import sidrapy


    pib_code = series.query("variable == 'pib'").astype({'code':int})['code'].values[0]

    # Sem ajuste sazonal
    # https://apisidra.ibge.gov.br/values/t/1620/n1/all/v/all/p/all/c11255/90707/d/v583%202

    # Com ajuste sazonal
    # https://apisidra.ibge.gov.br/values/t/1621/n1/all/v/all/p/all/c11255/90707/d/v584%204

    pib = sidrapy.get_table(
                                table_code=f'{pib_code}', # 1620 
                                territorial_level='1', 
                                ibge_territorial_code='all', 
                                period='all',
                                variable='all', 
                                classification='11255/90707', 
                            )
    
    pib = pib[['D2C', 'V']].iloc[1:].reset_index(drop=True)
    pib.columns = ['Date', 'pib']
    pib = pib.assign(
                        pib = lambda df: pd.to_numeric(df['pib'], errors='coerce'),
                        year = pib['Date'].str[:4],
                        month = pib['Date'].str[-2:].astype(int).multiply(3).astype(str).str.zfill(2),
                        Date = lambda df: df['year'] + df['month'],
                    ) \
            .assign(
                        Date = lambda df: pd.to_datetime(df['Date'], format='%Y%m')
                    ) \
            .drop(columns=['year', 'month']) \
            .set_index('Date')

    if start_date is not None:
        pib = pib.loc[start_date:]

    return pib 

# %%
# Run
if __name__ == '__main__':
    import pandas as pd
    from pib_nowcast.config import SERIES_SPEC

    ### Especifica caminho e primeira data
    specs_df = pd.read_csv(SERIES_SPEC, sep=';')
    start_date = '1996-01-01'

    pib = get_pib_v2(specs_df)

    pib
