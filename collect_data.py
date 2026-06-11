# %% get_bcb
from tenacity import wait_fixed, retry, stop_after_attempt

@retry(stop=stop_after_attempt(5))
def get_bcb(serie:dict|None = None, start:str|None = None):

    import pandas as pd
    from bcb import sgs

    series = pd.read_csv('series_spec.csv', sep=';') \
                .query("variable != 'pib'") \
                .astype({'sgs_code': int})

    series = series.set_index('variable')['sgs_code'].to_dict()
    
    return sgs.get(series, start)

get_bcb()

# %% get_pib

@retry(stop=stop_after_attempt(5))
def get_pib():
    import pandas as pd
    import sidrapy
    # https://apisidra.ibge.gov.br/values/t/1620/n1/all/v/all/p/all/c11255/90707/d/v583%202

    pib = sidrapy.get_table(
                                table_code='1620', 
                                territorial_level='1', 
                                ibge_territorial_code='all', 
                                period='all',
                                variable='all', 
                                classification='11255/90707', 
                            )
    
    pib = pib[['D2C', 'V']].iloc[1:].reset_index(drop=True)
    pib.columns = ['Date', 'pib']
    pib = pib.assign(
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
