# %% get_bcb
from tenacity import wait_fixed, retry, stop_after_attempt

@retry(stop=stop_after_attempt(5))
def get_bcb(series: str| dict| None = None, start: str|None = None, **kwargs):

    import pandas as pd
    from bcb import sgs


    if (isinstance(series, str)) and ('.csv' in series):

        series = (
                    pd.read_csv(series, sep=';')
                    .query("source == 'sgs'")
                    .astype({'code': int})
                )

        series = series.set_index('variable')['code'].to_dict()
    
    return sgs.get(series, start)

# get_bcb('series_spec.csv')

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

get_pib()

# %% get_ipeadata

@retry(stop=stop_after_attempt(5))
def get_ipeadata(series: str| dict| None = None, start: str|None = None, **kwargs):
    import pandas as pd
    import ipeadatapy as ipea
    
    if (isinstance(series, str)) and ('.csv' in series):

        series = (
                    pd.read_csv(series, sep=';')
                    .query("source == 'ipeadata'")
                )

        series = series.set_index('variable')['code'].to_dict()

    ipea_dfs_list = []
    for name, code in series.items():
        temp = ipea.timeseries(series=code)
        temp = temp.iloc[:, [-1]]
        temp.columns = [name]

        ipea_dfs_list.append(temp)

    ipea_series = pd.concat(ipea_dfs_list, axis=1, join='outer')

    return ipea_series


get_ipeadata(
    {
        'cesta_basica_dieese': 'DIEESE12_CBSP12',
        'caged_saldo_adj': 'CAGED12_SALDONAJU12',
    
    }

).plot(subplots=True, figsize=(14,8))
    



# %%
