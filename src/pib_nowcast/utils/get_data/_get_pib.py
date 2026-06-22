from tenacity import wait_fixed, retry, stop_after_attempt

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