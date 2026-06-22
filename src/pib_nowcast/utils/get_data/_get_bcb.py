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