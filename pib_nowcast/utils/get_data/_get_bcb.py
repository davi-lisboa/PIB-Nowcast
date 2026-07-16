# %% get_bcb
import pandas as pd
from tenacity import wait_fixed, retry, stop_after_attempt

@retry(stop=stop_after_attempt(5))
def get_bcb(
            series: str | dict| pd.DataFrame | None = None, 
            start: str|None = None, 
            last:int|None=None, **kwargs
            ):

    import pandas as pd
    from bcb import sgs

    if (isinstance(series, pd.DataFrame)):

        series = (
                    series
                    .query("source == 'sgs'")
                    .astype({'code': int})
                )

        series = series.set_index('variable')['code'].to_dict()

    return sgs.get(series, start, last)

@retry(stop=stop_after_attempt(5))
def get_bcb_parallel(
            series: str | dict| pd.DataFrame | None = None, 
            start: str|None = None, 
            **kwargs
            ):

    import pandas as pd
    from bcb import sgs
    import asyncio

    if (isinstance(series, pd.DataFrame)):
        series = (
                    series
                    .query("source == 'sgs'")
                    .astype({'code': int})
                )
        series = series.set_index('variable')['code'].to_dict()

    if not series:
        return pd.DataFrame()

    # Utilize sgs.async_get to fetch all series concurrently
    try:
        bcb_series = asyncio.run(sgs.async_get(series, start=start))
    except Exception as e:
        print(f"Erro ao baixar as séries do BCB usando async_get: {e}")
        bcb_series = pd.DataFrame()
        
    return bcb_series

