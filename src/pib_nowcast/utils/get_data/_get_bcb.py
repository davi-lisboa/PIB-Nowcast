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