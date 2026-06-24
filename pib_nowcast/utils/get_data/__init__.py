# %%
import pandas as pd

from ._get_bcb import get_bcb
from ._get_pib import get_pib
from ._get_ipeadata import get_ipeadata

# %%
def get_data(specs_df: pd.DataFrame, start:str | None = None):

    import pandas as pd

    ### Efetua coletas no SGS, IPEA Data e PIB no SIDRA
    bcb_df = get_bcb(series=specs_df, start=start)
    ipea_df = get_ipeadata(series=specs_df, start=start)
    pib_df = get_pib()

    ### Junta tudo num df só
    full_data = pd.concat([bcb_df, ipea_df, pib_df], axis=1, join='outer').loc[start:, :]

    return full_data


# %%
if __name__ == '__main__':
    from pib_nowcast.config import SERIES_SPEC

    ### Especifica caminho e primeira data
    specs_df = pd.read_csv(SERIES_SPEC, sep=';')
    start_date = '1996-01-01'

    get_data(specs_df, start_date)