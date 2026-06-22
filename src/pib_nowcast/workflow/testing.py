# %% Libs
import os
from pathlib import Path

import pandas as pd
import numpy as np
import datetime as dt

# Funções auxiliares 
from pib_nowcast.utils.get_data import get_bcb, get_pib, get_ipeadata

# %%
series_path = r'/workspaces/PIB-Nowcast/src/pib_nowcast/data/series_spec.csv'

# %%

bcb_df = get_bcb(series=series_path, start='1996-01-01')

# %%
ipea_df = get_ipeadata(series=series_path, start='1996-01-01')

# %% 
pib_df = get_pib()
# %%

with pd.ExcelWriter(Path('../data/last_data_at_time.xlsx')) as writer:
    bcb_df.to_excel(writer, sheet_name='sgs')
    ipea_df.to_excel(writer, sheet_name='ipeadata' )
    pib_df.to_excel(writer, sheet_name='pib')

    (
        bcb_df
        .merge(
                ipea_df, left_index=True, right_index=True, how='outer'
            )
        .merge(
            pib_df, left_index=True, right_index=True, how='outer'
        )
    ).to_excel(writer, sheet_name='full_dataset')