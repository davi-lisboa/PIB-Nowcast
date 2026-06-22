from tenacity import wait_fixed, retry, stop_after_attempt

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