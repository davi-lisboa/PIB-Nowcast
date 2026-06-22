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

    if series == {}:
        print("Não foi localizada nenhuma série do IPEA Data no argumento `series`")
        return pd.DataFrame()

    ipea_dfs_list = []
    for name, code in series.items():
        temp = ipea.timeseries(series=code)
        temp = temp.iloc[:, [-1]]

        temp.columns = [name]

        ipea_dfs_list.append(temp)

    ipea_series = pd.concat(ipea_dfs_list, axis=1, join='outer')

    if (ipea_series is not None) and (start is not None) and (isinstance(start, str)):
        ipea_series = ipea_series.loc[start:, :]

    return ipea_series
