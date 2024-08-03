import pandas as pd

from tj_hyd_tank import TJHydTANK, TANKColNames

TJHydTANK(
    'CedarCreek.basin',
    pd.read_csv('data_example.csv'),
    TANKColNames(
        date='Date',
        precipitation='P',
        evapotranspiration='E',
        discharge='Q'
    )
)
