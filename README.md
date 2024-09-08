# TJ_HYD_TANK
![PyPI - Version](https://img.shields.io/pypi/v/tj-hyd-tank)


**Python implementation of the Tank Hydrological model by Sugawara and Funiyuki (1956), based on the original code
from [tank-model](https://github.com/nzahasan/tank-model) by [hckaraman](https://github.com/nzahasan).**

## Installation

To install the package, run:

```bash
pip install tj_hyd_tank
```

## Getting Started

### Prepare the Dataset

#### Dataset Requirements

Your dataset should contain the following columns: **Date**, **Precipitation**, **Evapotranspiration**, and **Discharge
**. The column names are customizable.

Example:

| Date       | Q       | P   | E    |
|------------|---------|-----|------|
| 10/9/2016  | 0.25694 | 0   | 2.79 |
| 10/10/2016 | 0.25812 | 0   | 3.46 |
| 10/11/2016 | 0.30983 | 0   | 3.65 |
| 10/12/2016 | 0.31422 | 0   | 3.46 |
| 10/13/2016 | 0.30866 | 0   | 5.64 |
| 10/14/2016 | 0.30868 | 0   | 3.24 |
| 10/15/2016 | 0.31299 | 0   | 3.41 |
| ...        | ...     | ... | ...  |

Ensure the time intervals between dates are consistent (e.g., 24 hours) for accurate model performance.

#### Basin File

Use a HEC-HMS basin file.

### Quick Start

```python
import pandas as pd
from tj_hyd_tank import TJHydTANK, TANKColNames, TANKConfig

df = pd.read_csv('assets/data_example.csv')
tank_cols_name = TANKColNames(
    date='Date',
    precipitation='P',
    evapotranspiration='E',
    discharge='Q'
)
tank_config = TANKConfig(
    start_date=None,
    end_date=None,
    interval=24.0
)

tank = TJHydTANK(
    basin_file='assets/CedarCreek.basin',
    df=df,
    tank_col_names=tank_cols_name,
    tank_config=tank_config
)
```

### Accessing Basin Definitions

```python
from tj_hyd_tank import Subbasin, Reach

for basin_def in tank.basin_defs:
    print(basin_def.name, basin_def.type)
    print(basin_def.stats)
    if isinstance(basin_def, (Subbasin, Reach)):
        print(basin_def.params)
```

### Accessing Root Nodes

```python
from tj_hyd_tank import Subbasin, Reach

for root_node in tank.root_node:
    print(root_node.name, root_node.type)
    print(root_node.stats)
    if isinstance(root_node, (Subbasin, Reach)):
        print(root_node.params)
```

### Plotting Q_obs vs Q_sim for a Basin Definition

```python
outlet1 = tank.get_basin_def_by_name('Outlet1')
if outlet1 is not None:
    tank.show_discharge(outlet1)
```

### Reconfiguring and Displaying Subbasin Properties

```python
from tj_hyd_tank import SubbasinParams

w170 = tank.get_basin_def_by_name('W170')
if w170 is not None:
    if isinstance(w170, Subbasin):
        tank.reconfig_subbasin_params(
            w170,
            SubbasinParams(
                t0_is=0.02,
                t0_soc_uo=80.0
            )
        )
        print('Q_tank_0', w170.Q_tank_0.tolist())
        print('Q_tank_1', w170.Q_tank_1.tolist())
        print('Q_tank_2', w170.Q_tank_2.tolist())
        print('Q_tank_3', w170.Q_tank_3.tolist())
        print('bottom_outlet_flow_tank_0', w170.bottom_outlet_flow_tank_0.tolist())
        print('bottom_outlet_flow_tank_1', w170.bottom_outlet_flow_tank_1.tolist())
        print('bottom_outlet_flow_tank_2', w170.bottom_outlet_flow_tank_2.tolist())
```

### Reconfiguring the TANK Model

```python
tank.reconfig_tank(
    TANKConfig(
        start_date=pd.to_datetime('09/10/2016', dayfirst=True, utc=True),
        end_date=pd.to_datetime('20/10/2016', dayfirst=True, utc=True)
    )
)
```

### Exporting Data to DataFrame

```python
outlet1_df = tank.to_dataframe(outlet1)
outlet1_df
```

### Viewing Logs

```python
print(tank.logs)
```

## BasinDef Subclass

### `Subbasin`

- Params: `SubbasinParams`

### `Junction`

- Params: `BasinDefParams`

### `Sink`

- Params: `BasinDefParams`

### `Reach`

- Params: `ReachParams`

## Exception Classes

### `InvalidBasinFileException`

Raised when a basin file is invalid, possibly due to incorrect format or corrupted data.

### `FileNotFoundException`

Raised when a specified file cannot be located. The missing file's name is included for easy identification.

### `MissingColumnsException`

Raised when required columns are missing from the dataset. The exception specifies which column is absent.

### `ColumnContainsEmptyDataException`

Raised when a specified column contains empty data, ensuring all necessary fields are populated.

### `InvalidDatetimeException`

Raised for invalid datetime entries, such as incorrect formatting or out-of-range values.

### `InvalidDatetimeIntervalException`

Raised when the provided datetime interval is invalid, ensuring consistency in date ranges or intervals.

### `InvalidStartDateException`

Raised for invalid start dates, particularly for time-based events or ranges.

### `InvalidEndDateException`

Raised for invalid end dates, handling errors related to the conclusion of time-based events or ranges.

### `InvalidDateRangeException`

Raised when the date range is invalid, such as when the start date is after the end date. An optional message may
provide additional context.
