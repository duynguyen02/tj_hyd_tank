import dataclasses
import datetime
from asyncio import Queue
from typing import Any, List

import numpy as np
import pandas as pd
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from pandas import DataFrame, Series

from .tank_exception import ColumnContainsEmptyDataException, InvalidDatetimeException, \
    MissingColumnsException, InvalidDatetimeIntervalException, InvalidStartDateException, InvalidEndDateException, \
    InvalidDateRangeException


@dataclasses.dataclass
class TANKColNames:
    date: str = 'Date'
    precipitation: str = 'Precipitation'
    evapotranspiration: str = 'Evapotranspiration'
    discharge: str = 'Discharge'


@dataclasses.dataclass
class TANKConfig:
    area: float
    start_date: datetime.datetime | None = None
    end_date: datetime.datetime | None = None
    root_node: List[str] = dataclasses.field(default_factory=lambda: [])
    interval: float = 24.0
    t0_is: float = 0.01
    t0_boc: float = 0.1
    t0_soc_uo: float = 0.1
    t0_soc_lo: float = 0.1
    t0_soh_uo: float = 75.0
    t0_soh_lo: float = 0.0
    t1_is: float = 0.01
    t1_boc: float = 0.01
    t1_soc: float = 0.01
    t1_soh: float = 0.0
    t2_is: float = 0.01
    t2_boc: float = 0.01
    t2_soc: float = 0.01
    t2_soh: float = 0.0
    t3_is: float = 0.01
    t3_soc: float = 0.0

    def to_initial_params(self):
        return {
            "t0_is": self.t0_is,
            "t0_boc": self.t0_boc,
            "t0_soc_uo": self.t0_soc_uo,
            "t0_soc_lo": self.t0_soc_lo,
            "t0_soh_uo": self.t0_soh_uo,
            "t0_soh_lo": self.t0_soh_lo,
            "t1_is": self.t1_is,
            "t1_boc": self.t1_boc,
            "t1_soc": self.t1_soc,
            "t1_soh": self.t1_soh,
            "t2_is": self.t2_is,
            "t2_boc": self.t2_boc,
            "t2_soc": self.t2_soc,
            "t2_soh": self.t2_soh,
            "t3_is": self.t3_is,
            "t3_soc": self.t3_soc
        }


@dataclasses.dataclass
class TANKStatistic:
    nse: float = None
    rmse: float = None
    fbias: float = None
    r2: float = None


class TJHydTANK:
    def __init__(
            self,
            dataset: DataFrame,
            tank_col_names: TANKColNames,
            tank_config: TANKConfig
    ):
        self._dataset = dataset.copy()
        self._tank_col_names = tank_col_names
        self._tank_configs: TANKConfig = tank_config
        self._tank_statistic: TANKStatistic = TANKStatistic()

        self._size: int | None = None
        self._date: Series | None = None
        self._T: Series | None = None
        self._P: Series | None = None
        self._E: Series | None = None
        self._Q_obs: Series | None = None
        self._Q_sim: ndarray[Any, dtype[floating[_64Bit]]] | None = None  # simulator_discharge

    def _validate_dataset(self):
        required_columns = [
            self._tank_col_names.date,
            self._tank_col_names.precipitation,
            self._tank_col_names.evapotranspiration,
            self._tank_col_names.discharge
        ]

        for col in required_columns:
            if col not in self._dataset.columns:
                raise MissingColumnsException(col)

        for column in self._dataset.columns:
            if self._dataset[column].isnull().any():
                raise ColumnContainsEmptyDataException(column)

        try:
            self._dataset[self._tank_col_names.date] = pd.to_datetime(
                self._dataset[self._tank_col_names.date],
                utc=True
            )
        except Exception as _:
            str(_)
            raise InvalidDatetimeException()

        self._dataset['Interval'] = self._dataset[self._tank_col_names.date].diff()
        interval_hours = pd.Timedelta(hours=self._tank_configs.interval)
        is_valid_interval_hours = self._dataset['Interval'].dropna().eq(interval_hours).all()
        if not is_valid_interval_hours:
            raise InvalidDatetimeIntervalException()
        self._dataset = self._dataset.drop(columns=['Interval'])

        start = 0
        end = self._dataset.size

        if self._tank_configs.start_date is not None:
            valid = False
            for i in range(len(self._dataset.index)):
                if str(self._dataset[self._tank_col_names.date][i]) == str(self._tank_configs.start_date):
                    valid = True
                    start = i
                    break

            if not valid:
                raise InvalidStartDateException()

        if self._tank_configs.end_date is not None:
            valid = False
            for i in range(len(self._dataset.index)):
                if str(self._dataset[self._tank_col_names.date][i]) == str(self._tank_configs.end_date):
                    valid = True
                    end = i + 1
                    break

            if not valid:
                raise InvalidEndDateException()

        if self._tank_configs.start_date is not None and self._tank_configs.end_date is not None:
            if self._tank_configs.start_date > self._tank_configs.end_date:
                raise InvalidDateRangeException()

        return start, end

    def _init_data(self, start: int, end: int):
        proc_dataset = self._dataset.copy()
        proc_dataset = proc_dataset.iloc[start: end]

        self._size: int = proc_dataset.size
        self._date: Series = proc_dataset[self._tank_col_names.date]
        self._P: Series = proc_dataset[self._tank_col_names.precipitation]  # precipitation
        self._E: Series = proc_dataset[self._tank_col_names.evapotranspiration]  # evapotranspiration
        self._Q_obs: Series = proc_dataset[self._tank_col_names.discharge]  # observed_discharge
        self._Q_sim: ndarray[Any, dtype[floating[_64Bit]]] = np.zeros(self._size)  # simulator_discharge

    def _build_computation_stack(self):
        computation_stack = []

        node_queue: Queue[str] = Queue()

        for root_node in self._tank_configs.root_node:
            node_queue.put(root_node)

        while not node_queue.empty():
            node = node_queue.get()
            computation_stack.append(node)
            if project['basin_def'][node].get('upstream', False):
                childs = project['basin_def'][node]['upstream']

                for child in childs:
                    node_queue.put(child)

        return computation_stack

    def _calculate(self):
        ...

    def _run(self):
        start, end = self._validate_dataset()
        self._init_data(start, end)
        self._calculate()
