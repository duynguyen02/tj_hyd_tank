import dataclasses
from enum import Enum
from typing import Optional, List

import numpy as np
import pandas as pd


class BasinDefType(Enum):
    SUBBASIN = 'SUBBASIN'
    REACH = 'REACH'
    JUNCTION = 'JUNCTION'
    SINK = 'SINK'


class BasinDefParams:
    def __str__(self):
        return str(self.__dict__)

    # def __repr__(self):
    #     return str(self.__dict__)


@dataclasses.dataclass
class BasinDefStatistics:
    rmse: float = None
    nse: float = None
    r2: float = None
    pbias: float = None


def MSE(x: np.ndarray, y: np.ndarray):
    return ((x - y) ** 2).sum() / x.shape[0]


def RMSE(x: np.ndarray, y: np.ndarray):
    return np.sqrt(MSE(x, y))


def NSE(sim: np.ndarray, obs: np.ndarray):
    obs_mean = obs.mean()
    return 1 - (np.square(obs - sim).sum() / np.square(obs - obs_mean).sum())


def R2(x: np.ndarray, y: np.ndarray):
    n = x.shape[0]

    NU = (n * ((x * y).sum()) - (x.sum()) * (y.sum())) ** 2
    DE = (n * ((x ** 2).sum()) - (x.sum()) ** 2) * (n * ((y ** 2).sum()) - (y.sum()) ** 2)

    return NU / DE


def PBIAS(obs: np.ndarray, sim: np.ndarray):
    return (obs - sim).sum() * 100 / obs.sum()


class BasinDef:
    def __init__(self, name: str, hyd_e_type: BasinDefType):
        self._name = name
        self._type = hyd_e_type
        self._params: BasinDefParams = BasinDefParams()
        self._stats: BasinDefStatistics = BasinDefStatistics()
        self._downstream: Optional['BasinDef'] = None
        self._upstream: List['BasinDef'] = []
        self._Q_sim: Optional[np.ndarray] = None

    def calculate_stats(self, Q_obs: np.ndarray):
        self._stats.rmse = RMSE(Q_obs, self._Q_sim)
        self._stats.nse = NSE(self._Q_sim, Q_obs)
        self._stats.r2 = R2(self._Q_sim, Q_obs)
        self._stats.pbias = PBIAS(self._Q_sim, Q_obs)

    @property
    def params(self):
        return self._params

    @property
    def stats(self):
        return self._stats

    @property
    def Q_sim(self):
        return self._Q_sim

    @Q_sim.setter
    def Q_sim(self, value: Optional[np.ndarray]):
        self._Q_sim = value

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def upstream(self):
        return self._upstream

    @upstream.setter
    def upstream(self, upstream: List['BasinDef']):
        self._upstream = upstream

    @property
    def downstream(self):
        return self._downstream

    @downstream.setter
    def downstream(self, downstream: 'BasinDef'):
        self._downstream = downstream

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


@dataclasses.dataclass
class SubbasinParams(BasinDefParams):
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
    t3_soc: float = 0.01

    @classmethod
    def MaxBoundParams(cls):
        return cls(
            t0_is=100.0,
            t0_boc=0.5,
            t0_soc_uo=0.5,
            t0_soc_lo=0.5,
            t0_soh_uo=100.0,
            t0_soh_lo=50.0,
            t1_is=100.0,
            t1_boc=0.5,
            t1_soc=0.5,
            t1_soh=100.0,
            t2_is=100.0,
            t2_boc=0.5,
            t2_soc=0.5,
            t2_soh=100.0,
            t3_is=100.0,
            t3_soc=0.5,
        )

    def to_list(self):
        return [
            self.t0_is,
            self.t0_boc,
            self.t0_soc_uo,
            self.t0_soc_lo,
            self.t0_soh_uo,
            self.t0_soh_lo,
            self.t1_is,
            self.t1_boc,
            self.t1_soc,
            self.t1_soh,
            self.t2_is,
            self.t2_boc,
            self.t2_soc,
            self.t2_soh,
            self.t3_is,
            self.t3_soc,
        ]


class Subbasin(BasinDef):
    def __init__(
            self,
            name: str,
            area: float
    ):
        super().__init__(name, BasinDefType.SUBBASIN)
        self._area = area
        self._params = SubbasinParams()

        self._Q_tank_0: Optional[np.ndarray] = None
        self._Q_tank_1: Optional[np.ndarray] = None
        self._Q_tank_2: Optional[np.ndarray] = None
        self._Q_tank_3: Optional[np.ndarray] = None

        self._side_outlet_flow_tank_0: Optional[np.ndarray] = None
        self._side_outlet_flow_tank_1: Optional[np.ndarray] = None
        self._side_outlet_flow_tank_2: Optional[np.ndarray] = None
        self._side_outlet_flow_tank_3: Optional[np.ndarray] = None

        self._bottom_outlet_flow_tank_0: Optional[np.ndarray] = None
        self._bottom_outlet_flow_tank_1: Optional[np.ndarray] = None
        self._bottom_outlet_flow_tank_2: Optional[np.ndarray] = None

    @property
    def bottom_outlet_flow_tank_0(self) -> Optional[np.ndarray]:
        return self._bottom_outlet_flow_tank_0

    @bottom_outlet_flow_tank_0.setter
    def bottom_outlet_flow_tank_0(self, value: Optional[np.ndarray]):
        self._bottom_outlet_flow_tank_0 = value

    @property
    def bottom_outlet_flow_tank_1(self) -> Optional[np.ndarray]:
        return self._bottom_outlet_flow_tank_1

    @bottom_outlet_flow_tank_1.setter
    def bottom_outlet_flow_tank_1(self, value: Optional[np.ndarray]):
        self._bottom_outlet_flow_tank_1 = value

    @property
    def bottom_outlet_flow_tank_2(self) -> Optional[np.ndarray]:
        return self._bottom_outlet_flow_tank_2

    @bottom_outlet_flow_tank_2.setter
    def bottom_outlet_flow_tank_2(self, value: Optional[np.ndarray]):
        self._bottom_outlet_flow_tank_2 = value

    @property
    def side_outlet_flow_tank_1(self) -> Optional[np.ndarray]:
        return self._side_outlet_flow_tank_1

    @side_outlet_flow_tank_1.setter
    def side_outlet_flow_tank_1(self, value: Optional[np.ndarray]):
        self._side_outlet_flow_tank_1 = value

    @property
    def side_outlet_flow_tank_2(self) -> Optional[np.ndarray]:
        return self._side_outlet_flow_tank_2

    @side_outlet_flow_tank_2.setter
    def side_outlet_flow_tank_2(self, value: Optional[np.ndarray]):
        self._side_outlet_flow_tank_2 = value

    @property
    def side_outlet_flow_tank_3(self) -> Optional[np.ndarray]:
        return self._side_outlet_flow_tank_3

    @side_outlet_flow_tank_3.setter
    def side_outlet_flow_tank_3(self, value: Optional[np.ndarray]):
        self._side_outlet_flow_tank_3 = value

    @property
    def Q_tank_0(self) -> Optional[np.ndarray]:
        return self._Q_tank_0

    @Q_tank_0.setter
    def Q_tank_0(self, value: Optional[np.ndarray]):
        self._Q_tank_0 = value

    @property
    def Q_tank_1(self) -> Optional[np.ndarray]:
        return self._Q_tank_1

    @Q_tank_1.setter
    def Q_tank_1(self, value: Optional[np.ndarray]):
        self._Q_tank_1 = value

    @property
    def Q_tank_2(self) -> Optional[np.ndarray]:
        return self._Q_tank_2

    @Q_tank_2.setter
    def Q_tank_2(self, value: Optional[np.ndarray]):
        self._Q_tank_2 = value

    @property
    def Q_tank_3(self) -> Optional[np.ndarray]:
        return self._Q_tank_3

    @Q_tank_3.setter
    def Q_tank_3(self, value: Optional[np.ndarray]):
        self._Q_tank_3 = value

    @property
    def area(self):
        return self._area

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value: BasinDefParams):
        self._params = value

    @area.setter
    def area(self, area: float):
        self._area = area


class Junction(BasinDef):
    def __init__(
            self,
            name: str
    ):
        super().__init__(name, BasinDefType.JUNCTION)


class Sink(BasinDef):
    def __init__(
            self,
            name: str
    ):
        super().__init__(name, BasinDefType.SINK)


@dataclasses.dataclass
class ReachParams(BasinDefParams):
    k: float = 0.01
    x: float = 0.01

    @classmethod
    def MaxBoundParams(cls):
        return cls(
            k=5.0,
            x=0.5
        )

    def to_list(self):
        return [
            self.k,
            self.x
        ]


class Reach(BasinDef):
    def __init__(
            self,
            name: str
    ):
        super().__init__(name, BasinDefType.REACH)
        self._params = ReachParams()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value: ReachParams):
        self._params = value
