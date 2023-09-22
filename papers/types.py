from typing import Any

import flax
import numpy as np

DataType = np.ndarray | dict[str, "DataType"]
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
