from typing import Any, Union

import flax
import numpy as np

DataType = Union[np.ndarray, dict[str, "DataType"]]
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
