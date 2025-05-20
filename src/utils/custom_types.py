import numpy as np
from typing import Union


DataType = Union[np.ndarray, dict[str, "DataType"]]

DatasetDict = dict[str, DataType]
PRNGKey = object
