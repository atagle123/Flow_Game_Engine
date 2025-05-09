import numpy as np
from typing import Dict, Iterable, Optional, Tuple, Union, Any


DataType = Union[np.ndarray, Dict[str, "DataType"]]

DatasetDict = Dict[str, DataType]
PRNGKey = Any
