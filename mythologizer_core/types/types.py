from typing import Any, Callable, Union
from numpy.typing import ArrayLike, NDArray
from uuid import UUID
import numpy as np

type Embedding = list[float] | NDArray[np.floating]
type Embeddings = list[Embedding] | NDArray[np.floating]
type id_type = int | str | UUID
type Mythmatrix = ArrayLike
type Weight = float | np.floating
type Weights = list[Weight] | NDArray[np.floating]

type EpochChangeFunction = Callable[[np.ndarray, Any, Any], np.ndarray]
type EmbeddingFunction = Callable[[str], Embedding]