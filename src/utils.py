import uuid
import numpy as np


def get_id() -> str:
    id: str = str(uuid.uuid4())
    return id


def convert_numpy_array_to_bytes(arr: np.ndarray) -> bytes:
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)  # a 2d array.
    data_bytes: bytes = arr.tobytes()
    return data_bytes


def convert_bytes_to_numpy_array(
    data_bytes: bytes, arr_type: np.dtype, vector_dimension: int
) -> np.ndarray:
    arr: np.ndarray = np.frombuffer(data_bytes, dtype=arr_type).reshape(
        -1, vector_dimension
    )  # a 2d array
    return arr
