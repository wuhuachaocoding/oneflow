"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
from oneflow.framework.tensor import register_tensor_op


@register_tensor_op("roll")
def roll_op(input, shifts, dims=None):
    """Roll the tensor along the given dimension(s). 
    
    Elements that are shifted beyond the last position are re-introduced at the first position. 
    
    If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.

    Args:
        input (oneflow.Tensor): The input Tensor.
        shifts (int or tuple of python:ints): The number of places by which the elements of the tensor are shifted. 
                                              If shifts is a tuple, dims must be a tuple of the same size, 
                                              and each dimension will be rolled by the corresponding value.
        dims (int or tuple of python:ints): Axis along which to roll.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[1, 2],
        ...               [3, 4],
        ...               [5, 6],
        ...               [7, 8]])
        >>> input = flow.Tensor(x)
        >>> input.shape
        oneflow.Size([4, 2])
        >>> out = flow.roll(input, 1, 0)
        >>> out.shape
        oneflow.Size([4, 2])

    """
    assert isinstance(shifts, (int, tuple))
    if dims is not None:
        assert isinstance(dims, (int, tuple))

    if input.dtype == flow.int8:
        input = flow.cast(input, flow.int32)
    shifts = [shifts] if isinstance(shifts, int) else list(shifts)
    dims = [-1] if dims is None else [dims] if isinstance(dims, int) else list(dims)

    return flow._C.roll(input, shifts, dims)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
