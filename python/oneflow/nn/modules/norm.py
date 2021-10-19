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
from oneflow.nn.module import Module


class Vector_Norm(Module):
    def __init__(self, ord=2, dim=None, keepdim=False, dtype=None) -> None:
        super().__init__()
        self.ord=ord
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype

    def forward(self, x):
        return flow._C.vector_norm(x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, dtype = self.dtype)
        

class Matrix_Norm(Module):
    def __init__(self, ord="fro", dim=(-2, -1), keepdim=False, dtype=None) -> None:
        super().__init__()
        self.ord = ord           
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype

    def forward(self, x):  
        return flow._C.matrix_norm(x, self.ord, self.dim, self.keepdim, dtype=self.dtype)

        
def norm_op(input, ord=None, dim=None, keepdim=False, dtype=None):
    """linalg.norm(input, ord=None, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor
    Returns the matrix norm or vector norm of a given tensor.

    This function can calculate one of eight different types of matrix norms, or one
    of an infinite number of vector norms, depending on both the number of reduction
    dimensions and the value of the `ord` parameter.

    Args:
        input (Tensor): The input tensor. If dim is None, input must be 1-D or 2-D, unless :attr:`ord`
            is None. If both :attr:`dim` and :attr:`ord` are None, the 2-norm of the input flattened to 1-D
            will be returned. Its data type must be either a floating point or complex type. For complex
            inputs, the norm is calculated on of the absolute values of each element. If the input is
            complex and neither :attr:`dtype` nor :attr:`out` is specified, the result's data type will
            be the corresponding floating point type (e.g. float if :attr:`input` is complexfloat).
        ord (int, float, inf, -inf, 'fro', 'nuc', optional): The order of norm.
            inf refers to :attr:`float('inf')`, numpy's :attr:`inf` object, or any equivalent object.
            The following norms can be calculated:
            =====  ============================  ==========================
            ord    norm for matrices             norm for vectors
            =====  ============================  ==========================
            None   Frobenius norm                2-norm
            'fro'  Frobenius norm                -- not supported --
            'nuc'  -- not supported yet --       -- not supported --
            inf    max(sum(abs(x), dim=1))       max(abs(x))
            -inf   min(sum(abs(x), dim=1))       min(abs(x))
            0      -- not supported --           sum(x != 0)
            1      max(sum(abs(x), dim=0))       as below
            -1     min(sum(abs(x), dim=0))       as below
            2      -- not supported yet --       as below
            -2     -- not supported yet --       as below
            other  -- not supported --           sum(abs(x)**ord)**(1./ord)
            =====  ============================  ==========================
            Default: ``None``
        dim (int, 2-tuple of ints, 2-list of ints, optional): If :attr:`dim` is an int,
            vector norm will be calculated over the specified dimension. If :attr:`dim`
            is a 2-tuple of ints, matrix norm will be calculated over the specified
            dimensions. If :attr:`dim` is None, matrix norm will be calculated
            when the input tensor has two dimensions, and vector norm will be
            calculated when the input tensor has one dimension. Default: ``None``
        keepdim (bool, optional): If set to True, the reduced dimensions are retained
            in the result as dimensions with size one. Default: ``False``
        out (Tensor, optional): The output tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> from oneflow import linalg as LA
        >>> import numpy as np
        >>> a = flow.tensor(np.arange(9, dtype=np.float32) - 4)
        >>> a
        tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.], dtype=oneflow.float32)
        >>> b = a.reshape(3, 3)
        >>> b
        tensor([[-4., -3., -2.],
                [-1.,  0.,  1.],
                [ 2.,  3.,  4.]], dtype=oneflow.float32)
        >>> LA.norm(a)
        tensor(7.7460, dtype=oneflow.float32)
        >>> LA.norm(b)
        tensor(7.7460, dtype=oneflow.float32)
        >>> LA.norm(b, 'fro')
        tensor(7.7460, dtype=oneflow.float32)
        >>> LA.norm(a, float('inf'))
        tensor(4., dtype=oneflow.float32)
        >>> LA.norm(b, float('inf'))
        tensor(9., dtype=oneflow.float32)
        >>> LA.norm(a, -float('inf'))
        tensor(0., dtype=oneflow.float32)
        >>> LA.norm(b, -float('inf'))
        tensor(2., dtype=oneflow.float32)
        >>> LA.norm(a, 1)
        tensor(20., dtype=oneflow.float32)
        >>> LA.norm(b, 1)
        tensor(7., dtype=oneflow.float32)
        >>> LA.norm(a, -1)
        tensor(0., dtype=oneflow.float32)
        >>> LA.norm(b, -1)
        tensor(6., dtype=oneflow.float32)
        >>> LA.norm(a, 2)
        tensor(7.7460, dtype=oneflow.float32)
        >>> LA.norm(a, -2)
        tensor(0., dtype=oneflow.float32)
        >>> LA.norm(a, 3)
        tensor(5.8480, dtype=oneflow.float32)
        >>> LA.norm(a, -3)
        tensor(0., dtype=oneflow.float32)
        >>> c = flow.tensor([[1., 2., 3.],
        ...                   [-1, 1, 4]])
        >>> LA.norm(c, dim=0)
        tensor([1.4142, 2.2361, 5.0000], dtype=oneflow.float32)
        >>> LA.norm(c, dim=1, keepdim = True)
        tensor([[3.7417],
                [4.2426]], dtype=oneflow.float32)
        >>> LA.norm(c, ord=1, dim=1)
        tensor([6., 6.], dtype=oneflow.float32)
        >>> m = flow.tensor(np.arange(8, dtype=np.float32)).reshape(2, 2, 2)
        >>> LA.norm(m, dim=(1,2))
        tensor([ 3.7417, 11.2250], dtype=oneflow.float32)
    """
    return flow._C.norm(input, ord, dim, keepdim, dtype=dtype)

def vector_norm_tensor_op(input, ord=2, dim=None, keepdim=False, dtype=None):
    """
    linalg.vector_norm(input, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor

    Computes a vector norm.

    Supports input of float, double dtypes.

    This function does not necessarily treat multidimensonal attr:`input` as a batch of
    vectors, instead:

    - If :attr:`dim`\\ `= None`, :attr:`input` will be flattened before the norm is computed.
    - If :attr:`dim` is an `int` or a `tuple`, the norm will be computed over these dimensions and the other dimensions will be treated as batch dimensions.

    This behavior is for consistency with :func:`flow.linalg.norm`.

    :attr:`ord` defines the vector norm that is computed. The following norms are supported:

    ======================   ========================================================
    :attr:`ord`              vector norm
    ======================   ========================================================
    `2` (default)            `2`-norm (see below)
    `inf`                    `max(abs(x))`
    `-inf`                   `min(abs(x))`
    `0`                      `sum(x != 0)`
    other `int` or `float`   `sum(abs(x)^{ord})^{(1 / ord)}`
    ======================   ========================================================

    where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.


    Args:
        input (Tensor): tensor, flattened by default, but this behavior can be
            controlled using :attr:`dim`.
        ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `2`
        dim (int, Tuple[int], optional): dimensions over which to compute
            the norm. See above for the behavior when :attr:`dim`\\ `= None`.
            Default: `None`
        keepdim (bool, optional): If set to `True`, the reduced dimensions are retained
            in the result as dimensions with size one. Default: `False`

    Returns:
        A real-valued tensor.

    Examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> from oneflow import linalg as LA
        >>> import numpy as np
        >>> a = flow.tensor(np.arange(9, dtype=np.float32) - 4)
        >>> a
        tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.], dtype=oneflow.float32)
        >>> b = a.reshape(3, 3)
        >>> b
        tensor([[-4., -3., -2.],
                [-1.,  0.,  1.],
                [ 2.,  3.,  4.]], dtype=oneflow.float32)
        >>> LA.vector_norm(a, ord=3.5)
        tensor(5.4345, dtype=oneflow.float32)
        >>> LA.vector_norm(b, ord=3.5)
        tensor(5.4345, dtype=oneflow.float32)
    """
    return Vector_Norm(ord, dim, keepdim, dtype)(input)


def matrix_norm_tensor_op(input, ord="fro", dim=(-2, -1), keepdim=False,dtype=None):
    """
    linalg.matrix_norm(input, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None) -> Tensor

    Computes a matrix norm.

    Support input of float, double, cfloat and cdouble dtypes.
    Also supports batches of matrices: the norm will be computed over the
    dimensions specified by the 2-tuple :attr:`dim` and the other dimensions will
    be treated as batch dimensions. The output will have the same batch dimensions.

    :attr:`ord` defines the matrix norm that is computed. The following norms are supported:

    ======================   ========================================================
    :attr:`ord`              matrix norm
    ======================   ========================================================
    `'fro'` (default)        Frobenius norm
    `'nuc'`                  -- not supported yet --
    `inf`                    `max(sum(abs(x), dim=1))`
    `-inf`                   `min(sum(abs(x), dim=1))`
    `1`                      `max(sum(abs(x), dim=0))`
    `-1`                     `min(sum(abs(x), dim=0))`
    `2`                      'largest singular value'
    `-2`                     'smallest singular value'
    ======================   ========================================================

    where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

    Args:
        input (Tensor): tensor with two or more dimensions. By default its
            shape is interpreted as `(*, m, n)` where `*` is zero or more
            batch dimensions, but this behavior can be controlled using :attr:`dim`.
        ord (int, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `'fro'`
        dim (Tuple[int, int], optional): dimensions over which to compute the norm. Default: `(-2, -1)`
        keepdim (bool, optional): If set to `True`, the reduced dimensions are retained
            in the result as dimensions with size one. Default: `False`


    Returns:
        A real-valued tensor.

    Examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> from oneflow import linalg as LA
        >>> import numpy as np
        >>> a = flow.tensor(np.arange(9, dtype=np.float32)).reshape(3,3)
        >>> a
        tensor([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]], dtype=oneflow.float32)
        >>> LA.matrix_norm(a)
        tensor(14.2829, dtype=oneflow.float32)
        >>> LA.matrix_norm(a, ord=-1)
        tensor(9., dtype=oneflow.float32)
        >>> b = a.expand(2, -1, -1)
        >>> b
        tensor([[[0., 1., 2.],
                 [3., 4., 5.],
                 [6., 7., 8.]],
        <BLANKLINE>
                [[0., 1., 2.],
                 [3., 4., 5.],
                 [6., 7., 8.]]], dtype=oneflow.float32)
        >>> LA.matrix_norm(b)
        tensor([14.2829, 14.2829], dtype=oneflow.float32)
        >>> LA.matrix_norm(b, dim=(0, 2))
        tensor([ 3.1623, 10.0000, 17.2627], dtype=oneflow.float32)
    """
    return Matrix_Norm(ord, dim, keepdim,dtype)(input)


def l2_normalize(input, dim=0, epsilon=1e-12):
    """Use L2 norm to normalizes along dimension `dim`

    The equation is:

    .. math::
        out = \\frac{x}{max(\\sqrt{\\Sigma{x^2}}, \\epsilon)}

    Args:
        input (oneflow.Tensor): Input Tensor
        dim (int): The axis on which to apply L2 normalization. Defaults to 0.
        epsilon (float, optional): The epsilon value is used to avoid division by zero. Defaults to 1e-12.

    Returns:
        oneflow.Tensor: The normalized Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([[1, 2], [3, 4]], dtype=flow.float32)
        >>> out = flow.nn.functional.l2_normalize(x, 0)
        >>> out
        tensor([[0.3162, 0.4472],
                [0.9487, 0.8944]], dtype=oneflow.float32)
        >>> out = flow.nn.functional.l2_normalize(x, 1)
        >>> out
        tensor([[0.4472, 0.8944],
                [0.6000, 0.8000]], dtype=oneflow.float32)

    """
    y, _ = flow._C.l2_normalize(input, dim, epsilon)
    return y


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
