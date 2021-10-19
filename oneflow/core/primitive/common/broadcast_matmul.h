/*
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
*/
#ifndef ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_MATMUL_H_
#define ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_MATMUL_H_

#include "oneflow/core/primitive/include/broadcast_matmul.h"

namespace oneflow {

namespace primitive {

namespace broadcast_matmul {

inline void Simplify(size_t num_a_dims, const int64_t* a_dims, size_t num_b_dims,
                     const int64_t* b_dims, BlasTransposeType transpose_a,
                     BlasTransposeType transpose_b, int64_t* m, int64_t* n, int64_t* k,
                     int64_t* num_batch_dims, int64_t* a_batch_dims, int64_t* b_batch_dims) {
  CHECK_GE(num_a_dims, 2);
  CHECK_GE(num_b_dims, 2);
  if (transpose_a == BlasTransposeType::N) {
    *m = a_dims[num_a_dims - 2];
    *k = a_dims[num_a_dims - 1];
  } else if (transpose_a == BlasTransposeType::T) {
    *m = a_dims[num_a_dims - 1];
    *k = a_dims[num_a_dims - 2];
  } else {
    UNIMPLEMENTED();
  }
  if (transpose_b == BlasTransposeType::N) {
    CHECK_EQ(b_dims[num_b_dims - 2], *k);
    *n = b_dims[num_b_dims - 1];
  } else if (transpose_b == BlasTransposeType::T) {
    CHECK_EQ(b_dims[num_b_dims - 1], *k);
    *n = b_dims[num_b_dims - 2];
  } else {
    UNIMPLEMENTED();
  }
  const int64_t num_a_batch_dims = num_a_dims - 2;
  const int64_t num_b_batch_dims = num_b_dims - 2;
  const int64_t num_max_batch_dims = std::max(num_a_batch_dims, num_b_batch_dims);
  const int64_t num_a_padding_dims = num_max_batch_dims - num_a_batch_dims;
  const int64_t num_b_padding_dims = num_max_batch_dims - num_b_batch_dims;
  *num_batch_dims = 0;
  for (int64_t i = 0; i < num_max_batch_dims; ++i) {
    const int64_t a_dim = i < num_a_padding_dims ? 1 : a_dims[i - num_a_padding_dims];
    const int64_t b_dim = i < num_b_padding_dims ? 1 : b_dims[i - num_b_padding_dims];
    if (a_dim == 1 && b_dim == 1) {
      continue;
    } else if (*num_batch_dims != 0
               && ((a_dim == 1 && a_batch_dims[*num_batch_dims - 1] == 1)
                   || (b_dim == 1 && b_batch_dims[*num_batch_dims - 1] == 1)
                   || (a_dim == b_dim
                       && a_batch_dims[*num_batch_dims - 1]
                              == b_batch_dims[*num_batch_dims - 1]))) {
      a_batch_dims[*num_batch_dims - 1] *= a_dim;
      b_batch_dims[*num_batch_dims - 1] *= b_dim;
    } else {
      CHECK(a_dim == b_dim || a_dim == 1 || b_dim == 1);
      a_batch_dims[*num_batch_dims] = a_dim;
      b_batch_dims[*num_batch_dims] = b_dim;
      *num_batch_dims += 1;
    }
  }
  if (*num_batch_dims >= 1 && a_batch_dims[*num_batch_dims - 1] != 1
      && b_batch_dims[*num_batch_dims - 1] == 1 && transpose_a == BlasTransposeType::N) {
    *m *= a_batch_dims[*num_batch_dims - 1];
    *num_batch_dims -= 1;
  }
}

}  // namespace broadcast_matmul

}  // namespace primitive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_MATMUL_H_
