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
#include "oneflow/core/primitive/include/primitive.h"
#include "oneflow/core/primitive/include/broadcast_matmul.h"
#include "oneflow/core/primitive/common/broadcast_matmul.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/common/blas.h"

namespace oneflow {

namespace primitive {

namespace {

constexpr size_t kMaxNumDims = 8;

CBLAS_TRANSPOSE GetCblasTranspose(BlasTransposeType transpose_type) {
  if (transpose_type == BlasTransposeType::N) {
    return CblasNoTrans;
  } else if (transpose_type == BlasTransposeType::T) {
    return CblasTrans;
  } else {
    UNIMPLEMENTED();
    return CblasNoTrans;
  }
}

template<typename T>
void CblasMatmul(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b, int m, int n, int k, T alpha,
                 const T* a, const T* b, T beta, T* c) {
  int lda = 0;
  if (trans_a == CblasNoTrans) {
    lda = k;
  } else if (trans_a == CblasTrans) {
    lda = m;
  } else {
    UNIMPLEMENTED();
  }
  int ldb = 0;
  if (trans_b == CblasNoTrans) {
    ldb = n;
  } else if (trans_b == CblasTrans) {
    ldb = k;
  } else {
    UNIMPLEMENTED();
  }
  const int ldc = n;
  cblas_gemm<T>(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template<typename T>
void LaunchCblasBroadcastMatmul(StreamContext* stream_ctx, DataType data_type,
                                BlasTransposeType transpose_a, BlasTransposeType transpose_b,
                                int64_t num_batch_dims, const int64_t* a_batch_dims,
                                const int64_t* b_batch_dims, int64_t m, int64_t n, int64_t k,
                                Scalar alpha, const T* a, const T* b, Scalar beta, T* c) {
  const CBLAS_TRANSPOSE cblas_trans_a = GetCblasTranspose(transpose_a);
  const CBLAS_TRANSPOSE cblas_trans_b = GetCblasTranspose(transpose_b);
  const T alpha_value = alpha.Value<T>();
  const T beta_value = beta.Value<T>();
  if (num_batch_dims == 0) {
    CblasMatmul<T>(cblas_trans_a, cblas_trans_b, m, n, k, alpha_value, a, b, beta_value, c);
  } else {
    const int64_t stride_a = m * k;
    const int64_t stride_b = k * n;
    const int64_t stride_c = m * n;
    int64_t c_batch_count = 1;
    int64_t c_batch_dims[kMaxNumDims];
    for (int64_t i = 0; i < num_batch_dims; ++i) {
      const int64_t a_batch_dim = a_batch_dims[i];
      const int64_t b_batch_dim = b_batch_dims[i];
      CHECK(a_batch_dim == 1 || b_batch_dim == 1 || a_batch_dim == b_batch_dim);
      c_batch_dims[i] = std::max(a_batch_dim, b_batch_dim);
      c_batch_count *= c_batch_dims[i];
    }
    NdIndexOffsetHelper<int64_t, kMaxNumDims> c_index_helper(c_batch_dims, num_batch_dims);
    NdIndexOffsetHelper<int64_t, kMaxNumDims> a_index_helper(a_batch_dims, num_batch_dims);
    NdIndexOffsetHelper<int64_t, kMaxNumDims> b_index_helper(b_batch_dims, num_batch_dims);
    int64_t c_batch_index[kMaxNumDims];
    int64_t a_batch_index[kMaxNumDims];
    int64_t b_batch_index[kMaxNumDims];
    for (int64_t c_batch_offset = 0; c_batch_offset < c_batch_count; ++c_batch_offset) {
      c_index_helper.OffsetToNdIndex(c_batch_offset, c_batch_index);
      for (int64_t i = 0; i < num_batch_dims; ++i) {
        if (a_batch_dims[i] == 1) {
          a_batch_index[i] = 0;
        } else {
          a_batch_index[i] = c_batch_dims[i];
        }
        if (b_batch_dims[i] == 1) {
          b_batch_index[i] = 0;
        } else {
          b_batch_index[i] = c_batch_dims[i];
        }
      }
      const int64_t a_batch_offset = a_index_helper.NdIndexToOffset(a_batch_index);
      const int64_t b_batch_offset = b_index_helper.NdIndexToOffset(b_batch_index);
      CblasMatmul<T>(cblas_trans_a, cblas_trans_b, m, n, k, alpha_value,
                     a + a_batch_offset * stride_a, b + b_batch_offset * stride_b, beta_value,
                     c + c_batch_offset * stride_c);
    }
  }
}

template<typename T>
class BroadcastMatmulImpl : public BroadcastMatmul {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMatmulImpl);
  BroadcastMatmulImpl(DataType data_type, BlasTransposeType transpose_a,
                      BlasTransposeType transpose_b)
      : data_type_(data_type), transpose_a_(transpose_a), transpose_b_(transpose_b) {}
  ~BroadcastMatmulImpl() override = default;

  void Launch(StreamContext* stream_ctx, Scalar alpha, size_t num_a_dims, const int64_t* a_dims,
              const void* a, size_t num_b_dims, const int64_t* b_dims, const void* b, Scalar beta,
              void* c) override {
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t num_batch_dims = 0;
    int64_t a_batch_dims[kMaxNumDims]{};
    int64_t b_batch_dims[kMaxNumDims]{};
    broadcast_matmul::Simplify(num_a_dims, a_dims, num_b_dims, b_dims, transpose_a_, transpose_b_,
                               &m, &n, &k, &num_batch_dims, a_batch_dims, b_batch_dims);
    LaunchCblasBroadcastMatmul<T>(stream_ctx, data_type_, transpose_a_, transpose_b_,
                                  num_batch_dims, a_batch_dims, b_batch_dims, m, n, k, alpha,
                                  static_cast<const T*>(a), static_cast<const T*>(b), beta,
                                  static_cast<T*>(c));
  }

 private:
  DataType data_type_;
  BlasTransposeType transpose_a_;
  BlasTransposeType transpose_b_;
};

class BroadcastMatmulFactoryImpl : public BroadcastMatmulFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMatmulFactoryImpl);
  BroadcastMatmulFactoryImpl() = default;
  ~BroadcastMatmulFactoryImpl() override = default;

  std::unique_ptr<BroadcastMatmul> New(DataType data_type, BlasTransposeType transpose_a,
                                       BlasTransposeType transpose_b,
                                       size_t max_num_dims) override {
    if (max_num_dims > kMaxNumDims) { return nullptr; }
    if (data_type == DataType::kFloat) {
      return std::make_unique<BroadcastMatmulImpl<float>>(data_type, transpose_a, transpose_b);
    } else if (data_type == DataType::kDouble) {
      return std::make_unique<BroadcastMatmulImpl<double>>(data_type, transpose_a, transpose_b);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, BroadcastMatmulFactory, BroadcastMatmulFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
