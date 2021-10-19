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
#ifdef WITH_CUDA

#include "oneflow/core/primitive/include/primitive.h"
#include "oneflow/core/primitive/include/matmul.h"
#include "oneflow/core/primitive/include/batch_matmul.h"
#include "oneflow/core/primitive/include/broadcast_matmul.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/device/cuda_util.h"
#include <cuda.h>

namespace oneflow {

namespace primitive {

namespace {

Optional<cudaDataType_t> GetCudaDataType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUDA_R_32F;
    case kDouble: return CUDA_R_64F;
    case kFloat16: return CUDA_R_16F;
#if CUDA_VERSION >= 11000
    case kBFloat16: return CUDA_R_16BF;
#endif  // CUDA_VERSION >= 11000
    default: return NullOpt;
  }
}

Optional<cublasComputeType_t> GetComputeType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUBLAS_COMPUTE_32F;
    case kDouble: return CUBLAS_COMPUTE_64F;
    case kFloat16: return CUBLAS_COMPUTE_32F;
#if CUDA_VERSION >= 11000
    case kBFloat16: return CUBLAS_COMPUTE_32F;
#endif  // CUDA_VERSION >= 11000
    default: return NullOpt;
  }
}

void GetCublasGemmExArgs(BlasTransposeType transpose_a, BlasTransposeType transpose_b,
                         int64_t num_a_batches, int64_t num_b_batches, int64_t m, int64_t n,
                         int64_t k, const void* a, const void* b, cublasOperation_t* cublas_trans_a,
                         cublasOperation_t* cublas_trans_b, int* cublas_m, int* cublas_n,
                         int* cublas_k, const void** cublas_a, int* cublas_lda,
                         long long int* cublas_stride_a, const void** cublas_b, int* cublas_ldb,
                         long long int* cublas_stride_b, int* cublas_ldc,
                         long long int* cublas_stride_c) {
  CHECK(num_a_batches == 1 || num_b_batches == 1 || num_a_batches == num_b_batches);
  const auto ToCublasOperation = [](BlasTransposeType transpose_type,
                                    cublasOperation_t* cublas_trans) {
    if (transpose_type == BlasTransposeType::N) {
      *cublas_trans = CUBLAS_OP_T;
    } else if (transpose_type == BlasTransposeType::T) {
      *cublas_trans = CUBLAS_OP_N;
    } else {
      UNIMPLEMENTED();
    }
  };
  if (cublas_trans_a != nullptr) { ToCublasOperation(transpose_b, cublas_trans_a); }
  if (cublas_trans_b != nullptr) { ToCublasOperation(transpose_a, cublas_trans_b); }
  if (cublas_m != nullptr) { *cublas_m = n; }
  if (cublas_n != nullptr) { *cublas_n = m; }
  if (cublas_k != nullptr) { *cublas_k = k; }
  if (cublas_a != nullptr) { *cublas_a = b; }
  if (cublas_lda != nullptr) {
    if (transpose_b == BlasTransposeType::N) {
      *cublas_lda = n;
    } else if (transpose_b == BlasTransposeType::T) {
      *cublas_lda = k;
    } else {
      UNIMPLEMENTED();
    }
  }
  if (cublas_stride_a != nullptr) {
    if (num_b_batches == 1) {
      *cublas_stride_a = 0;
    } else {
      *cublas_stride_a = n * k;
    }
  }
  if (cublas_b != nullptr) { *cublas_b = a; }
  if (cublas_ldb != nullptr) {
    if (transpose_a == BlasTransposeType::N) {
      *cublas_ldb = k;
    } else if (transpose_a == BlasTransposeType::T) {
      *cublas_ldb = m;
    } else {
      UNIMPLEMENTED();
    }
  }
  if (cublas_stride_b != nullptr) {
    if (num_a_batches == 1) {
      *cublas_stride_b = 0;
    } else {
      *cublas_stride_b = k * m;
    }
  }
  if (cublas_ldc != nullptr) { *cublas_ldc = n; }
  if (cublas_stride_c != nullptr) {
    if (num_a_batches == 1 && num_b_batches == 1) {
      *cublas_stride_c = 0;
    } else {
      *cublas_stride_c = n * m;
    }
  }
}

constexpr size_t kMaxNumDims = 8;

void SimplifyBroadcastMatmul(size_t num_a_dims, const int64_t* a_dims, size_t num_b_dims,
                             const int64_t* b_dims, BlasTransposeType transpose_a,
                             BlasTransposeType transpose_b, int64_t* m, int64_t* n, int64_t* k,
                             int64_t* num_batch_dims, int64_t* a_batch_dims,
                             int64_t* b_batch_dims) {
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

void DoMatmul(StreamContext* stream_ctx, DataType data_type, int64_t m, int64_t n, int64_t k,
              Scalar alpha, const void* a, const void* b, Scalar beta, void* c) {}

void DoBatchMatmul(StreamContext* stream_ctx, DataType data_type, int64_t num_a_batches,
                   int64_t num_b_batches, int64_t m, int64_t n, int64_t k, Scalar alpha,
                   const void* a, const void* b, Scalar beta, void* c) {}

void DoBroadcastMatmul(StreamContext* stream_ctx, DataType data_type, int64_t num_batch_dims,
                       const int64_t* a_batch_dims, const int64_t* b_batch_dims, int64_t m,
                       int64_t n, int64_t k, Scalar alpha, const void* a, const void* b,
                       Scalar beta, void* c) {}

}  // namespace

class BroadcastMatmulImpl : public BroadcastMatmul {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMatmulImpl);
  BroadcastMatmulImpl(DataType data_type, BlasTransposeType transpose_a,
                      BlasTransposeType transpose_b)
      : data_type_(data_type), transpose_a_(transpose_a), transpose_b_(transpose_b) {}
  ~BroadcastMatmulImpl() override = default;

  DataType a_type() const override { return data_type_; }
  DataType b_type() const override { return data_type_; }
  DataType c_type() const override { return data_type_; }
  BlasTransposeType transpose_a() const override { return transpose_a_; }
  virtual BlasTransposeType transpose_b() const override { return transpose_b_; }

  void Launch(StreamContext* stream_ctx, Scalar alpha, size_t num_a_dims, const int64_t* a_dims,
              const void* a, size_t num_b_dims, const int64_t* b_dims, const void* b, Scalar beta,
              void* c) override {
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t num_batch_dims = 0;
    int64_t a_batch_dims[kMaxNumDims]{};
    int64_t b_batch_dims[kMaxNumDims]{};
    SimplifyBroadcastMatmul(num_a_dims, a_dims, num_b_dims, b_dims, transpose_a_, transpose_b_, &m,
                            &n, &k, &num_batch_dims, a_batch_dims, b_batch_dims);
    if (num_batch_dims == 0) {
      DoMatmul(stream_ctx, data_type_, m, n, k, alpha, a, b, beta, c);
    } else if (num_batch_dims == 1) {
      DoBatchMatmul(stream_ctx, data_type_, a_batch_dims[0], b_batch_dims[0], m, n, k, alpha, a, b,
                    beta, c);
    } else {
      // DoBroadcastMatmul();
    }
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
    auto cuda_data_type = GetCudaDataType(data_type);
    if (max_num_dims <= kMaxNumDims && cuda_data_type.has_value()) {
      return std::make_unique<BroadcastMatmulImpl>(data_type, transpose_a, transpose_b);
    } else {
      return nullptr;
    }
  }
};

}  // namespace primitive

}  // namespace oneflow

#endif  // WITH_CUDA
