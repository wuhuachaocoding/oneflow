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
#include "oneflow/core/stream/cuda_stream_context.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include <cuda.h>

namespace oneflow {

namespace primitive {

namespace {

constexpr size_t kMaxNumDims = 8;

Optional<cudaDataType_t> OptCudaDataType(DataType data_type) {
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

cudaDataType_t GetCudaDataType(DataType data_type) {
  auto cuda_data_type = OptCudaDataType(data_type);
  CHECK(cuda_data_type.has_value());
  return cuda_data_type.value_or(CUDA_R_32F);
}

union CublasScalarParameter {
  double d;
  float s;
};

CublasScalarParameter GetCublasScalarParameter(Scalar scalar, cublasComputeType_t compute_type) {
  CublasScalarParameter sp{};
  if (compute_type == CUBLAS_COMPUTE_64F) {
    sp.d = scalar.Value<double>();
  } else if (compute_type == CUBLAS_COMPUTE_32F) {
    sp.s = scalar.Value<float>();
  } else {
    UNIMPLEMENTED();
  }
  return sp;
}

cublasComputeType_t GetComputeType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUBLAS_COMPUTE_32F;
    case kDouble: return CUBLAS_COMPUTE_64F;
    case kFloat16: return CUBLAS_COMPUTE_32F;
#if CUDA_VERSION >= 11000
    case kBFloat16: return CUBLAS_COMPUTE_32F;
#endif  // CUDA_VERSION >= 11000
    default: UNIMPLEMENTED(); return CUBLAS_COMPUTE_32F;
  }
}

void LaunchCublasBroadcastMatmul(StreamContext* stream_ctx, DataType data_type,
                                 BlasTransposeType transpose_a, BlasTransposeType transpose_b,
                                 int64_t num_batch_dims, const int64_t* a_batch_dims,
                                 const int64_t* b_batch_dims, int64_t m, int64_t n, int64_t k,
                                 Scalar alpha, const void* a, const void* b, Scalar beta, void* c) {
  auto* cuda_stream_ctx = CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx));
  const auto cuda_data_type = GetCudaDataType(data_type);
  const auto cublas_compute_type = GetComputeType(data_type);
  const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_type);
  const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_type);
  const auto GetCublasOperation = [](BlasTransposeType transpose_type) {
    if (transpose_type == BlasTransposeType::N) {
      return CUBLAS_OP_T;
    } else if (transpose_type == BlasTransposeType::T) {
      return CUBLAS_OP_N;
    } else {
      UNIMPLEMENTED();
      return CUBLAS_OP_N;
    }
  };
  const cublasOperation_t cublas_trans_a = GetCublasOperation(transpose_b);
  const cublasOperation_t cublas_trans_b = GetCublasOperation(transpose_a);
  const int cublas_m = n;
  const int cublas_n = m;
  const int cublas_k = k;
  int cublas_lda = 0;
  if (transpose_b == BlasTransposeType::N) {
    cublas_lda = n;
  } else if (transpose_b == BlasTransposeType::T) {
    cublas_lda = k;
  } else {
    UNIMPLEMENTED();
  }
  int cublas_ldb = 0;
  if (transpose_a == BlasTransposeType::N) {
    cublas_ldb = k;
  } else if (transpose_a == BlasTransposeType::T) {
    cublas_ldb = m;
  } else {
    UNIMPLEMENTED();
  }
  const int cublas_ldc = n;
  if (num_batch_dims == 0) {
    const void* cublas_a = b;
    const void* cublas_b = a;
    void* cublas_c = c;
    OF_CUBLAS_CHECK(cublasGemmEx(cuda_stream_ctx->cublas_handle(), cublas_trans_a, cublas_trans_b,
                                 cublas_m, cublas_n, cublas_k, &sp_alpha, cublas_a, cuda_data_type,
                                 cublas_lda, cublas_b, cuda_data_type, cublas_ldb, &sp_beta,
                                 cublas_c, cuda_data_type, cublas_ldc, cublas_compute_type,
                                 CUBLAS_GEMM_DEFAULT));
  } else if (num_batch_dims == 1) {
    const void* cublas_a = b;
    const void* cublas_b = a;
    void* cublas_c = c;
    const int64_t a_batch_count = a_batch_dims[0];
    const int64_t b_batch_count = b_batch_dims[0];
    CHECK(a_batch_count == 1 || b_batch_count == 1 || a_batch_count == b_batch_count);
    CHECK_GT(a_batch_count, 0);
    CHECK_GT(b_batch_count, 0);
    const int batch_count = std::max(a_batch_count, b_batch_count);
    const long long int cublas_stride_a = b_batch_count == 1 ? 0 : cublas_m * cublas_k;
    const long long int cublas_stride_b = a_batch_count == 1 ? 0 : cublas_k * cublas_n;
    const long long int cublas_stride_c = cublas_m * cublas_n;
    OF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cuda_stream_ctx->cublas_handle(), cublas_trans_a, cublas_trans_b, cublas_m, cublas_n,
        cublas_k, &sp_alpha, cublas_a, cuda_data_type, cublas_lda, cublas_stride_a, cublas_b,
        cuda_data_type, cublas_ldb, cublas_stride_b, &sp_beta, cublas_c, cuda_data_type, cublas_ldc,
        cublas_stride_c, batch_count, cublas_compute_type, CUBLAS_GEMM_DEFAULT));
  } else {
    const int64_t stride_a = m * k;
    const int64_t stride_b = k * n;
    const int64_t stride_c = m * n;
    const size_t size_of_data_type = GetSizeOfDataType(data_type);
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
      const void* cublas_a =
          static_cast<const unsigned char*>(b) + b_batch_offset * stride_b * size_of_data_type;
      const void* cublas_b =
          static_cast<const unsigned char*>(a) + a_batch_offset * stride_a * size_of_data_type;
      void* cublas_c =
          static_cast<unsigned char*>(c) + c_batch_offset * stride_c * size_of_data_type;
      OF_CUBLAS_CHECK(cublasGemmEx(cuda_stream_ctx->cublas_handle(), cublas_trans_a, cublas_trans_b,
                                   cublas_m, cublas_n, cublas_k, &sp_alpha, cublas_a,
                                   cuda_data_type, cublas_lda, cublas_b, cuda_data_type, cublas_ldb,
                                   &sp_beta, cublas_c, cuda_data_type, cublas_ldc,
                                   cublas_compute_type, CUBLAS_GEMM_DEFAULT));
    }
  }
}

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

}  // namespace

class BroadcastMatmulImpl : public Matmul, public BatchMatmul, public BroadcastMatmul {
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

  void Launch(StreamContext* stream_ctx, size_t m, size_t n, size_t k, Scalar alpha, const void* a,
              const void* b, Scalar beta, void* c) override {
    Launch(stream_ctx, 1, m, n, k, alpha, a, b, beta, c);
  }

  // BatchMatmul
  void Launch(StreamContext* stream_ctx, size_t num_batches, size_t m, size_t n, size_t k,
              Scalar alpha, const void* a, const void* b, Scalar beta, void* c) override {
    int64_t a_dims[3];
    int64_t b_dims[3];
    a_dims[0] = num_batches;
    b_dims[0] = num_batches;
    if (transpose_a_ == BlasTransposeType::N) {
      a_dims[1] = m;
      a_dims[2] = k;
    } else if (transpose_a_ == BlasTransposeType::T) {
      a_dims[1] = k;
      a_dims[2] = m;
    } else {
      UNIMPLEMENTED();
    }
    if (transpose_b_ == BlasTransposeType::N) {
      b_dims[1] = k;
      b_dims[2] = n;
    } else if (transpose_b_ == BlasTransposeType::T) {
      b_dims[1] = n;
      b_dims[2] = k;
    } else {
      UNIMPLEMENTED();
    }
    Launch(stream_ctx, alpha, 3, a_dims, a, 3, b_dims, b, beta, c);
  }

  // BroadcastMatmul
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
    LaunchCublasBroadcastMatmul(stream_ctx, data_type_, transpose_a_, transpose_b_, num_batch_dims,
                                a_batch_dims, b_batch_dims, m, n, k, alpha, a, b, beta, c);
  }

 private:
  DataType data_type_;
  BlasTransposeType transpose_a_;
  BlasTransposeType transpose_b_;
};

class MatmulFactoryImpl : public MatmulFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MatmulFactoryImpl);
  MatmulFactoryImpl() = default;
  ~MatmulFactoryImpl() override = default;

  std::unique_ptr<Matmul> New(DataType data_type, BlasTransposeType transpose_a,
                              BlasTransposeType transpose_b) override {
    auto cuda_data_type = OptCudaDataType(data_type);
    if (cuda_data_type.has_value()) {
      return std::make_unique<BroadcastMatmulImpl>(data_type, transpose_a, transpose_b);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, MatmulFactory, MatmulFactoryImpl);

class BatchMatmulFactoryImpl : public BatchMatmulFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchMatmulFactoryImpl);
  BatchMatmulFactoryImpl() = default;
  ~BatchMatmulFactoryImpl() override = default;

  std::unique_ptr<BatchMatmul> New(DataType data_type, BlasTransposeType transpose_a,
                                   BlasTransposeType transpose_b) override {
    auto cuda_data_type = OptCudaDataType(data_type);
    if (cuda_data_type.has_value()) {
      return std::make_unique<BroadcastMatmulImpl>(data_type, transpose_a, transpose_b);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, BatchMatmulFactory, BatchMatmulFactoryImpl);

class BroadcastMatmulFactoryImpl : public BroadcastMatmulFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMatmulFactoryImpl);
  BroadcastMatmulFactoryImpl() = default;
  ~BroadcastMatmulFactoryImpl() override = default;

  std::unique_ptr<BroadcastMatmul> New(DataType data_type, BlasTransposeType transpose_a,
                                       BlasTransposeType transpose_b,
                                       size_t max_num_dims) override {
    auto cuda_data_type = OptCudaDataType(data_type);
    if (max_num_dims <= kMaxNumDims && cuda_data_type.has_value()) {
      return std::make_unique<BroadcastMatmulImpl>(data_type, transpose_a, transpose_b);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, BroadcastMatmulFactory, BroadcastMatmulFactoryImpl);

}  // namespace primitive

}  // namespace oneflow

#endif  // WITH_CUDA
