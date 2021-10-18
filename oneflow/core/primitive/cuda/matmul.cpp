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

namespace oneflow {

namespace primitive {

namespace {

constexpr size_t kMaxNumDims = 8;

void SimplifyBroadcastMatmul(size_t num_a_dims, int64_t* a_dims,
                             const void* a, size_t num_b_dims, int64_t* b_dims) {

}

}

Optional<cudaDataType_t> GetCudaDataType(DataType data_type) {
  switch (data_type) {
    case kFloat: return CUDA_R_32F;
    case kDouble: return CUDA_R_64F;
    case kFloat16: return CUDA_R_16F;
    case kBFloat16: return CUDA_R_16BF;
    default: return NullOpt;
  }
}

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

  void Launch(StreamContext* stream_ctx, Scalar alpha, size_t num_a_dims, int64_t* a_dims,
              const void* a, size_t num_b_dims, int64_t* b_dims, const void* b, Scalar beta,
              void* c) override {}

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
