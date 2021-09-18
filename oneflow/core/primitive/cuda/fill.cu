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
#include "oneflow/core/primitive/fill.h"
#include "oneflow/core/primitive/cuda/type_seq.h"
#include "oneflow/core/primitive/cuda/cuda_graph_support.h"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow {

namespace primitive {

namespace {

template<typename T>
__global__ void FillGpu(T* dst, T value, size_t count) {
  CUDA_1D_KERNEL_LOOP_T(size_t, i, count) { dst[i] = value; }
}

template<typename T>
class FillImpl : public Fill, public CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FillImpl);
  FillImpl() = default;
  ~FillImpl() override = default;

  void Launch(StreamContext* stream_ctx, void* dst, Scalar value, size_t count) override {
    cudaStream_t cuda_stream =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
    FillGpu<T><<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
        reinterpret_cast<T*>(dst), CHECK_JUST(value.As<T>()), count);
  }
};

template<typename T>
std::unique_ptr<Fill> NewFill() {
  return std::unique_ptr<Fill>(new FillImpl<T>());
}

class FillFactoryImpl : public FillFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FillFactoryImpl);
  FillFactoryImpl() = default;
  ~FillFactoryImpl() override = default;

  std::unique_ptr<Fill> New(DataType data_type) override {
#define MAKE_NEW_FILL_ENTRY(type_cpp, type_proto) {type_proto, NewFill<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<Fill>()>> new_fill_handle{
        OF_PP_FOR_EACH_TUPLE(MAKE_NEW_FILL_ENTRY, CUDA_PRIMITIVE_ALL_TYPE_SEQ)};

#undef MAKE_NEW_FILL_ENTRY

    const auto it = new_fill_handle.find(data_type);
    if (it != new_fill_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, FillFactory, FillFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
